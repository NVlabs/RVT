# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

import rvt.mvt.utils as mvt_utils
from rvt.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding,
)
from rvt.mvt.raft_utils import ConvexUpSample



class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        rend_three_views,
        use_point_renderer,
        pe_fix,
        feat_ver,
        wpt_img_aug,
        inp_pre_pro,
        inp_pre_con,
        cvx_up,
        xops,
        rot_ver,
        num_rot,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param norm_corr: wether or not to normalize the correspondece values.
            this matters when pc is outide -1, 1 like for the two stage mvt
        :param add_pixel_loc:
        :param add_depth:
        :param rend_three_views: True/False. Render only three views,
            i.e. top, right and front. Overwrites other configurations.
        :param use_point_renderer: whether to use the point renderer or not
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        :param feat_ver: whether to max pool final features or use soft max
            values using the heamtmap
        :param wpt_img_aug: how much noise is added to the wpt_img while
            training, expressed as a percentage of the image size
        :param inp_pre_pro: whether or not we have the intial input
            preprocess layer. this was used in peract but not not having it has
            cost advantages. if false, we replace the 1D convolution in the
            orginal design with identity
        :param inp_pre_con: whether or not the output of the inital
            preprocessing layer is concatenated with the ouput of the
            upsampling layer for the "final" layer
        :param cvx_up: whether to use learned convex upsampling
        :param xops: whether to use xops or not
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param num_rot: number of discrete rotations per axis, used only when
            rot_ver is 1
        :param no_feat: whether to return features or not
        """

        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix
        self.feat_ver = feat_ver
        self.wpt_img_aug = wpt_img_aug
        self.inp_pre_pro = inp_pre_pro
        self.inp_pre_con = inp_pre_con
        self.cvx_up = cvx_up
        self.use_point_renderer = use_point_renderer
        self.rot_ver = rot_ver
        self.num_rot = num_rot
        self.no_feat = no_feat

        if self.cvx_up:
            assert not self.inp_pre_con, (
                "When using the convex upsampling, we do not concatenate"
                " features from input_preprocess to the features used for"
                " prediction"
            )

        print(f"MVT Vars: {vars(self)}")

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 128 / 8 = 16

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 2
        else:
            self.input_dim_before_seq = self.im_channels

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        if self.inp_pre_pro:
            self.input_preprocess = Conv2DBlock(
                inp_img_feat_dim,
                self.im_channels,
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )
            inp_pre_out_dim = self.im_channels
        else:
            # identity
            self.input_preprocess = lambda x: x
            inp_pre_out_dim = inp_img_feat_dim

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                self.im_channels,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 2,
                norm="group",
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                use_fast=xops,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        if cvx_up:
            self.up0 = ConvexUpSample(
                in_dim=self.input_dim_before_seq,
                out_dim=1,
                up_ratio=self.img_patch_size,
            )
        else:
            self.up0 = Conv2DUpsampleBlock(
                self.input_dim_before_seq,
                self.im_channels,
                kernel_sizes=self.img_patch_size,
                strides=self.img_patch_size,
                norm=None,
                activation=activation,
                out_size=self.img_size,
            )

            if self.inp_pre_con:
                final_inp_dim = self.im_channels + inp_pre_out_dim
            else:
                final_inp_dim = self.im_channels

            # final layers
            self.final = Conv2DBlock(
                final_inp_dim,
                self.im_channels,
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=activation,
            )

            self.trans_decoder = Conv2DBlock(
                self.final_dim,
                1,
                kernel_sizes=3,
                strides=1,
                norm=None,
                activation=None,
            )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.input_dim_before_seq
            if self.cvx_up:
                feat_fc_dim += self.input_dim_before_seq
            else:
                feat_fc_dim += self.final_dim

            def get_feat_fc(
                _feat_in_size,
                _feat_out_size,
                _feat_fc_dim=feat_fc_dim,
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim

            if self.rot_ver == 0:
                self.feat_fc = get_feat_fc(
                    self.num_img * feat_fc_dim,
                    feat_out_size,
                )
            elif self.rot_ver == 1:
                assert self.num_rot * 3 <= feat_out_size
                feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
                if feat_out_size_ex_rot > 0:
                    self.feat_fc_ex_rot = get_feat_fc(
                        self.num_img * feat_fc_dim, feat_out_size_ex_rot
                    )

                self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
                self.feat_fc_pe = FixedPositionalEncoding(
                    self.num_img * feat_fc_dim, feat_scale_factor=1
                )
                self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
                self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)

            else:
                assert False

        if self.use_point_renderer:
            from point_renderer.rvt_ops import select_feat_from_hm
        else:
            from mvt.renderer import select_feat_from_hm
        global select_feat_from_hm

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        wpt_local=None,
        rot_x_y=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param rot_x_y: (bs, 2)
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )

        # concat proprio
        _, _, _d, _h, _w = ins.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([ins, p], dim=1)  # [B, 128, num_img, np, np]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, num_img * np * np + 77, 128]

        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )
        if self.cvx_up:
            trans = self.up0(x)
            trans = trans.view(bs, self.num_img, h, w)
        else:
            u0 = self.up0(x)
            if self.inp_pre_con:
                u0 = torch.cat([u0, d0], dim=1)
            u = self.final(u0)

            # translation decoder
            trans = self.trans_decoder(u).view(bs, self.num_img, h, w)

        if not self.no_feat:
            if self.feat_ver == 0:
                hm = F.softmax(trans.detach().view(bs, self.num_img, h * w), 2).view(
                    bs * self.num_img, 1, h, w
                )

                if self.cvx_up:
                    # since we donot predict u, we need to get u from x
                    # x is at a lower resolution than hm, therefore we average
                    # hm using the fold operation
                    _hm = F.unfold(
                        hm,
                        kernel_size=self.img_patch_size,
                        padding=0,
                        stride=self.img_patch_size,
                    )
                    assert _hm.shape == (
                        bs * self.num_img,
                        self.img_patch_size * self.img_patch_size,
                        num_pat_img * num_pat_img,
                    )
                    _hm = torch.mean(_hm, 1)
                    _hm = _hm.view(bs * self.num_img, 1, num_pat_img, num_pat_img)
                    _u = x
                else:
                    # (bs * num_img, self.input_dim_before_seq, h, w)
                    # we use the u directly
                    _hm = hm
                    _u = u

                _feat = torch.sum(_hm * _u, dim=[2, 3])
                _feat = _feat.view(bs, -1)

            elif self.feat_ver == 1:
                # get wpt_local while testing
                if not self.training:
                    wpt_local = self.get_wpt(
                        out={"trans": trans.clone().detach()},
                        dyn_cam_info=None,
                    )

                # projection
                # (bs, 1, num_img, 2)
                wpt_img = self.get_pt_loc_on_img(
                    wpt_local.unsqueeze(1),
                    dyn_cam_info=None,
                )
                wpt_img = wpt_img.reshape(bs * self.num_img, 2)

                # add noise to wpt image while training
                if self.training:
                    wpt_img = mvt_utils.add_uni_noi(
                        wpt_img, self.wpt_img_aug * self.img_size
                    )
                    wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

                if self.cvx_up:
                    _wpt_img = wpt_img / self.img_patch_size
                    _u = x
                    assert (
                        0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
                    ), print(_wpt_img, x.shape)
                else:
                    _u = u
                    _wpt_img = wpt_img

                _wpt_img = _wpt_img.unsqueeze(1)
                _feat = select_feat_from_hm(_wpt_img, _u)[0]
                _feat = _feat.view(bs, -1)

            else:
                assert False, NotImplementedError

            feat.append(_feat)

            feat = torch.cat(feat, dim=-1)

            if self.rot_ver == 0:
                feat = self.feat_fc(feat)
                out = {"feat": feat}
            elif self.rot_ver == 1:
                # features except rotation
                feat_ex_rot = self.feat_fc_ex_rot(feat)

                # batch normalized features for rotation
                feat_rot = self.feat_fc_init_bn(feat)
                feat_x = self.feat_fc_x(feat_rot)

                if self.training:
                    rot_x = rot_x_y[..., 0].view(bs, 1)
                else:
                    # sample with argmax
                    rot_x = feat_x.argmax(dim=1, keepdim=True)

                rot_x_pe = self.feat_fc_pe(rot_x)
                feat_y = self.feat_fc_y(feat_rot + rot_x_pe)

                if self.training:
                    rot_y = rot_x_y[..., 1].view(bs, 1)
                else:
                    rot_y = feat_y.argmax(dim=1, keepdim=True)
                rot_y_pe = self.feat_fc_pe(rot_y)
                feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)
                out = {
                    "feat_ex_rot": feat_ex_rot,
                    "feat_x": feat_x,
                    "feat_y": feat_y,
                    "feat_z": feat_z,
                }
        else:
            out = {}

        out.update({"trans": trans})

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)
        if self.use_point_renderer:
            pred_wpt = pred_wpt.squeeze(1)

        assert y_q is None

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
