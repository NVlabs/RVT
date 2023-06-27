# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from yacs.config import CfgNode as CN

_C = CN()

_C.depth = 8
_C.img_size = 220
_C.add_proprio = True
_C.proprio_dim = 4
_C.add_lang = True
_C.lang_dim = 512
_C.lang_len = 77
_C.img_feat_dim = 3
_C.feat_dim = (72 * 3) + 2 + 2
_C.im_channels = 64
_C.attn_dim = 512
_C.attn_heads = 8
_C.attn_dim_head = 64
_C.activation = "lrelu"
_C.weight_tie_layers = False
_C.attn_dropout = 0.1
_C.decoder_dropout = 0.0
_C.img_patch_size = 11
_C.final_dim = 64
_C.self_cross_ver = 1
_C.add_corr = True
_C.add_pixel_loc = True
_C.add_depth = True
_C.pe_fix = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()
