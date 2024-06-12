# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
import torch.nn as nn
import torch.nn.functional as F
from mvt.utils import ForkedPdb


class ConvexUpSample(nn.Module):
    """
    Learned convex upsampling
    """

    def __init__(
        self, in_dim, out_dim, up_ratio, up_kernel=3, mask_scale=0.1, with_bn=False
    ):
        """

        :param in_dim:
        :param out_dim:
        :param up_ratio:
        :param up_kernel:
        :param mask_scale:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_ratio = up_ratio
        self.up_kernel = up_kernel
        self.mask_scale = mask_scale
        self.with_bn = with_bn

        assert (self.up_kernel % 2) == 1

        if with_bn:
            self.net_out_bn1 = nn.BatchNorm2d(2 * in_dim)
            self.net_out_bn2 = nn.BatchNorm2d(2 * in_dim)

        self.net_out = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, out_dim, 3, padding=1),
        )

        mask_dim = (self.up_ratio**2) * (self.up_kernel**2)
        self.net_mask = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, mask_dim, 1, padding=0),
        )

    def forward(self, x):
        """

        :param x: (bs, in_dim, h, w)
        :return: (bs, out_dim, h*up_ratio, w*up_ratio)
        """

        bs, c, h, w = x.shape
        assert c == self.in_dim, c

        # low resolution output
        if self.with_bn:
            out_low = self.net_out[0](x)
            out_low = self.net_out_bn1(out_low)
            out_low = self.net_out[1](out_low)
            out_low = self.net_out[2](out_low)
            out_low = self.net_out_bn2(out_low)
            out_low = self.net_out[3](out_low)
            out_low = self.net_out[4](out_low)
        else:
            out_low = self.net_out(x)

        mask = self.mask_scale * self.net_mask(x)
        mask = mask.view(bs, 1, self.up_kernel**2, self.up_ratio, self.up_ratio, h, w)
        mask = torch.softmax(mask, dim=2)

        out = F.unfold(
            out_low,
            kernel_size=[self.up_kernel, self.up_kernel],
            padding=self.up_kernel // 2,
        )
        out = out.view(bs, self.out_dim, self.up_kernel**2, 1, 1, h, w)

        out = torch.sum(out * mask, dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(bs, self.out_dim, h * self.up_ratio, w * self.up_ratio)

        return out


if __name__ == "__main__":
    net = ConvexUpSample(2, 5, 20).cuda()
    x = torch.rand(4, 5, 10, 10).cuda()
    y = net(x)
    breakpoint()
