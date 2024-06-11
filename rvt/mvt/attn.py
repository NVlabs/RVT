# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# Sources:
# https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
# https://github.com/peract/peract/blob/main/helpers/network_utils.py

import math
from math import log
from functools import wraps
from packaging import version

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn, einsum
from einops import rearrange, repeat

try:
    import xformers.ops as xops
except ImportError as e:
    xops = None

LRELU_SLOPE = 0.02


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):  # is all you need. Living up to its name.
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, use_fast=False):

        super().__init__()
        self.use_fast = use_fast
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout_p = dropout
        # dropout left in use_fast for backward compatibility
        self.dropout = nn.Dropout(self.dropout_p)

        self.avail_xf = False
        if self.use_fast:
            if not xops is None:
                self.avail_xf = True
            else:
                self.use_fast = False

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        if self.use_fast:
            # using py2 if available
            dropout_p = self.dropout_p if self.training else 0.0
            # using xf if available
            if self.avail_xf:
                out = xops.memory_efficient_attention(
                    query=q, key=k, value=v, p=dropout_p
                )
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim.masked_fill_(~mask, max_neg_value)
            # attention
            attn = sim.softmax(dim=-1)
            # dropout
            attn = self.dropout(attn)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        return out


def act_layer(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("%s not recognized." % act)


def norm_layer2d(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == "group":
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    elif norm == "group":
        return nn.GroupNorm(4, num_channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,
    ):
        super().__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        else:
            self.norm = None
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class Conv2DUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_sizes=3,
        norm=None,
        activation=None,
        out_size=None,
    ):
        super().__init__()
        layer = [
            Conv2DBlock(in_channels, out_channels, kernel_sizes, 1, norm, activation)
        ]
        if strides > 1:
            if out_size is None:
                layer.append(
                    nn.Upsample(scale_factor=strides, mode="bilinear", align_corners=False)
                )
            else:
                layer.append(
                    nn.Upsample(size=out_size, mode="bilinear", align_corners=False)
                )

        if out_size is not None:
            if kernel_sizes % 2 == 0:
                kernel_sizes += 1

        convt_block = Conv2DBlock(
            out_channels, out_channels, kernel_sizes, 1, norm, activation
        )
        layer.append(convt_block)
        self.conv_up = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv_up(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.linear.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


# based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FixedPositionalEncoding(nn.Module):
    def __init__(self, feat_per_dim: int, feat_scale_factor: int):
        super().__init__()
        self.feat_scale_factor = feat_scale_factor
        # shape [1, feat_per_dim // 2]
        div_term = torch.exp(
            torch.arange(0, feat_per_dim, 2) * (-math.log(10000.0) /
                                                feat_per_dim)
        ).unsqueeze(0)
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        :param x: Tensor, shape [batch_size, input_dim]
        :return: Tensor, shape [batch_size, input_dim * feat_per_dim]
        """
        assert len(x.shape) == 2
        batch_size, input_dim = x.shape
        x = x.view(-1, 1)
        x = torch.cat((
            torch.sin(self.feat_scale_factor * x * self.div_term),
            torch.cos(self.feat_scale_factor * x * self.div_term)), dim=1)
        x = x.view(batch_size, -1)
        return x
