#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# QA-CSPNet building blocks: QGE module, multi-scale conv bottleneck,
# QA-CSP layer and adaptive FPN weighting (AFW) module.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_blocks import BaseConv, get_activation


class QualityGuidedEnhancement(nn.Module):
    """Quality-guided enhancement (QGE).

    Inverted quality gate x spatial attention, blended with the input via a
    learnable factor alpha.

    qge_type="global": the edge magnitude |DWConv(x)| is globally average-pooled,
                       giving one quality value per channel and image.
    qge_type="local":  |DWConv(x)| is average-pooled over a local_kernel x
                       local_kernel window around every position (stride 1),
                       and the same two FC layers are applied at each position,
                       giving a quality value per channel and position.
                       local_kernel=1 uses each position on its own.
    Both types have the same layers and parameters.

    invert=True uses the inverted gate 1 - w; invert=False uses w.
    alpha_mode="free": learnable, unbounded; "bounded": learnable, sigmoid(.) in
    (0, 1); "fixed": constant alpha_init. reduction sets the FC hidden size
    max(C / reduction, 8); spatial_kernel is the spatial-branch kernel (odd).
    """

    def __init__(self, channels, reduction=8, alpha_init=0.5, use_quality=True,
                 qge_type="global", local_kernel=7, invert=True, alpha_mode="free",
                 spatial_kernel=7, use_spatial=True):
        super().__init__()
        assert qge_type in ("global", "local"), qge_type
        assert local_kernel >= 1 and local_kernel % 2 == 1, "local_kernel must be odd"
        assert spatial_kernel >= 1 and spatial_kernel % 2 == 1, "spatial_kernel must be odd"
        assert alpha_mode in ("free", "bounded", "fixed"), alpha_mode
        self.use_quality = use_quality
        self.use_spatial = use_spatial  # ablation: False = quality gate only (no spatial attention map)
        assert use_quality or use_spatial, "QGE needs the quality branch or the spatial branch"
        self.qge_type, self.local_kernel = qge_type, local_kernel
        self.invert, self.alpha_mode = invert, alpha_mode
        if use_quality:
            hidden = max(channels // reduction, 8)
            self.edge = nn.Conv2d(
                channels, channels, 3, 1, 1, groups=channels, bias=False
            )
            self.fc1 = nn.Linear(channels, hidden)
            self.fc2 = nn.Linear(hidden, channels)
        if use_spatial:
            self.spatial = nn.Conv2d(2, 1, spatial_kernel, 1, spatial_kernel // 2, bias=False)
        a = float(alpha_init)
        if alpha_mode == "free":
            self.alpha = nn.Parameter(torch.tensor(a))
        elif alpha_mode == "bounded":
            assert 0.0 < a < 1.0, "bounded alpha needs 0 < alpha_init < 1"
            self.alpha = nn.Parameter(torch.tensor(math.log(a / (1 - a))))  # sigmoid -> a
        else:
            self.register_buffer("alpha", torch.tensor(a))

    def blend_factor(self):
        return torch.sigmoid(self.alpha) if self.alpha_mode == "bounded" else self.alpha

    def forward(self, x):
        # spatial attention: [mean_c; max_c] -> 7x7 conv -> sigmoid
        if self.use_spatial:
            f_avg = x.mean(dim=1, keepdim=True)
            f_max = x.amax(dim=1, keepdim=True)
            attn = torch.sigmoid(self.spatial(torch.cat([f_avg, f_max], dim=1)))
        else:
            attn = x.new_ones(())

        if self.use_quality:
            edge = self.edge(x).abs()
            if self.qge_type == "global":
                # |DWConv(x)| -> GAP -> FC-SiLU-FC -> sigmoid -> 1 - w   (B x C x 1 x 1)
                z = edge.mean(dim=(2, 3))
                w_quality = torch.sigmoid(self.fc2(F.silu(self.fc1(z))))
                a_quality = (1.0 - w_quality if self.invert else w_quality)[:, :, None, None]
            else:
                # |DWConv(x)| -> local average -> FC-SiLU-FC per position -> 1 - w   (B x C x H x W)
                k = self.local_kernel
                if k > 1:
                    edge = F.avg_pool2d(edge, k, 1, k // 2, count_include_pad=False)
                z = edge.permute(0, 2, 3, 1)  # B x H x W x C
                w_quality = torch.sigmoid(self.fc2(F.silu(self.fc1(z))))
                w_quality = w_quality.permute(0, 3, 1, 2)
                a_quality = 1.0 - w_quality if self.invert else w_quality
            attn = a_quality * attn

        return x * (1 + self.blend_factor() * (attn - 1))


class DilatedDWConv(nn.Module):
    """Dilated depthwise 3x3 conv + pointwise conv (each with BN + act)."""

    def __init__(self, in_channels, out_channels, ksize=3, dilation=2, act="silu"):
        super().__init__()
        pad = dilation * (ksize - 1) // 2
        self.dconv = nn.Conv2d(
            in_channels, in_channels, ksize, 1, pad,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = get_activation(act, inplace=True)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1, act=act)

    def forward(self, x):
        return self.pconv(self.act(self.bn(self.dconv(x))))


class MultiScaleBottleneck(nn.Module):
    """Two parallel paths: 3x3 conv and dilated 3x3 DWConv, concatenated."""

    def __init__(self, channels, shortcut=True, act="silu", dilation=2):
        super().__init__()
        half = channels // 2
        self.conv_path = BaseConv(channels, half, 3, 1, act=act)
        self.dilated_path = DilatedDWConv(channels, channels - half, 3, dilation, act=act)
        self.use_add = shortcut

    def forward(self, x):
        y = torch.cat([self.conv_path(x), self.dilated_path(x)], dim=1)
        if self.use_add:
            y = y + x
        return y


class QACSPLayer(nn.Module):
    """Quality-aware CSP layer.

    main:  Conv1 1x1 -> QGE -> N x MultiScaleBottleneck
    short: Conv2 1x1
    fuse:  concat -> DWConv 3x3 -> Conv 3x3
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        act="silu",
        use_quality=True,
        qge_cfg=None,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        qge_cfg = dict(qge_cfg or {})
        dilation = qge_cfg.pop("dilation", 2)  # bottleneck dilation rate
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden, 1, stride=1, act=act)
        self.qge = QualityGuidedEnhancement(hidden, use_quality=use_quality, **qge_cfg)
        self.m = nn.Sequential(
            *[MultiScaleBottleneck(hidden, shortcut, act=act, dilation=dilation) for _ in range(n)]
        )
        self.dwconv = BaseConv(2 * hidden, 2 * hidden, 3, 1, groups=2 * hidden, act=act)
        self.conv3 = BaseConv(2 * hidden, out_channels, 3, 1, act=act)

    def forward(self, x):
        x_1 = self.m(self.qge(self.conv1(x)))
        x_2 = self.conv2(x)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(self.dwconv(x))


class AdaptiveFPNWeighting(nn.Module):
    """Adaptive FPN weighting (AFW): GAP of each level -> weight predictor ->
    softmax over levels -> each level scaled by its weight.

    afw_type="mlp":  levels' GAP vectors concatenated -> `num_layers` fully
                     connected layers (ReLU between them); `hidden` units in the
                     hidden layers (default sum(C) / 4).
    afw_type="attn": each level's GAP vector is projected to `attn_dim` and
                     treated as one token (plus a learned level embedding);
                     `num_layers` Transformer self-attention layers with
                     `num_heads` heads; a linear layer scores each token.
    """

    def __init__(self, in_channels, afw_type="mlp", num_layers=3, hidden=None,
                 attn_dim=128, num_heads=4, dropout=0.0, min_weight=0.0, norm_input=False,
                 base=None, share=0.1, per_column=False):
        super().__init__()
        assert afw_type in ("mlp", "attn"), afw_type
        assert num_layers >= 1, "num_layers must be >= 1"
        self.afw_type = afw_type
        n = len(in_channels)
        # floor on each level's weight: w = m + (1 - n*m) * softmax (still sums to 1)
        assert 0.0 <= min_weight * n < 1.0, "min_weight must be < 1 / number of levels"
        self.min_weight = min_weight
        # normalize each level's pooled vector (parameter-free LayerNorm) before predicting
        # weights, so levels with different value ranges are compared fairly
        self.norm_input = norm_input
        # fixed base weight per level with AFW deciding only `share` of the total:
        # w = (1 - share) * base + share * softmax  (sums to 1; each level within base +- share)
        if base is not None:
            assert len(base) == n and abs(sum(base) - 1.0) < 1e-3, "base: one weight per level, sum 1"
            assert 0.0 < share <= 1.0
            self.register_buffer("base", torch.tensor(base, dtype=torch.float32), persistent=False)
        else:
            self.base = None
        self.share = share
        # per_column: weights predicted for every column (width position) instead of once per
        # image: each level is averaged over height only, the coarser levels are linearly
        # upsampled to the finest level's width, and the same MLP runs per column
        self.per_column = per_column
        if afw_type == "mlp":
            total = sum(in_channels)
            hidden = hidden or max(total // 4, 8)
            dims = [total] + [hidden] * (num_layers - 1) + [n]
            layers = []
            for i in range(num_layers):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < num_layers - 1:
                    layers.append(nn.ReLU(inplace=True))
            self.mlp = nn.Sequential(*layers)
        else:
            assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
            self.proj = nn.ModuleList(nn.Linear(c, attn_dim) for c in in_channels)
            self.level_emb = nn.Parameter(torch.zeros(1, n, attn_dim))
            nn.init.trunc_normal_(self.level_emb, std=0.02)
            layer = nn.TransformerEncoderLayer(
                attn_dim, num_heads, hidden or 2 * attn_dim, dropout,
                activation="gelu", batch_first=True,
            )
            self.attn = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
            self.score = nn.Linear(attn_dim, 1)

    def _column_weights(self, feats):
        w0 = feats[0].shape[-1]
        cols = []
        for f in feats:
            v = f.mean(dim=2)  # B x C x W
            if self.norm_input:
                v = F.layer_norm(v.transpose(1, 2), v.shape[1:2]).transpose(1, 2)
            if v.shape[-1] != w0:
                v = F.interpolate(v, size=w0, mode="linear", align_corners=False)
            cols.append(v)
        p = torch.cat(cols, dim=1).transpose(1, 2)  # B x W0 x total
        return torch.softmax(self.mlp(p), dim=-1).transpose(1, 2)  # B x levels x W0

    def forward(self, feats):
        if self.afw_type == "mlp" and self.per_column:
            w = self._column_weights(feats)
            if self.base is not None:
                w = (1.0 - self.share) * self.base.to(w.dtype)[None, :, None] + self.share * w
            elif self.min_weight > 0:
                w = self.min_weight + (1.0 - len(feats) * self.min_weight) * w
            out = []
            for i, f in enumerate(feats):
                wi = w[:, i:i + 1]  # B x 1 x W0
                if wi.shape[-1] != f.shape[-1]:
                    wi = F.adaptive_avg_pool1d(wi, f.shape[-1])
                out.append(f * wi[:, :, None, :])
            return tuple(out)
        if self.afw_type == "mlp":
            pooled = [f.mean(dim=(2, 3)) for f in feats]
            if self.norm_input:
                pooled = [F.layer_norm(v, v.shape[-1:]) for v in pooled]
            p = torch.cat(pooled, dim=1)
            logits = self.mlp(p)
        else:
            tokens = torch.stack(
                [proj(f.mean(dim=(2, 3))) for proj, f in zip(self.proj, feats)], dim=1
            )  # B x levels x attn_dim
            logits = self.score(self.attn(tokens + self.level_emb)).squeeze(-1)
        w = torch.softmax(logits, dim=1)
        if self.base is not None:
            w = (1.0 - self.share) * self.base.to(w.dtype) + self.share * w
        elif self.min_weight > 0:
            w = self.min_weight + (1.0 - len(feats) * self.min_weight) * w
        return tuple(f * w[:, i, None, None, None] for i, f in enumerate(feats))


class GlobalContextEncoder(nn.Module):
    """Global Context Encoder (GCE): Transformer encoder layers on the coarsest
    backbone map (last stage; stride 32 in the detector), before the neck.
    Optional; off by default.

    The C x H x W map is flattened into H*W tokens of size C. Each layer applies
    multi-head self-attention (2D sine-cosine positional embedding added to the
    queries and keys) and a feed-forward network, each with a residual connection
    followed by LayerNorm. The output is reshaped back to C x H x W.
    """

    def __init__(self, channels, num_layers=1, num_heads=8, ff_ratio=2.0, dropout=0.0,
                 temperature=10000.0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        assert channels % 4 == 0, "channels must be divisible by 4 (2D positional embedding)"
        self.temperature = temperature
        ff = int(channels * ff_ratio)
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(channels, num_heads, dropout=dropout,
                                              batch_first=True),
                "norm1": nn.LayerNorm(channels),
                "ffn": nn.Sequential(
                    nn.Linear(channels, ff), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(ff, channels),
                ),
                "norm2": nn.LayerNorm(channels),
            })
            for _ in range(num_layers)
        )
        self.drop = nn.Dropout(dropout)

    def pos_embed(self, h, w, c, device, dtype):
        """2D sine-cosine positional embedding, (h*w) x c."""
        gy, gx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        d = c // 4
        omega = 1.0 / self.temperature ** (torch.arange(d, device=device, dtype=torch.float32) / d)
        ox = gx.flatten()[:, None] * omega[None]
        oy = gy.flatten()[:, None] * omega[None]
        pe = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=1)
        return pe.to(dtype)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)  # B x HW x C
        pos = self.pos_embed(h, w, c, x.device, x.dtype)[None]
        for layer in self.layers:
            q = k = t + pos
            t = layer["norm1"](t + self.drop(layer["attn"](q, k, t, need_weights=False)[0]))
            t = layer["norm2"](t + self.drop(layer["ffn"](t)))
        return t.transpose(1, 2).reshape(b, c, h, w)
