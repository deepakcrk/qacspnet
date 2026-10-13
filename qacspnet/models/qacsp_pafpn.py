#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# QA-CSPNet backbone and PAFPN neck.

import torch
import torch.nn as nn

from .network_blocks import BaseConv, Focus, SPPBottleneck
from .qacsp_blocks import AdaptiveFPNWeighting, GlobalContextEncoder, QACSPLayer


class QACSPDarknet(nn.Module):
    """CSPDarknet with every CSPLayer replaced by a QA-CSP layer.

    use_focus=False gives the stride-1 stem used by QA-CSPTextNet (32x96 input).
    With use_focus=True the stride-2 stem is either
      stem_type="focus": space-to-depth (C -> 4C at half resolution) + 3x3 conv, or
      stem_type="conv":  one stem_kernel x stem_kernel conv with stride 2 on the
                         image; stem_kernel=6 has the same parameters and FLOPs
                         as the Focus stem.
    """

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        in_chans=3,
        use_focus=True,
        use_spp=False,
        act="silu",
        use_quality=True,
        qge_cfg=None,
        stem_type="focus",
        stem_kernel=6,
    ):
        super().__init__()
        assert stem_type in ("focus", "conv"), stem_type
        self.out_features = out_features
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        if use_focus and stem_type == "focus":
            self.stem = Focus(in_chans, base_channels, ksize=3, act=act)
        elif use_focus:
            self.stem = BaseConv(in_chans, base_channels, stem_kernel, 2, act=act)
        else:
            self.stem = BaseConv(in_chans, base_channels, 3, 1, act=act)

        def stage(cin, cout, n, shortcut=True, spp=False):
            layers = [BaseConv(cin, cout, 3, 2, act=act)]
            if spp:
                layers.append(SPPBottleneck(cout, cout, activation=act))
            layers.append(
                QACSPLayer(cout, cout, n=n, shortcut=shortcut, act=act,
                           use_quality=use_quality, qge_cfg=qge_cfg)
            )
            return nn.Sequential(*layers)

        self.dark2 = stage(base_channels, base_channels * 2, base_depth)
        self.dark3 = stage(base_channels * 2, base_channels * 4, base_depth * 3)
        self.dark4 = stage(base_channels * 4, base_channels * 8, base_depth * 3)
        self.dark5 = stage(
            base_channels * 8, base_channels * 16, base_depth,
            shortcut=False, spp=use_spp,
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class QACSPPAFPN(nn.Module):
    """PAFPN neck built from QA-CSP layers, followed by the AFW module."""

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        in_chans=3,
        use_focus=True,
        use_spp=False,
        act="silu",
        use_quality=True,
        use_afw=True,
        afw_cfg=None,
        qge_cfg=None,
        gce_cfg=None,
        stem_type="focus",
        stem_kernel=6,
    ):
        super().__init__()
        self.backbone = QACSPDarknet(
            depth, width, in_features, in_chans=in_chans, use_focus=use_focus,
            use_spp=use_spp, act=act, use_quality=use_quality, qge_cfg=qge_cfg,
            stem_type=stem_type, stem_kernel=stem_kernel,
        )
        self.in_features = in_features
        self.in_channels = in_channels
        c0, c1, c2 = [int(c * width) for c in in_channels]
        n = round(3 * depth)

        def csp(cin, cout):
            return QACSPLayer(cin, cout, n=n, shortcut=False, act=act,
                              use_quality=use_quality, qge_cfg=qge_cfg)

        # optional Global Context Encoder on the stride-32 backbone output
        self.gce = GlobalContextEncoder(c2, **gce_cfg) if gce_cfg else None

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(c2, c1, 1, 1, act=act)
        self.C3_p4 = csp(2 * c1, c1)
        self.reduce_conv1 = BaseConv(c1, c0, 1, 1, act=act)
        self.C3_p3 = csp(2 * c0, c0)
        self.bu_conv2 = BaseConv(c0, c0, 3, 2, act=act)
        self.C3_n3 = csp(2 * c0, c1)
        self.bu_conv1 = BaseConv(c1, c1, 3, 2, act=act)
        self.C3_n4 = csp(2 * c1, c2)

        self.afw = AdaptiveFPNWeighting([c0, c1, c2], **(afw_cfg or {})) if use_afw else None

    def forward(self, input):
        out_features = self.backbone(input)
        [x2, x1, x0] = [out_features[f] for f in self.in_features]
        if self.gce is not None:
            x0 = self.gce(x0)

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = torch.cat([self.upsample(fpn_out0), x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = torch.cat([self.upsample(fpn_out1), x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        outputs = (pan_out2, pan_out1, pan_out0)
        if self.afw is not None:
            outputs = self.afw(outputs)
        return outputs
