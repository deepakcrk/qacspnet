#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# QA-CSPNet balanced detection head.

import torch
import torch.nn as nn

from .losses import IOUloss
from .network_blocks import BaseConv
from .yolo_head import YOLOXHead


def class_weights_from_counts(counts, mode="sqrt_share", eps=1e-8):
    """Per-class positive weights for the classification BCE.

    mode="sqrt_share": w_c = 1 - sqrt(n_c) / sum_i sqrt(n_i)  (eps if n_c == 0).
                       Gives w = 0 for a single class.
    mode="inv_sqrt":   w_c = (1 / sqrt(n_c)) normalised to mean 1 over present classes.
    """
    counts = torch.as_tensor(counts, dtype=torch.float32)
    present = counts > 0
    sq = counts.sqrt()
    if mode == "sqrt_share":
        w = 1.0 - sq / sq.sum().clamp(min=eps)
    elif mode == "inv_sqrt":
        w = torch.zeros_like(counts)
        w[present] = 1.0 / sq[present]
        w = w / w[present].mean()
    else:
        raise ValueError(f"unknown class weight mode: {mode}")
    return torch.where(present, w, torch.full_like(w, eps))


class QACSPHead(YOLOXHead):
    """Stem 1x1 -> shared 3 x (Conv3x3 + act) -> {cls, reg, obj} predictions.

    Losses: ARW-SIoU for boxes (weight 4), class-weighted BCE for classes,
    plain BCE for objectness. SimOTA assignment is inherited from YOLOXHead.
    """

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        num_shared_convs=3,
        iou_type="arw_siou",
        arw_ar_cap=1.0,
        arw_iou_power=2.0,
        reg_weight=4.0,
        class_counts=None,
        class_weight_mode="sqrt_share",
        feat_width=None,
    ):
        super().__init__(num_classes, width, strides, in_channels, act)
        hidden = int(256 * width)
        # head channels follow `width`; the stems read the neck outputs, whose
        # channels follow the model width (feat_width) when the two differ
        if feat_width is not None and feat_width != width:
            self.stems = nn.ModuleList(
                BaseConv(int(c * feat_width), hidden, 1, 1, act=act) for c in in_channels
            )
        # replace the decoupled cls/reg branches by one shared branch per level
        del self.cls_convs, self.reg_convs
        self.shared_convs = nn.ModuleList(
            nn.Sequential(
                *[BaseConv(hidden, hidden, 3, 1, act=act) for _ in range(num_shared_convs)]
            )
            for _ in in_channels
        )
        self.iou_loss = IOUloss(reduction="none", loss_type=iou_type,
                                arw_ar_cap=arw_ar_cap, arw_iou_power=arw_iou_power)
        self.reg_weight = reg_weight
        self.set_class_weights(class_counts, class_weight_mode)

    def set_class_weights(self, class_counts=None, mode="sqrt_share"):
        pos_weight = None
        if class_counts is not None:
            pos_weight = class_weights_from_counts(class_counts, mode)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (shared_conv, stride_this_level, x) in enumerate(
            zip(self.shared_convs, self.strides, xin)
        ):
            feat = shared_conv(self.stems[k](x))
            cls_output = self.cls_preds[k](feat)
            reg_output = self.reg_preds[k](feat)
            obj_output = self.obj_preds[k](feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, 1, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs
