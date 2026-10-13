#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math

import torch
import torch.nn as nn


IOU_LOSSES = ("iou", "giou", "diou", "ciou", "eiou", "siou", "arw_siou")


class IOUloss(nn.Module):
    """Box regression losses on (cx, cy, w, h) boxes.

    arw_ar_cap caps the aspect-ratio difference in arw_siou and arw_iou_power is
    the IoU exponent in arw_siou.
    """

    def __init__(self, reduction="none", loss_type="iou", arw_ar_cap=1.0, arw_iou_power=2.0):
        super(IOUloss, self).__init__()
        assert loss_type in IOU_LOSSES, loss_type
        self.reduction = reduction
        self.loss_type = loss_type
        self.arw_ar_cap = arw_ar_cap
        self.arw_iou_power = arw_iou_power

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == "arw_siou":
            # aspect-ratio weighted squared-IoU.
            # Computed in fp32: 1e-12 underflows to 0 in fp16 and gives inf/inf = NaN.
            pw, ph = pred[:, 2].float(), pred[:, 3].float().clamp(min=1e-6)
            tw, th = target[:, 2].float(), target[:, 3].float().clamp(min=1e-6)
            ar_pred = pw / ph
            ar_gt = tw / th
            delta_ar = (ar_gt - ar_pred).abs() / (
                (ar_gt.abs() + ar_pred.abs()) / 2 + 1e-12
            )
            delta_ar = torch.nan_to_num(delta_ar, nan=self.arw_ar_cap, posinf=self.arw_ar_cap)
            w_ar = 1.0 + delta_ar.clamp(max=self.arw_ar_cap)
            loss = (1 - iou.float() ** self.arw_iou_power) * w_ar
        else:
            loss = _distance_iou_loss(pred.float(), target.float(), iou.float(), self.loss_type)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def _distance_iou_loss(pred, target, iou, kind, eps=1e-7):
    """DIoU, CIoU, EIoU and SIoU losses for (cx, cy, w, h) boxes (fp32)."""
    pw, ph = pred[:, 2].clamp(min=eps), pred[:, 3].clamp(min=eps)
    tw, th = target[:, 2].clamp(min=eps), target[:, 3].clamp(min=eps)
    dx, dy = target[:, 0] - pred[:, 0], target[:, 1] - pred[:, 1]
    # smallest enclosing box
    cw = torch.max(pred[:, 0] + pw / 2, target[:, 0] + tw / 2) - \
        torch.min(pred[:, 0] - pw / 2, target[:, 0] - tw / 2)
    ch = torch.max(pred[:, 1] + ph / 2, target[:, 1] + th / 2) - \
        torch.min(pred[:, 1] - ph / 2, target[:, 1] - th / 2)
    cw, ch = cw.clamp(min=eps), ch.clamp(min=eps)
    c2 = cw ** 2 + ch ** 2
    rho2 = dx ** 2 + dy ** 2

    if kind == "diou":
        return 1 - iou + rho2 / c2
    if kind == "ciou":
        v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return 1 - iou + rho2 / c2 + alpha * v
    if kind == "eiou":
        return 1 - iou + rho2 / c2 + (pw - tw) ** 2 / cw ** 2 + (ph - th) ** 2 / ch ** 2
    # siou: angle, distance and shape costs
    sigma = torch.sqrt(rho2) + eps
    sin_a = torch.where(dx.abs() / sigma > math.sqrt(2) / 2, dy.abs() / sigma, dx.abs() / sigma)
    angle = torch.cos(2 * torch.arcsin(sin_a.clamp(max=1.0)) - math.pi / 2)
    gamma = angle - 2
    dist = 2 - torch.exp(gamma * (dx / cw) ** 2) - torch.exp(gamma * (dy / ch) ** 2)
    ow = (pw - tw).abs() / torch.max(pw, tw)
    oh = (ph - th).abs() / torch.max(ph, th)
    shape = (1 - torch.exp(-ow)) ** 4 + (1 - torch.exp(-oh)) ** 4
    return 1 - iou + 0.5 * (dist + shape)
