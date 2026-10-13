#!/usr/bin/env python3
# QA-CSPNet detection experiment (object detection on COCO / text detection on COCO-Text).

import json
import os

import torch.nn as nn
from loguru import logger

from .yolox_base import Exp


def as_bool(v):
    """exp.merge() turns "False" into True via bool("False"); parse strings properly."""
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


class QACSPNetExp(Exp):
    def __init__(self):
        super().__init__()
        # multi-scale training between 320 and 800 (multiples of 32)
        self.random_size = (10, 25)
        # component switches
        self.iou_type = "arw_siou"  # "arw_siou" | "iou" | "giou"
        self.use_quality = True     # quality attention branch in QGE
        self.qge_type = "global"    # "global" (GAP, one value per channel) |
                                    # "local" (k x k average, one value per channel and position)
        self.qge_local_kernel = 7   # window size for qge_type="local" (odd)
        self.qge_invert = True      # inverted gate 1 - w (False: w)
        self.qge_alpha = "free"     # "free" | "bounded" (0-1) | "fixed"
        self.qge_alpha_init = 0.5   # initial (or fixed) alpha
        self.qge_reduction = 8      # QGE FC hidden size = max(C / reduction, 8)
        self.qge_spatial_kernel = 7  # QGE spatial-branch kernel (odd)
        # "sigmoid": gate in (0, 1), output scaled within [1 - alpha, 1];
        # "free": no sigmoid, raw scores, scale per channel/position unconstrained
        self.qge_gate = "sigmoid"
        self.bottleneck_dilation = 2  # dilation of the bottleneck's depthwise path
        self.arw_ar_cap = 1.0       # cap on the aspect-ratio difference in arw_siou
        self.arw_iou_power = 2.0    # IoU exponent in arw_siou
        # stride-2 stem: "focus" (space-to-depth + 3x3 conv) | "conv" (one strided conv)
        self.stem_type = "focus"
        self.stem_kernel = 6        # kernel of the "conv" stem; 6 matches Focus in params and FLOPs
        # Global Context Encoder (Transformer layers on the stride-32 backbone map), off by default
        self.use_gce = False
        self.gce_layers = 1
        self.gce_heads = 8
        self.gce_ff_ratio = 2.0     # feed-forward size = channels x gce_ff_ratio
        self.gce_dropout = 0.0
        self.use_afw = True         # adaptive FPN weighting
        self.afw_type = "mlp"       # "mlp" | "deep" (residual MLP blocks) | "attn" (self-attention)
        self.afw_layers = 3         # MLP layers, residual blocks for "deep", or attention layers
        self.afw_hidden = None      # MLP hidden units (None: sum of level channels / 4);
                                    # width for "deep" (None: sum of level channels);
                                    # feed-forward size for "attn" (None: 2 * afw_attn_dim)
        self.afw_attn_dim = 128     # token size for "attn"
        self.afw_heads = 4          # attention heads for "attn"
        # "softmax": level weights sum to 1 (levels share a fixed total);
        # "sigmoid": independent gate in (0, 1) per level;
        # "free": each level scaled by 1 + score, no normalization, floor or cap
        self.afw_weight = "softmax"
        self.use_class_weights = True
        self.class_weight_mode = "sqrt_share"  # "sqrt_share" | "inv_sqrt"
        self.reg_weight = 4.0
        self.num_shared_convs = 3
        # backbone/head compute split.
        # head_width scales the head channels relative to the model width
        # (head channels = 256 * width * head_width). If target_head_flops is set
        # (fraction of total GFLOPs, e.g. 0.20) or target_head_params (fraction of
        # total parameters), head_width is chosen automatically to reach it and the
        # head_width value is ignored. Set at most one of the two targets.
        self.head_width = 1.0
        self.target_head_flops = None
        self.target_head_params = None
        # keep the QGE gate's linear layers out of weight decay (set to () to disable);
        # with decay the gate collapses to a constant A_quality = 0.5
        self.no_decay_modules = ("qge.fc1", "qge.fc2", "qge.edge", "qge.spatial")

        # image folders (relative to data_dir) for each annotation file
        self.train_img_dir = "train2017"
        self.val_img_dir = "val2017"
        self.test_img_dir = "test2017"

        # repeated runs: QACSP_SEED=<int> python tools/train.py ...
        if os.environ.get("QACSP_SEED"):
            self.seed = int(os.environ["QACSP_SEED"])

    def qge_cfg(self):
        return dict(
            qge_type=str(self.qge_type), local_kernel=int(self.qge_local_kernel),
            invert=as_bool(self.qge_invert), alpha_mode=str(self.qge_alpha),
            alpha_init=float(self.qge_alpha_init), reduction=int(self.qge_reduction),
            spatial_kernel=int(self.qge_spatial_kernel), dilation=int(self.bottleneck_dilation),
            gate_mode=str(self.qge_gate),
        )

    def gce_cfg(self):
        if not as_bool(self.use_gce):
            return None
        return dict(num_layers=int(self.gce_layers), num_heads=int(self.gce_heads),
                    ff_ratio=float(self.gce_ff_ratio), dropout=float(self.gce_dropout))

    def afw_cfg(self):
        hidden = self.afw_hidden
        return dict(
            afw_type=str(self.afw_type), num_layers=int(self.afw_layers),
            hidden=None if hidden in (None, "", "None") else int(hidden),
            attn_dim=int(self.afw_attn_dim), num_heads=int(self.afw_heads),
            weight_mode=str(self.afw_weight),
        )

    def get_class_counts(self):
        """Per-class instance counts from the training annotation file."""
        from qacspnet.data import get_yolox_datadir

        data_dir = self.data_dir or os.path.join(get_yolox_datadir(), "COCO")
        ann_file = os.path.join(data_dir, "annotations", self.train_ann)
        if not os.path.isfile(ann_file):
            logger.warning(f"{ann_file} not found, class weighting disabled")
            return None
        with open(ann_file) as f:
            ann = json.load(f)
        cat_ids = sorted(c["id"] for c in ann["categories"])
        index = {cid: i for i, cid in enumerate(cat_ids)}
        counts = [0] * len(cat_ids)
        for a in ann["annotations"]:
            counts[index[a["category_id"]]] += 1
        return counts

    def get_model(self):
        from qacspnet.models import YOLOX, QACSPHead, QACSPPAFPN

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = QACSPPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act,
                use_quality=as_bool(self.use_quality), use_afw=as_bool(self.use_afw),
                afw_cfg=self.afw_cfg(),
                qge_cfg=self.qge_cfg(),
                gce_cfg=self.gce_cfg(),
                stem_type=str(self.stem_type), stem_kernel=int(self.stem_kernel),
            )
            # the sqrt_share weight is 0 for a single class, so skip it there
            counts = None
            if as_bool(self.use_class_weights) and self.num_classes > 1:
                counts = self.get_class_counts()
            def make_head(hw):
                return QACSPHead(
                    self.num_classes, self.width * hw, in_channels=in_channels,
                    act=self.act, num_shared_convs=int(self.num_shared_convs),
                    iou_type=self.iou_type, reg_weight=self.reg_weight,
                    arw_ar_cap=float(self.arw_ar_cap), arw_iou_power=float(self.arw_iou_power),
                    class_counts=counts, class_weight_mode=self.class_weight_mode,
                    feat_width=self.width,
                )

            hw = float(self.head_width)
            unset = (None, "", "None")
            if self.target_head_flops not in unset and self.target_head_params not in unset:
                raise ValueError("set only one of target_head_flops / target_head_params")
            if self.target_head_flops not in unset:
                hw = solve_head_width(backbone, make_head, float(self.target_head_flops),
                                      self.test_size, key="gflops")
            elif self.target_head_params not in unset:
                hw = solve_head_width(backbone, make_head, float(self.target_head_params),
                                      self.test_size, key="params")
            self.head_width = hw
            head = make_head(hw)
            self.model = YOLOX(backbone, head)
            self.log_split()

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def log_split(self):
        """Log how the GFLOPs and parameters are shared between backbone+neck and head."""
        try:
            b, h = compute_split(self.model, self.test_size)
        except Exception as e:  # thop missing, etc.: the split is informational only
            logger.warning(f"backbone/head split not computed: {e}")
            return
        gf, pa = b["gflops"] + h["gflops"], b["params"] + h["params"]
        logger.info(
            f"head_width {float(self.head_width):.3f} | GFLOPs {gf:.2f} "
            f"(backbone+neck {b['gflops']:.2f} = {b['gflops'] / gf:.1%}, "
            f"head {h['gflops']:.2f} = {h['gflops'] / gf:.1%}) | params {pa:.2f}M "
            f"(backbone+neck {b['params']:.2f}M = {b['params'] / pa:.1%}, "
            f"head {h['params']:.2f}M = {h['params'] / pa:.1%})"
        )

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from qacspnet.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_img_dir,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        """val_ann is used for model selection during training (dev split);
        test_ann (tools/eval.py --test) is the held-out final evaluation split."""
        from qacspnet.data import COCODataset, ValTransform

        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name=self.test_img_dir if testdev else self.val_img_dir,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )


def compute_split(model, tsize=(640, 640)):
    """GFLOPs (thop, 2 x MACs) and params (M) of backbone+neck+AFW and of the head."""
    import torch
    from copy import deepcopy
    from thop import profile

    dev = next(model.parameters()).device
    img = torch.zeros((1, 3, tsize[0], tsize[1]), device=dev)
    bb = deepcopy(model.backbone).eval()
    with torch.no_grad():
        feats = bb(img)
    b_macs, b_par = profile(bb, inputs=(img,), verbose=False)
    head = deepcopy(model.head).eval()
    head.training = False
    h_macs, h_par = profile(head, inputs=([f.clone() for f in feats],), verbose=False)
    to = lambda macs, par: {"gflops": 2 * macs / 1e9, "params": par / 1e6}
    return to(b_macs, b_par), to(h_macs, h_par)


def solve_head_width(backbone, make_head, target, tsize=(640, 640), key="gflops",
                     lo=0.05, hi=4.0, iters=12):
    """Bisection on head_width so that head share of `key` ("gflops" | "params") ~= target."""
    from qacspnet.models import YOLOX

    def share(hw):
        b, h = compute_split(YOLOX(backbone, make_head(hw)), tsize)
        return h[key] / (b[key] + h[key])

    if not share(lo) <= target <= share(hi):
        raise ValueError(f"target head {key} share {target} outside reachable range "
                         f"[{share(lo):.3f}, {share(hi):.3f}]; change num_shared_convs")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if share(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def setup_cocotext(exp, data_dir="datasets/COCOText"):
    """COCO-Text (converted by tools/data_prep/convert_cocotext.py), 1 class.

    Development: train on train_dev, select checkpoints on dev (val_ann).
    Final: tools/eval.py --test evaluates once on the official val split (test_ann).
    """
    exp.num_classes = 1
    exp.data_dir = data_dir
    exp.train_ann = "cocotext_train_dev.json"
    exp.val_ann = "cocotext_dev.json"
    exp.test_ann = "cocotext_val.json"
    exp.train_img_dir = exp.val_img_dir = exp.test_img_dir = "train2014"


class QACSPNetCocoTextExp(QACSPNetExp):
    def __init__(self):
        super().__init__()
        setup_cocotext(self)


class YOLOXCocoTextExp(Exp):
    """Unmodified YOLOX baseline trained with the same data, splits and schedule."""

    def __init__(self):
        super().__init__()
        setup_cocotext(self)
        self.random_size = (10, 25)
        if os.environ.get("QACSP_SEED"):
            self.seed = int(os.environ["QACSP_SEED"])

    get_dataset = QACSPNetExp.get_dataset
    get_eval_dataset = QACSPNetExp.get_eval_dataset
