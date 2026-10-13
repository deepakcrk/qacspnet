#!/usr/bin/env python3
# Ablation: QA-CSPNet-S with GIoU instead of ARW-SIoU.

import os

from qacspnet.exp import QACSPNetCocoTextExp


class Exp(QACSPNetCocoTextExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.iou_type = "giou"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
