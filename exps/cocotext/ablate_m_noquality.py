#!/usr/bin/env python3
# Ablation: QA-CSPNet-M without the quality attention branch in QGE.

import os

from qacspnet.exp import QACSPNetCocoTextExp


class Exp(QACSPNetCocoTextExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.use_quality = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
