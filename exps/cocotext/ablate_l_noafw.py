#!/usr/bin/env python3
# Ablation: QA-CSPNet-L without AFW.

import os

from qacspnet.exp import QACSPNetCocoTextExp


class Exp(QACSPNetCocoTextExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.use_afw = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
