#!/usr/bin/env python3
# QA-CSPNet-L on COCO-Text, full model.

import os

from qacspnet.exp import QACSPNetCocoTextExp


class Exp(QACSPNetCocoTextExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
