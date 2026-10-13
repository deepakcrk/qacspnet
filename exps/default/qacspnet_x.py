#!/usr/bin/env python3
# QA-CSPNet-X detection. For COCO-Text text detection set num_classes = 1
# and point data_dir / train_ann / val_ann at the COCO-Text (COCO-format) files.

import os

from qacspnet.exp import QACSPNetExp


class Exp(QACSPNetExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
