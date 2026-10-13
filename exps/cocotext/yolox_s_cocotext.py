#!/usr/bin/env python3
# YOLOX-S baseline on COCO-Text with the same splits and schedule.

import os

from qacspnet.exp import YOLOXCocoTextExp


class Exp(YOLOXCocoTextExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
