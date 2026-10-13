#!/usr/bin/env python3
# YOLOX-L baseline on COCO-Text with the same splits and schedule.

import os

from qacspnet.exp import YOLOXCocoTextExp


class Exp(YOLOXCocoTextExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
