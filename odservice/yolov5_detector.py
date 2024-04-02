"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2024-04-02
"""
from odservice.base_detector import BaseDetector


class YoloV5Detector(BaseDetector):
    def __init__(self):
        pass

    def predict(self, image, thresh=0.5, detect_class=None):
        """
        :param detect_class: object that needs to be detected
        :param image: numpy array
        :param thresh: confidence (data type floating point)
        :return: [bbox,conf,class]
        """
        pass
