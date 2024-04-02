"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2024-04-02
"""
from typing import List


class BaseDetector:
    def predict(self, image, thresh=0.5, detect_class=None):
        raise NotImplementedError
