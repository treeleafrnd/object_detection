
from typing import List

class BaseDetector:
    def predict(self, image, thresh=0.5, detect_class=None):
        raise NotImplementedError
