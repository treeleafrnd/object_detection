
from object_detection.odservice.detect_object import DetectedObjectResult
from typing import List

class BaseDetector:
    def predict(self, image, class_thresh= 0.4) -> List[DetectedObjectResult]:

        raise NotImplementedError
