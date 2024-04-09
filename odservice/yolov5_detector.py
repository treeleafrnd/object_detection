from typing import List

from object_detection.odservice.base_detector import BaseDetector
import torch
from object_detection.odservice.detect_object import DetectedObjectResult

class YoloV5Detector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def predict(self, image, class_thresh =0.6) -> List[DetectedObjectResult]:
        results = self.model(image)

        detected_objects = []

        # .xyxy[0] access the detected objects information from the result
        '''
        index[0] - [3] represents the bbox 
        index [4] - represents the confidence score
        index [5] - represents the class_name
        '''
        for obj in results.xyxy[0]:
            conf_score = obj[4]
            class_name = self.model.names[int(obj[5])]

            if conf_score >= class_thresh:
                bbox = [int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])]
                detected_objects.append(DetectedObjectResult(bbox=bbox, confidence=conf_score, class_name=class_name))

        return detected_objects
