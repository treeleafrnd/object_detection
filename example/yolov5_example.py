"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2024-04-02
"""
import cv2

from odservice.yolov5_detector import YoloV5Detector


def yolov5_example():
    image_url = "?path?to?image_url"
    image = cv2.imread(image_url)
    yolov5_detector = YoloV5Detector()
    result = yolov5_detector.predict(image)
    print(result)


if __name__ == '__main__':
    yolov5_example()
