
import cv2
from object_detection.odservice.base_detector import BaseDetector
import torch


class YoloV5Detector(BaseDetector):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def predict(self, image, detect_class_thresh = None):
        """
                detect_class_thresh= None  --- this parameter is a dictionary containing class as a key and thresh as value

                -> (List[tuple], List[float], List[str]) --- it indicates that the method returns a tuple containing three list
                -- list of bounding box(As tuples)
                -- list of conf score as float
                -- list of class names as strings
                """

        results = self.model(image)
        # converting predicted image into pandas dataframe
        data_frame = results.pandas().xyxy[0]

        # checks if its true then it filters the dataframe and if the value from the detect_class
        # matches then it assigns the filtered dataframe to the user_df

        bbox_list, conf_list, class_list =[],[],[]
        indexes = data_frame.index

        for index in indexes:
            conf_score = data_frame['confidence'][index]
            class_name = data_frame['name'][index]

            """
            detect_class_thresh contains the class name as key and thresh as values
            class_name in detect_class_thresh -- checks whether the class_name is present in detect_class_thresh or not
            if true then we retrieve the thresh value of the corresponding clas_name
            """
            if detect_class_thresh  and class_name in detect_class_thresh:
                class_thresh = detect_class_thresh[class_name]

            # if not in dictionary then the default thresh value will be 0.5
            else:
                class_thresh = 0.5
            # setting condition if the conf_score >= thresh then to display the bbox

            if conf_score >= class_thresh:
                x1 = data_frame['xmin'][index]
                y1 = data_frame['ymin'][index]
                x2 = data_frame['xmax'][index]
                y2 = data_frame['ymax'][index]

                label = class_name
                text = f"{label} {conf_score:.2f}"

                bbox_list.append((x1, y1, x2, y2))
                conf_list.append(conf_score)
                class_list.append(label)

        return bbox_list, conf_list, class_list