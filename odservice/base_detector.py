
from typing import List

from typing import List

class BaseDetector:
    def predict(self, image, detect_class_thresh=None) -> (List[tuple], List[float], List[str]):
        """
        detect_class_thresh= None  --- this parameter is a dictionary containing class as a key and thresh as value

        -> (List[tuple], List[float], List[str]) --- it indicates that the method returns a tuple containing three list
        -- list of bounding box(As tuples)
        -- list of conf score as float
        -- list of class names as strings
        """
        raise NotImplementedError
