import cv2
from object_detection.odservice.yolov5_detector import YoloV5Detector

def yolov5_example():
    image_path = '/home/mahesh/Downloads/animals.jpg'
    image = cv2.imread(image_path)

    # Creating an instance of YoloV5Detector class
    yolov5_detector = YoloV5Detector()

    detect_class_thresh = {'dog': 0.5, 'cat': 0.5}
    bbox, conf, class_names = yolov5_detector.predict(image, detect_class_thresh=detect_class_thresh)

    # Drawing bbox and labels on the image
    for (x1, y1, x2, y2), label, conf_score in zip(bbox, class_names, conf):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, f"{label} {conf_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0))

    # Displaying the image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    yolov5_example()
