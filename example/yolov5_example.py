import cv2
from object_detection.odservice.yolov5_detector import YoloV5Detector

def draw_detected_objects(image, detected_objects):
    for obj in detected_objects:
        bbox = obj.bbox
        label = obj.class_name
        conf_score = obj.confidence

        # Convert bbox coordinates to integers
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, f"{label} {conf_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    return image

def display_detected_objects(image_path, objects_to_detect):
    # Load the image
    image = cv2.imread(image_path)

    # Create an instance of YoloV5Detector class
    yolov5_detector = YoloV5Detector()

    # Perform object detection
    detected_objects = yolov5_detector.predict(image)

    # Filter out only the specified objects
    filtered_objects = [obj for obj in detected_objects if obj.class_name in objects_to_detect]

    #draw detected objects on the image
    image_with_objects = draw_detected_objects(image.copy(), filtered_objects)

    # Display the image with detected objects in a window
    cv2.imshow("Detected Objects", image_with_objects)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = '/home/mahesh/Downloads/animals.jpg'
    objects_to_detect = ['dog','cat']
    display_detected_objects(image_path, objects_to_detect)
