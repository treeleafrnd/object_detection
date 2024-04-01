'''
This code is written for detecting the specific object from the image
as input from the user
 creator -- Mahesh Pela
'''

import cv2
import torch

# download yolov5 model from the github
model = torch.hub.load('ultralytics/yolov5','yolov5s')

def example_imageDetection():
    # providing the image_path
    image = cv2.imread('../experiment/Images/animals.jpg')
    image = cv2.resize(image, (1000,650))

    # performing detection on the image
    results = model(image)

    # converting the detected image into pandas dataframe
    data_frame = results.pandas().xyxy[0]

    # specifying class to be filtered from the detected image
    user_input = ['dog','bowl']

    # filtering the dataframe to include only dog
    user_df = data_frame[data_frame['name'].isin(user_input)]

    # get indexes of all rows in the dataframe
    # only the information about the dog will be stored here
    indexes = user_df.index

    # iterating over each detected object
    for index in indexes:
        # for top left corner
        x1 = int(user_df['xmin'][index])
        y1 = int(user_df['ymin'][index])

        # for bottom right corner
        x2 = int(user_df['xmax'][index])
        y2 = int(user_df['ymax'][index])

        label = user_df['name'][index]
        # if dog is detected at index 0 then it searches for the confidence score of first index
        conf_score = user_df['confidence'][index]
        text = f"{label} {conf_score:.2f}"

    # drawing the bounding box and text on the image
        cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)


    cv2.imshow(f"Detected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# calling a function
example_imageDetection()
