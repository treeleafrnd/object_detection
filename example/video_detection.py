
import torch
import cv2

# Download model from github
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def example_videoDetection():
    cap = cv2.VideoCapture('../experiment/Videos/road_traffic_video.mp4')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on image
        result = model(frame)

        # specifying the object to be detected from the video
        user_input = ['car']

        # Convert detected result to pandas data frame
        data_frame = result.pandas().xyxy[0]
        print('data_frame:')
        print(data_frame)

        # filtering the user specified object
        user_df = data_frame[data_frame['name'].isin(user_input)]

        # Get indexes of all of the rows
        indexes = user_df.index
        for index in indexes:
            # Find the coordinate of top left corner of bounding box
            x1 = int(user_df['xmin'][index])
            y1 = int(user_df['ymin'][index])
            # Find the coordinate of right bottom corner of bounding box
            x2 = int(user_df['xmax'][index])
            y2 = int(user_df['ymax'][index ])

            # Find label name
            label = user_df['name'][index ]
            # Find confidence score of the model
            conf = user_df['confidence'][index]
            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
            cv2.putText(frame, text, (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255,255,0), 2)

        cv2.imshow('Video Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


example_videoDetection()