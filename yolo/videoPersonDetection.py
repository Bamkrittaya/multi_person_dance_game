from ultralytics import YOLO
import cv2
import math

# 🚨 CHANGE 1: Specify the path to your .mov file 🚨
# Replace 'path/to/your/video.mov' with the actual file path.
video_file_path = "testingData/peopleWalking.mp4" 

# Start video capture using the file path
cap = cv2.VideoCapture(video_file_path)

# You can keep these, but they might be ignored if the video file has a different resolution
# If the video is smaller than 640x480, it will use the video's resolution.
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the YOLO model for pose detection
model = YOLO("yolo_weights/yolo11n-pose.pt")

# Class names (for this model, we're only interested in "person")
classNames = ["person"]  # Only tracking "person" objects

while True:
    success, img = cap.read()
    
    # 🚨 CHANGE 2: Check for end of video file 🚨
    # If success is False, it means we reached the end of the video.
    if not success:
        print("End of video file reached. Exiting loop.")
        break

    # Perform inference
    results = model(img, stream=True, conf=0.3)  # Set confidence threshold to 0.3

    # Process results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Only continue if the detected class is "person"
            if int(box.cls[0]) == 0:  # Assuming "person" class index is 0
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int values
                confidence = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to two decimals

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Display label and confidence
                label = f"{classNames[0]} {confidence:.2f}"
                org = (x1, y1 - 10)  # Position label above the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, label, org, font, fontScale, color, thickness)

    # Display the processed video frame
    cv2.imshow('Video - Person Detection', img)

    # Break the loop if 'q' is pressed (you may need to increase cv2.waitKey() 
    # value for video playback speed adjustment)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()