from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime

UKBlue = (0, 51, 160)
Red = (255, 0, 0)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the YOLO model for pose detection
model = YOLO("yolo_weights/yolo11n-pose.pt")

# Class names (for this model, we're only interested in "person")
classNames = ["person"]  # Only tracking "person" objects

# Get frame height
_, frame = cap.read()
frame_height = frame.shape[0]
halfway_point = frame_height // 2  # Midpoint of the screen

# Cooldown setup
last_fall_time = 0  # Last logged fall time
cooldown_seconds = 3  # Cooldown duration

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform inference
    results = model(img, stream=True, conf=0.3, verbose=False)  # Set confidence threshold to 0.3, verbose false removes outputs every 4ms
    

    fall_detected_full = False
    fall_detected_clipped = False

    # Process results
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Only continue if the detected class is "person"
            if int(box.cls[0]) == 0:  # Assuming "person" class index is 0
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int values
                confidence = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to two decimals

                # Draw the full bounding box (original detection)
                cv2.rectangle(img, (x1, y1), (x2, y2), UKBlue, 3)

                # Display label and confidence
                label = f"{classNames[0]} {confidence:.2f}"
                org = (x1, y1 - 10)  # Position label above the box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                thickness = 2
                cv2.putText(img, label, org, font, fontScale, UKBlue, thickness)

                full_height = y2 - y1
                full_width = x2 - x1

                # Check fall condition for the full bounding box
                if full_width > full_height:
                    cv2.putText(img, "Fall Detected", (x1, y1 - 30), font, fontScale, UKBlue, thickness)
                    fall_detected_full = True  # Mark fall detected in full box

                # Restrict bounding box to bottom half of the screen
                if y2 > halfway_point:
                    clipped_y1 = max(y1, halfway_point)  # Adjust the top of the box if needed

                    # Draw the clipped bounding box in red
                    cv2.rectangle(img, (x1, clipped_y1), (x2, y2), Red, 3)

                    clipped_height = y2 - clipped_y1
                    clipped_width = x2 - x1

                    # Check fall condition for the clipped bounding box
                    if clipped_width > clipped_height:
                        cv2.putText(img, "Fall Detected", (x1, clipped_y1 - 30), font, fontScale, Red, thickness)
                        fall_detected_clipped = True  # Mark fall detected in clipped box

    # Handle fall detection cooldown
    current_time = time.time()
    if (fall_detected_full or fall_detected_clipped) and (current_time - last_fall_time > cooldown_seconds):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if fall_detected_full:
            print(f"[{timestamp}] Fall Detected in Full Box")
        if fall_detected_clipped:
            print(f"[{timestamp}] Fall Detected in Clipped Box")
        last_fall_time = current_time  # Reset cooldown timer

    # Display the webcam feed with bounding boxes and labels
    cv2.imshow('Webcam - Person Detection with Clipped Box', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
