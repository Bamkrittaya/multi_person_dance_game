from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the YOLO model for pose detection
model = YOLO("yolo_weights/yolo11n-pose.pt")

# Class names (for this model, we're only interested in "person")
classNames = ["person"]  # Only tracking "person" objects

while True:
    success, img = cap.read()
    if not success:
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

    # Display the webcam feed with bounding boxes and labels
    cv2.imshow('Webcam - Person Detection', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
