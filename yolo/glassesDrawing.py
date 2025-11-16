from ultralytics import YOLO
import cv2
import numpy as np
import math

# Load the pose model
model = YOLO("yolo_weights/yolo11n-pose.pt")

# Define keypoint labels (focus on the ones relevant for glasses)
keypoint_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear"
]
BLACK = (0,0,0)

# Calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Draw glasses on detected face
def draw_glasses(frame, keypoints_list, lens_size_factor=1.0):
    # Ensure required keypoints (eyes, nose, ears) are detected
    if all(kp != [0.0, 0.0] for kp in [keypoints_list[0], keypoints_list[1], keypoints_list[2], keypoints_list[3], keypoints_list[4]]):
        # Get the coordinates of the nose, eyes, and ears
        nose = tuple(map(int, keypoints_list[0]))
        left_eye = tuple(map(int, keypoints_list[1]))
        right_eye = tuple(map(int, keypoints_list[2]))
        left_ear = tuple(map(int, keypoints_list[3]))
        right_ear = tuple(map(int, keypoints_list[4]))

        # Calculate the distance between the eyes
        eye_distance = calculate_distance(left_eye, right_eye)

        # Scale the size of the glasses based on the eye distance
        lens_radius = int(eye_distance * lens_size_factor)  # Adjust the lens size with a factor
        frame_size = int((eye_distance * lens_size_factor)/4)

        # Draw circles around the eyes (adjust size based on distance)
        cv2.circle(frame, left_eye, lens_radius, BLACK, -1)
        cv2.circle(frame, right_eye, lens_radius, BLACK, -1)

        # Calculate the edge points of the frames based on the lens radius
        left_frame_start = (left_eye[0] + lens_radius, left_eye[1])  
        left_frame_end = (left_ear[0], left_ear[1] - lens_radius) 
        right_frame_start = (right_eye[0] - lens_radius, right_eye[1])  
        right_frame_end = (right_ear[0], right_ear[1] - lens_radius)  

        # Draw lines (glasses frames) from each eye to the corresponding ear
        cv2.line(frame, left_frame_start, left_frame_end, BLACK, frame_size)  # Left frame
        cv2.line(frame, right_frame_start, right_frame_end, BLACK, frame_size)  # Right frame

        # Optional: Draw a line connecting the two eyes to simulate a bridge
        cv2.line(frame, left_frame_start, right_frame_start, BLACK, frame_size)  # Bridge of the glasses

    else:
        # Handle case where keypoints are missing
        cv2.putText(frame, "Missing Face Keypoints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return frame


# Start video capture (webcam)
cap = cv2.VideoCapture(0)

# Define the adjustable constant for lens size scaling
lens_size_factor = 0.4  

while True:
    # ret is bool for if frame is available, frame is an image array
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on the current frame
    results = model(frame)

    # Process each detected person in the results
    for r in results:
        # Check if keypoints are available
        if hasattr(r, "keypoints"):
            # Access XY coordinates directly from the tensor
            keypoints = r.keypoints.xy[0] 

            # Convert tensor list to Python list
            keypoints_list = keypoints.tolist()

            # Process glasses if all required keypoints are available
            frame = draw_glasses(frame, keypoints_list, lens_size_factor)

    # Display the output with glasses
    cv2.imshow("Pose Estimation with Dynamic Glasses", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

