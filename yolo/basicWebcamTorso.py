from ultralytics import YOLO
import cv2
import torch
import numpy as np
import math

# Load the pose model
model = YOLO("yolo_weights/yolo11n-pose.pt")

# Define keypoint labels (update according to your model's expected keypoints order)
keypoint_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Calculate midpoint between two points
def calculate_midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

# Calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Main function to draw parallelogram and calculate distances
def process_torso_parallelogram(frame, keypoints_list):

        # 1--2
        # |  |
        # 4--3

    # Define the four vertices of the parallelogram
    p1 = tuple(map(int, keypoints_list[5]))  # Convert to int and tuple
    p2 = tuple(map(int, keypoints_list[6]))
    p3 = tuple(map(int, keypoints_list[12]))
    p4 = tuple(map(int, keypoints_list[11]))

    # Draw the parallelogram
    points = [p1, p2, p3, p4]
    cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Calculate midpoints of each side
    midpoint_1_2 = calculate_midpoint(p1, p2)
    midpoint_2_3 = calculate_midpoint(p2, p3)
    midpoint_3_4 = calculate_midpoint(p3, p4)
    midpoint_4_1 = calculate_midpoint(p4, p1)

    # Draw midpoints on the frame for visualization
    for midpoint in [midpoint_1_2, midpoint_2_3, midpoint_3_4, midpoint_4_1]:
        cv2.circle(frame, (int(midpoint[0]), int(midpoint[1])), radius=5, color=(255, 0, 0), thickness=-1)

    # Calculate distances between opposing midpoints
    distance_height = calculate_distance(midpoint_1_2, midpoint_3_4)
    distance_width = calculate_distance(midpoint_2_3, midpoint_4_1)

    # Draw height and width lines with text printing dimension
    cv2.line(frame, tuple(map(int, midpoint_1_2)), tuple(map(int, midpoint_3_4)), (0, 255, 0), 1)
    cv2.line(frame, tuple(map(int, midpoint_2_3)), tuple(map(int, midpoint_4_1)), (0, 255, 0), 1)

    # Convert distances to strings and display them on the frame
    cv2.putText(frame, "Height: " + str(round(distance_height, 2)), (int(midpoint_1_2[0]), int(midpoint_1_2[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.putText(frame, "Width: " + str(round(distance_width, 2)), (int(midpoint_2_3[0]), int(midpoint_2_3[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    

    if(distance_height < distance_width):
        cv2.putText(frame, "Fall Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    
    # # Print distances for reference
    # print("Distance between midpoint 1-2 and 3-4:", distance_height)
    # print("Distance between midpoint 2-3 and 4-1:", distance_width)

    return frame


# Start video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    # ret is bool for if frame is availible, frame is an image array vector
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model on the current frame
    results = model(frame)

    # Process each detected person in the results
    for r in results:
        # hasattr returns true if the object as the attribute
        if hasattr(r, "keypoints"):
            # Access XY coordinates directly from the tensor
            keypoints = r.keypoints.xy[0]  # Get keypoints for the first person (change index if needed)\

            # convert tensor list to python list
            keypoints_list = keypoints.tolist()

            # Check if specific keypoints are non-zero
            if all(kp != [0.0, 0.0] for kp in [keypoints_list[5], keypoints_list[6], keypoints_list[11], keypoints_list[12]]):
               process_torso_parallelogram(frame, keypoints_list)

            # Display text for when the proper points arent in the view of the camera
            else:   
                cv2.putText(frame, "Missing Datapoints", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Iterate through keypoints and plot them
            for i, (x, y) in enumerate(keypoints):
                x, y = int(x), int(y)  # Convert tensor values to integers for display

                # HELPER print out keypoints for labeling 
                # print(keypoint_labels[i], ": ()", x, ", ", y, ")\n")

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, keypoint_labels[i], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Display the output with keypoints
    cv2.imshow("Pose Estimation", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
