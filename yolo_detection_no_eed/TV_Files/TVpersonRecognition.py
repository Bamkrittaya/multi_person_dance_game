from ultralytics import YOLO
import cv2
import numpy as np
import math
import time

# ! SAVED COLORS
BLACK = (0,0,0)
UKBLUE = (0,51,160)
WHITE = (255,255,255)

# ! OTHER STYLES
FONT = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
THICKNESS = 2

# Load the pose model
model = YOLO("yolo_weights/yolo11n-pose.pt")

# Define keypoint labels (focus on the ones relevant for glasses)
keypoint_labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear"
]
classNames = ["person"]  # Only tracking "person" objects


# Calculate midpoint between two points
def calculate_midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

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

        # Draw mustache as a line extending left from the nose
        mustache_length = int(eye_distance * lens_size_factor * 1.2)  # Length of the mustache line
        mustache_offset_y = int(eye_distance * lens_size_factor * 0.8)  # Vertical offset below the nose

        # Define start and end points for the left mustache line
        mustache_start = (nose[0], nose[1] + int(mustache_offset_y * 1.2) ) # Start just below the nose
        mustache_end = (nose[0] - mustache_length, nose[1] + mustache_offset_y + int(mustache_length * 0.3))  # Angle down to the left

        # Draw the mustache line
        cv2.line(frame, mustache_start, mustache_end, BLACK, frame_size)
        
        # Define start and end points for the  right mustache line
        mustache_start = (nose[0], nose[1] + int(mustache_offset_y * 1.2) ) # Start just below the nose
        mustache_end = (nose[0] + mustache_length, nose[1] + mustache_offset_y + int(mustache_length * 0.3))  # Angle down to the left

        # Draw the mustache line
        cv2.line(frame, mustache_start, mustache_end, BLACK, frame_size)

    else:
        # Handle case where keypoints are missing
        cv2.putText(frame, "Missing Face Keypoints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return frame


def main():
    person_detected_count = 0
    last_seen_time = None
    frame_interval = 2
    detection_time_threshold = 5
    person_label = 0
    lens_size_factor = 0.4
    start_time = time.time()

    cap = cv2.VideoCapture(0)

    print("Starting detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Check if enough time has passed for the next frame
        if (time.time() - start_time > frame_interval):
            start_time = time.time()

            # take an estimation from the model
            results = model(frame)

            # check if any of the results are identified as a "person", if so add one to counter, and update last_seen_time
            if any(int(box.cls[0]) == person_label for box in results[0].boxes):
                person_detected_count += 1
                last_seen_time = time.time()

                # if a person has been seen for two consecutive frames
                if person_detected_count >= 2:
                    last_seen_time = time.time() # possibly redundant
                    print("Seen person wishing to interact. Stopping detection, starting interaction.")

                    # while the time since we last saw a person is less than threshold
                    # 1. take another frame
                    # 2. run another estimation
                    # 3. check for person
                    # 4. draw glasses
                    while (time.time() - last_seen_time) < detection_time_threshold:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = model(frame)

                        
                        if any(int(box.cls[0]) == person_label for box in results[0].boxes):
                            last_seen_time = time.time()

                            for r in results:
                                if hasattr(r, "keypoints"):
                                    keypoints = r.keypoints.xy[0]
                                    keypoints_list = keypoints.tolist()
                                    frame = draw_glasses(frame, keypoints_list, lens_size_factor)

                            cv2.imshow("Pose Estimation with Dynamic Glasses", frame)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                    person_detected_count = 0
                    print("Threshold passed, no longer detecting person. Stopping interaction, resuming detection")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()