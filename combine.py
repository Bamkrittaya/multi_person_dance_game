"""
combine.py
-----------
YOLO (Ultralytics) + Just Dance (TFLite) integration.

Detects multiple players in real-time using YOLO,
draws pose skeletons, feeds each player's cropped image into the
Just Dance model, and displays pose-based scores for each player.
Pauses automatically if more than 2 people are detected.
"""

import cv2
import numpy as np
import sys, os
from ultralytics import YOLO
import tensorflow as tf

# --- 1. Import your Just Dance modules ---
sys.path.append(os.path.join(os.getcwd(), 'just_dance'))

try:
    from just_dance_model import JustDanceModel
    from just_dance_score import calculate_score  # optional
except ImportError:
    print("⚠️ Could not import Just Dance modules. Check folder structure and file names.")

# --- 2. Load YOLO and Just Dance Models ---
print("🔍 Loading YOLO model...")
yolo_model = YOLO("yolo/yolo_weights/yolo11n-pose.pt")

print("🎵 Loading Just Dance TFLite model...")
tflite_path = "just_dance/model/model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Automatically detect model input size
input_shape = input_details[0]['shape']
input_height, input_width = int(input_shape[1]), int(input_shape[2])
print(f"📏 Model expects input size: {input_width}x{input_height}")

# --- 3. Helper: Run the Just Dance model ---
def run_just_dance_inference(frame):
    """Preprocess cropped person image and get dance confidence score."""
    img = cv2.resize(frame, (input_width, input_height))
    img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    score = float(np.mean(output_data))
    return score


# --- 4. Start webcam feed ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

print("✅ Press 'q' to quit, 'r' to resume after pause.")

paused = False
max_players = 2

# --- 5. Main game loop ---
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        # Mirror the camera (horizontal flip)
        frame = cv2.flip(frame, 1)

        # Step 1: Detect people + keypoints with YOLO
        results = yolo_model(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        keypoints_all = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []
        num_people = len(boxes)

        display = frame.copy()
        

        # Step 2: Pause if too many players
        if num_people > max_players:
            paused = True
            cv2.putText(display, f"⚠️ Too many players ({num_people})! Max = {max_players}.",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, "Press 'r' to resume.",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("🕺 Multi-Person Just Dance", display)
            continue  # skip scoring this frame

        # Step 3: Draw skeletons + run scoring
        for i, (box, keypoints) in enumerate(zip(boxes, keypoints_all)):
            x1, y1, x2, y2 = map(int, box[:4])
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # --- Draw keypoints ---
            for kp in keypoints:
                if kp[0] > 0 and kp[1] > 0:
                    cv2.circle(display, (int(kp[0]), int(kp[1])), 3, (0, 255, 255), -1)

            # --- Draw skeleton lines (COCO keypoint pairs) ---
            skeleton_pairs = [
                (5, 7), (7, 9),     # left arm
                (6, 8), (8, 10),    # right arm
                (11, 13), (13, 15), # left leg
                (12, 14), (14, 16), # right leg
                (5, 6), (11, 12),   # shoulders, hips
                (5, 11), (6, 12)    # torso
            ]
            for a, b in skeleton_pairs:
                if keypoints[a][0] > 0 and keypoints[b][0] > 0:
                    cv2.line(display,
                             (int(keypoints[a][0]), int(keypoints[a][1])),
                             (int(keypoints[b][0]), int(keypoints[b][1])),
                             (255, 0, 0), 2)

            # --- Run Just Dance model ---
            score = run_just_dance_inference(person_crop)

            # --- Draw bounding box + score ---
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(display, f"Player {i+1}: {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Step 4: Show window
        cv2.imshow("🕺 Multi-Person Just Dance", display)

    # --- Key controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        paused = False  # resume

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
