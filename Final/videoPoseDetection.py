# videoPoseDetection.py
"""
Precompute tutorial video keypoints + joint angles using YOLOv8 Pose.

Outputs:
- precomputed/<song>_reference_angles.npy      (T, J)
- precomputed/<song>_reference_keypoints.npy  (T, 17, 2)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# Path to YOLOv8 pose weights (put the .pt file here)
MODEL_PATH = "yolo_weights/yolov8s-pose.pt"

# Load YOLOv8 model once
_MODEL = YOLO(MODEL_PATH)

# ============================================================
#  JOINT LIST AND ANGLE COMPUTATION
# ============================================================

# COCO keypoint indices (0–16)
# We'll compute angles at elbows and knees:
# 5-7-9 : left elbow, 6-8-10 : right elbow
# 11-13-15 : left knee, 12-14-16 : right knee
ANGLE_TRIPLETS = [
    (5, 7, 9),      # left elbow
    (6, 8, 10),     # right elbow
    (11, 13, 15),   # left knee
    (12, 14, 16),   # right knee
]


def _joint_angle(a, b, c):
    """
    Compute angle ABC (in degrees) given points a, b, c (2D).
    """
    ba = a - b
    bc = c - b
    num = np.dot(ba, bc)
    den = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.clip(num / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def keypoints_to_angles(kpts):
    """
    Convert full set of 17 keypoints (x,y) to our joint angles.

    Args:
        kpts: np.ndarray of shape (17, 2)
    Returns:
        np.ndarray of shape (J,) where J = len(ANGLE_TRIPLETS)
    """
    angles = []
    for i, j, k in ANGLE_TRIPLETS:
        a, b, c = kpts[i], kpts[j], kpts[k]
        angles.append(_joint_angle(a, b, c))
    return np.array(angles, dtype=np.float32)


# ============================================================
#  PRECOMPUTE FUNCTION
# ============================================================

def precompute_video_angles(video_path, out_dir="precomputed"):
    """
    Run YOLOv8 pose over the entire tutorial video and store:
        - per-frame joint angles
        - per-frame 17 keypoints

    Args:
        video_path (str): path to tutorial video
        out_dir (str): directory for .npy files

    Returns:
        angles: np.ndarray (T, J)
        keypoints: np.ndarray (T, 17, 2)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    out_angles = os.path.join(out_dir, f"{base}_reference_angles.npy")
    out_keypoints = os.path.join(out_dir, f"{base}_reference_keypoints.npy")

    # Load from cache if both files exist
    if os.path.exists(out_angles) and os.path.exists(out_keypoints):
        print(f"[precompute] Using cached angles + keypoints for {base}")
        return np.load(out_angles), np.load(out_keypoints)

    print(f"[precompute] Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    all_angles = []
    all_keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = _MODEL(frame, conf=0.3, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            # No detection → zeros (so frame can be ignored in scoring)
            all_angles.append(np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32))
            all_keypoints.append(np.zeros((17, 2), dtype=np.float32))
        else:
            # Take the first detected person
            kpts = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2)
            all_keypoints.append(kpts)
            all_angles.append(keypoints_to_angles(kpts))

    cap.release()

    angles = np.stack(all_angles)      # (T, J)
    keypoints = np.stack(all_keypoints)  # (T, 17, 2)

    np.save(out_angles, angles)
    np.save(out_keypoints, keypoints)

    print(f"[precompute] Saved angles   → {out_angles}")
    print(f"[precompute] Saved kpts     → {out_keypoints}")
    print(f"[precompute] Frames: {angles.shape[0]}, joints: {angles.shape[1]}")
    return angles, keypoints

def precompute_video_keypoints(video_path, out_dir="precomputed"):
    """
    Extract full keypoints (17,2) for each frame of the video.
    Saves to .npy so we can reuse them for FAST playback.

    Returns:
        keypoints_arr: np.ndarray of shape (T, 17, 2)
        out_path: str (saved .npy path)
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_keypoints.npy")

    # load cached
    if os.path.exists(out_path):
        kpts = np.load(out_path)
        print(f"[precompute] Loaded cached keypoints from {out_path}")
        return kpts, out_path

    cap = cv2.VideoCapture(video_path)
    all_kpts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = _MODEL(frame, conf=0.3, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            all_kpts.append(np.zeros((17,2), dtype=np.float32))
        else:
            k = results[0].keypoints.xy[0].cpu().numpy()  # (17,2)
            all_kpts.append(k)

    cap.release()
    keypoints_arr = np.stack(all_kpts, axis=0)  # (T,17,2)
    np.save(out_path, keypoints_arr)
    print(f"[precompute] Saved keypoints {keypoints_arr.shape} → {out_path}")
    return keypoints_arr, out_path


# ============================================================
#  CLI TEST
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    precompute_video_angles(args.video)
    print("Done.")
