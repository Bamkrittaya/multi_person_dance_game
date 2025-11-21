# just_dance_controller.py
"""
Main controller for Just Dance YOLOv8 Pose game.

- Left: tutorial video (precomputed keypoints, no YOLO at runtime → fast)
- Right: live webcam with YOLOv8 pose
- Full-body skeleton drawing
- Asynchronous music playback from songs_audio/<song>.mp3
- Angle-based scoring for elbows + knees
"""

import os
import cv2
import numpy as np
from threading import Thread
from playsound import playsound
from ultralytics import YOLO

from videoPoseDetection import (
    precompute_video_angles,
    keypoints_to_angles,
    ANGLE_TRIPLETS,
)

ANGLE_TOLERANCE = 45.0  # degrees, lower = stricter

# Path to YOLOv8 pose model (same as in videoPoseDetection.py)
MODEL_PATH = "yolo_weights/yolov8s-pose.pt"


# ============================================================
#  FULL-BODY SKELETON
# ============================================================

# COCO full-body connections (0–16)
FULL_BODY_PAIRS = [
    (0, 1), (1, 3),        # nose → eyes/ears
    (0, 2), (2, 4),
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 6),                # shoulders
    (5, 11), (6, 12),      # torso
    (11, 12),              # hips
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
]


def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    """
    Draws 17 COCO keypoints + full-body skeleton.

    Args:
        frame: BGR image
        keypoints: np.ndarray (17, 2)
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return frame

    # Draw keypoints
    for x, y in keypoints:
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    # Draw skeleton lines
    for a, b in FULL_BODY_PAIRS:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return frame


# ============================================================
#  MUSIC (ASYNC)
# ============================================================

def play_music_async(mp3_path: str):
    """
    Plays music on a background thread so video/YOLO don't freeze.
    """
    if not os.path.exists(mp3_path):
        print(f"[warning] Music file not found: {mp3_path}")
        return

    def _play():
        try:
            playsound(mp3_path)
        except Exception as e:
            print(f"[warning] playsound error: {e}")

    Thread(target=_play, daemon=True).start()


# ============================================================
#  CONTROLLER CLASS
# ============================================================

class JustDanceController:
    def __init__(self, reference_video: str, camera_index: int = 0):
        self.reference_video = reference_video
        self.camera_index = camera_index

        # YOLOv8 pose for webcam
        self.model = YOLO(MODEL_PATH)

        # Filled after precompute
        self.ref_angles = None      # (T, J)
        self.ref_keypoints = None   # (T, 17, 2)
        self.player_angles = []     # list of (J,) over time

    # ---------------------------------------------------------
    # PRECOMPUTE
    # ---------------------------------------------------------
    def precompute_reference(self):
        """
        Loads or generates reference angles + keypoints from tutorial video.
        """
        self.ref_angles, self.ref_keypoints = precompute_video_angles(
            self.reference_video
        )

    # ---------------------------------------------------------
    # SCORING UTILITIES
    # ---------------------------------------------------------
    @staticmethod
    def score_calculator(ref_joint, player_joint):
        """
        Score for a single joint over time (0–100).

        ref_joint, player_joint: 1D arrays of angles over T frames.
        """
        T = min(len(ref_joint), len(player_joint))
        ref_joint = ref_joint[:T]
        player_joint = player_joint[:T]

        # ignore frames with zeros (no detection)
        mask = ~((ref_joint == 0) | (player_joint == 0))
        if mask.sum() == 0:
            return 0.0

        diff = np.abs(ref_joint[mask] - player_joint[mask])
        sim = np.clip(1.0 - diff / ANGLE_TOLERANCE, 0.0, 1.0)
        return float(sim.mean() * 100.0)

    def final_score(self, ref_angles, player_angles):
        """
        Average score over all joints (0–100).

        ref_angles, player_angles: (T, J)
        """
        T = min(ref_angles.shape[0], player_angles.shape[0])
        ref = ref_angles[:T]
        player = player_angles[:T]

        joint_scores = []
        for j in range(ref.shape[1]):
            s = self.score_calculator(ref[:, j], player[:, j])
            joint_scores.append(s)

        return float(np.mean(joint_scores)) if joint_scores else 0.0

    # ---------------------------------------------------------
    # MAIN GAME
    # ---------------------------------------------------------
    def run_game(self, show_window: bool = True) -> float:
        """
        Run the full game:
        - Precompute reference
        - Play tutorial + webcam
        - Draw full-body skeletons
        - Compute final score
        Returns:
            final score (0–100)
        """
        # 1) Precompute angles + keypoints
        self.precompute_reference()

        # 2) Start music (if there is a matching mp3)
        base = os.path.splitext(os.path.basename(self.reference_video))[0]
        mp3_path = os.path.join("songs_audio", base + ".mp3")
        play_music_async(mp3_path)

        # 3) Open video + webcam
        cap_ref = cv2.VideoCapture(self.reference_video)
        cap_cam = cv2.VideoCapture(self.camera_index)

        running_score = 0.0

        while True:
            ret_ref, frame_ref = cap_ref.read()
            ret_cam, frame_cam = cap_cam.read()

            if not ret_ref or not ret_cam:
                break

            frame_idx = len(self.player_angles)

            # --------- Tutorial side: use PRECOMPUTED keypoints ---------
            if frame_idx < len(self.ref_keypoints):
                kpts_ref = self.ref_keypoints[frame_idx]
                frame_ref = draw_skeleton(frame_ref, kpts_ref, color=(0, 0, 255))

            # --------- Webcam side: YOLOv8 live detection ---------
            results = self.model(frame_cam, conf=0.3, verbose=False)

            if len(results) > 0 and results[0].keypoints is not None:
                kpts_player = results[0].keypoints.xy[0].cpu().numpy()  # (17,2)
                frame_cam = draw_skeleton(frame_cam, kpts_player, color=(0, 255, 0))
                angles_player = keypoints_to_angles(kpts_player)
            else:
                angles_player = np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32)

            self.player_angles.append(angles_player)

            # --------- Live score preview (per-frame approximate) ---------
            if frame_idx < len(self.ref_angles):
                ref_angles_frame = self.ref_angles[frame_idx]
                frame_score = self.final_score(
                    ref_angles_frame.reshape(1, -1),
                    angles_player.reshape(1, -1),
                )
                running_score = 0.9 * running_score + 0.1 * frame_score

            # --------- Render side-by-side ---------
            if show_window:
                h = 480
                frame_ref_r = cv2.resize(
                    frame_ref,
                    (int(frame_ref.shape[1] * h / frame_ref.shape[0]), h),
                )
                frame_cam_r = cv2.resize(
                    frame_cam,
                    (int(frame_cam.shape[1] * h / frame_cam.shape[0]), h),
                )
                combined = np.hstack((frame_ref_r, frame_cam_r))

                cv2.putText(
                    combined,
                    f"Score: {running_score:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Just Dance YOLOv8 Pose", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap_ref.release()
        cap_cam.release()
        cv2.destroyAllWindows()

        # --------- Final Score ---------
        if not self.player_angles:
            return 0.0

        player_arr = np.stack(self.player_angles, axis=0)  # (T, J)
        final = self.final_score(self.ref_angles, player_arr)
        print(f"\n💯 Final Score: {final:.2f}\n")
        return final


# ============================================================
#  CLI TEST
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    controller = JustDanceController(args.video, args.camera)
    score = controller.run_game(show_window=True)
    print("✨ Game finished with score:", score)
