# just_dance_controller.py

import os
import cv2
import numpy as np
from ultralytics import YOLO
import pygame

from videoPoseDetection import (
    precompute_video_angles,
    keypoints_to_angles,
    ANGLE_TRIPLETS,  # now full-body angle set
)

# Beginner-friendly: larger tolerance → easier scoring
ANGLE_TOLERANCE = 45  # degrees, higher = more forgiving
MODEL_PATH = "yolo_weights/yolov8s-pose.pt"

# Path to countdown video (5 sec, e.g. "3-2-1-start")
COUNTDOWN_VIDEO_PATH = "videos/countdown.mp4"


# ---------------------------------------------------------
# Full-body skeleton connections (COCO 17 keypoints)
# ---------------------------------------------------------
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
    Draws 17 COCO keypoints + full skeleton on frame.

    keypoints: (17, 2) array
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return frame

    # Draw joints
    for x, y in keypoints:
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    # Draw bones
    for a, b in FULL_BODY_PAIRS:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return frame


class JustDanceController:
    def __init__(
        self,
        reference_video: str,
        audio_file: str,
        reference_angles_path: str,
        reference_keypoints_path: str,
        camera_index: int = 0,
        countdown_video: str = COUNTDOWN_VIDEO_PATH,
    ):
        self.reference_video = reference_video
        self.audio_file = audio_file
        self.reference_angles_path = reference_angles_path
        self.reference_keypoints_path = reference_keypoints_path
        self.camera_index = camera_index
        self.countdown_video = countdown_video

        # YOLOv8 pose model for webcam
        self.model = YOLO(MODEL_PATH)

        # Filled in precompute_reference
        self.ref_angles = None      # (T, J)
        self.ref_keypoints = None   # (T, 17, 2)
        self.player_angles = []     # list of (J,) arrays

    # ---------------------------------------------------------
    # Precompute / load reference arrays
    # ---------------------------------------------------------
    def precompute_reference(self):
        """
        Load cached precomputed angles + keypoints if npy files exist.
        If not, compute and save them using videoPoseDetection.
        """
        have_npy = (
            os.path.exists(self.reference_angles_path)
            and os.path.exists(self.reference_keypoints_path)
        )

        if have_npy:
            print("[precompute] Loading cached FAST reference angles/keypoints…")
            self.ref_angles = np.load(self.reference_angles_path)
            self.ref_keypoints = np.load(self.reference_keypoints_path)
            return

        print("[precompute] npy files missing, computing from video…")
        angles, keypoints = precompute_video_angles(self.reference_video)

        self.ref_angles = angles
        self.ref_keypoints = keypoints

        # Save under the provided paths
        np.save(self.reference_angles_path, angles)
        np.save(self.reference_keypoints_path, keypoints)
        print(f"[precompute] Saved angles to {self.reference_angles_path}")
        print(f"[precompute] Saved keypoints to {self.reference_keypoints_path}")

    # ---------------------------------------------------------
    # Countdown video
    # ---------------------------------------------------------
    def play_countdown(self, show_window: bool = True):
        """
        Plays a countdown video (e.g. 5s) before the game starts.
        """
        if not show_window:
            return

        if not self.countdown_video or not os.path.exists(self.countdown_video):
            print(f"[countdown] Countdown video not found: {self.countdown_video}")
            return

        cap_cd = cv2.VideoCapture(self.countdown_video)
        if not cap_cd.isOpened():
            print(f"[countdown] Could not open countdown video: {self.countdown_video}")
            return

        while True:
            ret, frame = cap_cd.read()
            if not ret:
                break
            cv2.imshow("Get Ready!", frame)
            # ~30 FPS
            if cv2.waitKey(33) & 0xFF == ord("q"):
                break

        cap_cd.release()
        cv2.destroyWindow("Get Ready!")

    # ---------------------------------------------------------
    # Scoring utilities
    # ---------------------------------------------------------
    @staticmethod
    def score_calculator(ref_joint, player_joint):
        """
        Compute score (0–100) for a single joint over time.
        ref_joint, player_joint: 1D arrays with angles over frames.
        """
        T = min(len(ref_joint), len(player_joint))
        ref_joint = ref_joint[:T]
        player_joint = player_joint[:T]

        # Mask out frames with no detection
        mask = ~((ref_joint == 0) | (player_joint == 0))
        if mask.sum() == 0:
            return 0.0

        diff = np.abs(ref_joint[mask] - player_joint[mask])

        # Beginner mode: larger tolerance → easier
        sim = np.clip(1.0 - diff / ANGLE_TOLERANCE, 0.0, 1.0)
        return float(sim.mean() * 100.0)

    def final_score(self, ref_angles, player_angles):
        """
        Compute final score over all joints (0–100).
        ref_angles, player_angles: (T, J)
        """
        T = min(ref_angles.shape[0], player_angles.shape[0])
        ref = ref_angles[:T]
        player = player_angles[:T]

        scores = []
        for j in range(ref.shape[1]):
            s = self.score_calculator(ref[:, j], player[:, j])
            scores.append(s)

        return float(np.mean(scores)) if scores else 0.0

    # ---------------------------------------------------------
    # Main game loop with pause logic
    # ---------------------------------------------------------
    def run_game(self, show_window: bool = True) -> float:
        """
        Full game:
        - Load or compute reference arrays
        - Play countdown
        - Start music (controlled via pause/unpause)
        - Open tutorial video (FAST) + webcam
        - Draw skeletons
        - Pause video/audio when:
            * 0 humans → pause
            * 1 human → play
            * ≥2 humans → pause
        - Compute score only when exactly 1 human.
        """
        # 1) Prepare reference
        self.precompute_reference()

        # 2) Countdown
        self.play_countdown(show_window=show_window)

        # 3) Init audio (pygame) — start paused
        pygame.mixer.init()
        has_audio = False
        try:
            if self.audio_file and os.path.exists(self.audio_file):
                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.play()
                pygame.mixer.music.pause()
                has_audio = True
                print(f"[audio] Loaded and started (paused): {self.audio_file}")
            else:
                print(f"[audio] Audio file not found: {self.audio_file}")
        except Exception as e:
            print(f"[audio] Error loading audio: {e}")

        # 4) Open video + webcam
        cap_ref = cv2.VideoCapture(self.reference_video)
        cap_cam = cv2.VideoCapture(self.camera_index)

        running_score = 0.0
        status_text = "Waiting for player…"
        paused = True
        dance_idx = 0            # index into reference arrays
        last_ref_frame = None    # to freeze video when paused

        while True:
            # --- Webcam frame ---
            ret_cam, frame_cam = cap_cam.read()
            if not ret_cam:
                print("[game] Webcam ended or not available.")
                break

            # Mirror webcam for display + YOLO (safe for angles)
            frame_cam = cv2.flip(frame_cam, 1)

            # --- YOLO on webcam: crash-proof ---
            results = self.model(frame_cam, conf=0.3, verbose=False)
            num_people = 0
            kpts_player = None

            if len(results) > 0 and results[0].keypoints is not None:
                xy = results[0].keypoints.xy
                # xy: (num_people, 17, 2) or None
                if xy is not None and xy.numel() > 0:
                    num_people = xy.shape[0]
                    # Use first person for scoring
                    kpts_player = xy[0].cpu().numpy()

            # --- Decide pause/play state from num_people ---
            if num_people == 1:
                # Resume
                if paused:
                    paused = False
                    if has_audio:
                        try:
                            pygame.mixer.music.unpause()
                        except Exception as e:
                            print(f"[audio] unpause error: {e}")
                status_text = "Dancing! (1 player)"
            elif num_people == 0:
                # Pause: no person
                if not paused:
                    paused = True
                    if has_audio:
                        try:
                            pygame.mixer.music.pause()
                        except Exception as e:
                            print(f"[audio] pause error: {e}")
                status_text = "No player detected – step into view"
            else:
                # Pause: too many people
                if not paused:
                    paused = True
                    if has_audio:
                        try:
                            pygame.mixer.music.pause()
                        except Exception as e:
                            print(f"[audio] pause error: {e}")
                status_text = "Too many players – only 1 allowed"

            # --- Tutorial video frame ---
            if not paused:
                # advance tutorial
                ret_ref, frame_ref = cap_ref.read()
                if not ret_ref:
                    print("[game] Tutorial video finished.")
                    break
                last_ref_frame = frame_ref
            else:
                # freeze on last frame, or grab first frame if none yet
                if last_ref_frame is None:
                    ret_ref, frame_ref = cap_ref.read()
                    if not ret_ref:
                        print("[game] Tutorial video finished.")
                        break
                    last_ref_frame = frame_ref
                frame_ref = last_ref_frame

            # --- Draw tutorial skeleton from reference keypoints ---
            if dance_idx < len(self.ref_keypoints):
                kpts_ref = self.ref_keypoints[dance_idx]
                frame_ref_draw = draw_skeleton(frame_ref.copy(), kpts_ref, color=(0, 0, 255))
            else:
                frame_ref_draw = frame_ref

            # --- Webcam skeleton + scoring ---
            if kpts_player is not None and num_people == 1 and not paused:
                frame_cam_draw = draw_skeleton(frame_cam.copy(), kpts_player, color=(0, 255, 0))
                angles_player = keypoints_to_angles(kpts_player)
                self.player_angles.append(angles_player)

                if dance_idx < len(self.ref_angles):
                    ref_angles_frame = self.ref_angles[dance_idx]
                    frame_score = self.final_score(
                        ref_angles_frame.reshape(1, -1),
                        angles_player.reshape(1, -1),
                    )
                    running_score = 0.9 * running_score + 0.1 * frame_score
                dance_idx += 1
            else:
                # No valid single person for scoring
                frame_cam_draw = frame_cam.copy()
                # do not modify running_score or dance_idx

            # --- Render UI ---
            if show_window:
                h = 480
                frame_ref_r = cv2.resize(
                    frame_ref_draw,
                    (int(frame_ref_draw.shape[1] * h / frame_ref_draw.shape[0]), h),
                )
                frame_cam_r = cv2.resize(
                    frame_cam_draw,
                    (int(frame_cam_draw.shape[1] * h / frame_cam_draw.shape[0]), h),
                )

                combined = np.hstack((frame_ref_r, frame_cam_r))

                # Score text
                cv2.putText(
                    combined,
                    f"Score: {running_score:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )

                # Status text
                cv2.putText(
                    combined,
                    status_text,
                    (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("Just Dance YOLOv8 Pose", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap_ref.release()
        cap_cam.release()
        cv2.destroyAllWindows()

        if has_audio:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            pygame.mixer.quit()

        # Final score
        if not self.player_angles:
            return 0.0

        player_arr = np.stack(self.player_angles, axis=0)
        final = self.final_score(self.ref_angles, player_arr)
        print(f"\n💯 Final Score: {final:.2f}\n")
        return final


if __name__ == "__main__":
    # Simple manual test (you can ignore this when using GUI)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="FAST tutorial video")
    parser.add_argument("--audio", type=str, required=False, default="", help="FAST audio file")
    parser.add_argument("--angles", type=str, required=False, default="", help="Angles .npy path")
    parser.add_argument("--keypoints", type=str, required=False, default="", help="Keypoints .npy path")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--countdown", type=str, default=COUNTDOWN_VIDEO_PATH, help="Countdown mp4 path")
    args = parser.parse_args()

    controller = JustDanceController(
        reference_video=args.video,
        audio_file=args.audio,
        reference_angles_path=args.angles,
        reference_keypoints_path=args.keypoints,
        camera_index=args.camera,
        countdown_video=args.countdown,
    )
    controller.run_game(show_window=True)
