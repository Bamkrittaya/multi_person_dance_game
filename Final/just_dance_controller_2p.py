# just_dance_controller_2p.py
"""
Strict 2-player Just Dance controller.

- Uses same precomputed reference (angles + keypoints) as 1P mode.
- Beginner-friendly scoring (17-joint angles via videoPoseDetection).
- ANGLE_TOLERANCE = 60°.
- Game only plays when EXACTLY 2 players are visible.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import pygame

from videoPoseDetection import (
    precompute_video_angles,
    keypoints_to_angles,
    ANGLE_TRIPLETS,  # full-body angle set
)

# Beginner-friendly: higher tolerance → easier score
ANGLE_TOLERANCE = 60.0  # degrees
MODEL_PATH = "yolo_weights/yolov8s-pose.pt"

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


class JustDanceController2P:
    """
    Strict 2-player controller:
    - Requires exactly 2 people to play.
    - Scores Player 1 and Player 2 separately.
    """

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

        # YOLOv8 pose model
        self.model = YOLO(MODEL_PATH)

        # Reference data
        self.ref_angles = None       # (T, J)
        self.ref_keypoints = None    # (T, 17, 2)

        # Player angle histories
        self.player1_angles = []     # list of (J,) arrays
        self.player2_angles = []     # list of (J,) arrays

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
            print("[precompute 2P] Loading cached FAST reference angles/keypoints…")
            self.ref_angles = np.load(self.reference_angles_path)
            self.ref_keypoints = np.load(self.reference_keypoints_path)
            return

        print("[precompute 2P] npy files missing, computing from video…")
        angles, keypoints = precompute_video_angles(self.reference_video)

        self.ref_angles = angles
        self.ref_keypoints = keypoints

        np.save(self.reference_angles_path, angles)
        np.save(self.reference_keypoints_path, keypoints)
        print(f"[precompute 2P] Saved angles to {self.reference_angles_path}")
        print(f"[precompute 2P] Saved keypoints to {self.reference_keypoints_path}")

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
            print(f"[countdown 2P] Countdown video not found: {self.countdown_video}")
            return

        cap_cd = cv2.VideoCapture(self.countdown_video)
        if not cap_cd.isOpened():
            print(f"[countdown 2P] Could not open countdown video: {self.countdown_video}")
            return

        while True:
            ret, frame = cap_cd.read()
            if not ret:
                break
            cv2.imshow("Get Ready! (2 Players)", frame)
            # ~30 FPS
            if cv2.waitKey(33) & 0xFF == ord("q"):
                break

        cap_cd.release()
        cv2.destroyWindow("Get Ready! (2 Players)")

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

        mask = ~((ref_joint == 0) | (player_joint == 0))
        if mask.sum() == 0:
            return 0.0

        diff = np.abs(ref_joint[mask] - player_joint[mask])
        sim = np.clip(1.0 - diff / ANGLE_TOLERANCE, 0.0, 1.0)
        return float(sim.mean() * 100.0)

    def final_score(self, ref_angles, player_angles):
        """
        Compute final score over all joints (0–100).
        ref_angles, player_angles: (T, J)
        """
        if player_angles.size == 0:
            return 0.0

        T = min(ref_angles.shape[0], player_angles.shape[0])
        ref = ref_angles[:T]
        player = player_angles[:T]

        scores = []
        for j in range(ref.shape[1]):
            s = self.score_calculator(ref[:, j], player[:, j])
            scores.append(s)

        return float(np.mean(scores)) if scores else 0.0

    # ---------------------------------------------------------
    # Main 2P game loop (strict 2 players)
    # ---------------------------------------------------------
    def run_game(self, show_window: bool = True):
        """
        Returns: (player1_final_score, player2_final_score)
        """
        # 1) Load / compute reference
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
                print(f"[audio 2P] Loaded and started (paused): {self.audio_file}")
            else:
                print(f"[audio 2P] Audio file not found: {self.audio_file}")
        except Exception as e:
            print(f"[audio 2P] Error loading audio: {e}")

        # 4) Open video + webcam
        cap_ref = cv2.VideoCapture(self.reference_video)
        cap_cam = cv2.VideoCapture(self.camera_index)

        running_score_p1 = 0.0
        running_score_p2 = 0.0
        status_text = "Waiting for 2 players…"
        paused = True
        dance_idx = 0
        last_ref_frame = None

        while True:
            # --- Webcam frame ---
            ret_cam, frame_cam = cap_cam.read()
            if not ret_cam:
                print("[game 2P] Webcam ended or not available.")
                break

            # Mirror webcam (safe for angles & more natural)
            frame_cam = cv2.flip(frame_cam, 1)

            # --- YOLO detection (multi-person) ---
            results = self.model(frame_cam, conf=0.3, verbose=False)
            num_people = 0
            kpts_p1 = None
            kpts_p2 = None

            if len(results) > 0 and results[0].keypoints is not None:
                xy = results[0].keypoints.xy
                if xy is not None and xy.numel() > 0:
                    num_people = xy.shape[0]
                    if num_people >= 1:
                        kpts_p1 = xy[0].cpu().numpy()
                    if num_people >= 2:
                        kpts_p2 = xy[1].cpu().numpy()

            # --- Strict 2-player logic ---
            if num_people == 2:
                if paused:
                    paused = False
                    if has_audio:
                        try:
                            pygame.mixer.music.unpause()
                        except Exception as e:
                            print(f"[audio 2P] unpause error: {e}")
                status_text = "2 players dancing!"
            elif num_people == 0:
                if not paused:
                    paused = True
                    if has_audio:
                        try:
                            pygame.mixer.music.pause()
                        except Exception as e:
                            print(f"[audio 2P] pause error: {e}")
                status_text = "No players detected – need 2 players"
            elif num_people == 1:
                if not paused:
                    paused = True
                    if has_audio:
                        try:
                            pygame.mixer.music.pause()
                        except Exception as e:
                            print(f"[audio 2P] pause error: {e}")
                status_text = "Only 1 player detected – need 2 players"
            else:  # num_people > 2
                if not paused:
                    paused = True
                    if has_audio:
                        try:
                            pygame.mixer.music.pause()
                        except Exception as e:
                            print(f"[audio 2P] pause error: {e}")
                status_text = "Too many players – only 2 allowed"

            # --- Tutorial video frame ---
            if not paused:
                ret_ref, frame_ref = cap_ref.read()
                if not ret_ref:
                    print("[game 2P] Tutorial video finished.")
                    break
                last_ref_frame = frame_ref
            else:
                if last_ref_frame is None:
                    ret_ref, frame_ref = cap_ref.read()
                    if not ret_ref:
                        print("[game 2P] Tutorial video finished.")
                        break
                    last_ref_frame = frame_ref
                frame_ref = last_ref_frame

            # --- Draw tutorial skeleton ---
            if dance_idx < len(self.ref_keypoints):
                kpts_ref = self.ref_keypoints[dance_idx]
                frame_ref_draw = draw_skeleton(frame_ref.copy(), kpts_ref, color=(0, 0, 255))
            else:
                frame_ref_draw = frame_ref

            # --- Draw player skeletons + scoring ---
            frame_cam_draw = frame_cam.copy()

            if not paused and num_people == 2 and kpts_p1 is not None and kpts_p2 is not None:
                # Player 1
                frame_cam_draw = draw_skeleton(frame_cam_draw, kpts_p1, color=(0, 255, 0))
                angles_p1 = keypoints_to_angles(kpts_p1)
                self.player1_angles.append(angles_p1)

                # Player 2
                frame_cam_draw = draw_skeleton(frame_cam_draw, kpts_p2, color=(0, 255, 255))
                angles_p2 = keypoints_to_angles(kpts_p2)
                self.player2_angles.append(angles_p2)

                # Frame-level score for both
                if dance_idx < len(self.ref_angles):
                    ref_angles_frame = self.ref_angles[dance_idx]

                    s1 = self.final_score(
                        ref_angles_frame.reshape(1, -1),
                        angles_p1.reshape(1, -1),
                    )
                    s2 = self.final_score(
                        ref_angles_frame.reshape(1, -1),
                        angles_p2.reshape(1, -1),
                    )

                    running_score_p1 = 0.9 * running_score_p1 + 0.1 * s1
                    running_score_p2 = 0.9 * running_score_p2 + 0.1 * s2

                dance_idx += 1
            else:
                # No valid 2-player frame → no scoring update
                pass

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

                # Player scores
                cv2.putText(
                    combined,
                    f"P1: {running_score_p1:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    combined,
                    f"P2: {running_score_p2:.1f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
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

                cv2.imshow("Just Dance YOLOv8 Pose – 2 Players", combined)
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

        # Final scores
        if not self.player1_angles:
            p1_final = 0.0
        else:
            p1_arr = np.stack(self.player1_angles, axis=0)
            p1_final = self.final_score(self.ref_angles, p1_arr)

        if not self.player2_angles:
            p2_final = 0.0
        else:
            p2_arr = np.stack(self.player2_angles, axis=0)
            p2_final = self.final_score(self.ref_angles, p2_arr)

        print(f"\n💯 Player 1 Final Score: {p1_final:.2f}")
        print(f"💯 Player 2 Final Score: {p2_final:.2f}\n")

        return p1_final, p2_final


if __name__ == "__main__":
    # Manual CLI test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="FAST tutorial video")
    parser.add_argument("--audio", type=str, required=False, default="", help="FAST audio file")
    parser.add_argument("--angles", type=str, required=False, default="", help="Angles .npy path")
    parser.add_argument("--keypoints", type=str, required=False, default="", help="Keypoints .npy path")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--countdown", type=str, default=COUNTDOWN_VIDEO_PATH)
    args = parser.parse_args()

    controller = JustDanceController2P(
        reference_video=args.video,
        audio_file=args.audio,
        reference_angles_path=args.angles,
        reference_keypoints_path=args.keypoints,
        camera_index=args.camera,
        countdown_video=args.countdown,
    )
    controller.run_game(show_window=True)
