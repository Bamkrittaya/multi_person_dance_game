"""
just_dance_controller.py
------------------------
Handles reference pose precomputation, real-time scoring, and cleanup.

Workflow:
1. When the game starts, extract reference keypoints from the tutorial video.
2. Run live scoring using webcam vs. reference poses.
3. Show scores in real-time.
4. Delete temporary reference_poses.npy after the game ends.
"""

import cv2
import numpy as np
import os, time
from just_dance_model import JustDanceModel
from just_dance_score import calculate_score
from keypoint import normalize_pose


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model/model.tflite"
REFERENCE_VIDEO = "songs/uptownfunk.mp4"  # your tutorial video
REFERENCE_POSES_FILE = "model/reference_poses.npy"
MAX_WINDOW = 3   # tolerance for frame matching


# ============================================================
# 1. PRECOMPUTE REFERENCE POSES
# ============================================================
def precompute_reference_poses(video_path=REFERENCE_VIDEO,
                               model_path=MODEL_PATH,
                               save_path=REFERENCE_POSES_FILE):
    """Extract pose keypoints for each frame of the reference video."""
    print(f"🎬 Precomputing reference poses from {video_path}")
    model = JustDanceModel(model_path)
    cap = cv2.VideoCapture(video_path)
    poses = []
    frame_count = 0

    if not cap.isOpened():
        raise FileNotFoundError(f"❌ Could not open reference video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = model.extract_keypoints(frame)
        poses.append(keypoints)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  processed {frame_count} frames...")

    cap.release()
    np.save(save_path, np.array(poses))
    print(f"✅ Saved {len(poses)} reference poses → {save_path}")
    return poses


# ============================================================
# 2. LIVE SCORING CONTROLLER
# ============================================================
class JustDanceController:
    def __init__(self,
                 model_path=MODEL_PATH,
                 reference_path=REFERENCE_POSES_FILE,
                 reference_video=REFERENCE_VIDEO):
        """Initialize controller, ensuring reference poses exist."""
        self.model = JustDanceModel(model_path)
        self.reference_path = reference_path
        self.reference_video = reference_video

        # Auto-generate reference poses if missing
        if not os.path.exists(reference_path):
            print("⚙️ No reference poses found — generating from tutorial video...")
            precompute_reference_poses(video_path=reference_video,
                                       model_path=model_path,
                                       save_path=reference_path)

        self.reference_poses = np.load(reference_path, allow_pickle=True)
        print(f"📂 Loaded {len(self.reference_poses)} precomputed reference poses.")

    def run_live_scoring(self, camera_index=0):
        """Run webcam loop and score player pose vs reference."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("❌ Could not open webcam")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = int(1000 / fps)
        frame_idx = 0
        total_score = 0
        frame_scores = []

        print("💃 Start dancing! Press 'q' to quit.")

        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- Player pose ---
            player_pose = self.model.extract_keypoints(frame)
            player_norm = normalize_pose(player_pose)

            # --- Reference pose (frame-matched) ---
            ref_idx = min(frame_idx, len(self.reference_poses) - 1)
            start = max(0, ref_idx - MAX_WINDOW)
            end = min(len(self.reference_poses), ref_idx + MAX_WINDOW)
            ref_candidates = self.reference_poses[start:end]

            # --- Score ---
            scores = []
            for ref_pose in ref_candidates:
                ref_norm = normalize_pose(ref_pose)
                s = calculate_score(player_norm, ref_norm)
                scores.append(s)
            frame_score = max(scores)
            frame_scores.append(frame_score)
            total_score += frame_score

            # --- Draw ---
            cv2.putText(frame, f"Frame: {frame_idx}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Score: {frame_score:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("🎶 Just Dance - Live", frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1
            now = time.time()
            if frame_idx % 30 == 0:
                print(f"FPS ≈ {1/(now - prev_time):.1f}")
            prev_time = now

        cap.release()
        cv2.destroyAllWindows()

        avg_score = total_score / max(1, len(frame_scores))
        print(f"🏁 Final average score: {avg_score:.2f}")

        # --- Cleanup temporary reference poses ---
        if os.path.exists(self.reference_path):
            try:
                os.remove(self.reference_path)
                print(f"🗑️ Deleted temporary file: {self.reference_path}")
            except Exception as e:
                print(f"⚠️ Could not delete reference poses file: {e}")

        return avg_score


# ============================================================
# 3. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Just Dance game controller")
    parser.add_argument(
        "--video",
        type=str,
        default=REFERENCE_VIDEO,
        help="Path to the tutorial/reference dance video",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index (default 0)",
    )
    args = parser.parse_args()

    # Create controller instance
    controller = JustDanceController(reference_video=args.video)
    # Run the game and scoring
    final_score = controller.run_live_scoring(camera_index=args.camera)

    # Final message
    print("✨ Game finished successfully!")
    print(f"💯 Final Average Score: {final_score:.2f}")
