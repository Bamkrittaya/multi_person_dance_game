import os
import subprocess
import numpy as np
from videoPoseDetection import precompute_video_angles, precompute_video_keypoints
from path_helper import get_fast_paths


# ===============================
# CONFIG — EASY SPEED CONTROL
# ===============================

VIDEO_SPEED = 4       # 4× faster video
AUDIO_DELAY_MS = 10000   # 10 seconds delay


# ===============================
# FFmpeg Helpers
# ===============================

def run_ffmpeg(cmd):
    print("\n[FFmpeg] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ===============================
# MAIN OPTIMIZER
# ===============================

def optimize_song(original_video_path):
    """
    Automatically creates fast video, fast audio, angles, keypoints.
    This is triggered automatically the FIRST TIME a song is played.
    """

    fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

    base = os.path.splitext(os.path.basename(original_video_path))[0]

    # --- folders ---
    os.makedirs("videos_fast", exist_ok=True)
    os.makedirs("songs_audio_fast", exist_ok=True)
    os.makedirs("precomputed", exist_ok=True)

    # ===============================
    # 1. FAST VIDEO
    # ===============================
    if not os.path.exists(fast_video):
        # setpts = 1/VIDEO_SPEED
        setpts_value = 1.0 / VIDEO_SPEED
        setpts_str = f"setpts={setpts_value}*PTS"

        filter_v = f"{setpts_str},scale=640:-1"

        cmd_video = [
            "ffmpeg", "-y",
            "-i", original_video_path,
            "-filter:v", filter_v,
            "-preset", "ultrafast",
            fast_video
        ]
        run_ffmpeg(cmd_video)
    else:
        print(f"[skip] Fast video exists → {fast_video}")

    # ===============================
    # 2. FAST AUDIO (NORMAL SPEED) + 10 SEC DELAY
    # ===============================
    if not os.path.exists(fast_audio):

        # Extract audio from original video
        temp_wav = f"songs_audio_fast/{base}_temp.wav"
        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", original_video_path,
            "-vn",
            temp_wav
        ])

        # AUDIO FILTER:
        # - normal speed (no atempo)
        # - 10 sec delay (10000 ms)
        audio_filter = f"adelay={AUDIO_DELAY_MS}|{AUDIO_DELAY_MS}"

        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", temp_wav,
            "-af", audio_filter,
            fast_audio
        ])

        # remove temp file
        os.remove(temp_wav)

    else:
        print(f"[skip] Fast audio exists → {fast_audio}")


    # ===============================
    # 3. PRECOMPUTE KEYPOINTS
    # ===============================
    if not os.path.exists(fast_keypoints):
        kpts, _ = precompute_video_keypoints(fast_video)
        np.save(fast_keypoints, kpts)
        print(f"[save] {fast_keypoints}")
    else:
        print(f"[skip] Keypoints exist → {fast_keypoints}")

    # ===============================
    # 4. PRECOMPUTE ANGLES
    # ===============================
    if not os.path.exists(fast_angles):
        ang, _ = precompute_video_angles(fast_video)
        np.save(fast_angles, ang)
        print(f"[save] {fast_angles}")
    else:
        print(f"[skip] Angles exist → {fast_angles}")

    print("\n✅ Optimization complete for:", original_video_path)
