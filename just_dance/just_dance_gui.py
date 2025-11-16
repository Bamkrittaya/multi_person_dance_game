"""
just_dance_gui.py
-----------------
Tkinter GUI for the Just Dance Game
Displays webcam feed + reference dance video inside the same window,
runs live pose scoring, and shows the player's score in real time.
"""

import cv2
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from just_dance_controller import JustDanceController
from just_dance_model import JustDanceModel
# from just_dance_score import calculate_score
from keypoint import normalize_pose


class JustDanceGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎵 Just Dance Game")
        self.root.geometry("1280x750")
        self.root.configure(bg="#181818")

        # --- Header ---
        tk.Label(
            self.root,
            text="💃 Just Dance AI Game 🕺",
            font=("Helvetica", 28, "bold"),
            fg="#ff66cc",
            bg="#181818",
        ).pack(pady=10)

        # --- Video frames ---
        self.left_label = tk.Label(self.root, bg="black")
        self.left_label.place(x=80, y=100, width=540, height=400)

        self.right_label = tk.Label(self.root, bg="black")
        self.right_label.place(x=660, y=100, width=540, height=400)

        # --- Scoreboard ---
        self.score_label = tk.Label(
            self.root, text="Score: 0", font=("Helvetica", 20),
            fg="white", bg="#181818"
        )
        self.score_label.place(x=560, y=520)

        # --- Song selection ---
        self.song_var = tk.StringVar(value="songs/uptownfunk.mp4")
        ttk.Label(
            self.root, text="Select Song:", background="#181818", foreground="white"
        ).place(x=80, y=40)
        self.song_entry = ttk.Entry(self.root, textvariable=self.song_var, width=50)
        self.song_entry.place(x=180, y=40)

        # --- Buttons ---
        self.start_button = ttk.Button(self.root, text="▶️ Start Game", command=self.start_game)
        self.start_button.place(x=900, y=40)
        self.stop_button = ttk.Button(self.root, text="⏹ Stop", command=self.stop_game)
        self.stop_button.place(x=1040, y=40)

        # --- Controller and model setup ---
        self.running = False
        self.controller = None
        self.player_model = JustDanceModel("model/model.tflite")

    def start_game(self):
        """Start the dance game in a background thread."""
        if self.running:
            return
        self.running = True
        self.start_button.config(state="disabled")
        self.score_label.config(text="Score: 0")
        selected_song = self.song_var.get()

        threading.Thread(target=self.run_game, args=(selected_song,), daemon=True).start()

    def stop_game(self):
        """Stop the dance session."""
        self.running = False
        self.start_button.config(state="normal")

    def run_game(self, selected_song):
        try:
            # Load controller + reference video
            self.controller = JustDanceController(reference_video=selected_song)
            ref_video = cv2.VideoCapture(selected_song)
            ref_fps = ref_video.get(cv2.CAP_PROP_FPS) or 30
            delay = 1 / ref_fps

            webcam = cv2.VideoCapture(0)
            if not webcam.isOpened():
                raise RuntimeError("❌ Could not open webcam")

            ref_frame_idx = 0
            total_score = 0
            frame_count = 0

            while self.running:
                ret_ref, ref_frame = ref_video.read()
                ret_cam, cam_frame = webcam.read()
                if not ret_ref or not ret_cam:
                    break
                
                cam_frame = cv2.flip(cam_frame, 1)  # 👈 Mirror camera here

                # Resize both feeds
                ref_disp = cv2.resize(ref_frame, (540, 400))
                cam_disp = cv2.resize(cam_frame, (540, 400))

                # Pose inference
                player_pose = self.player_model.extract_keypoints(cam_frame)
                ref_idx = min(ref_frame_idx, len(self.controller.reference_poses) - 1)
                reference_pose = self.controller.reference_poses[ref_idx]

                # Compute score
                # --- Compute per-frame score using the model ---
                # Normalize both poses before comparison
                player_norm = normalize_pose(player_pose)
                ref_norm = normalize_pose(reference_pose)

                # --- Compute per-frame score using the model ---
                frame_score = self.player_model.score_calculator(
                    [np.mean(player_norm[:, 0]) * 100],  # simplified stand-in angles
                    [np.mean(ref_norm[:, 0]) * 100],
                    threshold=15
)


                # --- Smooth the score with a running average ---
                if not hasattr(self, "recent_scores"):
                    self.recent_scores = []
                self.recent_scores.append(frame_score)
                if len(self.recent_scores) > 20:  # keep last 20 frame scores
                    self.recent_scores.pop(0)
                avg_score = np.mean(self.recent_scores)

                color = "#00ff00" if avg_score > 80 else "#ffff00" if avg_score > 50 else "#ff3333"
                self.score_label.config(text=f"Score: {avg_score:.1f}", fg=color)




                # Draw keypoints on webcam
                cam_disp = self.player_model.draw_keypoints(cam_disp, player_pose)

                # Convert for Tkinter display
                ref_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(ref_disp, cv2.COLOR_BGR2RGB)))
                cam_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cam_disp, cv2.COLOR_BGR2RGB)))

                # Update GUI frames
                self.left_label.imgtk = ref_img
                self.left_label.configure(image=ref_img)
                self.right_label.imgtk = cam_img
                self.right_label.configure(image=cam_img)

                # Update live score
                self.score_label.config(text=f"Score: {avg_score:.2f}")

                ref_frame_idx += 1
                time.sleep(delay)

            webcam.release()
            ref_video.release()
            self.end_game(avg_score)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.running = False
            self.start_button.config(state="normal")

    def end_game(self, final_score):
        """Show final score and reset UI."""
        # self.running = False
        # self.start_button.config(state="normal")
        # messagebox.showinfo("🏁 Game Over", f"Your final score: {final_score:.2f}")
        # final_score = np.mean(self.recent_scores)
        # self.end_game(final_score)
        final_score = np.mean(self.recent_scores)
        messagebox.showinfo("🏁 Game Over", f"Your final score: {final_score:.2f}")
        self.running = False
        self.start_button.config(state="normal")


    def run(self):
        """Start Tkinter main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    app = JustDanceGUI()
    app.run()
