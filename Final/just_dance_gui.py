# just_dance_gui.py

import tkinter as tk
from tkinter import messagebox
import os

from just_dance_controller import JustDanceController
from just_dance_controller_2p import JustDanceController2P
from just_dance_score import save_score
from just_dance_gui_score import Score
from path_helper import get_fast_paths
from auto_optimize_videos import optimize_song


SONGS = [
    ("Call Me Maybe", "videos/callmemaybe.mp4"),
    ("Cheap Thrills", "videos/cheapthrills.mp4"),
    ("Don't Start Now", "videos/dontstartnow.mp4"),
    ("Ghungroo", "videos/ghungroo.mp4"),
    ("Uptown Funk", "videos/uptownfunk.mp4"),
]


class JustDanceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Just Dance – YOLO Edition")
        self.geometry("500x400")
        self._build_widgets() 
    
    def _build_widgets(self): 
        title = tk.Label( self, text="Just Dance – YOLO Pose", font=("Helvetica", 20, "bold"), ) 
        title.pack(pady=10)

        # store selections
        self.num_players = 1
        self.player1 = ""
        self.player2 = ""
        self.selected_song = None

        # Start at STEP 1
        self.show_step1_choose_players()

    # ============================================================
    # STEP 1 — Select 1 or 2 players
    # ============================================================
    def show_step1_choose_players(self):
        self.clear_window()

        tk.Label(self, text="How many players?", font=("Helvetica", 18, "bold")).pack(pady=20)

        self.num_players_var = tk.IntVar(value=1)

        tk.Radiobutton(self, text="1 Player", variable=self.num_players_var, value=1,
                       font=("Helvetica", 14)).pack(pady=5)
        tk.Radiobutton(self, text="2 Players", variable=self.num_players_var, value=2,
                       font=("Helvetica", 14)).pack(pady=5)

        tk.Button(self, text="Next", font=("Helvetica", 14),
                  command=self.go_step2_names).pack(pady=20)

    # ============================================================
    # STEP 2 — Enter player name(s)
    # ============================================================
    def go_step2_names(self):
        self.num_players = self.num_players_var.get()
        self.clear_window()

        tk.Label(self, text="Enter Player Name(s)", font=("Helvetica", 18, "bold")).pack(pady=20)

        # Player 1
        frame1 = tk.Frame(self)
        frame1.pack(pady=10)
        tk.Label(frame1, text="Player 1:", font=("Helvetica", 14)).pack(side="left", padx=5)
        self.entry_p1 = tk.Entry(frame1, width=25)
        self.entry_p1.pack(side="left", padx=5)

        # Player 2 (only if needed)
        if self.num_players == 2:
            frame2 = tk.Frame(self)
            frame2.pack(pady=10)
            tk.Label(frame2, text="Player 2:", font=("Helvetica", 14)).pack(side="left", padx=5)
            self.entry_p2 = tk.Entry(frame2, width=25)
            self.entry_p2.pack(side="left", padx=5)

        tk.Button(self, text="Next", font=("Helvetica", 14),
                  command=self.go_step3_songs).pack(pady=20)

    # ============================================================
    # STEP 3 — Song selection
    # ============================================================
    def go_step3_songs(self):
        self.player1 = self.entry_p1.get().strip() or "Player 1"
        if self.num_players == 2:
            self.player2 = self.entry_p2.get().strip() or "Player 2"

        self.clear_window()

        tk.Label(self, text="Choose a Song", font=("Helvetica", 18, "bold")).pack(pady=15)

        self.song_var = tk.StringVar(value=SONGS[0][1])

        for display, path in SONGS:
            tk.Radiobutton(
                self, text=display, value=path, variable=self.song_var,
                font=("Helvetica", 14)
            ).pack(anchor="w", padx=40)

        tk.Button(self, text="Start!", font=("Helvetica", 16, "bold"),
                  command=self.start_dance).pack(pady=25)

    # ============================================================
    # STEP 4 — Run game (1P or 2P)
    # ============================================================
    def start_dance(self):
        original_video_path = self.song_var.get()
        if not os.path.exists(original_video_path):
            messagebox.showerror("Error", "Song video not found.")
            return

        fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

        if not all(os.path.exists(p) for p in [fast_video, fast_audio, fast_angles, fast_keypoints]):
            optimize_song(original_video_path)
            fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

        # --- RUN MODE ---
        try:
            if self.num_players == 1:
                controller = JustDanceController(
                    reference_video=fast_video,
                    audio_file=fast_audio,
                    reference_angles_path=fast_angles,
                    reference_keypoints_path=fast_keypoints,
                    camera_index=0,
                )
                score = controller.run_game(show_window=True)

                save_score(self.player1, os.path.basename(original_video_path), score)

                messagebox.showinfo("Result", f"{self.player1}, your score: {score:.1f}")

            else:
                controller = JustDanceController2P(
                    reference_video=fast_video,
                    audio_file=fast_audio,
                    reference_angles_path=fast_angles,
                    reference_keypoints_path=fast_keypoints,
                    camera_index=0,
                )
                p1, p2 = controller.run_game(show_window=True)

                save_score(self.player1 + " (P1)", os.path.basename(original_video_path), p1)
                save_score(self.player2 + " (P2)", os.path.basename(original_video_path), p2)

                messagebox.showinfo(
                    "Results",
                    f"{self.player1} (P1): {p1:.1f}\n{self.player2} (P2): {p2:.1f}"
                )

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{e}")
            return

        self.open_leaderboard()

    # ============================================================
    def open_leaderboard(self):
        Score(self)

    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()


def main():
    app = JustDanceGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
