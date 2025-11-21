# just_dance_gui.py

import tkinter as tk
from tkinter import messagebox
import os

from just_dance_controller import JustDanceController
from just_dance_score import save_score
from just_dance_gui_score import Score
from path_helper import get_fast_paths
from auto_optimize_videos import optimize_song



# List of songs the player can choose.
# Left: label on the GUI, Right: ORIGINAL video path.
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
        title = tk.Label(
            self,
            text="Just Dance – YOLO Pose",
            font=("Helvetica", 20, "bold"),
        )
        title.pack(pady=10)

        # Player name
        frame_name = tk.Frame(self)
        frame_name.pack(pady=5)
        tk.Label(frame_name, text="Player name:").pack(side="left", padx=5)
        self.entry_name = tk.Entry(frame_name, width=25)
        self.entry_name.pack(side="left", padx=5)

        # Song list
        tk.Label(self, text="Choose a song:").pack(pady=5)
        self.song_var = tk.StringVar(value=SONGS[0][1])

        for display, path in SONGS:
            rb = tk.Radiobutton(
                self,
                text=display,
                value=path,
                variable=self.song_var,
            )
            rb.pack(anchor="w", padx=40)

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)

        start_btn = tk.Button(
            btn_frame,
            text="Start Dance",
            command=self.start_selected_song,
            width=15,
        )
        start_btn.pack(side="left", padx=5)

        leaderboard_btn = tk.Button(
            btn_frame,
            text="View Leaderboard",
            command=self.open_leaderboard,
            width=18,
        )
        leaderboard_btn.pack(side="left", padx=5)

        quit_btn = tk.Button(
            self,
            text="Quit",
            command=self.destroy,
            width=10,
        )
        quit_btn.pack(pady=5)

    def start_selected_song(self):
        # Original tutorial video path (slow, full version)
        original_video_path = self.song_var.get()

        if not os.path.exists(original_video_path):
            messagebox.showerror("Error", f"Original video not found:\n{original_video_path}")
            return

        # Map to fast video/audio/precompute paths
        fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

        # Auto-generate fast versions if missing
        if not os.path.exists(fast_video) or not os.path.exists(fast_audio) \
        or not os.path.exists(fast_angles) or not os.path.exists(fast_keypoints):

            messagebox.showinfo(
                "Preparing...",
                "Optimizing video, audio, and precompute files.\n\n"
                "This happens only the first time."
            )

            optimize_song(original_video_path)  # <-- this creates EVERYTHING automatically

            # re-fetch paths because new files now exist
            fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)


        player_name = self.entry_name.get().strip() or "Player"

        try:
            controller = JustDanceController(
                reference_video=fast_video,
                audio_file=fast_audio,
                reference_angles_path=fast_angles,
                reference_keypoints_path=fast_keypoints,
                camera_index=0,
            )

            score = controller.run_game(show_window=True)

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{e}")
            return

        # Save score and notify
        song_display = os.path.basename(original_video_path)
        save_score(player_name, song_display, score)

        messagebox.showinfo(
            "Dance finished!",
            f"{player_name}, your final score is: {score:.1f}",
        )

        # Show leaderboard after each game
        self.open_leaderboard()

    def open_leaderboard(self):
        Score(self)


def main():
    app = JustDanceGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
