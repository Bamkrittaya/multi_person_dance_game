# just_dance_gui.py
"""
Main GUI for Just Dance – YOLOv8 Pose Edition.

Features:
- Choose player name
- Choose song
- Launch game (calls JustDanceController)
- View leaderboard
"""

import tkinter as tk
from tkinter import messagebox
from just_dance_controller import JustDanceController
from just_dance_score import save_score, get_leaderboard_scores


# ============================================================
#  SONG OPTIONS
# ============================================================

SONGS = {
    "Call Me Maybe": "videos/callmemaybe.mp4",
    "Cheap Thrills": "videos/cheapthrills.mp4",
    "Don't Start Now": "videos/dontstartnow.mp4",
    "Ghungroo": "videos/ghungroo.mp4",
    "Uptown Funk": "videos/uptownfunk.mp4"
}


# ============================================================
#  GUI CLASS
# ============================================================

class DanceGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Just Dance – YOLO Pose")
        self.geometry("650x500")
        self.configure(bg="#1e1e1e")

        self.selected_song = tk.StringVar()
        self.player_name = tk.StringVar()

        title = tk.Label(
            self,
            text="Just Dance – YOLO Pose",
            font=("Helvetica", 22, "bold"),
            fg="white",
            bg="#1e1e1e",
        )
        title.pack(pady=20)

        # Player name input
        frame_name = tk.Frame(self, bg="#1e1e1e")
        frame_name.pack(pady=10)

        tk.Label(
            frame_name,
            text="Player name:",
            font=("Helvetica", 14),
            fg="white",
            bg="#1e1e1e",
        ).pack(side="left")

        tk.Entry(
            frame_name,
            textvariable=self.player_name,
            font=("Helvetica", 14),
            width=20,
        ).pack(side="left", padx=10)

        # Song selection
        tk.Label(
            self,
            text="Choose a song:",
            font=("Helvetica", 14),
            fg="white",
            bg="#1e1e1e",
        ).pack(pady=10)

        for name in SONGS.keys():
            tk.Radiobutton(
                self,
                text=name,
                variable=self.selected_song,
                value=name,
                font=("Helvetica", 13),
                fg="white",
                bg="#1e1e1e",
                selectcolor="#333",
                activebackground="#1e1e1e",
            ).pack(anchor="w", padx=120)

        # Buttons
        tk.Button(
            self,
            text="Start Dance",
            command=self.start_dance,
            font=("Helvetica", 15, "bold"),
            bg="#4caf50",
            fg="white",
            width=15,
        ).pack(pady=20)

        tk.Button(
            self,
            text="View Leaderboard",
            command=self.show_leaderboard,
            font=("Helvetica", 13),
            bg="#3a3a3a",
            fg="white",
            width=18,
        ).pack()

        tk.Button(
            self,
            text="Quit",
            command=self.quit,
            font=("Helvetica", 12),
            bg="#8b0000",
            fg="white",
            width=10,
        ).pack(pady=15)

    # ============================================================
    #  START DANCE
    # ============================================================

    def start_dance(self):
        name = self.player_name.get().strip()
        song = self.selected_song.get()

        if name == "":
            messagebox.showerror("Error", "Please enter a player name.")
            return

        if song not in SONGS:
            messagebox.showerror("Error", "Please choose a song.")
            return

        video_path = SONGS[song]

        # Run game
        controller = JustDanceController(video_path)
        final_score = controller.run_game(show_window=True)

        # Save to leaderboard
        save_score(name, song, final_score)

        # Show popup
        messagebox.showinfo(
            "Finished!",
            f"{name}, your final score is: {final_score:.2f}",
        )

    # ============================================================
    #  LEADERBOARD POPUP
    # ============================================================

    def show_leaderboard(self):
        scores = get_leaderboard_scores()

        popup = tk.Toplevel(self)
        popup.title("Leaderboard")
        popup.geometry("420x350")
        popup.configure(bg="#1e1e1e")

        tk.Label(
            popup,
            text="Leaderboard (Top 10)",
            font=("Helvetica", 18, "bold"),
            fg="white",
            bg="#1e1e1e",
        ).pack(pady=10)

        frame = tk.Frame(popup, bg="#1e1e1e")
        frame.pack()

        if not scores:
            tk.Label(
                popup,
                text="No scores yet.",
                font=("Helvetica", 14),
                fg="white",
                bg="#1e1e1e",
            ).pack(pady=20)
            return

        for idx, entry in enumerate(scores, start=1):
            text = f"{idx}. {entry['player']} – {entry['song']} – {entry['score']:.1f}"
            tk.Label(
                frame,
                text=text,
                font=("Helvetica", 13),
                fg="white",
                bg="#1e1e1e",
            ).pack(anchor="w", pady=3)


# ============================================================
#  LAUNCH GUI
# ============================================================

if __name__ == "__main__":
    gui = DanceGUI()
    gui.mainloop()
