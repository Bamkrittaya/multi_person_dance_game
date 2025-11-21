# just_dance_gui_score.py
"""
Score & Leaderboard GUI (JSON-compatible)
"""

import tkinter as tk
from tkinter import font as tk_font
from just_dance_score import get_latest_score, get_leaderboard_scores


class Score(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tk_font.Font(
            family="Helvetica", size=18, weight="bold"
        )

        # Window container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.geometry("600x300")

        self.frames = {}
        for F in (ScorePage, LeaderboardPage):
            page = F(parent=container, controller=self)
            self.frames[F.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_frame("ScorePage")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()


class ScorePage(tk.Frame):
    """
    First page: show latest score
    """

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        controller.title("Just Dance - Score")

        latest = get_latest_score()
        if latest:
            score_value = latest["score"]
            player = latest["player"]
            song = latest["song"]
            text = f"{player}, your score for {song} is:\n{score_value:.1f}"
        else:
            text = "No score recorded yet."

        label = tk.Label(self, text=text, font=controller.title_font)
        label.pack(pady=20)

        leaderboard_button = tk.Button(
            self,
            text="View Leaderboard",
            command=lambda: controller.show_frame("LeaderboardPage"),
        )
        leaderboard_button.pack()

        quit_button = tk.Button(
            self,
            text="Quit Game",
            command=controller.destroy,
        )
        quit_button.pack(pady=10)


class LeaderboardPage(tk.Frame):
    """
    Shows top 10 leaderboard scores
    """

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        controller.title("Just Dance - Leaderboard")

        tk.Label(
            self,
            text="Leaderboard",
            font=controller.title_font
        ).pack(pady=20)

        scores = get_leaderboard_scores()

        if not scores:
            tk.Label(
                self,
                text="No scores yet.",
                font=("Helvetica", 14)
            ).pack()
        else:
            for i, s in enumerate(scores, start=1):
                line = f"{i}. {s['player']} - {s['song']} : {s['score']:.1f}"
                tk.Label(self, text=line, font=("Helvetica", 14)).pack()

        back_button = tk.Button(
            self,
            text="Back to Score",
            command=lambda: controller.show_frame("ScorePage"),
        )
        back_button.pack(pady=15)
