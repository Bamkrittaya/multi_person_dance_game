"""
Module for extracting leaderboard csv and score functions for
the Just Dance Game.
"""
import csv
import numpy as np



def get_leaderboard(filename):
    """
    Retrieve the leaderboard data from a CSV file.

    Args:
        filename (str): The name of the CSV file to read
            the leaderboard data from.

    Returns:
        A list of lists representing the leaderboard data read from
         the CSV file. Each sub-list represents a row of the CSV file.
    """
    data = []

    with open(filename, "r") as file:  # pylint: disable=unspecified-encoding
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    return data


def get_current_score(filename):
    """
    This function reads the leaderboard data from a file with the
    given filename and returns the last (current) score in the leaderboard.

    Args:
        filename (str): The name of the file containing the leaderboard data.

    Returns:
        int: The last (current) score in the leaderboard.
    """
    score_data = get_leaderboard(filename)
    return int(float(score_data[-1][-1]))


def get_leaderboard_scores(filename):
    """
    Retrieve the top 5 maximum scores from a CSV file.

    Args:
        filename (str): The name of the CSV file to read
            the leaderboard data from.

    Returns:
        A list of integers representing the top 5 maximum scores
        in the CSV file.
    """
    data = get_leaderboard(filename)
    sorted_data = sorted(
        data[1:], key=lambda row: int(float(row[0])), reverse=True
    )
    top_scores = [int(float(row[0])) for row in sorted_data]
    return top_scores[:5]

def calculate_score(player_pose, reference_pose):
    """
    Compare player and reference poses and return a numeric score (0–100).
    Uses Euclidean distance between corresponding keypoints.

    Args:
        player_pose (np.ndarray): Array of shape (17, 3) with [x, y, confidence]
        reference_pose (np.ndarray): Array of shape (17, 3)

    Returns:
        float: A score between 0 and 100 (higher is better)
    """
    if player_pose is None or reference_pose is None:
        return 0.0

    try:
        diff = np.linalg.norm(player_pose[:, :2] - reference_pose[:, :2])
        score = max(0.0, 100.0 - diff * 100.0)  # smaller distance = higher score
        return score
    except Exception:
        return 0.0
