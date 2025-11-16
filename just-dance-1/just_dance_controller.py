"""
Set 'JustDanceController' class for the application
"""
import sys
import time
import numpy as np
import cv2
from playsound import playsound
from just_dance_view import JustDanceView
import pygame


class JustDanceController:
    """A class that controls the execution of the Just Dance game.

    Attributes:
        model (object): A JustDanceModel object used to run inference on frames
        view (object): A JustDanceView object used to visualize the game
        angles_video (dict): A dictionary containing the angles of the body
            parts detected in the video frames
        angles_camera (dict): A dictionary containing the angles of the body
            parts detected in the camera frames
        cap1 (object): A VideoCapture object for the video file
        cap2 (object): A VideoCapture object for the camera

    Methods:
        __init__: Initialize a new `JustDanceController` object
        process_frame: Process a single frame of the video or camera capture
        process_frames: Process the frames from the video and camera capture
        release_capture: Release the video and camera captures
        close_windows: Close all open windows
        play_sound: Play a sound file
    """

    def __init__(self, model, video_path, camera_index=0):
        """
        Initialize a new `JustDanceController` object

        Args:
            model: A `JustDanceModel` object used for pose estimation
            video_path: A string representing the path to the video file
            camera_index: An integer representing the index of the camera
        """
        self.model = model
        self.view = JustDanceView(model=self.model)
        self.angles_video = {
            "left_arm": [],
            "right_arm": [],
            "left_elbow": [],
            "right_elbow": [],
            "left_thigh": [],
            "right_thigh": [],
            "left_leg": [],
            "right_leg": [],
        }
        self.angles_camera = {
            "left_arm": [],
            "right_arm": [],
            "left_elbow": [],
            "right_elbow": [],
            "left_thigh": [],
            "right_thigh": [],
            "left_leg": [],
            "right_leg": [],
        }
        self.cap1 = cv2.VideoCapture(video_path)  # pylint: disable=no-member
        self.cap2 = cv2.VideoCapture(camera_index)  # pylint: disable=no-member
        self.frame1_rate = self.cap1.get(
            cv2.CAP_PROP_FPS
        )  # pylint: disable=no-member

    def process_frame(self, frame):
        """
        Process a single frame of the video or camera capture

        Args:
            frame: A `numpy.ndarray` object representing the image frame

        Returns:
            key_points_with_scores: A `numpy.ndarray` object
                representing the key points with scores
        """
        img = cv2.resize(frame, (192, 192))  # pylint: disable=no-member
        img = np.expand_dims(img, axis=0)
        key_points_with_scores = self.model.run_inference(img)
        return key_points_with_scores



    def process_frames(self):
        """
        Process the frames from the video and camera capture
        """
        counter = 0
        skip_base = 1   # initial frame skip
        speed_factor = 1.0
        increase_rate = 0.01  # how quickly it speeds up
        max_skip = 10           # cap so it doesn’t become a blur

        while self.cap1.isOpened():
            start_time = time.time()
            _, frame1 = self.cap1.read()
            _, frame2 = self.cap2.read()

            frame2 = cv2.flip(frame2, 1)  # pylint: disable=no-membe
            
            
            # Update playback speed dynamically
            speed_factor += increase_rate
            N = min(int(skip_base * speed_factor), max_skip)

            # Only increase skipping occasionally, not every frame
            if counter % 3 == 0:  # every ~30 frames (~1 second)
                N = min(int(skip_base * speed_factor), max_skip)
            else:
                N = 0  # no skipping this frame

            # # Skip frames only sometimes
            # for _ in range(N):
            #     self.cap1.grab()


            # Skip N frames smoothly by reading and discarding
            for _ in range(N):
                self.cap1.grab()  # grab() is lightweight: moves forward without decode


            # if counter == 0:
            #     key_points_with_scores_video = self.process_frame(frame1)
            #     key_points_with_scores_camera = self.process_frame(frame2)

            #     self.model.store_angles(
            #         self.angles_video, frame2, key_points_with_scores_video
            #     )
            #     self.model.store_angles(
            #         self.angles_camera, frame2, key_points_with_scores_camera
            #     )

            # counter = (counter + 1) % 100
            # Every 3rd frame, do pose detection

            # if counter % 3 == 0:
            #     key_points_with_scores_video = self.process_frame(frame1)
            #     key_points_with_scores_camera = self.process_frame(frame2)

            #     self.model.store_angles(
            #         self.angles_video, frame2, key_points_with_scores_video
            #     )
            #     self.model.store_angles(
            #         self.angles_camera, frame2, key_points_with_scores_camera
            #     )

            # counter += 1

            if counter % 3 == 0:
                # Run pose estimation every 3rd frame
                key_points_with_scores_video = self.process_frame(frame1)
                key_points_with_scores_camera = self.process_frame(frame2)

                # ✅ Draw skeleton (tutorial in blue, webcam in green)
                self.model.draw_connections(frame1, key_points_with_scores_video)
                self.model.draw_keypoints(frame1, key_points_with_scores_video, color=(255, 0, 0))  # tutorial = blue

                self.model.draw_connections(frame2, key_points_with_scores_camera)
                self.model.draw_keypoints(frame2, key_points_with_scores_camera, color=(0, 255, 0))  # webcam = green

                # Store joint angles for later scoring
                self.model.store_angles(
                    self.angles_video, frame2, key_points_with_scores_video
                )
                self.model.store_angles(
                    self.angles_camera, frame2, key_points_with_scores_camera
                )

            counter += 1



            if frame1 is not None and frame2 is not None:
                # Get the dimensions of frame1 and frame2
                height1, _, _ = frame1.shape
                height2, width2, _ = frame2.shape
            else:
                break

            # Resize frame2 to have the same height as frame1
            if height1 != height2:
                scale_factor = height1 / height2
                width2 = int(width2 * scale_factor)
                height2 = height1
                frame2 = cv2.resize(
                    frame2, (width2, height2)
                )  # pylint: disable=no-member

            # Combine the video and camera frames horizontally
            combined_frame = np.concatenate((frame1, frame2), axis=1)

            # Resize the combined frame to fit the window size
            combined_frame = cv2.resize(  # pylint: disable=no-member
                combined_frame,
                (1280, 480),
                interpolation=cv2.INTER_LINEAR,  # pylint: disable=no-member
            )

            # Display the combined frame in a named window
            cv2.namedWindow(
                "Just Dance", cv2.WINDOW_NORMAL
            )  # pylint: disable=no-member
            cv2.imshow(
                "Just Dance", combined_frame
            )  # pylint: disable=no-member

            if cv2.waitKey(1) & 0xFF == ord("q"):  # pylint: disable=no-member
                # Exit program if 'q' key is pressed
                sys.exit()
            elapsed_time = time.time() - start_time

            # Estimate effective FPS after skipping
            fps = self.cap1.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            # Adjust for skipping (N + 1 frames per loop)
            effective_fps = fps / max(1, N + 1)

            # Compute frame delay based on effective FPS
            frame_delay = max(0.001, (1.0 / effective_fps) - elapsed_time)
            time.sleep(frame_delay)

            # elapsed_time = time.time() - start_time
            # frame_delay = max(
            #     1, int(1000 / self.frame1_rate) - int(elapsed_time * 1000)
            # )
            # time.sleep(frame_delay / 1000.0)







# def process_frames(self):
#     """
#     Process frames from the dance video and webcam feed.
#     Keeps playback close to real-time while maintaining pose accuracy.
#     """
#     counter = 0
#     N = 3  # Process every 3rd frame for speed

#     # Get the FPS of the video
#     fps = self.frame1_rate if self.frame1_rate > 0 else 30.0
#     frame_delay = 1.0 / fps

#     print(f"[INFO] Target FPS: {fps:.2f}")

#     while self.cap1.isOpened():
#         start_time = time.time()

#         ret1, frame1 = self.cap1.read()
#         ret2, frame2 = self.cap2.read()
#         if not ret1 or not ret2:
#             break

#         frame2 = cv2.flip(frame2, 1)

#         # Run TensorFlow inference less often to avoid lag
#         if counter % N == 0:
#             key_points_with_scores_video = self.process_frame(frame1)
#             key_points_with_scores_camera = self.process_frame(frame2)

#             self.model.store_angles(
#                 self.angles_video, frame2, key_points_with_scores_video
#             )
#             self.model.store_angles(
#                 self.angles_camera, frame2, key_points_with_scores_camera
#             )

#         counter += 1

#         # Match both frame sizes
#         height1, _, _ = frame1.shape
#         height2, width2, _ = frame2.shape
#         if height1 != height2:
#             scale_factor = height1 / height2
#             width2 = int(width2 * scale_factor)
#             frame2 = cv2.resize(frame2, (width2, height1))

#         # Combine horizontally for display
#         combined_frame = np.concatenate((frame1, frame2), axis=1)
#         combined_frame = cv2.resize(combined_frame, (1920, 800))

#         # Display in window
#         cv2.namedWindow("Just Dance", cv2.WINDOW_NORMAL)
#         cv2.imshow("Just Dance", combined_frame)

#         # Show real-time FPS in terminal
#         elapsed = time.time() - start_time
#         fps_now = 1.0 / elapsed if elapsed > 0 else 0
#         print(f"FPS: {fps_now:.2f}")

#         # Let OpenCV handle display timing; skip custom sleep
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             sys.exit()




    # def release_capture(self):
    #     """
    #     Release the video and camera captures
    #     """
    #     self.cap1.release()
    #     self.cap2.release()

    # @staticmethod
    # def close_windows():
    #     """
    #     Close all open windows
    #     """
    #     cv2.destroyAllWindows()  # pylint: disable=no-member

    # @staticmethod
    # def play_sound(song):
    #     """
    #     Play a sound file

    #     Args:
    #         song: A string representing the path to the sound file
    #     """
    #     playsound(song, False)

    def release_capture(self):
        """
        Release the video and camera captures
        """
        self.cap1.release()
        self.cap2.release()

    @staticmethod
    def close_windows():
        """
        Close all open windows
        """
        cv2.destroyAllWindows()  # pylint: disable=no-member

    @staticmethod
    def play_sound(song):
        """
        Play a sound file

        Args:
            song: A string representing the path to the sound file
        """
        playsound(song, False)
