# YOLO Python Scripts Collection

This repository contains multiple Python scripts utilizing the YOLO model for various real-time object detection and pose estimation tasks, including fall detection and general object recognition.

## Overview
The scripts here use the YOLO model from the [Ultralytics library](https://github.com/ultralytics/ultralytics) for detecting objects and poses in video feeds, primarily via webcam. YOLO (You Only Look Once) is an advanced, high-speed object detection model, making it suitable for real-time applications.

## Features
- **Pose Detection**: Detects human poses and highlights keypoints to monitor body orientation, with fall detection alerts when unusual body posture is identified.
- **Facial Detection**: Recognizes facial features and keypoints to monitor entry way or apply filters.
- **Object Recognition**: Identifies various object classes (e.g., person, vehicle, animals) in real-time video.
- **Customizable Thresholds**: Adjusts confidence thresholds for object detection and classification.

## Requirements
- Python
- OpenCV
- Ultralytics YOLO library

## Usage
1. Clone this repository.
2. Set up a virtual environment and install dependencies:
   ```bash
   # Step 1: Navigate to the project directory
   cd path/to/your/project

   # Step 2: Create a virtual environment named 'venv'
   py -m venv venv

   # Step 3: Activate the virtual environment
   # On macOS and Linux
   # source venv/bin/activate
   # On Windows
   venv\Scripts\activate

   # Step 4: Install dependencies from requirements.txt
   pip install -r requirements.txt

   # Step 5 (Optional): Verify installed packages
   pip list

   # Step 6: Deactivate the virtual environment when done
   deactivate
