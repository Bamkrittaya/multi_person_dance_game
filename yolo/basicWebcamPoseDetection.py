from ultralytics import YOLO

# Load a model
model = YOLO("yolo_weights/yolo11n-pose.pt")  # load an official model

# Use the file path instead of the camera index (0)
# video_file_path = "YOLO_Examples/test_videos/fall_video.mp4" 

# Predict with the model
results = model(source=0, show=True, conf=0.3, save=False)  # webcam input, show results, confidence threshold 0.3, save output video
