# Sports-Tracking-Toolkit
A modular computer vision toolkit for sports video player & ball tracking, trajectory analysis and kinematic measurement.

## Features
✅ YOLOv8 detector for player and ball detection
✅ ByteTrack multi-object tracking with ID recovery under occlusion
✅ Automatic court homography transformation (pixel → real court coordinate)
✅ Trajectory smoothing by Kalman filter
✅ Calculate speed, displacement and movement metrics for each athlete
✅ Pose keypoint extraction (optional)
✅ Export annotated video + trajectory CSV
✅ Better ID persistence than roboflow sports demo, less ID switch during occlusion

## Install
```
pip install -r requirements.txt
```

## Quick Start
```
python demo.py --video ./test.mp4 --model yolov8n.pt --conf 0.3
```

## Parameters
- --video: input video path
- --model: yolov8 weight path
- --conf: detection confidence threshold
- --save_csv: whether save trajectory csv (default True)
- --save_video: whether save visualized video (default True)

## Output
- outputs/trajectory.csv: frame, track_id, class, pixel_x, pixel_y, court_x, court_y, speed
- outputs/result.mp4: video with bounding box, track id and trajectory line

## Module Introduction
1. detector: YOLOv8 detect players and ball
2. tracker: ByteTrack + kalman filter for multi-object tracking
3. court_homography: perspective transform from image plane to court plane
4. trajectory: smooth trajectory and compute movement metrics
5. pose_estimator: mediapipe pose keypoint (optional)
6. visualizer: draw box, id, trajectory on frame

## Limitation
- Automatic court calibration works best for top-down broadcast view
- For low light / heavy occlusion scene, detection performance drops
