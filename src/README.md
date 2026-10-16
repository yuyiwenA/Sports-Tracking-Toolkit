# `src/` — module map

This directory previously carried its own full copy of the project README, which
then disagreed with the root one. Documentation now lives in a single place:
see [`../README.md`](../README.md).

| Module | Responsibility |
|---|---|
| `detector.py` | YOLOv8 wrapper, returns `[x1,y1,x2,y2,conf,cls]` |
| `tracker.py` | `Track` objects, ByteTrack with an IoU+Kalman fallback |
| `kalman.py` | pure-NumPy constant-velocity Kalman filter |
| `court_homography.py` | pixel ⇄ court metres, calibration strategies |
| `trajectory.py` | smoothing, distance, speed, trail buffer |
| `analytics.py` | session report, heat-map, summary board |
| `visualizer.py` | boxes, labels, trails, HUD, court mini-map |
| `pose_estimator.py` | optional MediaPipe pose keypoints |
| `pipeline.py` | the run loop and the CLI |
| `demo.py` | compatibility entry point → `pipeline.main` |
| `detect.py` | compatibility shim → `detector.Detector` |
