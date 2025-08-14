# SpeedCam-Mobile (Pipeline B: Moving/Wearable Camera Speed + TTC)

Pipeline B estimates vehicle speeds and time-to-crossing (TTC) from a moving camera using Visual–Inertial Odometry (VIO) trajectory input and a local ground plane.

Core components:
- VIO trajectory ingest (CSV/JSON): camera pose T_wc(t) in a metric world frame
- Ground plane estimation and tracking in world frame
- YOLOv8 detection + tracker (ID, bbox)
- Ray–plane intersection to get vehicle 3D contact points p_w(t)
- Velocity and TTC to a specified crossing plane
- Overlay UI and CSV logging

## Install

Use the same venv as Pipeline A or create a new one.

```bash
source /workspace/speedcam/.venv/bin/activate  # or create a new venv
pip install -r requirements.txt
```

## Inputs
- Video: moving/wearable camera stream/file
- VIO: CSV with header `time_s,tx,ty,tz,qx,qy,qz,qw` (world_T_camera)
- Crossing plane: a JSON with `{ "point": [x,y,z], "normal": [nx,ny,nz] }`

## Quick start

```bash
python -m speedcam_mobile.run \
  --source /path/to/moving_video.mp4 \
  --poses /path/to/vio_poses.csv \
  --crossing /path/to/crossing.json \
  --model yolov8n.pt --display 1 --save 1 --csv-out /workspace/mobile_log.csv
```

## Notes
- Poses must be metric and time-synced with video (same clock; we linearly interpolate in time).
- If you don’t have a crossing plane file, add `--autoplane` to fit a ground plane from SLAM/VIO map points exported, or provide `--cam-height-m` to recover plane from gravity.

## License
MIT