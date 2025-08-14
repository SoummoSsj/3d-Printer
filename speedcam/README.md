# SpeedCam (Pipeline A: Fixed-Camera Monocular Speed Estimation)

A production-ready reference implementation of Pipeline A: fully automated speed measurement from a single, fixed camera using self-calibration (vanishing points), road-plane via ray–plane intersection, multi-prior metric scale estimation, tracking-by-detection, and Kalman-smoothed speed.

Features:
- Automatic vanishing-point based camera orientation and road-plane recovery
- Metric scale fusion (lane-mark priors + vehicle priors)
- YOLOv8 detection (configurable model) + IOU tracker
- Ground-plane mapping and per-vehicle speed estimation with smoothing
- Real-time overlay UI

## Install

```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./.venv/bin/python get-pip.py
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python -m speedcam.run --source /path/to/video.mp4 --model yolov8n.pt --display 1 --save 1
```

Arguments:
- `--source`: video file or stream URL
- `--model`: YOLOv8 model weights
- `--device`: e.g. `cuda:0` or empty for auto
- `--display`: 1 to show a window, 0 headless
- `--save`: 1 to write annotated video
- `--out`: output path (default: alongside input)
- `--lock-scale-m-per-unit`: if known, lock metric scale to bypass auto scale

## How it works
1. Calibration: detect line segments across frames, estimate two orthogonal vanishing points; derive intrinsics and the road-plane normal; intersect image rays with the plane.
2. Scale: fuse lane-mark priors (typical lane width) and vehicle width priors to obtain metric scale.
3. Detection+tracking: run YOLOv8 and a lightweight tracker.
4. World mapping: map bbox bottom-center to road-plane coordinates via ray–plane intersection.
5. Speed estimation: differentiate positions over time; exponential smoothing; report km/h.

## Notes
- For best results, camera should observe lane markings and vehicles over 10+ seconds.
- Night/rain scenes benefit from fine-tuning the detector.
- You can lock scale using `--lock-scale-m-per-unit` if you know the scene’s scale.

## License
MIT