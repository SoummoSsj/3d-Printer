# Vehicle Speed Kit (VS Code Pipeline)

A production-ready pipeline to estimate vehicle speeds from a fixed camera with:
- YOLOv8 detector (latest)
- ByteTrack-style tracker with ReID hooks (upgradeable to BoT-SORT)
- Self-calibration (vanishing points) to recover intrinsics and road-plane
- Ground-contact footpoint head (optional; improves stability)
- Sequence smoother (LSTM) on geometric features for fluid speeds
- CLI + overlay + CSV outputs

Weights are trained on Kaggle (cells provided separately), then plugged into this pipeline.

## Install (local/VS Code)
```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./.venv/bin/python get-pip.py
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (inference)
```bash
python -m vehicle_speed_kit.run \
  --source /path/to/video.mp4 \
  --det yolov8x.pt \
  --smooth-weights /path/to/smoother.pt \
  --fp-weights /path/to/footpoint.pt \
  --display 1 --save 1 --csv-out speeds.csv
```

## Weights
- `smoother.pt`: trained on Kaggle via provided cells; smooths per-track speed sequences
- `footpoint.pt` (optional): trained on cropped vehicles with footpoint labels; improves ray–plane intersection stability

## Notes
- Works without weights; adding them improves fluidity.
- ReID / BoT-SORT can be enabled later; hooks are present in `tracker.py`.