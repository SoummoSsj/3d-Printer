# SpeedCam-Pro: End-to-end Vehicle Speed Estimation (Fixed Camera)

This project integrates a state-of-the-art fixed-camera speed pipeline with:
- Latest YOLOv8 detector
- ByteTrack for MOT (or BoT-SORT optional)
- ReID embeddings for robust tracking
- Footpoint keypoint head (predict ground-contact point)
- Sequence smoother (LSTM/TCN) on geometric features
- Optional: flow-conditioned smoothing (RAFT-lite hook)
- Optional: end-to-end speed regressor (temporal CNN) fused with geometry
- BrnoCompSpeed data ingestion and evaluation tooling

## Install

```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./.venv/bin/python get-pip.py
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets
- Place BrnoCompSpeed under `data/brno/`.
- Provide splits in `configs/brno.yaml`.

## Quick run (inference)
```bash
python -m speedcam_pro.run \
  --source /path/to/video.mp4 \
  --det yolov8x.pt \
  --tracker bytetrack \
  --display 1 --save 1 --csv-out /workspace/speeds.csv
```

## Training hooks
- Footpoint head: `python -m speedcam_pro.train_footpoint --cfg configs/footpoint.yaml`
- Sequence smoother: `python -m speedcam_pro.train_smoother --cfg configs/smoother.yaml`
- ReID: `python -m speedcam_pro.train_reid --cfg configs/reid.yaml`
- End-to-end regressor: `python -m speedcam_pro.train_regressor --cfg configs/regressor.yaml`

## Notes
- The pipeline works out-of-the-box without training. Training modules improve fluidity & robustness.
- RAFT flow hook is stubbed; enable if hardware allows.

## License
MIT