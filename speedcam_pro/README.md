# SpeedCam-Pro: End-to-end Vehicle Speed Estimation (Fixed Camera)

This project integrates a state-of-the-art fixed-camera speed pipeline with:
- Latest YOLOv8 detector
- ByteTrack-style MOT with ReID hooks
- Footpoint keypoint head (predict ground-contact point)
- Sequence smoother (LSTM/TCN) on geometric features
- Optional: flow-conditioned smoothing and end-to-end speed regressor hooks
- BrnoCompSpeed (Kaggle subset) ingestion and evaluation scaffolding

## Dataset layout (as provided)
```
<DATA_ROOT>/brno_kaggle_subset/dataset/
  session0_center/
    gt_data.pkl
    screen.png
    video.avi
    video_mask.png
  session0_left/
    ...
  session0_right/
    ...
  ... up to session1_right
```
`gt_data.pkl` is a dict containing `measurementLines`, `cars` (with `carId`, `intersections[].videoTime`, `speed`), and `fps`.

## Environment
- Python 3.10+ with CUDA if available. On Kaggle, enable GPU in the Notebook settings.

```bash
# in Kaggle cell (or local)
%pip install --no-cache-dir ultralytics==8.3.56 opencv-python scikit-image filterpy torch torchvision tqdm pydantic matplotlib PyYAML
```

Or using the repo venv locally:
```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
./.venv/bin/python get-pip.py
source .venv/bin/activate
pip install -r requirements.txt
```

## Training on Kaggle (sequence smoother)
1) Upload or mount your dataset root as `/kaggle/input/brnocomp` (contains `brno_kaggle_subset/dataset/...`).
2) Extract features:
```bash
python -m speedcam_pro.extract_features \
  --root /kaggle/input/brnocomp \
  --det yolov8n.pt \
  --out-npz /kaggle/working/smoother_feats.npz \
  --max-frames 0
```
3) Train smoother:
```bash
python -m speedcam_pro.train_smoother \
  --npz /kaggle/working/smoother_feats.npz \
  --epochs 15 --batch 128 --lr 1e-3 \
  --out /kaggle/working/smoother.pt
```
4) (Optional) Prepare footpoint crops with `--save-crops /kaggle/working/foot_crops`; annotate a small set offline and train `footpoint.py` head (script for footpoint training can be added similarly; model loads via `--fp-weights`).

## Inference on Kaggle
```bash
python -m speedcam_pro.run \
  --source /kaggle/input/brnocomp/brno_kaggle_subset/dataset/session0_center/video.avi \
  --det yolov8x.pt \
  --smooth-weights /kaggle/working/smoother.pt \
  --display 0 --save 1 --csv-out /kaggle/working/speeds.csv
```
- Outputs: annotated video next to source, and per-frame CSV with raw/smoothed speeds.

## Notes
- Works without training; smoother and footpoint increase stability.
- ReID embeddings and BoT-SORT can be integrated later (hooks present in `tracker.py`).
- Flow and end-to-end regressor are optional extras; enable when GPU budget allows.

## Next steps (optional)
- Add evaluator to compute MAE/RMSE vs `gt_data.pkl` speeds using intersection times.
- Add footpoint training script once annotated points are available.

## License
MIT