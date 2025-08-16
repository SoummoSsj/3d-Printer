# SpeedNet: 3D-Aware Vehicle Speed Estimation from Video

A state-of-the-art neural network system for accurate vehicle speed estimation from monocular video feeds. This project implements a novel 3D-aware architecture that addresses common issues in traditional speed detection systems.

## 🌟 Key Features

- **3D-Aware Architecture**: Uses geometric understanding to provide accurate speed estimates regardless of camera position
- **Perspective Correction**: Automatically handles camera calibration and perspective distortion
- **Temporal Modeling**: LSTM + Attention mechanism for robust speed estimation across video sequences
- **Real-time Performance**: Optimized for live video feeds with YOLOv8 backbone
- **Uncertainty Estimation**: Provides confidence scores for each speed prediction
- **Multi-vehicle Tracking**: Handles multiple vehicles simultaneously with consistent tracking

## 🚗 Why This Approach?

Traditional speed detection systems suffer from several critical issues:

1. **Distance-dependent errors**: Speed estimates increase as vehicles approach the camera
2. **Stationary vehicle confusion**: Still objects get assigned non-zero speeds
3. **Manual calibration dependency**: Requires precise manual camera setup

**SpeedNet solves these problems by:**

- Learning perspective geometry through neural networks
- Using 3D vehicle models for depth-aware measurements
- Temporal fusion to distinguish motion from perspective changes
- Automatic camera calibration from video content

## 📁 Project Structure

```
speednet/
├── models/
│   └── speednet.py           # Core neural network architecture
├── dataset_analyzer.py       # BrnCompSpeed dataset processing
├── train_speednet.py         # Training pipeline for Kaggle
├── inference_pipeline.py     # Real-time inference system
├── environment.yml           # Conda environment setup
├── requirements.txt          # Pip dependencies
├── setup.py                  # Package installation
└── README.md                 # This file
```

## 🛠️ Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/speednet.git
cd speednet

# Create conda environment
conda env create -f environment.yml
conda activate speednet

# Install the package
pip install -e .
```

### Option 2: Using Pip

```bash
# Clone the repository
git clone https://github.com/yourusername/speednet.git
cd speednet

# Create virtual environment
python -m venv speednet_env
source speednet_env/bin/activate  # On Windows: speednet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 📊 Dataset Preparation

### Download BrnCompSpeed Dataset

1. **Kaggle**: Download from [BrnCompSpeed Dataset](https://www.kaggle.com/datasets/jakubsochor/brnocompspeed)
2. **Extract**: Unzip to your desired location
3. **Structure**: Ensure the following structure:

```
brno_kaggle_subset/
└── dataset/
    ├── session0_center/
    │   ├── gt_data.pkl
    │   ├── video.avi
    │   ├── video_mask.png
    │   └── screen.png
    ├── session0_left/
    ├── session0_right/
    └── ... (more sessions)
```

### Analyze Dataset

```bash
python dataset_analyzer.py
```

This will:
- Scan all available sessions
- Extract speed statistics
- Generate visualizations
- Prepare training data

## 🚀 Training

### On Kaggle

1. **Upload** the training script and model files to Kaggle
2. **Add** the BrnCompSpeed dataset to your Kaggle kernel
3. **Run** the training:

```python
# In Kaggle notebook
!python train_speednet.py \
    --dataset_root /kaggle/input/brnocompspeed/brno_kaggle_subset/dataset \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --save_dir /kaggle/working/checkpoints
```

### Locally

```bash
python train_speednet.py \
    --dataset_root path/to/brno_kaggle_subset/dataset \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --device cuda \
    --save_dir checkpoints
```

### Training Parameters

- `--dataset_root`: Path to the BrnCompSpeed dataset
- `--batch_size`: Batch size (4-8 recommended for GPU memory)
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate (1e-4 works well)
- `--sequence_length`: Number of frames per sequence (8 default)
- `--image_size`: Input image dimensions (640x640 default)

## 🎯 Inference

### Real-time Video Processing

```bash
# Process video file
python inference_pipeline.py \
    --model_path checkpoints/best_model.pth \
    --source path/to/video.mp4 \
    --output results/output_video.mp4

# Use webcam
python inference_pipeline.py \
    --model_path checkpoints/best_model.pth \
    --source 0

# RTSP stream
python inference_pipeline.py \
    --model_path checkpoints/best_model.pth \
    --source rtsp://camera_ip:port/stream
```

### Python API

```python
from inference_pipeline import SpeedDetectionPipeline

# Initialize pipeline
pipeline = SpeedDetectionPipeline(
    model_path='checkpoints/best_model.pth',
    device='cuda',
    confidence_threshold=0.5
)

# Process video
pipeline.run_on_video('input_video.mp4', 'output_video.mp4')

# Process single frame
import cv2
frame = cv2.imread('frame.jpg')
results = pipeline.process_frame(frame)

print(f"Detected {len(results['active_tracks'])} vehicles")
for track_id, speed_info in results['speed_estimates'].items():
    print(f"Vehicle {track_id}: {speed_info['speed']:.1f} km/h")
```

## 📈 Model Architecture

### Core Components

1. **Vehicle Detection**: YOLOv8n for fast, accurate vehicle detection
2. **Camera Calibration Module**: Neural network for automatic perspective correction
3. **Vehicle 3D Module**: Estimates 3D bounding boxes and depth
4. **Temporal Fusion**: LSTM + Multi-head attention for sequence modeling
5. **Speed Regression**: Final speed prediction with uncertainty estimation

### Key Innovations

- **Geometric Feature Integration**: Combines 2D detection with 3D understanding
- **Uncertainty-Aware Loss**: Heteroscedastic loss function for reliable predictions
- **Multi-Scale Processing**: Handles vehicles at different distances uniformly
- **Temporal Consistency**: Smooth speed estimates across time

## 🎛️ Configuration

### Model Hyperparameters

```python
# Core architecture
backbone = 'yolov8n'              # Detection backbone
sequence_length = 8               # Temporal window size
confidence_threshold = 0.5        # Detection confidence

# Training
learning_rate = 1e-4              # AdamW learning rate
weight_decay = 1e-5               # L2 regularization
batch_size = 4                    # Batch size

# Loss weights
speed_weight = 1.0                # Speed regression loss
uncertainty_weight = 0.1          # Uncertainty regularization
camera_weight = 0.5               # Camera calibration loss
```

## 📊 Performance Metrics

The model is evaluated using:

- **MAE (Mean Absolute Error)**: Primary metric for speed accuracy
- **RMSE (Root Mean Square Error)**: Sensitivity to large errors
- **Uncertainty Calibration**: Reliability of confidence estimates
- **Temporal Consistency**: Smoothness of speed estimates

### Expected Performance

- **MAE**: < 5 km/h on BrnCompSpeed dataset
- **RMSE**: < 8 km/h on BrnCompSpeed dataset
- **Real-time FPS**: 15-25 FPS on RTX 3080
- **Detection Range**: 5-150 km/h

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 2
   
   # Use smaller images
   --image_size 512 512
   ```

2. **Slow Training**
   ```bash
   # Enable mixed precision
   # (automatically enabled in training script)
   
   # Reduce sequence length
   --sequence_length 4
   ```

3. **Poor Speed Estimates**
   ```bash
   # Check model path
   --model_path path/to/correct/model.pth
   
   # Verify camera calibration
   # (automatic in SpeedNet)
   ```

### Performance Optimization

- **GPU Memory**: Use gradient checkpointing for large models
- **CPU Usage**: Increase `num_workers` for data loading
- **Inference Speed**: Use TensorRT for deployment optimization

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{speednet2024,
  title={SpeedNet: 3D-Aware Vehicle Speed Estimation from Monocular Video},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/speednet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/speednet/discussions)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- **BrnCompSpeed Dataset**: Thanks to Jakub Sochor et al. for the comprehensive dataset
- **YOLOv8**: Ultralytics team for the excellent detection framework
- **PyTorch Community**: For the robust deep learning framework

---

**Made with ❤️ for the computer vision community**