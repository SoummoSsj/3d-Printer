# 🚀 SpeedNet Training Quick Start Guide

This guide will walk you through training SpeedNet on the BrnCompSpeed dataset, from setup to getting your first results.

## 📋 Prerequisites

- **Kaggle account** (for training with free GPUs)
- **BrnCompSpeed dataset** access on Kaggle
- **Local machine** with conda/VS Code (for development and inference)

## 🗂️ Step 1: Download and Prepare Dataset

### Option A: Using Kaggle (Recommended)

1. **Go to Kaggle Dataset**: [BrnCompSpeed Dataset](https://www.kaggle.com/datasets/jakubsochor/brnocompspeed)

2. **Add to your Kaggle account** by clicking "Copy and Edit"

3. **Verify dataset structure** in Kaggle notebook:
   ```python
   import os
   
   # Check dataset structure
   dataset_path = "/kaggle/input/brnocompspeed"
   
   for root, dirs, files in os.walk(dataset_path):
       level = root.replace(dataset_path, '').count(os.sep)
       indent = ' ' * 2 * level
       print(f"{indent}{os.path.basename(root)}/")
       subindent = ' ' * 2 * (level + 1)
       for file in files[:3]:  # Show first 3 files
           print(f"{subindent}{file}")
       if len(files) > 3:
           print(f"{subindent}... and {len(files)-3} more files")
   ```

### Option B: Local Download

1. **Download from Kaggle** using Kaggle API:
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Download dataset (requires Kaggle API token)
   kaggle datasets download -d jakubsochor/brnocompspeed
   unzip brnocompspeed.zip
   ```

## 🛠️ Step 2: Setup Development Environment

### Local Setup (for development)

1. **Clone/create project directory**:
   ```bash
   mkdir speednet-training
   cd speednet-training
   
   # Copy all SpeedNet files here
   ```

2. **Create conda environment**:
   ```bash
   # Create environment from file
   conda env create -f environment.yml
   conda activate speednet
   
   # OR create manually
   conda create -n speednet python=3.9
   conda activate speednet
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "from ultralytics import YOLO; print('YOLO imported successfully')"
   ```

## 📊 Step 3: Analyze Dataset

Before training, let's understand the dataset:

```bash
# Run dataset analysis
python dataset_analyzer.py

# This will:
# - Scan all sessions
# - Extract speed statistics
# - Generate visualizations
# - Show data distribution
```

**Expected output:**
```
Scanning dataset structure...
Found 6 sessions

Analyzing session0_center...
  Total cars: 33
  Valid cars: 33
  Speed range: 68.7 - 102.3 km/h
  Mean speed: 80.2 km/h
```

## 🎯 Step 4: Training on Kaggle

### Upload Files to Kaggle

1. **Create new Kaggle notebook**
2. **Upload these files** as a dataset:
   ```
   speednet-code/
   ├── models/
   │   ├── __init__.py
   │   └── speednet.py
   ├── dataset_analyzer.py
   ├── train_speednet.py
   ├── requirements.txt
   └── README.md
   ```

### Kaggle Training Notebook

Create a new notebook with this code:

```python
# Cell 1: Install dependencies
!pip install ultralytics>=8.0.0
!pip install supervision>=0.16.0
!pip install albumentations>=1.3.0

# Cell 2: Import and setup
import sys
sys.path.append('/kaggle/input/speednet-code')  # Adjust path to your uploaded code

import torch
import numpy as np
from train_speednet import main, BrnCompSpeedDataset, SpeedNetTrainer
from models.speednet import SpeedNet

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Cell 3: Check dataset
dataset_root = "/kaggle/input/brnocompspeed/brno_kaggle_subset/dataset"
import os
print("Dataset contents:")
for item in os.listdir(dataset_root):
    print(f"  {item}")

# Cell 4: Quick dataset test
from dataset_analyzer import BrnCompSpeedAnalyzer

analyzer = BrnCompSpeedAnalyzer(dataset_root)
sessions = analyzer.scan_dataset()
print(f"Found {len(sessions)} sessions")

if sessions:
    # Analyze first session
    session_name = sessions[0]['name']
    analysis = analyzer.analyze_ground_truth(session_name)
    print(f"\nSession {session_name}:")
    print(f"  Cars: {analysis['num_cars']}")
    print(f"  Speed range: {analysis['speed_stats'].get('min_speed', 0):.1f} - {analysis['speed_stats'].get('max_speed', 0):.1f} km/h")

# Cell 5: Start training
import subprocess
import sys

# Training command
cmd = [
    sys.executable, '/kaggle/input/speednet-code/train_speednet.py',
    '--dataset_root', dataset_root,
    '--batch_size', '4',  # Start small
    '--num_epochs', '20',  # Quick test first
    '--learning_rate', '1e-4',
    '--save_dir', '/kaggle/working/checkpoints',
    '--num_workers', '2'
]

# Run training
result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

## 📈 Step 5: Monitor Training

### Expected Training Output

```
Dataset train: 156 samples
Dataset val: 39 samples
Training samples: 156
Validation samples: 39
Model parameters: 8,234,567

Starting training...

Epoch 1/20
Training: 100%|██████████| 39/39 [02:15<00:00,  3.47s/it, loss=2.3456, avg_loss=2.4123]
Validation: 100%|██████████| 10/10 [00:32<00:00,  3.24s/it]

Train Loss: 2.4123
Val Loss: 2.1234
Val MAE: 12.5 km/h
Val RMSE: 18.3 km/h

New best model saved! MAE: 12.5 km/h
```

### Training Tips

1. **Start with small batch size** (2-4) to avoid OOM errors
2. **Monitor GPU memory** usage in Kaggle
3. **Early stopping** if validation loss plateaus
4. **Save checkpoints** regularly

## 🔧 Step 6: Troubleshooting Common Issues

### CUDA Out of Memory
```python
# Reduce batch size
--batch_size 2

# Reduce sequence length
--sequence_length 4

# Reduce image size
--image_size 512 512
```

### Slow Training
```python
# Reduce workers if I/O bound
--num_workers 1

# Enable mixed precision (already enabled in code)
# Use smaller model backbone
```

### Poor Convergence
```python
# Adjust learning rate
--learning_rate 5e-5

# Check data augmentation
# Verify dataset loading
```

## 📥 Step 7: Download Trained Model

After training completes:

```python
# In Kaggle notebook - final cell
import shutil
import os

# Copy best model to output
shutil.copy('/kaggle/working/checkpoints/best_model.pth', '/kaggle/working/speednet_best.pth')

# Also copy training curves
if os.path.exists('/kaggle/working/checkpoints/training_curves.png'):
    shutil.copy('/kaggle/working/checkpoints/training_curves.png', '/kaggle/working/')

print("Training complete! Download speednet_best.pth from output")
```

## 🎯 Step 8: Local Inference Testing

Once you have the trained model:

```bash
# Test on local video
python inference_pipeline.py \
    --model_path speednet_best.pth \
    --source test_video.mp4 \
    --output results.mp4

# Test on webcam
python inference_pipeline.py \
    --model_path speednet_best.pth \
    --source 0
```

## 📊 Expected Results

After 50-100 epochs of training, you should see:

- **Training Loss**: < 1.0
- **Validation MAE**: < 8 km/h
- **Validation RMSE**: < 12 km/h
- **Inference Speed**: 15-25 FPS

## 🚀 Advanced Training Options

### Hyperparameter Tuning

```python
# Experiment with these parameters
--learning_rate 1e-4    # Try: 5e-5, 2e-4
--batch_size 4          # Try: 2, 6, 8
--sequence_length 8     # Try: 4, 12, 16
--weight_decay 1e-5     # Try: 1e-4, 1e-6
```

### Data Augmentation

The training pipeline includes:
- Color jitter (brightness, contrast, saturation)
- Gaussian blur
- Normalization

You can modify these in `create_data_transforms()`.

### Model Architecture

Try different backbones:
```python
# In speednet.py, change backbone
backbone='yolov8n'  # Fastest
backbone='yolov8s'  # Better accuracy
backbone='yolov8m'  # Best accuracy (if memory allows)
```

## 📞 Getting Help

If you encounter issues:

1. **Check the demo script**: `python demo.py --help`
2. **Verify dataset structure** with analyzer
3. **Start with minimal training** (5 epochs, batch_size=1)
4. **Check GPU memory** usage in Kaggle

## 📈 Next Steps

After successful training:

1. **Evaluate on test data**
2. **Compare with baseline methods**
3. **Optimize for deployment**
4. **Create thesis results**

---

**Good luck with your training! 🚗💨**