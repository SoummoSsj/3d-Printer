# 🚗 Vehicle Speed Estimation YOLOv6-3D: Complete Setup Guide

## 📁 **Step 1: Project Structure Setup**

First, let's organize your downloaded repository properly:

```
Vehicle-Speed-Estimation-YOLOv6-3D/
├── README.md
├── requirements.txt
├── models/
├── utils/
├── data/
├── checkpoints/
├── speed_estimation.py
├── tensorrt_estimation.py
├── train.py
└── eval.py
```

## 🔧 **Step 2: Environment Setup in VS Code**

### 2.1 Open the Project
1. Open VS Code
2. File → Open Folder → Select your `Vehicle-Speed-Estimation-YOLOv6-3D` folder
3. Open integrated terminal: `Ctrl+`` (backtick)

### 2.2 Create Python Environment
```bash
# Create virtual environment
python -m venv speednet_env

# Activate environment (Windows)
speednet_env\Scripts\activate

# Activate environment (Linux/Mac)
source speednet_env/bin/activate
```

### 2.3 Install Dependencies
```bash
# Install basic requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tqdm
pip install ultralytics
pip install onnx
pip install onnxruntime

# If there's a requirements.txt file:
pip install -r requirements.txt
```

## 📊 **Step 3: Prepare Your Video Data**

### 3.1 Create Data Directory Structure
```bash
mkdir -p data/videos
mkdir -p data/outputs
mkdir -p data/test_results
```

### 3.2 Place Your Video Files
```
data/
├── videos/
│   ├── your_video.mp4          # Your test video
│   ├── session1_center.avi     # BrnoCompSpeed format (if you have)
│   └── session1_center.pkl     # Ground truth (if you have)
└── outputs/
    └── results/
```

## 🎯 **Step 4: Download Pre-trained Models**

### 4.1 Check Available Models
Look in the `checkpoints/` directory or download from the repository:

```bash
# Navigate to checkpoints directory
cd checkpoints

# If models aren't included, download them:
# (Check the repository's README for download links)
```

### 4.2 Expected Model Files
```
checkpoints/
├── yolov6_3d_best.pt          # Main trained model
├── yolov6_3d_quantized.onnx   # Quantized model (optional)
└── tensorrt_model.engine      # TensorRT model (optional)
```

## 🚀 **Step 5: Test with Your Video**

### 5.1 Basic Speed Estimation Test

Create a test script `test_my_video.py`:

```python
#!/usr/bin/env python3
"""
Test script for running speed estimation on your video
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse

def test_video_speed_estimation(video_path, model_path, output_path):
    """
    Test speed estimation on a single video
    
    Args:
        video_path: Path to your video file
        model_path: Path to trained model
        output_path: Path to save results
    """
    
    print(f"🎥 Testing video: {video_path}")
    print(f"🧠 Using model: {model_path}")
    print(f"📁 Output path: {output_path}")
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # TODO: Load your trained model here
    # model = load_model(model_path)
    
    # Process video frame by frame
    frame_count = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # TODO: Run inference on frame
        # detections = model.predict(frame)
        # speeds = calculate_speeds(detections)
        
        # For now, just show progress
        if frame_count % 100 == 0:
            print(f"📈 Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # TODO: Store results
        # results.append({
        #     'frame': frame_count,
        #     'timestamp': timestamp,
        #     'detections': detections,
        #     'speeds': speeds
        # })
    
    cap.release()
    
    print(f"✅ Processing complete! Processed {frame_count} frames")
    
    # TODO: Save results
    # save_results(results, output_path)

def main():
    parser = argparse.ArgumentParser(description='Test speed estimation on video')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='checkpoints/yolov6_3d_best.pt', help='Path to model')
    parser.add_argument('--output', type=str, default='data/outputs/test_results.json', help='Output path')
    
    args = parser.parse_args()
    
    test_video_speed_estimation(args.video, args.model, args.output)

if __name__ == "__main__":
    main()
```

### 5.2 Run the Test Script

```bash
# Test with your video
python test_my_video.py --video data/videos/your_video.mp4 --model checkpoints/yolov6_3d_best.pt
```

## 🛠️ **Step 6: Using the Repository's Scripts**

### 6.1 Check Main Speed Estimation Script

Look for the main script (usually `speed_estimation.py` or `main.py`):

```bash
# Check what scripts are available
ls *.py

# Look at the main speed estimation script
cat speed_estimation.py
```

### 6.2 Understand Script Arguments

```bash
# Check script help
python speed_estimation.py --help

# Or check the script manually to see required arguments
```

### 6.3 Common Usage Patterns

The repository likely expects something like:

```bash
# Basic usage
python speed_estimation.py --input data/videos/your_video.mp4 --output data/outputs/

# With specific model
python speed_estimation.py --input data/videos/your_video.mp4 --model checkpoints/yolov6_3d_best.pt --output data/outputs/

# With configuration file
python speed_estimation.py --config configs/default.yaml --input data/videos/your_video.mp4
```

## 🎯 **Step 7: Handle Common Issues**

### 7.1 Missing Dependencies
```bash
# If you get import errors, install missing packages:
pip install [missing_package_name]

# Common missing packages:
pip install yaml
pip install albumentations
pip install tensorrt  # For TensorRT inference
```

### 7.2 CUDA/GPU Issues
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If GPU not available, the code should fall back to CPU
# but will be slower
```

### 7.3 Model Loading Issues
- Make sure the model file exists in the `checkpoints/` directory
- Check if the model format matches what the script expects (.pt, .pth, .onnx)
- Verify model compatibility with your PyTorch version

### 7.4 Video Format Issues
```bash
# Convert video to compatible format if needed
ffmpeg -i your_video.mov -c:v libx264 -c:a aac your_video.mp4
```

## 📊 **Step 8: Understanding the Output**

### 8.1 Expected Output Formats

The repository should produce:
- **Detection results**: Bounding boxes for detected vehicles
- **3D coordinates**: World coordinates for each vehicle
- **Speed estimates**: Speed in km/h for tracked vehicles
- **Visualization**: Annotated video with speed overlays

### 8.2 Output Files
```
data/outputs/
├── detections.json           # Raw detection results
├── speeds.json              # Speed estimation results
├── annotated_video.mp4      # Video with overlays
└── statistics.txt           # Summary statistics
```

## 🚀 **Step 9: Visualization and Analysis**

### 9.1 Create Visualization Script

```python
# visualize_results.py
import json
import cv2
import matplotlib.pyplot as plt

def visualize_speed_results(results_path, video_path, output_path):
    """Create visualization of speed estimation results"""
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create speed timeline plot
    timestamps = [r['timestamp'] for r in results]
    speeds = [r['avg_speed'] for r in results if 'avg_speed' in r]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, speeds)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speed (km/h)')
    plt.title('Vehicle Speed Over Time')
    plt.grid(True)
    plt.savefig(f'{output_path}/speed_timeline.png')
    plt.show()
    
    print(f"✅ Visualization saved to {output_path}/speed_timeline.png")

# Usage
# python -c "from visualize_results import visualize_speed_results; visualize_speed_results('data/outputs/speeds.json', 'data/videos/your_video.mp4', 'data/outputs')"
```

## 🎯 **Step 10: Troubleshooting Checklist**

### ✅ Environment Check
- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] CUDA available (if using GPU)

### ✅ Data Check  
- [ ] Video file exists and is readable
- [ ] Video format is supported (mp4, avi, mov)
- [ ] Video resolution is reasonable (not too large)

### ✅ Model Check
- [ ] Model file exists in checkpoints directory
- [ ] Model file is not corrupted
- [ ] Model format matches script expectations

### ✅ Script Check
- [ ] Main script exists and is executable
- [ ] Command line arguments are correct
- [ ] Output directory exists and is writable

## 🚀 **Quick Start Commands**

```bash
# 1. Setup environment
python -m venv speednet_env
speednet_env\Scripts\activate  # Windows
pip install torch torchvision opencv-python numpy tqdm

# 2. Test basic functionality
python -c "import torch; import cv2; print('✅ Environment ready!')"

# 3. Place your video
copy your_video.mp4 data\videos\

# 4. Run speed estimation (adjust command based on actual script)
python speed_estimation.py --input data/videos/your_video.mp4 --output data/outputs/

# 5. Check results
dir data\outputs\
```

## 🎉 **Success Indicators**

You'll know it's working when you see:
- ✅ Video loading successfully
- ✅ Model loading without errors  
- ✅ Frame-by-frame processing progress
- ✅ Output files being generated
- ✅ Speed estimates in reasonable ranges (0-120 km/h)

## 🆘 **Getting Help**

If you encounter issues:
1. Check the repository's README.md for specific instructions
2. Look at example commands in the repository
3. Check issue tracker on GitHub
4. Verify your video format and model files
5. Test with a shorter video clip first

**Let me know what specific errors you encounter, and I'll help you troubleshoot!** 🚀