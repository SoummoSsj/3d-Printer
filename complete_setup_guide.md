# 🚗 **Complete Setup Guide: Vehicle Speed Estimation YOLOv6-3D**

## 📁 **Phase 1: Repository Setup (10 minutes)**

### **Step 1.1: Open VS Code and Set Up Project**

1. **Open VS Code**
2. **Open the repository folder:**
   - File → Open Folder
   - Select your `Vehicle-Speed-Estimation-YOLOv6-3D` folder
3. **Open integrated terminal:**
   - `Ctrl + ` (backtick) or Terminal → New Terminal

### **Step 1.2: Verify Repository Structure**

Your repository should look like this:
```
Vehicle-Speed-Estimation-YOLOv6-3D/
├── README.md
├── requirements.txt
├── speed_estimation.py         # Main script
├── tensorrt_estimation.py      # TensorRT version
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── obtain_calibration.py       # Calibration helper
├── models/                     # Model definitions
├── utils/                      # Utility functions
├── data/                       # Data handling scripts
├── checkpoints/                # Pre-trained models (may be empty)
└── configs/                    # Configuration files
```

**Check what you have:**
```bash
# List all Python files
ls *.py

# Check directories
ls -la

# Check if checkpoints exist
ls checkpoints/
```

## 🐍 **Phase 2: Environment Setup (15 minutes)**

### **Step 2.1: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv speednet_env

# Activate environment
# Windows:
speednet_env\Scripts\activate

# Linux/Mac:
source speednet_env/bin/activate

# Verify activation (you should see (speednet_env) in terminal)
```

### **Step 2.2: Install Dependencies**

```bash
# First, check if requirements.txt exists
cat requirements.txt

# Install basic requirements
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install from requirements.txt
pip install -r requirements.txt

# If requirements.txt is missing or incomplete, install manually:
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tqdm
pip install ultralytics
pip install seaborn
pip install pandas
pip install scipy
pip install onnx
pip install onnxruntime
```

### **Step 2.3: Verify Installation**

```bash
# Test essential imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.x.x
OpenCV: 4.x.x
CUDA available: True
```

## 🧠 **Phase 3: Model Setup (20 minutes)**

### **Step 3.1: Check for Pre-trained Models**

```bash
# Check checkpoints directory
ls -la checkpoints/

# Look for model files anywhere in the repo
find . -name "*.pt" -o -name "*.pth" -o -name "*.onnx"
```

### **Step 3.2: Download Models (if needed)**

**If no models found, you need to:**

1. **Check the repository README:**
   ```bash
   cat README.md | grep -i download
   cat README.md | grep -i checkpoint
   cat README.md | grep -i model
   ```

2. **Look for model download links in:**
   - Repository releases (GitHub → Releases tab)
   - Google Drive links in README
   - Hugging Face model hub

3. **Create checkpoints directory if missing:**
   ```bash
   mkdir -p checkpoints
   ```

4. **Download models to checkpoints directory** (example):
   ```bash
   # If there's a download link, use wget or curl
   # wget https://example.com/model.pt -O checkpoints/best.pt
   ```

### **Step 3.3: Expected Model Files**

You should have at least one of these:
```
checkpoints/
├── best.pt              # Main trained model
├── yolov6_3d.pt        # Alternative name
├── last.pt             # Last checkpoint
└── quantized.onnx      # Optimized model (optional)
```

## 🎥 **Phase 4: Video Preparation (5 minutes)**

### **Step 4.1: Create Directory Structure**

```bash
# Create directories for your data
mkdir -p data/videos
mkdir -p data/results
mkdir -p data/outputs
```

### **Step 4.2: Prepare Your Video**

1. **Copy your video to the data directory:**
   ```bash
   cp /path/to/your/video.mp4 data/videos/
   ```

2. **Video requirements:**
   - **Format:** MP4, AVI, MOV
   - **Resolution:** Any (will be resized automatically)
   - **Duration:** Start with 10-60 seconds for testing
   - **Content:** Should contain vehicles on roads

3. **Test video loading:**
   ```bash
   python -c "
   import cv2
   cap = cv2.VideoCapture('data/videos/your_video.mp4')
   print(f'Video opened: {cap.isOpened()}')
   print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
   print(f'Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
   cap.release()
   "
   ```

## ⚙️ **Phase 5: Configuration (10 minutes)**

### **Step 5.1: Examine the Main Script**

```bash
# Look at the speed estimation script
head -50 speed_estimation.py

# Check for command line arguments
grep -n "argparse\|add_argument" speed_estimation.py
```

### **Step 5.2: Check Default Parameters**

**Look for these key parameters in `speed_estimation.py`:**
- `weights` - Path to model weights
- `source` - Input video path
- `output` - Output directory
- `conf-thres` - Confidence threshold
- `device` - GPU/CPU selection

### **Step 5.3: Understanding the Script Structure**

The script typically expects:
```python
# Common parameter patterns:
--weights checkpoints/best.pt
--source data/videos/your_video.mp4
--output data/results/
--device 0  # or 'cpu'
--conf-thres 0.5
```

## 🚀 **Phase 6: Running Speed Estimation (15 minutes)**

### **Step 6.1: Basic Test Run**

**Start with the simplest command:**

```bash
# Method 1: Check script help
python speed_estimation.py --help

# Method 2: Basic run (if script has defaults)
python speed_estimation.py

# Method 3: Specify minimum parameters
python speed_estimation.py --weights checkpoints/best.pt --source data/videos/your_video.mp4
```

### **Step 6.2: Full Parameter Run**

**Based on repository documentation, try:**

```bash
# Full command with all parameters
python speed_estimation.py \
    --weights checkpoints/best.pt \
    --source data/videos/your_video.mp4 \
    --output data/results/ \
    --conf-thres 0.5 \
    --device 0 \
    --batch-size 1 \
    --imgsz 640
```

### **Step 6.3: Alternative Script Usage**

**If the above doesn't work, try:**

```bash
# Check for different script patterns
python speed_estimation.py \
    --test-name my_test \
    --root_dir_video_path data/videos/ \
    --root_dir_results_path data/results/ \
    --weights checkpoints/best.pt

# Or TensorRT version (if available)
python tensorrt_estimation.py --source data/videos/your_video.mp4
```

### **Step 6.4: Monitor Progress**

**You should see output like:**
```
Loading model...
Model loaded successfully
Processing video: data/videos/your_video.mp4
Frame 1/1000 (0.1%) - Speed: 45.2 km/h
Frame 100/1000 (10.0%) - Speed: 52.1 km/h
...
Processing complete!
Results saved to: data/results/
```

## 📊 **Phase 7: Understanding Results (10 minutes)**

### **Step 7.1: Check Output Files**

```bash
# List generated files
ls -la data/results/

# Common output files:
ls data/results/*.json    # Detection results
ls data/results/*.txt     # Speed summaries
ls data/results/*.mp4     # Annotated videos
```

### **Step 7.2: Typical Output Structure**

```
data/results/
├── detections.json       # Raw vehicle detections
├── speeds.json          # Speed estimation results
├── statistics.txt       # Summary statistics
├── annotated_video.mp4  # Video with speed overlays
└── tracks.json          # Vehicle tracking data
```

### **Step 7.3: View Results**

```bash
# View statistics
cat data/results/statistics.txt

# Preview JSON results
head -20 data/results/speeds.json

# Check if annotated video was created
ls -la data/results/*.mp4
```

## 🎯 **Phase 8: Troubleshooting Common Issues**

### **Issue 1: "No module named 'xxx'"**

**Solution:**
```bash
# Activate environment first
speednet_env\Scripts\activate  # Windows
source speednet_env/bin/activate  # Linux/Mac

# Install missing module
pip install module_name
```

### **Issue 2: "Model file not found"**

**Solution:**
```bash
# Check if model exists
ls -la checkpoints/

# Download model (check README for links)
# Or use a different model path
python speed_estimation.py --weights path/to/your/model.pt
```

### **Issue 3: "CUDA out of memory"**

**Solution:**
```bash
# Use CPU instead
python speed_estimation.py --device cpu

# Or reduce batch size
python speed_estimation.py --batch-size 1
```

### **Issue 4: "Video format not supported"**

**Solution:**
```bash
# Convert video format
ffmpeg -i input_video.mov -c:v libx264 -c:a aac output_video.mp4
```

### **Issue 5: "No detections found"**

**Solution:**
```bash
# Lower confidence threshold
python speed_estimation.py --conf-thres 0.3

# Check if video contains vehicles
# Ensure good lighting and clear vehicle visibility
```

## 🔧 **Phase 9: Advanced Configuration**

### **Step 9.1: Custom Calibration (if needed)**

```bash
# If you need custom calibration for your camera
python obtain_calibration.py --video data/videos/your_video.mp4
```

### **Step 9.2: Batch Processing**

```bash
# Process multiple videos
for video in data/videos/*.mp4; do
    python speed_estimation.py --source "$video" --output "data/results/$(basename $video .mp4)/"
done
```

### **Step 9.3: Configuration Files**

```bash
# Check for config files
ls configs/

# Use custom config
python speed_estimation.py --cfg configs/custom.yaml
```

## ✅ **Phase 10: Verification Checklist**

### **Environment Check:**
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] PyTorch with CUDA working
- [ ] OpenCV installed

### **Data Check:**
- [ ] Video file accessible
- [ ] Video format supported
- [ ] Video contains vehicles
- [ ] Output directory writable

### **Model Check:**
- [ ] Model file exists
- [ ] Model file not corrupted
- [ ] Correct model path specified

### **Execution Check:**
- [ ] Script runs without errors
- [ ] Detections are found
- [ ] Speed estimates are reasonable
- [ ] Output files generated

## 🎉 **Success Indicators**

**You know it's working when you see:**

1. **Model loads successfully**
2. **Video processing starts**
3. **Progress updates showing frame processing**
4. **Speed estimates in reasonable ranges (0-120 km/h)**
5. **Output files generated in results directory**
6. **No error messages in terminal**

## 🆘 **Quick Help Commands**

```bash
# Environment diagnostics
python quick_test_script.py --skip-processing

# Basic video test
python -c "import cv2; cap=cv2.VideoCapture('data/videos/your_video.mp4'); print(f'OK: {cap.isOpened()}')"

# CUDA test
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Model file check
python -c "import torch; torch.load('checkpoints/best.pt'); print('Model loads OK')"
```

## 📝 **Next Steps After Success**

1. **Experiment with different videos**
2. **Adjust confidence thresholds**
3. **Try different model weights**
4. **Analyze speed estimation accuracy**
5. **Create visualizations of results**

---

## 🎯 **QUICK START SUMMARY**

```bash
# 1. Setup
python -m venv speednet_env
speednet_env\Scripts\activate
pip install torch torchvision opencv-python numpy tqdm

# 2. Test
python speed_estimation.py --help

# 3. Run
python speed_estimation.py --weights checkpoints/best.pt --source data/videos/your_video.mp4

# 4. Check
ls data/results/
```

**If you encounter any specific errors, share them with me and I'll provide targeted solutions!** 🚀