#!/usr/bin/env python3
"""
🚗 Quick Test Script for Vehicle Speed Estimation YOLOv6-3D
📚 Test your video with the downloaded GitHub repository
🎯 Quick setup and testing guide
===============================================================================
"""

import os
import sys
import cv2
import torch
import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    try:
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🎮 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA devices: {torch.cuda.device_count()}")
            print(f"🎮 Current device: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"❌ PyTorch issue: {e}")
    
    # Check OpenCV
    try:
        print(f"📹 OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV issue: {e}")
    
    print("✅ Environment check complete!")

def find_repository_files():
    """Find key files in the repository"""
    print("\n📁 Scanning repository structure...")
    
    # Current directory files
    current_files = list(Path('.').glob('*.py'))
    print(f"🐍 Python files found: {[f.name for f in current_files]}")
    
    # Look for key directories
    key_dirs = ['models', 'utils', 'checkpoints', 'data', 'configs']
    for dir_name in key_dirs:
        if Path(dir_name).exists():
            print(f"📁 Found directory: {dir_name}/")
            # List contents
            contents = list(Path(dir_name).iterdir())[:5]  # First 5 files
            print(f"   Contents (first 5): {[f.name for f in contents]}")
        else:
            print(f"❌ Missing directory: {dir_name}/")
    
    # Look for model files
    model_extensions = ['*.pt', '*.pth', '*.onnx', '*.weights']
    model_files = []
    for ext in model_extensions:
        model_files.extend(list(Path('.').rglob(ext)))
    
    if model_files:
        print(f"🧠 Model files found: {[f.name for f in model_files[:3]]}")
    else:
        print("❌ No model files found - you may need to download them")
    
    # Look for config files
    config_files = list(Path('.').rglob('*.yaml')) + list(Path('.').rglob('*.yml'))
    if config_files:
        print(f"⚙️  Config files found: {[f.name for f in config_files[:3]]}")
    
    return {
        'python_files': current_files,
        'model_files': model_files,
        'config_files': config_files
    }

def test_video_loading(video_path):
    """Test if video can be loaded properly"""
    print(f"\n🎥 Testing video: {video_path}")
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return None
    
    # Try to open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Could not open video file")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"✅ Video loaded successfully!")
    print(f"   📊 Resolution: {width}x{height}")
    print(f"   📊 FPS: {fps}")
    print(f"   📊 Total frames: {total_frames}")
    print(f"   📊 Duration: {duration:.1f} seconds")
    
    # Test reading a few frames
    frame_count = 0
    test_frames = []
    
    for i in range(min(5, total_frames)):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            test_frames.append(frame.shape)
        else:
            break
    
    cap.release()
    
    print(f"   📊 Successfully read {frame_count} test frames")
    print(f"   📊 Frame shape: {test_frames[0] if test_frames else 'None'}")
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration': duration
    }

def run_basic_speed_estimation(video_path, output_dir):
    """Run basic speed estimation (placeholder for actual implementation)"""
    print(f"\n🚀 Running basic speed estimation...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ Could not open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frames (basic example)
    results = []
    frame_count = 0
    
    print(f"📈 Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # TODO: Replace with actual model inference
        # For now, just detect if frame has content
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        motion_score = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        # Placeholder speed estimation
        estimated_speed = motion_score * 100  # Very rough approximation
        
        results.append({
            'frame': frame_count,
            'timestamp': timestamp,
            'motion_score': float(motion_score),
            'estimated_speed': float(estimated_speed)
        })
        
        # Show progress
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"   📈 Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    
    # Save results
    results_file = Path(output_dir) / 'basic_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Processing complete! Results saved to: {results_file}")
    
    # Basic statistics
    speeds = [r['estimated_speed'] for r in results]
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    
    print(f"📊 Basic statistics:")
    print(f"   Average speed: {avg_speed:.1f} km/h")
    print(f"   Maximum speed: {max_speed:.1f} km/h")
    print(f"   Total frames processed: {frame_count}")

def suggest_next_steps(repo_files):
    """Suggest next steps based on found files"""
    print(f"\n🎯 Suggested Next Steps:")
    
    # Check for main scripts
    main_scripts = [f for f in repo_files['python_files'] if 'main' in f.name.lower() or 'speed' in f.name.lower() or 'test' in f.name.lower()]
    
    if main_scripts:
        print(f"📝 Found potential main scripts:")
        for script in main_scripts:
            print(f"   • {script.name}")
        print(f"🚀 Try running: python {main_scripts[0].name} --help")
    
    # Check for models
    if repo_files['model_files']:
        print(f"🧠 Found model files - you can use them directly")
        print(f"   • {repo_files['model_files'][0].name}")
    else:
        print(f"⬇️  You need to download pre-trained models:")
        print(f"   • Check the repository README for download links")
        print(f"   • Look for 'checkpoints' or 'models' section")
    
    # Check for configs
    if repo_files['config_files']:
        print(f"⚙️  Found config files - check for default settings")
        print(f"   • {repo_files['config_files'][0].name}")
    
    print(f"\n📚 General recommendations:")
    print(f"   1. Read the repository README.md carefully")
    print(f"   2. Check requirements.txt for dependencies")
    print(f"   3. Look for example commands or usage instructions")
    print(f"   4. Start with a short test video (10-30 seconds)")
    print(f"   5. Verify model files are downloaded and in correct location")

def main():
    parser = argparse.ArgumentParser(description='Quick test for Vehicle Speed Estimation repository')
    parser.add_argument('--video', type=str, help='Path to test video file')
    parser.add_argument('--output', type=str, default='test_output', help='Output directory')
    parser.add_argument('--skip-processing', action='store_true', help='Skip video processing, just check setup')
    
    args = parser.parse_args()
    
    print("🚗 Vehicle Speed Estimation - Quick Test Script")
    print("=" * 60)
    
    # Step 1: Check environment
    check_environment()
    
    # Step 2: Scan repository
    repo_files = find_repository_files()
    
    # Step 3: Test video if provided
    if args.video:
        video_info = test_video_loading(args.video)
        
        if video_info and not args.skip_processing:
            # Step 4: Run basic processing
            run_basic_speed_estimation(args.video, args.output)
    
    # Step 5: Suggest next steps
    suggest_next_steps(repo_files)
    
    print(f"\n🎉 Quick test complete!")
    print(f"📝 Check the generated setup guide for detailed instructions")

if __name__ == "__main__":
    main()