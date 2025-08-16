#!/usr/bin/env python3
"""
🔍 Repository Diagnostic Script for Vehicle-Speed-Estimation-YOLOv6-3D
📚 Analyzes your setup and provides specific guidance for running the repository
🎯 Identifies issues and suggests exact solutions
===============================================================================
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def print_header(title):
    """Print section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_step(step, description):
    """Print step with emoji"""
    print(f"\n{step} {description}")

def check_file_exists(filepath, description):
    """Check if file exists and report"""
    if Path(filepath).exists():
        print(f"   ✅ {description}: Found")
        return True
    else:
        print(f"   ❌ {description}: Missing")
        return False

def run_command(command, description):
    """Run command and capture output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"   ✅ {description}: Success")
            return True, result.stdout.strip()
        else:
            print(f"   ❌ {description}: Failed")
            print(f"      Error: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {description}: Timeout")
        return False, "Timeout"
    except Exception as e:
        print(f"   ❌ {description}: Exception - {e}")
        return False, str(e)

def check_repository_structure():
    """Check if repository has expected structure"""
    print_header("Repository Structure Analysis")
    
    # Check key files
    key_files = {
        'speed_estimation.py': 'Main speed estimation script',
        'README.md': 'Repository documentation', 
        'requirements.txt': 'Dependencies list',
        'train.py': 'Training script',
        'eval.py': 'Evaluation script',
        'tensorrt_estimation.py': 'TensorRT optimization script',
        'obtain_calibration.py': 'Calibration helper'
    }
    
    missing_files = []
    for file, desc in key_files.items():
        if not check_file_exists(file, desc):
            missing_files.append(file)
    
    # Check directories
    key_dirs = {
        'models/': 'Model definitions',
        'utils/': 'Utility functions',
        'data/': 'Data handling scripts',
        'checkpoints/': 'Pre-trained models',
        'configs/': 'Configuration files'
    }
    
    missing_dirs = []
    for dir_path, desc in key_dirs.items():
        if not check_file_exists(dir_path, desc):
            missing_dirs.append(dir_path)
    
    # List Python files found
    python_files = list(Path('.').glob('*.py'))
    print(f"\n📁 Python files found: {[f.name for f in python_files]}")
    
    # Check subdirectories
    subdirs = [d for d in Path('.').iterdir() if d.is_dir()]
    print(f"📁 Directories found: {[d.name for d in subdirs]}")
    
    return missing_files, missing_dirs

def check_python_environment():
    """Check Python environment and dependencies"""
    print_header("Python Environment Analysis")
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("   ⚠️  Warning: Python 3.8+ recommended")
    else:
        print("   ✅ Python version compatible")
    
    # Check virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("   ✅ Virtual environment detected")
    else:
        print("   ⚠️  No virtual environment detected - recommended to use one")
    
    # Check essential packages
    essential_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing_packages = []
    for package, name in essential_packages.items():
        try:
            if package == 'cv2':
                import cv2
                print(f"   ✅ {name}: {cv2.__version__}")
            elif package == 'torch':
                import torch
                print(f"   ✅ {name}: {torch.__version__}")
                # Check CUDA
                if torch.cuda.is_available():
                    print(f"   ✅ CUDA available: {torch.cuda.device_count()} device(s)")
                    print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("   ⚠️  CUDA not available - will use CPU (slower)")
            elif package == 'numpy':
                import numpy
                print(f"   ✅ {name}: {numpy.__version__}")
            elif package == 'matplotlib':
                import matplotlib
                print(f"   ✅ {name}: {matplotlib.__version__}")
        except ImportError:
            print(f"   ❌ {name}: Not installed")
            missing_packages.append(package)
    
    return missing_packages

def check_model_files():
    """Check for model files and weights"""
    print_header("Model Files Analysis")
    
    # Look for model files
    model_patterns = ['*.pt', '*.pth', '*.onnx', '*.weights']
    model_files = []
    
    for pattern in model_patterns:
        found = list(Path('.').rglob(pattern))
        model_files.extend(found)
    
    if model_files:
        print("🧠 Model files found:")
        for model in model_files:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"   ✅ {model} ({size_mb:.1f} MB)")
    else:
        print("❌ No model files found")
        print("   You need to download pre-trained models")
        
        # Check README for download instructions
        if Path('README.md').exists():
            with open('README.md', 'r', encoding='utf-8', errors='ignore') as f:
                readme_content = f.read().lower()
                if 'download' in readme_content or 'checkpoint' in readme_content or 'model' in readme_content:
                    print("   💡 Check README.md for download instructions")
    
    # Check checkpoints directory specifically
    checkpoints_dir = Path('checkpoints')
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob('*'))
        if checkpoint_files:
            print(f"\n📁 Checkpoints directory contains {len(checkpoint_files)} files:")
            for file in checkpoint_files[:5]:  # Show first 5
                print(f"   • {file.name}")
        else:
            print("\n📁 Checkpoints directory is empty")
    
    return model_files

def analyze_main_script():
    """Analyze the main speed estimation script"""
    print_header("Main Script Analysis")
    
    main_script = Path('speed_estimation.py')
    if not main_script.exists():
        print("❌ speed_estimation.py not found")
        # Look for alternative main scripts
        alternatives = ['main.py', 'inference.py', 'detect.py', 'run.py']
        found_alternatives = [f for f in alternatives if Path(f).exists()]
        if found_alternatives:
            print(f"   🔍 Alternative scripts found: {found_alternatives}")
        return None
    
    print("✅ speed_estimation.py found")
    
    # Analyze script content
    try:
        with open(main_script, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for argument parser
        if 'argparse' in content:
            print("   ✅ Command line arguments supported")
        else:
            print("   ⚠️  No argparse detected - may use hardcoded parameters")
        
        # Look for key parameters
        key_params = ['weights', 'source', 'output', 'device', 'conf']
        found_params = []
        for param in key_params:
            if param in content.lower():
                found_params.append(param)
        
        if found_params:
            print(f"   📊 Key parameters found: {found_params}")
        
        # Look for imports to understand dependencies
        critical_imports = ['torch', 'cv2', 'numpy']
        missing_imports = []
        for imp in critical_imports:
            if f'import {imp}' not in content and f'from {imp}' not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"   ⚠️  Missing imports in script: {missing_imports}")
        
    except Exception as e:
        print(f"   ❌ Error reading script: {e}")
        return None
    
    return True

def suggest_next_steps(missing_files, missing_dirs, missing_packages, model_files):
    """Suggest specific next steps based on analysis"""
    print_header("Recommended Next Steps")
    
    steps = []
    
    # Environment setup
    if missing_packages:
        steps.append("🐍 Install missing Python packages:")
        steps.append(f"   pip install {' '.join(missing_packages)}")
        if 'torch' in missing_packages:
            steps.append("   # For PyTorch with CUDA:")
            steps.append("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Model files
    if not model_files:
        steps.append("🧠 Download pre-trained models:")
        steps.append("   1. Check README.md for download links")
        steps.append("   2. Look for GitHub releases tab")
        steps.append("   3. Create checkpoints/ directory if missing")
        steps.append("   4. Download model files to checkpoints/")
    
    # Repository structure
    if missing_files:
        if 'requirements.txt' in missing_files:
            steps.append("📄 Create requirements.txt or install manually:")
            steps.append("   pip install torch torchvision opencv-python numpy matplotlib tqdm")
    
    # Testing
    steps.append("🎥 Test with your video:")
    steps.append("   1. Create data/videos/ directory")
    steps.append("   2. Copy your video file there")
    steps.append("   3. Run: python speed_estimation.py --help")
    steps.append("   4. Run with your video (adjust paths as needed)")
    
    # Print all steps
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

def create_quick_setup_script():
    """Create a quick setup script"""
    print_header("Quick Setup Script Generation")
    
    setup_script = """#!/bin/bash
# Quick setup script for Vehicle-Speed-Estimation-YOLOv6-3D

echo "🚀 Setting up Vehicle Speed Estimation..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv speednet_env

# Activate environment
echo "✅ Activating environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    source speednet_env/Scripts/activate
else
    source speednet_env/bin/activate
fi

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy matplotlib tqdm

# Install from requirements.txt if exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p data/videos
mkdir -p data/results
mkdir -p checkpoints

echo "✅ Setup complete!"
echo "💡 Next steps:"
echo "   1. Download model files to checkpoints/"
echo "   2. Copy your video to data/videos/"
echo "   3. Run: python speed_estimation.py --help"
"""
    
    with open('quick_setup.sh', 'w') as f:
        f.write(setup_script)
    
    print("✅ Created quick_setup.sh")
    print("   Run with: bash quick_setup.sh")

def main():
    """Main diagnostic function"""
    print("🔍 Vehicle Speed Estimation YOLOv6-3D - Repository Diagnostic")
    print("=" * 70)
    print("📍 Analyzing your setup and providing specific guidance...")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Run all checks
    missing_files, missing_dirs = check_repository_structure()
    missing_packages = check_python_environment()
    model_files = check_model_files()
    analyze_main_script()
    
    # Provide recommendations
    suggest_next_steps(missing_files, missing_dirs, missing_packages, model_files)
    
    # Create setup script
    create_quick_setup_script()
    
    print_header("Summary")
    
    # Overall status
    issues = len(missing_files) + len(missing_packages) + (0 if model_files else 1)
    
    if issues == 0:
        print("🎉 Your setup looks good! You should be able to run the speed estimation.")
        print("💡 Try: python speed_estimation.py --help")
    elif issues <= 2:
        print("⚠️  Minor issues detected. Follow the recommended steps above.")
    else:
        print("❌ Several issues detected. Please follow the setup steps carefully.")
    
    print(f"\n📊 Issues found: {issues}")
    print("📝 Follow the numbered steps above to resolve issues.")
    print("\n🆘 If you encounter specific errors, share them for targeted help!")

if __name__ == "__main__":
    main()