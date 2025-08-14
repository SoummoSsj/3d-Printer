#!/usr/bin/env python3
"""
SpeedNet Training Setup Helper
Prepares your environment for training on BrnCompSpeed dataset
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil
import zipfile

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️  No GPU detected - training will be slow on CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - cannot check GPU")
        return False

def install_dependencies(environment='local'):
    """Install required dependencies"""
    print(f"\n📦 Installing dependencies for {environment} environment...")
    
    if environment == 'local':
        # Install from requirements.txt
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    elif environment == 'kaggle':
        # Kaggle-specific packages
        packages = [
            "ultralytics>=8.0.0",
            "supervision>=0.16.0", 
            "albumentations>=1.3.0",
            "scikit-learn",
            "matplotlib",
            "seaborn"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
        return True

def create_project_structure():
    """Create the project directory structure"""
    print("\n📁 Creating project structure...")
    
    directories = [
        "models",
        "data", 
        "checkpoints",
        "results",
        "logs"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created {dir_name}/")
    
    return True

def create_kaggle_package():
    """Create a zip file for easy Kaggle upload"""
    print("\n📦 Creating Kaggle upload package...")
    
    files_to_include = [
        "models/speednet.py",
        "models/__init__.py", 
        "dataset_analyzer.py",
        "train_speednet.py",
        "requirements.txt",
        "README.md",
        "kaggle_training_notebook.py"
    ]
    
    # Create kaggle upload directory
    kaggle_dir = Path("kaggle_upload")
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy files
    for file_path in files_to_include:
        if os.path.exists(file_path):
            dest_path = kaggle_dir / file_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)
            print(f"✅ Added {file_path}")
        else:
            print(f"⚠️  Missing {file_path}")
    
    # Create zip file
    zip_path = "speednet_kaggle_upload.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(kaggle_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, kaggle_dir)
                zipf.write(file_path, arcname)
    
    print(f"✅ Created {zip_path} for Kaggle upload")
    
    # Cleanup
    shutil.rmtree(kaggle_dir)
    return True

def verify_dataset(dataset_path):
    """Verify BrnCompSpeed dataset structure"""
    print(f"\n🔍 Verifying dataset at {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    
    # Expected sessions
    expected_sessions = [
        'session0_center', 'session0_left', 'session0_right',
        'session1_center', 'session1_left', 'session1_right'
    ]
    
    found_sessions = []
    for session in expected_sessions:
        session_path = os.path.join(dataset_path, session)
        if os.path.exists(session_path):
            # Check required files
            required_files = ['gt_data.pkl', 'video.avi']
            has_all_files = all(
                os.path.exists(os.path.join(session_path, f)) 
                for f in required_files
            )
            if has_all_files:
                found_sessions.append(session)
                print(f"✅ {session}")
            else:
                print(f"⚠️  {session} (missing files)")
        else:
            print(f"❌ {session} (not found)")
    
    if found_sessions:
        print(f"✅ Found {len(found_sessions)} valid sessions")
        return True
    else:
        print("❌ No valid sessions found")
        return False

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("✅ All imports successful")
        
        # Test model creation
        sys.path.append('.')
        from models.speednet import SpeedNet
        
        model = SpeedNet(sequence_length=4)  # Small for testing
        print("✅ Model creation successful")
        
        # Test forward pass
        dummy_input = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ Forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps(environment):
    """Print next steps for the user"""
    print(f"\n🚀 Setup complete! Next steps for {environment}:")
    print("=" * 50)
    
    if environment == 'local':
        print("1. Download BrnCompSpeed dataset:")
        print("   • Go to: https://www.kaggle.com/datasets/jakubsochor/brnocompspeed")
        print("   • Download and extract to ./data/")
        print("")
        print("2. Verify dataset:")
        print("   python setup_training.py --verify-dataset data/brno_kaggle_subset/dataset")
        print("")
        print("3. Start training:")
        print("   python train_speednet.py --dataset_root data/brno_kaggle_subset/dataset")
        print("")
        print("4. Monitor with:")
        print("   tensorboard --logdir logs/")
        
    elif environment == 'kaggle':
        print("1. Upload speednet_kaggle_upload.zip as a Kaggle dataset")
        print("")
        print("2. Create new Kaggle notebook")
        print("")
        print("3. Add these datasets:")
        print("   • BrnCompSpeed dataset")
        print("   • Your uploaded SpeedNet code")
        print("")
        print("4. Copy code from kaggle_training_notebook.py")
        print("")
        print("5. Run training (remember to enable GPU!)")

def main():
    parser = argparse.ArgumentParser(description='SpeedNet Training Setup')
    parser.add_argument('--environment', type=str, choices=['local', 'kaggle'], 
                       default='local', help='Target environment')
    parser.add_argument('--verify-dataset', type=str, help='Verify dataset at path')
    parser.add_argument('--quick-test', action='store_true', help='Run quick functionality test')
    parser.add_argument('--kaggle-package', action='store_true', help='Create Kaggle upload package')
    
    args = parser.parse_args()
    
    print("🚗 SpeedNet Training Setup")
    print("=" * 30)
    
    # Verify dataset only
    if args.verify_dataset:
        verify_dataset(args.verify_dataset)
        return
    
    # Quick test only  
    if args.quick_test:
        run_quick_test()
        return
    
    # Create Kaggle package only
    if args.kaggle_package:
        create_kaggle_package()
        return
    
    # Full setup
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check GPU (optional)
    check_gpu()
    
    # Create project structure
    if success and not create_project_structure():
        success = False
    
    # Install dependencies
    if success and not install_dependencies(args.environment):
        success = False
    
    # Create Kaggle package if needed
    if success and args.environment == 'kaggle':
        create_kaggle_package()
    
    # Run quick test
    if success and args.environment == 'local':
        run_quick_test()
    
    # Print next steps
    if success:
        print_next_steps(args.environment)
    else:
        print("\n❌ Setup incomplete. Please fix the errors above.")

if __name__ == "__main__":
    main()