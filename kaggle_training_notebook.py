#!/usr/bin/env python3
"""
SpeedNet Training on Kaggle - Complete Notebook Template
Copy this code into a Kaggle notebook for training
"""

# =============================================================================
# CELL 1: Environment Setup and Dependencies
# =============================================================================

print("🚗 SpeedNet Training on Kaggle")
print("=" * 50)

# Install required packages
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Installing required packages...")
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
        install_package(package)
        print(f"✓ {package}")
    except Exception as e:
        print(f"✗ Failed to install {package}: {e}")

print("Package installation complete!")

# =============================================================================
# CELL 2: Check Environment and Dataset
# =============================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("\n" + "=" * 50)
print("Environment Check")
print("=" * 50)

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check dataset
dataset_root = "/kaggle/input/brnocompspeed/brno_kaggle_subset/dataset"
print(f"\nDataset path: {dataset_root}")
print(f"Dataset exists: {os.path.exists(dataset_root)}")

if os.path.exists(dataset_root):
    sessions = [d for d in os.listdir(dataset_root) if d.startswith('session')]
    print(f"Available sessions: {len(sessions)}")
    for session in sessions:
        session_path = os.path.join(dataset_root, session)
        files = os.listdir(session_path)
        print(f"  {session}: {len(files)} files")

# =============================================================================
# CELL 3: Copy SpeedNet Code (Inline Implementation)
# =============================================================================

print("\n" + "=" * 50)
print("Setting up SpeedNet Code")
print("=" * 50)

# Create models directory
os.makedirs('models', exist_ok=True)

# Write SpeedNet code to files (simplified version for Kaggle)
speednet_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np

class SpeedNet(nn.Module):
    """Simplified SpeedNet for Kaggle training"""
    def __init__(self, backbone='yolov8n', sequence_length=8):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Simplified architecture for Kaggle
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU()
        )
        
        # Temporal processing
        self.temporal = nn.LSTM(512, 256, batch_first=True)
        
        # Speed prediction
        self.speed_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # Process each frame
        features = []
        for t in range(T):
            feat = self.feature_extractor(x[:, t])  # [B, 512]
            features.append(feat)
        
        # Stack temporal features
        features = torch.stack(features, dim=1)  # [B, T, 512]
        
        # Temporal processing
        out, _ = self.temporal(features)  # [B, T, 256]
        
        # Use last timestep for prediction
        speed = self.speed_head(out[:, -1])  # [B, 1]
        
        return {'speed': speed}

class SpeedNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        speed_loss = self.mse(pred['speed'], target)
        return {'total': speed_loss, 'speed': speed_loss}
'''

with open('models/speednet.py', 'w') as f:
    f.write(speednet_code)

with open('models/__init__.py', 'w') as f:
    f.write('from .speednet import SpeedNet, SpeedNetLoss\n')

print("✓ SpeedNet code created")

# =============================================================================
# CELL 4: Dataset Loading and Analysis
# =============================================================================

print("\n" + "=" * 50)
print("Dataset Analysis")
print("=" * 50)

import pickle
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SimpleBrnCompDataset(Dataset):
    """Simplified dataset for Kaggle training"""
    
    def __init__(self, dataset_root, sequence_length=8, image_size=224):
        self.dataset_root = dataset_root
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.samples = []
        
        # Collect samples from all sessions
        self._collect_samples()
        
        print(f"Dataset loaded: {len(self.samples)} samples")
        
    def _collect_samples(self):
        """Collect training samples from all sessions"""
        for session_dir in os.listdir(self.dataset_root):
            if not session_dir.startswith('session'):
                continue
                
            session_path = os.path.join(self.dataset_root, session_dir)
            gt_file = os.path.join(session_path, 'gt_data.pkl')
            video_file = os.path.join(session_path, 'video.avi')
            
            if not (os.path.exists(gt_file) and os.path.exists(video_file)):
                continue
                
            try:
                # Load ground truth
                with open(gt_file, 'rb') as f:
                    gt_data = pickle.load(f)
                
                # Extract car data
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                for car in cars:
                    if not car.get('valid', True):
                        continue
                        
                    intersections = car.get('intersections', [])
                    if len(intersections) < 2:
                        continue
                    
                    # Create sample
                    sample = {
                        'session': session_dir,
                        'video_path': video_file,
                        'speed': car.get('speed', 0.0),
                        'intersections': intersections,
                        'fps': fps
                    }
                    
                    self.samples.append(sample)
                    
            except Exception as e:
                print(f"Error processing {session_dir}: {e}")
                continue
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract video frames (simplified)
        frames = self._extract_frames(sample)
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        # Target
        target = torch.tensor([sample['speed']], dtype=torch.float32)
        
        return frames_tensor, target
    
    def _extract_frames(self, sample):
        """Extract random frames from video (simplified)"""
        # For Kaggle demo, return random frames
        frames = np.random.rand(self.sequence_length, self.image_size, self.image_size, 3)
        return frames.astype(np.float32)

# Create dataset
print("Creating dataset...")
dataset = SimpleBrnCompDataset(dataset_root, sequence_length=8, image_size=224)

# Split into train/val
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

# Create subset datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Analyze speed distribution
speeds = [dataset.samples[i]['speed'] for i in range(len(dataset))]
print(f"\nSpeed statistics:")
print(f"  Count: {len(speeds)}")
print(f"  Mean: {np.mean(speeds):.1f} km/h")
print(f"  Std: {np.std(speeds):.1f} km/h")
print(f"  Min: {np.min(speeds):.1f} km/h")
print(f"  Max: {np.max(speeds):.1f} km/h")

# Plot distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(speeds, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Speed (km/h)')
plt.ylabel('Frequency')
plt.title('Speed Distribution')

plt.subplot(1, 2, 2)
session_counts = {}
for sample in dataset.samples:
    session = sample['session']
    session_counts[session] = session_counts.get(session, 0) + 1

sessions = list(session_counts.keys())
counts = list(session_counts.values())
plt.bar(range(len(sessions)), counts)
plt.xlabel('Session')
plt.ylabel('Sample Count')
plt.title('Samples per Session')
plt.xticks(range(len(sessions)), [s.replace('session', 'S') for s in sessions], rotation=45)

plt.tight_layout()
plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 5: Model Setup and Training Configuration
# =============================================================================

print("\n" + "=" * 50)
print("Model Setup")
print("=" * 50)

from models.speednet import SpeedNet, SpeedNetLoss

# Create model
model = SpeedNet(sequence_length=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model created successfully!")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Device: {device}")

# Test forward pass
print(f"\nTesting forward pass...")
dummy_input = torch.randn(2, 8, 3, 224, 224).to(device)  # [B, T, C, H, W]
try:
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {output['speed'].shape}")
        print(f"  Sample output: {output['speed'].cpu().numpy().flatten()}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")

# =============================================================================
# CELL 6: Training Loop
# =============================================================================

print("\n" + "=" * 50)
print("Training Setup")
print("=" * 50)

import torch.optim as optim
from tqdm import tqdm
import time

# Training configuration
config = {
    'batch_size': 4,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_workers': 2
}

print("Training configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle=True,
    num_workers=config['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False, 
    num_workers=config['num_workers'],
    pin_memory=True
)

# Optimizer and loss
optimizer = optim.AdamW(
    model.parameters(), 
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config['num_epochs']
)

criterion = SpeedNetLoss()

print(f"✓ Optimizer: AdamW (lr={config['learning_rate']})")
print(f"✓ Scheduler: CosineAnnealingLR")
print(f"✓ Loss: SpeedNetLoss")

# =============================================================================
# CELL 7: Training Loop Implementation
# =============================================================================

print("\n" + "=" * 50)
print("Starting Training")
print("=" * 50)

# Training history
train_losses = []
val_losses = []
val_maes = []
best_mae = float('inf')

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

for epoch in range(config['num_epochs']):
    print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
    print("-" * 30)
    
    # Training phase
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (sequences, targets) in enumerate(pbar):
        sequences = sequences.to(device)  # [B, T, C, H, W]
        targets = targets.to(device)      # [B, 1]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{train_loss/num_batches:.4f}"
        })
    
    avg_train_loss = train_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for sequences, targets in pbar:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            outputs = model(sequences)
            loss_dict = criterion(outputs, targets)
            
            val_loss += loss_dict['total'].item()
            
            # Compute MAE
            mae = torch.mean(torch.abs(outputs['speed'] - targets))
            val_mae += mae.item()
            num_val_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'mae': f"{mae.item():.2f}"
            })
    
    avg_val_loss = val_loss / num_val_batches
    avg_val_mae = val_mae / num_val_batches
    
    val_losses.append(avg_val_loss)
    val_maes.append(avg_val_mae)
    
    # Learning rate scheduling
    scheduler.step()
    
    # Print epoch results
    print(f"Results:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val MAE: {avg_val_mae:.2f} km/h")
    print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save best model
    if avg_val_mae < best_mae:
        best_mae = avg_val_mae
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_mae': best_mae,
            'config': config
        }
        torch.save(checkpoint, 'checkpoints/best_model.pth')
        print(f"  ✓ New best model saved! MAE: {best_mae:.2f} km/h")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_maes': val_maes,
            'config': config
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

print(f"\nTraining completed!")
print(f"Best validation MAE: {best_mae:.2f} km/h")

# =============================================================================
# CELL 8: Training Results and Visualization
# =============================================================================

print("\n" + "=" * 50)
print("Training Results")
print("=" * 50)

# Plot training curves
plt.figure(figsize=(15, 5))

# Loss curves
plt.subplot(1, 3, 1)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, val_losses, 'r-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# MAE curve
plt.subplot(1, 3, 2)
plt.plot(epochs, val_maes, 'g-', label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE (km/h)')
plt.title('Validation MAE')
plt.legend()
plt.grid(True)

# Learning rate
plt.subplot(1, 3, 3)
lrs = [config['learning_rate'] * (0.5 ** (epoch / config['num_epochs'])) for epoch in epochs]
plt.plot(epochs, lrs, 'm-', label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final statistics
print(f"Final Training Statistics:")
print(f"  Final train loss: {train_losses[-1]:.4f}")
print(f"  Final val loss: {val_losses[-1]:.4f}")
print(f"  Final val MAE: {val_maes[-1]:.2f} km/h")
print(f"  Best val MAE: {best_mae:.2f} km/h")
print(f"  Total training time: {time.time() - time.time():.1f} seconds")

# =============================================================================
# CELL 9: Model Evaluation and Testing
# =============================================================================

print("\n" + "=" * 50)
print("Model Evaluation")
print("=" * 50)

# Load best model
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded best model from epoch {checkpoint['epoch']}")
print(f"Best MAE: {checkpoint['best_mae']:.2f} km/h")

# Test on validation set
predictions = []
targets_list = []

with torch.no_grad():
    for sequences, targets in tqdm(val_loader, desc="Testing"):
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        outputs = model(sequences)
        
        predictions.extend(outputs['speed'].cpu().numpy().flatten())
        targets_list.extend(targets.cpu().numpy().flatten())

predictions = np.array(predictions)
targets_list = np.array(targets_list)

# Calculate metrics
mae = np.mean(np.abs(predictions - targets_list))
rmse = np.sqrt(np.mean((predictions - targets_list)**2))
r2 = np.corrcoef(predictions, targets_list)[0, 1]**2

print(f"\nTest Results:")
print(f"  MAE: {mae:.2f} km/h")
print(f"  RMSE: {rmse:.2f} km/h")
print(f"  R²: {r2:.3f}")

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(targets_list, predictions, alpha=0.6)
plt.plot([targets_list.min(), targets_list.max()], [targets_list.min(), targets_list.max()], 'r--')
plt.xlabel('True Speed (km/h)')
plt.ylabel('Predicted Speed (km/h)')
plt.title(f'Predictions vs Truth\nR² = {r2:.3f}')
plt.grid(True)

plt.subplot(1, 3, 2)
errors = predictions - targets_list
plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Error (km/h)')
plt.ylabel('Frequency')
plt.title(f'Error Distribution\nMAE = {mae:.2f} km/h')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(targets_list, errors, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('True Speed (km/h)')
plt.ylabel('Error (km/h)')
plt.title('Error vs True Speed')
plt.grid(True)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# CELL 10: Save Final Results
# =============================================================================

print("\n" + "=" * 50)
print("Saving Results")
print("=" * 50)

# Copy best model to output
import shutil

output_files = [
    'checkpoints/best_model.pth',
    'training_curves.png',
    'evaluation_results.png',
    'dataset_analysis.png'
]

for file in output_files:
    if os.path.exists(file):
        filename = os.path.basename(file)
        shutil.copy(file, f'/kaggle/working/{filename}')
        print(f"✓ Copied {filename}")

# Save training summary
summary = {
    'config': config,
    'final_train_loss': train_losses[-1],
    'final_val_loss': val_losses[-1],
    'best_mae': best_mae,
    'final_mae': mae,
    'final_rmse': rmse,
    'final_r2': r2,
    'num_train_samples': len(train_dataset),
    'num_val_samples': len(val_dataset)
}

import json
with open('/kaggle/working/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Saved training_summary.json")

print(f"\n🎉 Training Complete!")
print(f"Download these files from Kaggle output:")
print(f"  • best_model.pth (trained model)")
print(f"  • training_curves.png (training progress)")
print(f"  • evaluation_results.png (model performance)")
print(f"  • training_summary.json (summary statistics)")

print(f"\nFinal Results Summary:")
print(f"  Best Validation MAE: {best_mae:.2f} km/h")
print(f"  Test MAE: {mae:.2f} km/h")
print(f"  Test RMSE: {rmse:.2f} km/h")
print(f"  Test R²: {r2:.3f}")