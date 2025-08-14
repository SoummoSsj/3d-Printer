#!/usr/bin/env python3
"""
🚗 Balanced SpeedNet for Kaggle
Optimized for quality vs performance balance
"""

import os
import sys
import json
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("🚗 Balanced SpeedNet for Kaggle")
print("⚖️ Quality vs Performance Balance")
print("=" * 60)

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Force GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🔧 GPU Status:")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()

# =============================================================================
# Balanced Dataset Configuration
# =============================================================================

def load_pickle_safe(file_path):
    """Safely load pickle files"""
    encodings_to_try = ['latin1', 'bytes', 'ASCII', 'utf-8']
    
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'rb') as f:
                if encoding == 'bytes':
                    data = pickle.load(f, encoding='bytes')
                else:
                    data = pickle.load(f, encoding=encoding)
                return data
        except:
            continue
    return None

def convert_bytes_keys(obj):
    """Convert bytes to strings"""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            if isinstance(key, bytes):
                key = key.decode('utf-8', errors='replace')
            new_dict[key] = convert_bytes_keys(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_bytes_keys(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    else:
        return obj

class BalancedBrnCompDataset(Dataset):
    """Balanced dataset - compromise between quality and efficiency"""
    
    def __init__(self, dataset_root, sequence_length=6, image_size=160, mode='train'):
        self.dataset_root = dataset_root
        self.sequence_length = sequence_length  # Balanced: 6 frames (vs 8 original, 4 lightweight)
        self.image_size = image_size             # Balanced: 160px (vs 224 original, 128 lightweight)
        self.mode = mode
        self.samples = []
        
        # Balanced transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self._collect_samples()
        print(f"✅ {mode} dataset: {len(self.samples)} samples")
        
    def _collect_samples(self):
        """Collect samples efficiently"""
        for session_dir in os.listdir(self.dataset_root):
            if not session_dir.startswith('session'):
                continue
                
            session_path = os.path.join(self.dataset_root, session_dir)
            gt_file = os.path.join(session_path, 'gt_data.pkl')
            video_file = os.path.join(session_path, 'video.avi')
            
            if not (os.path.exists(gt_file) and os.path.exists(video_file)):
                continue
                
            try:
                gt_data = load_pickle_safe(gt_file)
                if gt_data is None:
                    continue
                
                gt_data = convert_bytes_keys(gt_data)
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                for car in cars:
                    if not car.get('valid', True):
                        continue
                        
                    intersections = car.get('intersections', [])
                    if len(intersections) < 2:
                        continue
                    
                    times = [i['videoTime'] for i in intersections if 'videoTime' in i]
                    if len(times) < 2:
                        continue
                        
                    start_time = min(times) - 0.8  # Balanced window
                    end_time = max(times) + 0.8
                    
                    speed = car.get('speed', 0.0)
                    if not isinstance(speed, (int, float)) or np.isnan(speed) or speed < 0:
                        speed = 0.0
                    
                    self.samples.append({
                        'session': session_dir,
                        'video_path': video_file,
                        'speed': float(speed),
                        'start_time': start_time,
                        'end_time': end_time,
                        'fps': fps,
                    })
                    
            except Exception as e:
                print(f"❌ Error processing {session_dir}: {e}")
                continue
                
    def _extract_video_frames(self, video_path, start_time, end_time, fps):
        """Extract frames with balanced quality"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._create_dummy_frames()
            
            start_frame = max(0, int(start_time * fps))
            end_frame = int(end_time * fps)
            
            frame_indices = np.linspace(start_frame, end_frame, self.sequence_length, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
            
            cap.release()
            return np.array(frames[:self.sequence_length], dtype=np.float32) / 255.0
            
        except Exception as e:
            return self._create_dummy_frames()
    
    def _create_dummy_frames(self):
        """Create dummy frames"""
        return np.random.rand(self.sequence_length, self.image_size, self.image_size, 3).astype(np.float32)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        frames = self._extract_video_frames(
            sample['video_path'], 
            sample['start_time'], 
            sample['end_time'], 
            sample['fps']
        )
        
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        
        if self.transform:
            transformed_frames = []
            for t in range(frames_tensor.shape[0]):
                frame = self.transform(frames_tensor[t])
                transformed_frames.append(frame)
            frames_tensor = torch.stack(transformed_frames)
        
        target = torch.tensor([sample['speed']], dtype=torch.float32)
        return frames_tensor, target

# =============================================================================
# Balanced SpeedNet Architecture  
# =============================================================================

class BalancedSpeedNet(nn.Module):
    """Balanced SpeedNet - medium complexity for good quality/performance trade-off"""
    def __init__(self, sequence_length=6):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Balanced backbone - more complex than lightweight, simpler than full
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 3
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block 4
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # Enhanced temporal processing
        self.temporal = nn.LSTM(256, 128, batch_first=True, dropout=0.3, num_layers=2)
        
        # Attention mechanism (lightweight)
        self.attention = nn.MultiheadAttention(128, num_heads=4, dropout=0.2, batch_first=True)
        
        # Speed regression with residual connection
        self.speed_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, image_sequence):
        """
        Args:
            image_sequence: [B, T, C, H, W]
        """
        B, T, C, H, W = image_sequence.shape
        
        # Process each frame
        frame_features = []
        for t in range(T):
            feat = self.backbone(image_sequence[:, t])  # [B, 256]
            frame_features.append(feat)
        
        # Stack temporal features
        sequence_features = torch.stack(frame_features, dim=1)  # [B, T, 256]
        
        # LSTM processing
        temporal_out, _ = self.temporal(sequence_features)  # [B, T, 128]
        
        # Self-attention for better temporal modeling
        attended_out, _ = self.attention(temporal_out, temporal_out, temporal_out)  # [B, T, 128]
        
        # Combine LSTM + attention
        combined_features = temporal_out + attended_out  # Residual connection
        
        # Use mean pooling across time (better than just last timestep)
        final_features = torch.mean(combined_features, dim=1)  # [B, 128]
        
        # Predictions
        speed = self.speed_head(final_features)  # [B, 1]
        uncertainty = self.uncertainty_head(final_features)  # [B, 1]
        
        # Ensure positive speed
        speed = F.softplus(speed)
        
        return {
            'speed': speed,
            'uncertainty': uncertainty
        }

class BalancedSpeedNetLoss(nn.Module):
    """Enhanced loss function for balanced model"""
    def __init__(self, speed_weight=1.0, uncertainty_weight=0.15, smoothness_weight=0.05):
        super().__init__()
        self.speed_weight = speed_weight
        self.uncertainty_weight = uncertainty_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self, predictions, targets):
        pred_speeds = predictions['speed']
        pred_uncertainties = predictions['uncertainty']
        target_speeds = targets
        
        # Heteroscedastic loss (uncertainty-weighted)
        speed_diff = pred_speeds - target_speeds
        precision = torch.exp(-pred_uncertainties)
        speed_loss = torch.mean(precision * speed_diff**2 + pred_uncertainties)
        
        # Uncertainty regularization
        uncertainty_reg = torch.mean(torch.exp(pred_uncertainties))
        
        # Smoothness regularization for stable predictions
        smoothness_loss = torch.mean(torch.abs(pred_speeds[1:] - pred_speeds[:-1])) if len(pred_speeds) > 1 else 0.0
        
        total_loss = (self.speed_weight * speed_loss + 
                     self.uncertainty_weight * uncertainty_reg +
                     self.smoothness_weight * smoothness_loss)
        
        return {
            'total': total_loss,
            'speed': speed_loss,
            'uncertainty': uncertainty_reg,
            'smoothness': smoothness_loss
        }

# =============================================================================
# Balanced Training Loop
# =============================================================================

def main():
    dataset_root = "/kaggle/input/brnocomp/brno_kaggle_subset/dataset"
    
    print(f"\n📁 Loading balanced dataset...")
    
    # Create balanced datasets
    full_dataset = BalancedBrnCompDataset(
        dataset_root, 
        sequence_length=6,   # Balanced: more context than lightweight
        image_size=160,      # Balanced: better detail than lightweight
        mode='train'
    )
    
    # Train/val split
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42
    )
    
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    
    train_dataset = BalancedBrnCompDataset(dataset_root, sequence_length=6, image_size=160, mode='train')
    train_dataset.samples = train_samples
    
    val_dataset = BalancedBrnCompDataset(dataset_root, sequence_length=6, image_size=160, mode='val')
    val_dataset.samples = val_samples
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Balanced configuration
    config = {
        'batch_size': 6,        # Balanced batch size
        'num_epochs': 40,       # More epochs for better convergence
        'learning_rate': 1e-4,  # Conservative learning rate
        'weight_decay': 1e-5,
        'num_workers': 2,
        'sequence_length': 6,
        'image_size': 160
    }
    
    print(f"\n⚖️ Balanced Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    
    print(f"✅ Train loader: {len(train_loader)} batches")
    print(f"✅ Val loader: {len(val_loader)} batches")
    
    # Create balanced model
    model = BalancedSpeedNet(sequence_length=6).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 Balanced SpeedNet created!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {device}")
    
    # Test model
    print(f"\n🧪 Testing model...")
    dummy_input = torch.randn(2, 6, 3, 160, 160).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Model test successful!")
        print(f"  Speed shape: {output['speed'].shape}")
    
    # GPU memory check
    if torch.cuda.is_available():
        print(f"  GPU memory after model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = BalancedSpeedNetLoss()
    scaler = GradScaler('cuda')
    
    # Training loop
    print(f"\n🚀 Starting Balanced Training...")
    print("=" * 60)
    
    train_losses = []
    val_maes = []
    best_mae = float('inf')
    
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="🔥 Training")
        for sequences, targets in pbar:
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(sequences)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['total']
            
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in tqdm(val_loader, desc="📊 Validation"):
                sequences = sequences.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs = model(sequences)
                
                predictions = outputs['speed'].cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets_np)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        val_mae = mean_absolute_error(all_targets, all_predictions)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        val_r2 = np.corrcoef(all_targets, all_predictions)[0, 1] ** 2 if len(all_targets) > 1 else 0.0
        
        val_maes.append(val_mae)
        
        scheduler.step()
        
        # Results
        epoch_time = (time.time() - start_time) / (epoch + 1)
        remaining_time = (config['num_epochs'] - epoch - 1) * epoch_time / 3600
        
        print(f"📈 Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f} km/h")
        print(f"  Val RMSE: {val_rmse:.2f} km/h") 
        print(f"  Val R²: {val_r2:.3f}")
        print(f"  Epoch time: {epoch_time:.1f}s")
        print(f"  Est. remaining: {remaining_time:.1f}h")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        if torch.cuda.is_available():
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'mae': val_mae,
                'config': config
            }, '/kaggle/working/best_balanced_model.pth')
            print(f"  🏆 New best model saved! MAE: {best_mae:.2f} km/h")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Balanced Training completed!")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best validation MAE: {best_mae:.2f} km/h")
    
    # Save training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_maes, 'g-', label='Val MAE', linewidth=2)
    plt.axhline(y=best_mae, color='r', linestyle='--', label=f'Best: {best_mae:.2f}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (km/h)')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Speed distribution comparison
    plt.hist(all_predictions, bins=30, alpha=0.7, label='Predicted', density=True)
    plt.hist(all_targets, bins=30, alpha=0.7, label='Ground Truth', density=True)
    plt.xlabel('Speed (km/h)')
    plt.ylabel('Density')
    plt.title('Speed Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/balanced_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📁 Files saved to /kaggle/working/:")
    print(f"  • best_balanced_model.pth")
    print(f"  • balanced_training_results.png")

if __name__ == "__main__":
    main()