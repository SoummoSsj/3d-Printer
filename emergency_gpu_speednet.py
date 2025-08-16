#!/usr/bin/env python3
"""
🚨 Emergency GPU SpeedNet - Force GPU Usage
Minimal complexity, maximum GPU utilization
"""

# STEP 1: FORCE GPU SETUP
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU to avoid complications
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import numpy as np
import cv2
import random
import time
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# EMERGENCY GPU INITIALIZATION
print("🚨 EMERGENCY GPU SETUP")
print("=" * 50)

# Clear everything
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Force device selection - NO FALLBACK TO CPU
assert torch.cuda.is_available(), "❌ CUDA MUST be available!"
device = torch.device('cuda:0')
torch.cuda.set_device(0)

# Test GPU immediately
test_tensor = torch.randn(100, 100, device=device)
gpu_result = torch.mm(test_tensor, test_tensor)
print(f"✅ GPU FORCED: {gpu_result.device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# MINIMAL DATASET - NO COMPLEX AUGMENTATION
# =============================================================================

def load_pickle_safe(file_path):
    encodings = ['latin1', 'bytes', 'utf-8']
    for encoding in encodings:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding=encoding)
                return data
        except:
            continue
    return None

class MinimalDataset(Dataset):
    def __init__(self, dataset_root, mode='train'):
        self.dataset_root = dataset_root
        self.samples = []
        self.device = device  # Store device
        
        # Collect samples quickly
        for session_dir in os.listdir(dataset_root):
            if not session_dir.startswith('session'):
                continue
                
            gt_file = os.path.join(dataset_root, session_dir, 'gt_data.pkl')
            video_file = os.path.join(dataset_root, session_dir, 'video.avi')
            
            if not os.path.exists(gt_file):
                continue
                
            try:
                gt_data = load_pickle_safe(gt_file)
                if gt_data is None:
                    continue
                    
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                for car in cars[:50]:  # Limit samples for speed
                    if not car.get('valid', True):
                        continue
                        
                    intersections = car.get('intersections', [])
                    if len(intersections) < 2:
                        continue
                    
                    times = [i['videoTime'] for i in intersections if 'videoTime' in i]
                    if len(times) < 2:
                        continue
                        
                    start_time = min(times) - 0.5
                    end_time = max(times) + 0.5
                    speed = float(car.get('speed', 0.0))
                    
                    if speed > 0:
                        self.samples.append({
                            'video_path': video_file,
                            'speed': speed,
                            'start_time': start_time,
                            'end_time': end_time,
                            'fps': fps,
                        })
                        
            except Exception as e:
                print(f"Skip {session_dir}: {e}")
                continue
        
        print(f"✅ {mode}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # MINIMAL frame extraction - just 4 frames
        try:
            cap = cv2.VideoCapture(sample['video_path'])
            start_frame = int(sample['start_time'] * sample['fps'])
            end_frame = int(sample['end_time'] * sample['fps'])
            
            frame_indices = np.linspace(start_frame, end_frame, 4, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (128, 128))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
            
            cap.release()
            
            # Convert to tensor and normalize
            frames = np.array(frames, dtype=np.float32) / 255.0
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - mean) / std
            
        except:
            # Dummy data if video fails
            frames_tensor = torch.randn(4, 3, 128, 128)
        
        target = torch.tensor([sample['speed']], dtype=torch.float32)
        return frames_tensor, target

# =============================================================================
# ULTRA-MINIMAL MODEL - GUARANTEED GPU USAGE
# =============================================================================

class EmergencySpeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Minimal CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 128->64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64->32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32->16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        
        # Minimal temporal
        self.temporal = nn.LSTM(128 * 4 * 4, 64, batch_first=True)
        
        # Output
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True)  # Ensure positive speed
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Process frames
        features = []
        for t in range(T):
            feat = self.backbone(x[:, t])
            features.append(feat)
        
        # Temporal processing
        sequence = torch.stack(features, dim=1)
        lstm_out, _ = self.temporal(sequence)
        
        # Output
        speed = self.head(lstm_out[:, -1])
        return speed

# =============================================================================
# EMERGENCY TRAINING - FORCE GPU
# =============================================================================

def main():
    print("\n🚨 STARTING EMERGENCY TRAINING")
    
    # Dataset
    dataset_root = "/kaggle/input/brnocomp/brno_kaggle_subset/dataset"
    
    full_dataset = MinimalDataset(dataset_root, 'train')
    
    # Quick split
    split_idx = int(0.8 * len(full_dataset))
    train_samples = full_dataset.samples[:split_idx]
    val_samples = full_dataset.samples[split_idx:]
    
    train_dataset = MinimalDataset(dataset_root, 'train')
    train_dataset.samples = train_samples
    
    val_dataset = MinimalDataset(dataset_root, 'val') 
    val_dataset.samples = val_samples
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Data loaders - MINIMAL WORKERS
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Larger batch since model is smaller
        shuffle=True, 
        num_workers=0,  # NO multiprocessing - avoid issues
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # Model - FORCE TO GPU
    model = EmergencySpeedNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    # Test model GPU usage
    dummy_input = torch.randn(2, 4, 3, 128, 128, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"✅ Model test: {dummy_output.device} - {dummy_output.shape}")
    
    # TRAINING LOOP
    print(f"\n🔥 EMERGENCY TRAINING LOOP")
    best_mae = float('inf')
    
    for epoch in range(20):  # Short training
        print(f"\nEpoch {epoch+1}/20")
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        start_time = time.time()
        
        for sequences, targets in tqdm(train_loader, desc="Training"):
            # FORCE TO GPU
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(sequences)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
        
        epoch_time = time.time() - start_time
        avg_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs = model(sequences)
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        val_mae = mean_absolute_error(val_targets, val_preds)
        
        print(f"Loss: {avg_loss:.4f}, MAE: {val_mae:.2f} km/h, Time: {epoch_time:.1f}s")
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Save best
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), '/kaggle/working/emergency_speednet.pth')
            print(f"🏆 Best model saved! MAE: {best_mae:.2f}")
    
    print(f"\n🎉 EMERGENCY TRAINING COMPLETE!")
    print(f"Best MAE: {best_mae:.2f} km/h")
    print(f"Model saved: emergency_speednet.pth")

if __name__ == "__main__":
    main()