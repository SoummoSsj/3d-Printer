#!/usr/bin/env python3
"""
🚗 GPU + Anti-Overfitting SpeedNet WITH CHECKPOINTING
🛡️ Handles GPU issues AND 2-road overfitting AND Kaggle resets
============================================================
"""

import os
import sys
import time
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Force GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert torch.cuda.is_available(), "❌ CUDA not available! Enable GPU in Kaggle settings"

print("🚗 GPU + Anti-Overfitting SpeedNet WITH CHECKPOINTING")
print("🛡️ Handles GPU issues AND 2-road overfitting AND Kaggle resets")
print("=" * 60)
print(f"✅ GPU WORKING: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")

# Kaggle session management
KAGGLE_SESSION_START = datetime.now()
KAGGLE_MAX_HOURS = 11.5  # Leave buffer for cleanup

def get_time_remaining():
    """Get remaining time in Kaggle session"""
    elapsed = datetime.now() - KAGGLE_SESSION_START
    remaining = timedelta(hours=KAGGLE_MAX_HOURS) - elapsed
    return remaining.total_seconds() / 3600  # Return hours

def should_continue_training():
    """Check if we should continue training or save and exit"""
    remaining_hours = get_time_remaining()
    print(f"⏰ Kaggle session time remaining: {remaining_hours:.1f} hours")
    return remaining_hours > 0.5  # Stop with 30min buffer

# Safe pickle loading for dataset compatibility
def load_pickle_safe(file_path):
    """Load pickle with multiple encoding attempts"""
    encodings = ['latin1', 'bytes', 'ASCII', 'utf-8', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'rb') as f:
                if encoding == 'bytes':
                    data = pickle.load(f, encoding='bytes')
                else:
                    data = pickle.load(f, encoding=encoding)
            return convert_bytes_to_str(data)
        except Exception as e:
            continue
    
    # Final attempt with different pickle protocols
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, fix_imports=True, encoding='latin1')
        return convert_bytes_to_str(data)
    except Exception as e:
        raise RuntimeError(f"Could not load {file_path} with any encoding: {e}")

def convert_bytes_to_str(obj):
    """Recursively convert bytes to strings in nested structures"""
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except:
            return obj.decode('latin1')
    elif isinstance(obj, dict):
        return {convert_bytes_to_str(k): convert_bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_bytes_to_str(item) for item in obj)
    elif isinstance(obj, set):
        return {convert_bytes_to_str(item) for item in obj}
    return obj

# Simple augmentation without albumentations dependency
def simple_augment_frame(frame):
    """Simple augmentations without external dependencies"""
    if np.random.random() < 0.3:
        return frame  # No augmentation
    
    # Color jitter
    if np.random.random() < 0.5:
        frame = frame.astype(np.float32)
        # Brightness
        frame *= np.random.uniform(0.8, 1.2)
        # Contrast
        frame = (frame - frame.mean()) * np.random.uniform(0.8, 1.2) + frame.mean()
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # Add noise
    if np.random.random() < 0.3:
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Horizontal flip
    if np.random.random() < 0.3:
        frame = cv2.flip(frame, 1)
    
    # Rotation
    if np.random.random() < 0.3:
        angle = np.random.uniform(-5, 5)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h))
    
    # Blur
    if np.random.random() < 0.2:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
    
    return frame

class AntiOverfittingDataset(Dataset):
    def __init__(self, dataset_root, split='train', use_augmentation=True):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.sequence_length = 6
        self.image_size = (160, 160)
        
        self.samples = []
        if split == 'full':
            self._collect_samples()
        
        print(f"✅ {split} dataset: {len(self.samples)} samples")
        
        # Track session distribution for domain adaptation
        self.session_counts = {}
        for sample in self.samples:
            session = sample['session_id']
            self.session_counts[session] = self.session_counts.get(session, 0) + 1
        print(f"📊 Session distribution: {self.session_counts}")
    
    def _collect_samples(self):
        """Collect all valid samples from dataset"""
        print("📁 Loading full dataset...")
        
        for session_dir in sorted(self.dataset_root.iterdir()):
            if not session_dir.is_dir():
                continue
                
            session_name = session_dir.name
            print(f"📊 {session_name}: ", end="")
            
            gt_path = session_dir / "gt_data.pkl"
            video_path = session_dir / "video.avi"
            
            if not gt_path.exists() or not video_path.exists():
                print("❌ Missing files")
                continue
            
            try:
                # Load ground truth safely
                gt_data = load_pickle_safe(gt_path)
                
                # Extract session ID for domain adaptation
                session_id = session_name.split('_')[0]  # session0 or session1
                
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                valid_cars = [car for car in cars if car.get('valid', False) and 
                            len(car.get('intersections', [])) >= 2]
                
                print(f"{len(valid_cars)} cars")
                
                # Add samples with session tracking
                for car in valid_cars:
                    speed_kmh = car.get('speed', 0)
                    if 30 <= speed_kmh <= 150:  # Reasonable speed range
                        
                        intersections = car['intersections']
                        start_time = intersections[0]['videoTime']
                        end_time = intersections[-1]['videoTime']
                        
                        # Add temporal jitter for augmentation
                        if self.use_augmentation:
                            time_jitter = np.random.uniform(-0.2, 0.2)
                            start_time += time_jitter
                            end_time += time_jitter
                        
                        self.samples.append({
                            'video_path': str(video_path),
                            'session_id': session_id,  # Track session for domain adaptation
                            'speed_kmh': speed_kmh,
                            'start_time': start_time,
                            'end_time': end_time,
                            'fps': fps
                        })
                        
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract video frames
        frames = self._extract_frames(
            sample['video_path'], 
            sample['start_time'], 
            sample['end_time'], 
            sample['fps']
        )
        
        # Convert to tensor
        frames = torch.stack([
            torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0 
            for frame in frames
        ])
        
        # Session ID for domain adaptation (0 for session0, 1 for session1)
        session_domain = 0 if sample['session_id'] == 'session0' else 1
        
        return {
            'frames': frames,
            'speed': torch.tensor(sample['speed_kmh'], dtype=torch.float32),
            'session_id': sample['session_id'],
            'domain': torch.tensor(session_domain, dtype=torch.long)
        }
    
    def _extract_frames(self, video_path, start_time, end_time, fps):
        """Extract sequence of frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        try:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Calculate frame indices for sequence
            total_frames = end_frame - start_frame
            if total_frames < self.sequence_length:
                frame_indices = np.linspace(start_frame, end_frame, self.sequence_length).astype(int)
            else:
                frame_indices = np.linspace(start_frame, end_frame, self.sequence_length).astype(int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.resize(frame, self.image_size)
                    if self.use_augmentation:
                        frame = simple_augment_frame(frame)
                    frames.append(frame)
                else:
                    # Fallback: repeat last frame
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8))
            
            return frames
            
        finally:
            cap.release()

# Regularized model for anti-overfitting
class RegularizedSpeedNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # CNN backbone with heavy dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.5),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Temporal processing with dropout
        self.lstm = nn.LSTM(
            input_size=128 * 16,  # 128 channels * 4*4 spatial
            hidden_size=256,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # 2 sessions (domains)
        )
        
        # Speed regressor with heavy dropout
        self.speed_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, frames, alpha=1.0):
        batch_size, seq_len, channels, height, width = frames.shape
        
        # Process all frames through CNN
        frames_flat = frames.view(-1, channels, height, width)
        cnn_out = self.cnn(frames_flat)  # [batch*seq, 128, 4, 4]
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # [batch, seq, 128*16]
        
        # LSTM temporal processing
        lstm_out, _ = self.lstm(cnn_out)  # [batch, seq, 256]
        
        # Use mean pooling for stability (instead of just last timestep)
        features = lstm_out.mean(dim=1)  # [batch, 256]
        
        # Speed prediction
        speed = self.speed_regressor(features)
        
        # Domain prediction for adversarial training
        # Gradient reversal for domain adaptation
        from torch.autograd import Function
        
        class GradientReversalFunction(Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x.view_as(x)
            
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.alpha, None
        
        reversed_features = GradientReversalFunction.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        
        return speed.squeeze(-1), domain_pred

# Checkpointing functions
def save_checkpoint(epoch, split_idx, split_name, model, optimizer, scheduler, scaler,
                   train_losses, val_maes, val_rmses, best_mae, config):
    """Save comprehensive checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'split_idx': split_idx,
        'split_name': split_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_losses': train_losses,
        'val_maes': val_maes,
        'val_rmses': val_rmses,
        'best_mae': best_mae,
        'config': config,
        'timestamp': str(datetime.now()),
        'session_start_time': str(KAGGLE_SESSION_START),
        'pytorch_version': torch.__version__
    }
    
    # Save to multiple locations for safety
    checkpoint_path = '/kaggle/working/latest_checkpoint.pth'
    backup_path = f'/kaggle/working/checkpoint_epoch_{epoch}_split_{split_idx}.pth'
    
    try:
        torch.save(checkpoint, checkpoint_path, weights_only=False)
        torch.save(checkpoint, backup_path, weights_only=False)
        print(f"✅ Checkpoint saved: epoch {epoch}, split {split_idx}")
        return True
    except Exception as e:
        print(f"❌ Checkpoint save failed: {e}")
        return False

def load_checkpoint(checkpoint_path):
    """Load checkpoint safely"""
    if not os.path.exists(checkpoint_path):
        print(f"🆕 No checkpoint found at {checkpoint_path}")
        return None
    
    try:
        # Fix for PyTorch 2.6
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        except:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        print(f"✅ Checkpoint loaded from epoch {checkpoint.get('epoch', 0)}")
        print(f"   Split: {checkpoint.get('split_name', 'Unknown')}")
        print(f"   Best MAE: {checkpoint.get('best_mae', 'Unknown'):.2f}")
        
        return checkpoint
    except Exception as e:
        print(f"❌ Checkpoint load failed: {e}")
        return None

# Training with domain adaptation and checkpointing
def train_with_domain_adaptation_checkpointed(model, train_loader, val_loader, epochs=25, 
                                            split_idx=0, split_name="Split"):
    """Training with checkpointing and domain adaptation"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()
    
    # Try to resume from checkpoint
    checkpoint = load_checkpoint('/kaggle/working/latest_checkpoint.pth')
    start_epoch = 0
    train_losses = []
    val_maes = []
    val_rmses = []
    best_mae = float('inf')
    
    if checkpoint is not None and checkpoint.get('split_idx') == split_idx:
        print(f"🔄 Resuming training from checkpoint...")
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        val_maes = checkpoint.get('val_maes', [])
        val_rmses = checkpoint.get('val_rmses', [])
        best_mae = checkpoint.get('best_mae', float('inf'))
        print(f"✅ Resumed from epoch {start_epoch}, best MAE: {best_mae:.2f}")
    
    config = {
        'split_idx': split_idx,
        'split_name': split_name,
        'epochs': epochs,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'batch_size': 6,
        'sequence_length': 6,
        'image_size': [160, 160],
        'dropout_rate': 0.5
    }
    
    for epoch in range(start_epoch, epochs):
        # Check time remaining
        if not should_continue_training():
            print(f"⏰ Time limit approaching! Saving checkpoint and exiting...")
            save_checkpoint(epoch-1, split_idx, split_name, model, optimizer, scheduler, scaler,
                          train_losses, val_maes, val_rmses, best_mae, config)
            return best_mae, train_losses, val_maes
        
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        
        # Dynamic domain adaptation weight (cosine annealing)
        domain_weight = 0.1 * (1 + np.cos(np.pi * epoch / epochs)) / 2
        
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="🔥 Training")
        
        for batch in train_pbar:
            frames = batch['frames'].to(device)
            speeds = batch['speed'].to(device)
            domains = batch['domain'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                speed_pred, domain_pred = model(frames, alpha=domain_weight)
                
                # Multi-task loss
                speed_loss = F.mse_loss(speed_pred, speeds)
                domain_loss = F.cross_entropy(domain_pred, domains)
                
                total_loss = speed_loss + domain_weight * domain_loss
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            train_pbar.set_postfix({'loss': total_loss.item():.4f})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_sessions = []
        
        val_pbar = tqdm(val_loader, desc="📊 Validation")
        
        with torch.no_grad():
            for batch in val_pbar:
                frames = batch['frames'].to(device)
                speeds = batch['speed'].to(device)
                
                with autocast('cuda'):
                    speed_pred, _ = model(frames, alpha=0.0)  # No domain loss in validation
                
                val_predictions.extend(speed_pred.cpu().numpy())
                val_targets.extend(speeds.cpu().numpy())
                val_sessions.extend(batch['session_id'])
        
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # Overall metrics
        val_mae = np.mean(np.abs(val_predictions - val_targets))
        val_rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))
        
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        
        # Per-session analysis
        unique_sessions = list(set(val_sessions))
        for session in sorted(unique_sessions):
            session_mask = np.array([s == session for s in val_sessions])
            if np.any(session_mask):
                session_mae = np.mean(np.abs(val_predictions[session_mask] - val_targets[session_mask]))
                session_count = np.sum(session_mask)
                print(f"  📊 {session} MAE: {session_mae:.2f} km/h ({session_count} samples)")
        
        # Update scheduler
        scheduler.step()
        
        # Results
        print(f"📈 Overall Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f} km/h")
        print(f"  Domain weight: {domain_weight:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'mae': best_mae,
                'epoch': epoch,
                'split_name': split_name,
                'config': config
            }, f'/kaggle/working/best_model_split_{split_idx}.pth')
            print(f"  🏆 New best model saved! MAE: {best_mae:.2f} km/h")
        
        # Save checkpoint every epoch
        save_checkpoint(epoch, split_idx, split_name, model, optimizer, scheduler, scaler,
                       train_losses, val_maes, val_rmses, best_mae, config)
        
        # Clear cache
        torch.cuda.empty_cache()
    
    return best_mae, train_losses, val_maes

# Cross-session validation
def create_cross_session_splits(dataset):
    """Create cross-session validation splits"""
    
    # Group samples by session
    session0_samples = [s for s in dataset.samples if s['session_id'] == 'session0']
    session1_samples = [s for s in dataset.samples if s['session_id'] == 'session1']
    
    print(f"📊 Cross-session split:")
    print(f"  Session 0 samples: {len(session0_samples)}")
    print(f"  Session 1 samples: {len(session1_samples)}")
    
    splits = [
        (session0_samples, session1_samples, "Session0→Session1"),
        (session1_samples, session0_samples, "Session1→Session0"),
        (session0_samples + session1_samples[:len(session0_samples)], 
         session1_samples[len(session0_samples):], "Mixed")
    ]
    
    return splits

def main():
    """Main function with checkpointing support"""
    print("\n🚀 STARTING ANTI-OVERFITTING TRAINING WITH CHECKPOINTING")
    
    dataset_root = "/kaggle/input/brnocomp/brno_kaggle_subset/dataset"
    
    # Create dataset
    full_dataset = AntiOverfittingDataset(dataset_root, 'full', use_augmentation=True)
    
    # Create cross-session splits
    splits = create_cross_session_splits(full_dataset)
    
    results = {}
    
    # Check for existing progress
    checkpoint = load_checkpoint('/kaggle/working/latest_checkpoint.pth')
    start_split = 0
    
    if checkpoint is not None:
        start_split = checkpoint.get('split_idx', 0)
        print(f"🔄 Resuming from split {start_split}")
    
    # Test each split strategy
    for i, (train_samples, val_samples, split_name) in enumerate(splits[start_split:], start_split):
        print(f"\n🧪 Testing {split_name} ({i+1}/3)")
        print("=" * 60)
        
        # Create datasets
        train_dataset = AntiOverfittingDataset(dataset_root, 'train', use_augmentation=True)
        train_dataset.samples = train_samples
        
        val_dataset = AntiOverfittingDataset(dataset_root, 'val', use_augmentation=False)
        val_dataset.samples = val_samples
        
        print(f"📊 Split info:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=6, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=6, shuffle=False, num_workers=0, pin_memory=True
        )
        
        # Create fresh model for each split (unless resuming)
        model = RegularizedSpeedNet(dropout_rate=0.5).to(device)
        
        print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train with checkpointing
        best_mae, train_losses, val_maes = train_with_domain_adaptation_checkpointed(
            model, train_loader, val_loader, epochs=25, split_idx=i, split_name=split_name
        )
        
        results[split_name] = {
            'best_mae': best_mae,
            'train_losses': train_losses,
            'val_maes': val_maes
        }
        
        print(f"\n✅ {split_name} complete! Best MAE: {best_mae:.2f} km/h")
        
        # Save split-specific model
        torch.save(model.state_dict(), f'/kaggle/working/model_{split_name.replace("→", "_to_")}.pth')
        
        # Check if we need to stop due to time
        if not should_continue_training():
            print(f"⏰ Kaggle session time limit approaching!")
            print(f"📁 Download checkpoints and continue in new session")
            break
    
    # Summary
    print(f"\n🎉 TRAINING SESSION COMPLETE!")
    print("=" * 60)
    
    if results:
        print(f"📊 Results so far:")
        for split_name, result in results.items():
            print(f"  {split_name}: {result['best_mae']:.2f} km/h")
    
    # Save progress
    with open('/kaggle/working/training_progress.json', 'w') as f:
        json.dump({
            'completed_splits': list(results.keys()),
            'results': {k: {**v, 'train_losses': [float(x) for x in v['train_losses']], 
                                  'val_maes': [float(x) for x in v['val_maes']]} 
                       for k, v in results.items()},
            'session_end_time': str(datetime.now()),
            'remaining_time_hours': get_time_remaining()
        }, f, indent=2)
    
    print(f"\n📁 Files saved:")
    print(f"  • latest_checkpoint.pth (resume point)")
    print(f"  • training_progress.json (session summary)")
    for split_name in results.keys():
        print(f"  • model_{split_name.replace('→', '_to_')}.pth")
    
    print(f"\n⏰ Session info:")
    print(f"  Time remaining: {get_time_remaining():.1f} hours")
    print(f"  To continue: Download checkpoints → New session → Upload → Re-run")

if __name__ == "__main__":
    main()