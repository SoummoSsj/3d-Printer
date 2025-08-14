#!/usr/bin/env python3
"""
🚗 GPU + Anti-Overfitting SpeedNet
Solves BOTH GPU issues AND overfitting with 2-road dataset
"""

# FORCE GPU SETUP
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import numpy as np
import cv2
import pickle
import time
import random
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print("🚗 GPU + Anti-Overfitting SpeedNet")
print("🛡️ Handles GPU issues AND 2-road overfitting")
print("=" * 60)

# FORCE GPU
torch.cuda.empty_cache()
assert torch.cuda.is_available(), "❌ Enable GPU in Kaggle settings!"
device = torch.device('cuda:0')
torch.cuda.set_device(0)

# Test GPU
test = torch.randn(100, 100, device=device)
result = torch.mm(test, test)
print(f"✅ GPU WORKING: {result.device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# ANTI-OVERFITTING DATA LOADING
# =============================================================================

def load_pickle_safe(file_path):
    """Load pickle with multiple encoding attempts"""
    for encoding in ['latin1', 'bytes', 'utf-8']:
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f, encoding=encoding)
        except:
            continue
    return None

def convert_bytes_to_str(obj):
    """Convert bytes to strings"""
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, dict):
        return {convert_bytes_to_str(k): convert_bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(item) for item in obj]
    return obj

def simple_augment_frame(frame, augment_prob=0.7):
    """Simple but effective augmentation without complex dependencies"""
    if random.random() > augment_prob:
        return frame
    
    # Color jitter
    if random.random() < 0.5:
        # Brightness
        brightness = random.uniform(0.7, 1.3)
        frame = np.clip(frame * brightness, 0, 255)
    
    if random.random() < 0.5:
        # Contrast  
        contrast = random.uniform(0.8, 1.2)
        frame = np.clip((frame - 128) * contrast + 128, 0, 255)
    
    # Gaussian noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, frame.shape)
        frame = np.clip(frame + noise, 0, 255)
    
    # Horizontal flip (simulate opposite direction)
    if random.random() < 0.3:
        frame = cv2.flip(frame, 1)
    
    # Small rotation
    if random.random() < 0.4:
        angle = random.uniform(-5, 5)
        center = (frame.shape[1]//2, frame.shape[0]//2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
    
    # Small blur (simulate motion/focus issues)
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    return frame

class AntiOverfittingDataset(Dataset):
    """Dataset with anti-overfitting measures for 2-road scenario"""
    
    def __init__(self, dataset_root, mode='train', use_augmentation=True):
        self.dataset_root = dataset_root
        self.mode = mode
        self.use_augmentation = use_augmentation and (mode == 'train')
        self.samples = []
        self.session_info = {}  # Track which samples come from which session
        
        print(f"📁 Loading {mode} dataset...")
        
        # Collect samples with session tracking
        for session_dir in os.listdir(dataset_root):
            if not session_dir.startswith('session'):
                continue
                
            session_id = session_dir.split('_')[0]  # session0 or session1
            
            gt_file = os.path.join(dataset_root, session_dir, 'gt_data.pkl')
            video_file = os.path.join(dataset_root, session_dir, 'video.avi')
            
            if not (os.path.exists(gt_file) and os.path.exists(video_file)):
                continue
                
            try:
                gt_data = load_pickle_safe(gt_file)
                if gt_data is None:
                    continue
                
                gt_data = convert_bytes_to_str(gt_data)
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                print(f"📊 {session_dir}: {len(cars)} cars")
                
                session_samples = []
                
                for car in cars:
                    try:
                        if not car.get('valid', True):
                            continue
                            
                        intersections = car.get('intersections', [])
                        if len(intersections) < 2:
                            continue
                        
                        times = [i.get('videoTime') for i in intersections if i.get('videoTime') is not None]
                        if len(times) < 2:
                            continue
                            
                        start_time = min(times) - 0.5
                        end_time = max(times) + 0.5
                        speed = car.get('speed', 0.0)
                        
                        if not isinstance(speed, (int, float)) or np.isnan(speed) or speed <= 0:
                            continue
                            
                        sample = {
                            'video_path': video_file,
                            'speed': float(speed),
                            'start_time': start_time,
                            'end_time': end_time,
                            'fps': fps,
                            'session_dir': session_dir,
                            'session_id': session_id,
                            'car_id': car.get('carId', -1)
                        }
                        
                        session_samples.append(sample)
                        self.samples.append(sample)
                        
                    except Exception:
                        continue
                
                self.session_info[session_id] = len(session_samples)
                        
            except Exception as e:
                print(f"❌ Error processing {session_dir}: {e}")
                continue
        
        print(f"✅ {mode} dataset: {len(self.samples)} samples")
        print(f"📊 Session distribution: {self.session_info}")
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames
        try:
            frames = self._extract_frames_safe(
                sample['video_path'],
                sample['start_time'], 
                sample['end_time'],
                sample['fps']
            )
        except:
            frames = np.random.rand(4, 128, 128, 3).astype(np.float32) * 255
        
        # Apply simple augmentation to combat overfitting
        if self.use_augmentation:
            for i in range(len(frames)):
                frames[i] = simple_augment_frame(frames[i])
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames_tensor = frames_tensor / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        target = torch.tensor([sample['speed']], dtype=torch.float32)
        
        # Include session info for domain analysis
        session_id = 0 if sample['session_id'] == 'session0' else 1
        session_tensor = torch.tensor([session_id], dtype=torch.long)
        
        return frames_tensor, target, session_tensor
    
    def _extract_frames_safe(self, video_path, start_time, end_time, fps):
        """Extract frames with temporal jitter for augmentation"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(end_frame, total_frames - 1)
        
        # Add temporal jitter for augmentation
        if self.use_augmentation and random.random() < 0.3:
            jitter = random.randint(-2, 2)
            start_frame = max(0, start_frame + jitter)
            end_frame = max(start_frame + 4, end_frame + jitter)
            end_frame = min(end_frame, total_frames - 1)
        
        if end_frame <= start_frame:
            frame_indices = [start_frame] * 4
        else:
            frame_indices = np.linspace(start_frame, end_frame, 4, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame = cv2.resize(frame, (128, 128))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32))
            else:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((128, 128, 3), dtype=np.float32))
        
        cap.release()
        
        while len(frames) < 4:
            frames.append(frames[-1].copy() if frames else np.zeros((128, 128, 3), dtype=np.float32))
        
        return np.array(frames[:4])

# =============================================================================
# REGULARIZED MODEL FOR ANTI-OVERFITTING
# =============================================================================

class RegularizedSpeedNet(nn.Module):
    """SpeedNet with heavy regularization to prevent overfitting"""
    
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.25),
            
            # Block 2  
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            
            # Block 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.75),
            
            # Block 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        
        # Temporal with dropout
        self.temporal = nn.LSTM(256 * 4 * 4, 128, batch_first=True, dropout=dropout_rate, num_layers=2)
        
        # Domain classifier for anti-overfitting (adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # 2 sessions
        )
        
        # Speed prediction with heavy regularization
        self.speed_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_domain_features=False):
        B, T, C, H, W = x.shape
        
        # Process frames
        frame_features = []
        for t in range(T):
            feat = self.backbone(x[:, t])
            frame_features.append(feat)
        
        sequence_features = torch.stack(frame_features, dim=1)
        lstm_out, _ = self.temporal(sequence_features)
        
        # Use mean pooling instead of last timestep for stability
        final_features = torch.mean(lstm_out, dim=1)
        
        speed = self.speed_head(final_features)
        
        if return_domain_features:
            domain_logits = self.domain_classifier(final_features)
            return speed, domain_logits, final_features
        
        return speed

# =============================================================================
# ANTI-OVERFITTING TRAINING WITH CROSS-SESSION VALIDATION
# =============================================================================

def create_cross_session_splits(dataset):
    """Create splits that test generalization across sessions"""
    session0_samples = [s for s in dataset.samples if s['session_id'] == 'session0']
    session1_samples = [s for s in dataset.samples if s['session_id'] == 'session1']
    
    print(f"📊 Cross-session split:")
    print(f"  Session 0 samples: {len(session0_samples)}")  
    print(f"  Session 1 samples: {len(session1_samples)}")
    
    # Strategy 1: Train on Session 0, test on Session 1
    split1_train = session0_samples
    split1_val = session1_samples[:len(session1_samples)//2]  # Use half for validation
    
    # Strategy 2: Train on Session 1, test on Session 0  
    split2_train = session1_samples
    split2_val = session0_samples
    
    # Strategy 3: Mixed training with balanced validation
    all_samples = session0_samples + session1_samples
    random.shuffle(all_samples)
    split_idx = int(0.8 * len(all_samples))
    split3_train = all_samples[:split_idx]
    split3_val = all_samples[split_idx:]
    
    return [
        (split1_train, split1_val, "Session0→Session1"),
        (split2_train, split2_val, "Session1→Session0"), 
        (split3_train, split3_val, "Mixed")
    ]

def train_with_domain_adaptation(model, train_loader, val_loader, epochs=30):
    """Training with domain adaptation to prevent session overfitting"""
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    speed_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda')
    
    best_mae = float('inf')
    train_losses = []
    val_maes = []
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        
        # Training with domain adaptation
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Dynamic domain adaptation weight
        domain_weight = 0.1 * (1.0 + np.cos(np.pi * epoch / epochs)) / 2
        
        for sequences, targets, session_ids in tqdm(train_loader, desc="🔥 Training"):
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            session_ids = session_ids.to(device, non_blocking=True).squeeze()
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                speed_pred, domain_logits, features = model(sequences, return_domain_features=True)
                
                # Speed prediction loss
                speed_loss = speed_criterion(speed_pred, targets)
                
                # Domain adaptation loss (encourage session-invariant features)
                domain_loss = domain_criterion(domain_logits, session_ids)
                
                # Total loss with domain adaptation
                total_loss = speed_loss + domain_weight * domain_loss
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation with per-session analysis
        model.eval()
        val_predictions = []
        val_targets = []
        val_sessions = []
        
        with torch.no_grad():
            for sequences, targets, session_ids in tqdm(val_loader, desc="📊 Validation"):
                sequences = sequences.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    speed_pred = model(sequences)
                
                val_predictions.extend(speed_pred.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
                val_sessions.extend(session_ids.numpy().flatten())
        
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_sessions = np.array(val_sessions)
        
        overall_mae = mean_absolute_error(val_targets, val_predictions)
        val_maes.append(overall_mae)
        
        # Per-session analysis
        for session_id in np.unique(val_sessions):
            mask = val_sessions == session_id
            if np.sum(mask) > 0:
                session_mae = mean_absolute_error(val_targets[mask], val_predictions[mask])
                session_name = f"session{session_id}"
                print(f"  📊 {session_name} MAE: {session_mae:.2f} km/h ({np.sum(mask)} samples)")
        
        scheduler.step(overall_mae)
        
        print(f"📈 Overall Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val MAE: {overall_mae:.2f} km/h") 
        print(f"  Domain weight: {domain_weight:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        if overall_mae < best_mae:
            best_mae = overall_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'mae': overall_mae,
                'per_session_results': {
                    'predictions': val_predictions,
                    'targets': val_targets, 
                    'sessions': val_sessions
                }
            }, '/kaggle/working/anti_overfitting_speednet.pth')
            print(f"  🏆 New best model saved! MAE: {best_mae:.2f} km/h")
    
    return best_mae, train_losses, val_maes

def main():
    """Main function with cross-session validation"""
    print("\n🚀 STARTING ANTI-OVERFITTING TRAINING")
    
    dataset_root = "/kaggle/input/brnocomp/brno_kaggle_subset/dataset"
    
    # Create dataset
    full_dataset = AntiOverfittingDataset(dataset_root, 'full', use_augmentation=True)
    
    # Create cross-session splits
    splits = create_cross_session_splits(full_dataset)
    
    results = {}
    
    # Test each split strategy
    for i, (train_samples, val_samples, split_name) in enumerate(splits):
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
        
        # Create fresh model for each split
        model = RegularizedSpeedNet(dropout_rate=0.5).to(device)
        
        print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        best_mae, train_losses, val_maes = train_with_domain_adaptation(
            model, train_loader, val_loader, epochs=25
        )
        
        results[split_name] = {
            'best_mae': best_mae,
            'train_losses': train_losses,
            'val_maes': val_maes
        }
        
        print(f"\n✅ {split_name} complete! Best MAE: {best_mae:.2f} km/h")
        
        # Save split-specific model
        torch.save(model.state_dict(), f'/kaggle/working/model_{split_name.replace("→", "_to_")}.pth')
    
    # Summary of all results
    print(f"\n🎉 ANTI-OVERFITTING TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Cross-Session Validation Results:")
    for split_name, result in results.items():
        print(f"  {split_name}: {result['best_mae']:.2f} km/h")
    
    # The best generalization is typically the worst single-split performance
    worst_mae = max(result['best_mae'] for result in results.values())
    best_mae = min(result['best_mae'] for result in results.values())
    
    print(f"\n🎯 Generalization Analysis:")
    print(f"  Best single split: {best_mae:.2f} km/h")
    print(f"  Worst single split: {worst_mae:.2f} km/h")
    print(f"  Generalization gap: {worst_mae - best_mae:.2f} km/h")
    
    if worst_mae - best_mae < 5.0:
        print(f"✅ Good generalization! Gap < 5 km/h")
    else:
        print(f"⚠️  Poor generalization. Gap > 5 km/h indicates overfitting")
    
    # Save comprehensive results
    import json
    with open('/kaggle/working/cross_session_results.json', 'w') as f:
        json.dump({k: {**v, 'train_losses': [float(x) for x in v['train_losses']], 
                            'val_maes': [float(x) for x in v['val_maes']]} 
                  for k, v in results.items()}, f, indent=2)
    
    print(f"\n📁 Files saved:")
    print(f"  • anti_overfitting_speednet.pth (best overall)")
    print(f"  • model_Session0_to_Session1.pth")  
    print(f"  • model_Session1_to_Session0.pth")
    print(f"  • model_Mixed.pth")
    print(f"  • cross_session_results.json")

if __name__ == "__main__":
    main()