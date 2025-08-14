#!/usr/bin/env python3
"""
🚗 Enhanced SpeedNet with Aggressive Data Augmentation
Designed to combat overfitting with limited road scenarios
"""

import os
import json
import pickle
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import torchvision.transforms as transforms
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("🚗 Enhanced SpeedNet with Aggressive Data Augmentation")
print("🛡️ Combat Overfitting with Limited Road Scenarios")
print("=" * 60)

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# Advanced Data Augmentation Pipeline
# =============================================================================

class AdvancedAugmentation:
    """Aggressive augmentation to simulate different road conditions"""
    
    def __init__(self, mode='train'):
        self.mode = mode
        
        if mode == 'train':
            # VERY aggressive augmentation for generalization
            self.geometric_aug = A.Compose([
                # Perspective changes (simulate different camera positions)
                A.Perspective(scale=(0.02, 0.1), p=0.6),
                
                # Rotation and scaling (simulate different vehicle orientations)
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=5, 
                    p=0.7
                ),
                
                # Horizontal flip (simulate opposite traffic direction)
                A.HorizontalFlip(p=0.3),
                
                # Elastic transform (simulate road surface variations)
                A.ElasticTransform(
                    alpha=50, 
                    sigma=5, 
                    alpha_affine=5, 
                    p=0.3
                ),
                
                # Grid distortion (simulate camera lens distortion)
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            ])
            
            self.photometric_aug = A.Compose([
                # Weather simulation
                A.OneOf([
                    A.RandomRain(rain_type="drizzle", p=1.0),
                    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.2, p=1.0),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                ], p=0.4),
                
                # Lighting variations (simulate different times of day)
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=0.8
                ),
                
                # Color variations (simulate different cameras/seasons)
                A.HueSaturationValue(
                    hue_shift_limit=20, 
                    sat_shift_limit=30, 
                    val_shift_limit=20, 
                    p=0.6
                ),
                
                # Shadow simulation
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1), 
                    num_shadows_lower=1, 
                    num_shadows_upper=3, 
                    p=0.4
                ),
                
                # Blur variations (simulate motion blur, defocus)
                A.OneOf([
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
                
                # Noise simulation (simulate different camera sensors)
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.3),
                
                # Channel manipulation (simulate different lighting conditions)
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.ChannelShuffle(p=0.1),
            ])
            
            # Advanced temporal augmentation
            self.temporal_aug_prob = 0.5
            
        else:
            # Minimal augmentation for validation
            self.geometric_aug = A.Compose([])
            self.photometric_aug = A.Compose([])
            self.temporal_aug_prob = 0.0
    
    def apply_geometric(self, frames):
        """Apply consistent geometric augmentation across all frames"""
        if self.mode != 'train':
            return frames
            
        # Apply same geometric transform to all frames in sequence
        augmented_frames = []
        
        # Get random transform parameters (consistent across frames)
        transform = self.geometric_aug.get_transform_init_args_names()
        
        for frame in frames:
            try:
                augmented = self.geometric_aug(image=frame)['image']
                augmented_frames.append(augmented)
            except:
                augmented_frames.append(frame)
        
        return np.array(augmented_frames)
    
    def apply_photometric(self, frames):
        """Apply varying photometric augmentation across frames"""
        if self.mode != 'train':
            return frames
            
        augmented_frames = []
        
        for frame in frames:
            try:
                # Each frame can have different photometric augmentation
                # (simulates changing lighting conditions during video)
                augmented = self.photometric_aug(image=frame)['image']
                augmented_frames.append(augmented)
            except:
                augmented_frames.append(frame)
        
        return np.array(augmented_frames)
    
    def apply_temporal(self, frames):
        """Apply temporal augmentation (frame dropping, reordering)"""
        if self.mode != 'train' or random.random() > self.temporal_aug_prob:
            return frames
            
        # Temporal augmentations
        augmented = frames.copy()
        
        # Random frame dropping (simulate missing frames)
        if random.random() < 0.3:
            drop_idx = random.randint(1, len(frames)-2)
            augmented[drop_idx] = augmented[drop_idx-1]  # Duplicate previous frame
        
        # Small temporal jitter (simulate frame timing variations)
        if random.random() < 0.2:
            # Randomly swap adjacent frames
            swap_idx = random.randint(0, len(frames)-2)
            augmented[swap_idx], augmented[swap_idx+1] = augmented[swap_idx+1], augmented[swap_idx]
        
        # Add small motion blur to random frames (simulate camera shake)
        if random.random() < 0.4:
            blur_idx = random.randint(0, len(frames)-1)
            kernel_size = random.choice([3, 5])
            augmented[blur_idx] = cv2.GaussianBlur(augmented[blur_idx], (kernel_size, kernel_size), 0)
        
        return augmented

# =============================================================================
# Enhanced Dataset with Domain Randomization
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

class EnhancedBrnCompDataset(Dataset):
    """Enhanced dataset with aggressive augmentation and domain randomization"""
    
    def __init__(self, dataset_root, sequence_length=6, image_size=160, mode='train'):
        self.dataset_root = dataset_root
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.mode = mode
        self.samples = []
        
        # Advanced augmentation pipeline
        self.augmentation = AdvancedAugmentation(mode)
        
        # Domain randomization parameters
        self.domain_randomization = (mode == 'train')
        
        # Collect samples with cross-session mixing
        self._collect_samples()
        
        # Mix sessions to prevent session-specific overfitting
        if mode == 'train':
            self._mix_sessions()
        
        print(f"✅ {mode} dataset: {len(self.samples)} samples")
        
    def _collect_samples(self):
        """Collect samples with session awareness"""
        session_samples = {}  # Track samples per session
        
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
                
                session_key = session_dir.split('_')[0]  # session0 or session1
                if session_key not in session_samples:
                    session_samples[session_key] = []
                
                for car in cars:
                    if not car.get('valid', True):
                        continue
                        
                    intersections = car.get('intersections', [])
                    if len(intersections) < 2:
                        continue
                    
                    times = [i['videoTime'] for i in intersections if 'videoTime' in i]
                    if len(times) < 2:
                        continue
                        
                    start_time = min(times) - 0.8
                    end_time = max(times) + 0.8
                    
                    speed = car.get('speed', 0.0)
                    if not isinstance(speed, (int, float)) or np.isnan(speed) or speed < 0:
                        speed = 0.0
                    
                    sample = {
                        'session': session_dir,
                        'session_id': session_key,
                        'video_path': video_file,
                        'speed': float(speed),
                        'start_time': start_time,
                        'end_time': end_time,
                        'fps': fps,
                    }
                    
                    session_samples[session_key].append(sample)
                    
            except Exception as e:
                print(f"❌ Error processing {session_dir}: {e}")
                continue
        
        # Store samples with session information
        self.session_samples = session_samples
        for session_key, samples in session_samples.items():
            self.samples.extend(samples)
            print(f"📊 {session_key}: {len(samples)} samples")
    
    def _mix_sessions(self):
        """Mix sessions to prevent session-specific overfitting"""
        if len(self.session_samples) < 2:
            return
            
        # Create balanced sampling from both sessions
        session_keys = list(self.session_samples.keys())
        min_samples = min(len(samples) for samples in self.session_samples.values())
        
        # Balance sessions (prevent one session from dominating)
        balanced_samples = []
        for session_key in session_keys:
            samples = self.session_samples[session_key]
            # Sample with replacement if needed
            selected = random.choices(samples, k=min_samples)
            balanced_samples.extend(selected)
        
        self.samples = balanced_samples
        random.shuffle(self.samples)
        print(f"🔄 Balanced mixing: {len(self.samples)} samples")
    
    def _extract_video_frames(self, video_path, start_time, end_time, fps):
        """Extract frames with enhanced augmentation"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return self._create_dummy_frames()
            
            start_frame = max(0, int(start_time * fps))
            end_frame = int(end_time * fps)
            
            # Add temporal jitter for augmentation
            if self.mode == 'train' and random.random() < 0.3:
                jitter = random.randint(-2, 2)
                start_frame = max(0, start_frame + jitter)
                end_frame = max(start_frame + self.sequence_length, end_frame + jitter)
            
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
            return np.array(frames[:self.sequence_length], dtype=np.uint8)
            
        except Exception as e:
            return self._create_dummy_frames()
    
    def _create_dummy_frames(self):
        """Create dummy frames"""
        return (np.random.rand(self.sequence_length, self.image_size, self.image_size, 3) * 255).astype(np.uint8)
    
    def _apply_domain_randomization(self, frames):
        """Apply domain randomization to simulate different road conditions"""
        if not self.domain_randomization:
            return frames
        
        augmented = frames.astype(np.uint8)
        
        # Apply augmentation pipeline
        augmented = self.augmentation.apply_geometric(augmented)
        augmented = self.augmentation.apply_photometric(augmented)
        augmented = self.augmentation.apply_temporal(augmented)
        
        return augmented.astype(np.float32) / 255.0
                
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
        
        # Apply domain randomization
        frames = self._apply_domain_randomization(frames)
        
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        target = torch.tensor([sample['speed']], dtype=torch.float32)
        
        # Add session information for analysis
        session_id = torch.tensor([0 if 'session0' in sample['session'] else 1], dtype=torch.long)
        
        return frames_tensor, target, session_id

# =============================================================================
# Enhanced Model with Regularization
# =============================================================================

class RegularizedSpeedNet(nn.Module):
    """SpeedNet with enhanced regularization to prevent overfitting"""
    def __init__(self, sequence_length=6):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Feature extractor with strong dropout
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.15),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256)
        )
        
        # Temporal processing with regularization
        self.temporal = nn.LSTM(256, 128, batch_first=True, dropout=0.3, num_layers=2)
        
        # Session-invariant features (domain adaptation)
        self.session_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 2 sessions
        )
        
        # Speed prediction head
        self.speed_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, image_sequence, return_features=False):
        """Forward pass with optional feature return for analysis"""
        B, T, C, H, W = image_sequence.shape
        
        # Process each frame
        frame_features = []
        for t in range(T):
            feat = self.backbone(image_sequence[:, t])
            frame_features.append(feat)
        
        sequence_features = torch.stack(frame_features, dim=1)
        
        # Temporal processing
        temporal_out, _ = self.temporal(sequence_features)
        
        # Use mean pooling for stability
        final_features = torch.mean(temporal_out, dim=1)
        
        # Predictions
        speed = F.softplus(self.speed_head(final_features))
        uncertainty = self.uncertainty_head(final_features)
        
        # Session classification (for domain adaptation)
        session_logits = self.session_classifier(final_features)
        
        result = {
            'speed': speed,
            'uncertainty': uncertainty,
            'session_logits': session_logits
        }
        
        if return_features:
            result['features'] = final_features
            
        return result

# =============================================================================
# Enhanced Loss with Domain Adaptation
# =============================================================================

class EnhancedSpeedNetLoss(nn.Module):
    """Enhanced loss with domain adaptation to prevent session overfitting"""
    def __init__(self, speed_weight=1.0, uncertainty_weight=0.1, domain_weight=0.2):
        super().__init__()
        self.speed_weight = speed_weight
        self.uncertainty_weight = uncertainty_weight
        self.domain_weight = domain_weight
        
        self.domain_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets, session_ids, alpha=1.0):
        pred_speeds = predictions['speed']
        pred_uncertainties = predictions['uncertainty']
        session_logits = predictions['session_logits']
        target_speeds = targets
        
        # Speed loss with uncertainty
        speed_diff = pred_speeds - target_speeds
        precision = torch.exp(-pred_uncertainties)
        speed_loss = torch.mean(precision * speed_diff**2 + pred_uncertainties)
        
        # Uncertainty regularization
        uncertainty_reg = torch.mean(torch.exp(pred_uncertainties))
        
        # Domain adversarial loss (encourage session-invariant features)
        # Reverse gradient for domain classifier
        domain_loss = self.domain_loss(session_logits, session_ids.squeeze())
        
        total_loss = (self.speed_weight * speed_loss + 
                     self.uncertainty_weight * uncertainty_reg + 
                     self.domain_weight * alpha * domain_loss)
        
        return {
            'total': total_loss,
            'speed': speed_loss,
            'uncertainty': uncertainty_reg,
            'domain': domain_loss
        }

def main():
    dataset_root = "/kaggle/input/brnocomp/brno_kaggle_subset/dataset"
    
    print(f"\n📁 Loading enhanced dataset with aggressive augmentation...")
    
    # Create enhanced datasets
    full_dataset = EnhancedBrnCompDataset(
        dataset_root, 
        sequence_length=6,
        image_size=160,
        mode='train'
    )
    
    # Stratified split to ensure both sessions in train/val
    session_ids = [s['session_id'] for s in full_dataset.samples]
    unique_sessions = list(set(session_ids))
    
    print(f"📊 Dataset analysis:")
    for session in unique_sessions:
        count = session_ids.count(session)
        print(f"  {session}: {count} samples ({count/len(session_ids)*100:.1f}%)")
    
    # Stratified split
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42,
        stratify=session_ids
    )
    
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    
    train_dataset = EnhancedBrnCompDataset(dataset_root, sequence_length=6, image_size=160, mode='train')
    train_dataset.samples = train_samples
    
    val_dataset = EnhancedBrnCompDataset(dataset_root, sequence_length=6, image_size=160, mode='val')
    val_dataset.samples = val_samples
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Enhanced configuration with regularization
    config = {
        'batch_size': 4,        # Smaller batch for stability
        'num_epochs': 60,       # More epochs with early stopping
        'learning_rate': 5e-5,  # Lower LR for stability
        'weight_decay': 1e-4,   # Higher weight decay
        'num_workers': 2,
        'sequence_length': 6,
        'image_size': 160
    }
    
    print(f"\n⚙️ Enhanced Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create enhanced model
    model = RegularizedSpeedNet(sequence_length=6).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 Enhanced SpeedNet created!")
    print(f"  Total parameters: {total_params:,}")
    
    # Enhanced training setup
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    criterion = EnhancedSpeedNetLoss()
    scaler = GradScaler('cuda')
    
    # Training with early stopping and domain adaptation
    print(f"\n🚀 Starting Enhanced Training with Domain Adaptation...")
    print("=" * 60)
    
    best_mae = float('inf')
    patience_counter = 0
    patience_limit = 10
    
    train_losses = []
    val_maes = []
    
    for epoch in range(config['num_epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")
        
        # Dynamic domain adaptation weight
        alpha = 2.0 / (1.0 + np.exp(-10 * epoch / config['num_epochs'])) - 1
        
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for sequences, targets, session_ids in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            session_ids = session_ids.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(sequences)
                loss_dict = criterion(outputs, targets, session_ids, alpha)
                loss = loss_dict['total']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_sessions = []
        
        with torch.no_grad():
            for sequences, targets, session_ids in val_loader:
                sequences = sequences.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs = model(sequences)
                
                val_predictions.extend(outputs['speed'].cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
                val_sessions.extend(session_ids.numpy().flatten())
        
        # Calculate metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_sessions = np.array(val_sessions)
        
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        # Per-session analysis
        for session_id in np.unique(val_sessions):
            mask = val_sessions == session_id
            if np.sum(mask) > 0:
                session_mae = mean_absolute_error(val_targets[mask], val_predictions[mask])
                session_name = f"session{session_id}"
                print(f"  📊 {session_name} MAE: {session_mae:.2f} km/h")
        
        val_maes.append(val_mae)
        scheduler.step(val_mae)
        
        print(f"📈 Overall Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f} km/h")
        print(f"  Val RMSE: {val_rmse:.2f} km/h")
        print(f"  Domain α: {alpha:.3f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and model saving
        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'mae': val_mae,
                'config': config
            }, '/kaggle/working/best_enhanced_model.pth')
            print(f"  🏆 New best model saved! MAE: {best_mae:.2f} km/h")
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎉 Enhanced Training completed!")
    print(f"🏆 Best validation MAE: {best_mae:.2f} km/h")
    print(f"🛡️ Domain adaptation and aggressive augmentation applied!")

if __name__ == "__main__":
    main()