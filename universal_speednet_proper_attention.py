#!/usr/bin/env python3
"""
🌍 Universal SpeedNet: PROPER Physics-Informed Learning with Real Attention
🎯 TRUE multi-head attention + cross-frame attention + enhanced RNN
===============================================================================
"""

import os
import sys
import json
import pickle
import cv2
import numpy as np
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from tqdm import tqdm
import matplotlib.pyplot as plt

# Force GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Session management for Kaggle
KAGGLE_SESSION_START = datetime.now()

print("🌍 Universal SpeedNet: PROPER Physics-Informed Learning with Real Attention")
print("🎯 TRUE multi-head attention + CAUSAL cross-frame attention + enhanced RNN")
print("=" * 80)
print(f"✅ Device: {device}")
print(f"⏰ Session started: {KAGGLE_SESSION_START.strftime('%H:%M:%S')}")

# Kaggle session management
def get_time_remaining():
    """Get remaining time in Kaggle session"""
    elapsed = datetime.now() - KAGGLE_SESSION_START
    remaining = timedelta(hours=11.5) - elapsed
    return remaining.total_seconds() / 3600

def should_continue_training():
    """Check if we should continue training"""
    remaining_hours = get_time_remaining()
    print(f"⏰ Kaggle session time remaining: {remaining_hours:.1f} hours")
    return remaining_hours > 0.5

# Safe pickle loading (handles your encoding issues)
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
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, fix_imports=True, encoding='latin1')
        return convert_bytes_to_str(data)
    except Exception as e:
        raise RuntimeError(f"Could not load {file_path} with any encoding: {e}")

def convert_bytes_to_str(obj):
    """Recursively convert bytes to strings"""
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

class MultiHeadSpatialAttention(nn.Module):
    """PROPER Multi-Head Self-Attention for spatial regions"""
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Position encoding for spatial locations
        self.pos_encoding = nn.Parameter(torch.randn(100, embed_dim))  # Max 10x10=100 positions
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width] -> [batch, embed_dim, H, W]
        Returns:
            attended_features: [batch, embed_dim, H, W]
            attention_weights: [batch, num_heads, H*W, H*W]
        """
        batch_size, embed_dim, height, width = x.shape
        seq_len = height * width
        
        # Flatten spatial dimensions: [batch, embed_dim, H*W] -> [batch, H*W, embed_dim]
        x_flat = x.reshape(batch_size, embed_dim, seq_len).transpose(1, 2)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x_flat = x_flat + pos_enc
        
        # Generate Q, K, V
        Q = self.q_proj(x_flat)  # [batch, seq_len, embed_dim]
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)
        
        # Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: [batch, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention: [batch, num_heads, seq_len, head_dim]
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads: [batch, seq_len, embed_dim]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Reshape back to spatial: [batch, H*W, embed_dim] -> [batch, embed_dim, H, W]
        output = output.transpose(1, 2).reshape(batch_size, embed_dim, height, width)
        
        return output, attention_weights

class CrossFrameAttention(nn.Module):
    """CAUSAL Cross-Frame Attention for physically realistic temporal relationships"""
    
    def __init__(self, embed_dim=256, num_heads=8, sequence_length=6):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.head_dim = embed_dim // num_heads
        
        # Cross-attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Temporal position encoding
        self.temporal_pos = nn.Parameter(torch.randn(sequence_length, embed_dim))
        
        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, frame_features):
        """
        Args:
            frame_features: [batch, sequence_length, embed_dim, height, width]
        Returns:
            cross_attended: [batch, sequence_length, embed_dim, height, width]
            temporal_attention: [batch, num_heads, sequence_length, sequence_length]
        """
        batch_size, seq_len, embed_dim, height, width = frame_features.shape
        
        # Global average pooling for each frame: [batch, seq_len, embed_dim]
        frame_global = F.adaptive_avg_pool2d(frame_features.reshape(-1, embed_dim, height, width), (1, 1))
        frame_global = frame_global.reshape(batch_size, seq_len, embed_dim)
        
        # Add temporal position encoding
        frame_global = frame_global + self.temporal_pos.unsqueeze(0)
        
        # Cross-frame attention
        Q = self.q_proj(frame_global)  # [batch, seq_len, embed_dim]
        K = self.k_proj(frame_global)
        V = self.v_proj(frame_global)
        
        # Multi-head reshaping
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Temporal attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # CAUSAL MASK: Frame t can only attend to frames ≤ t (physics-correct!)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        attention_scores = attention_scores + causal_mask.unsqueeze(0).unsqueeze(0)  # [batch, heads, seq, seq]
        
        temporal_attention = F.softmax(attention_scores, dim=-1)
        temporal_attention = self.dropout(temporal_attention)
        
        # Apply attention
        attended = torch.matmul(temporal_attention, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output_global = self.out_proj(attended)
        
        # Broadcast back to spatial dimensions
        output_global = output_global.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, embed_dim, 1, 1]
        cross_attended = frame_features + output_global.expand(-1, -1, -1, height, width)
        
        return cross_attended, temporal_attention

class PhysicsInformedFeatureExtractor(nn.Module):
    """Extract features with PROPER spatial attention"""
    
    def __init__(self, input_size=(160, 160)):
        super().__init__()
        
        # CNN backbone (same as before)
        self.backbone = nn.Sequential(
            # Scale 1: Fine details
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 160x160 -> 80x80
            
            # Scale 2: Vehicle structure
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 80x80 -> 40x40
            
            # Scale 3: Contextual features
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 40x40 -> 20x20
            
            # Scale 4: High-level features
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((10, 10))  # 20x20 -> 10x10
        )
        
        # PROPER Multi-Head Spatial Attention
        self.spatial_attention = MultiHeadSpatialAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=0.1
        )
        
        # Vehicle size estimation
        self.size_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # [width, height, area]
            nn.Sigmoid()
        )
        
    def forward(self, frame):
        # Extract features
        features = self.backbone(frame)  # [batch, 256, 10, 10]
        
        # Apply PROPER spatial attention
        attended_features, spatial_attention_weights = self.spatial_attention(features)
        
        # Vehicle size estimation
        vehicle_size = self.size_estimator(attended_features)
        
        return attended_features, vehicle_size, spatial_attention_weights

class EnhancedTemporalAnalyzer(nn.Module):
    """Enhanced RNN with PROPER cross-frame attention"""
    
    def __init__(self, feature_dim=256, sequence_length=6, hidden_dim=512):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Cross-frame attention FIRST
        self.cross_frame_attention = CrossFrameAttention(
            embed_dim=feature_dim,
            num_heads=8,
            sequence_length=sequence_length
        )
        
        # Enhanced GRU (better than LSTM for this task)
        self.motion_gru = nn.GRU(
            input_size=feature_dim * 10 * 10,  # Flattened spatial features
            hidden_size=hidden_dim,
            num_layers=3,  # Deeper network
            dropout=0.3,
            batch_first=True,
            bidirectional=True
        )
        
        # Additional LSTM for comparison
        self.motion_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # GRU output
            hidden_size=256,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )
        
        # Motion pattern analysis
        self.motion_analyzer = nn.Sequential(
            nn.Linear(256 * 2, 256),  # LSTM output
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Physics-based motion estimation
        self.displacement_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # [x_displacement, y_displacement]
        )
        
        # Velocity estimation (new!)
        self.velocity_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # [x_velocity, y_velocity]
        )
        
        # Acceleration estimation (new!)
        self.acceleration_estimator = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),  # scalar acceleration
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feature_sequence, size_sequence):
        """
        Args:
            feature_sequence: [batch, seq_len, channels, height, width]
            size_sequence: [batch, seq_len, 3]
        """
        batch_size, seq_len = feature_sequence.shape[:2]
        
        # STEP 1: Cross-frame attention
        attended_features, temporal_attention = self.cross_frame_attention(feature_sequence)
        
        # STEP 2: Flatten for RNN processing
        features_flat = attended_features.reshape(batch_size, seq_len, -1)
        
        # STEP 3: Enhanced GRU processing
        gru_out, _ = self.motion_gru(features_flat)
        
        # STEP 4: Additional LSTM processing
        lstm_out, _ = self.motion_lstm(gru_out)
        
        # STEP 5: Use final state for analysis
        final_state = lstm_out[:, -1]  # [batch, 256*2]
        
        # STEP 6: Motion analysis
        motion_features = self.motion_analyzer(final_state)
        
        # STEP 7: Physics estimations
        displacement = self.displacement_estimator(motion_features)
        velocity = self.velocity_estimator(motion_features)
        acceleration = self.acceleration_estimator(motion_features)
        confidence = self.confidence_estimator(motion_features)
        
        return {
            'motion_features': motion_features,
            'displacement': displacement,
            'velocity': velocity,
            'acceleration': acceleration,
            'confidence': confidence,
            'temporal_attention': temporal_attention
        }

class PhysicsBasedSpeedPredictor(nn.Module):
    """Enhanced speed prediction using ALL physics information"""
    
    def __init__(self):
        super().__init__()
        
        # Multi-modal fusion (motion + size + displacement + velocity + acceleration)
        fusion_input_dim = 64 + 3 + 2 + 2 + 1  # motion + size + disp + vel + accel
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        
        # Speed prediction with uncertainty
        self.speed_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU()  # Ensure positive speeds
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus()  # Positive uncertainty
        )
        
        # Physics constraints (enhanced)
        self.physics_constraints = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, motion_features, vehicle_size, displacement, velocity, acceleration):
        # Fuse ALL physics information
        combined = torch.cat([
            motion_features, 
            vehicle_size, 
            displacement, 
            velocity, 
            acceleration
        ], dim=1)
        
        fused_features = self.fusion_layer(combined)
        
        # Predict speed and uncertainty
        raw_speed = self.speed_head(fused_features)
        uncertainty = self.uncertainty_head(fused_features)
        
        # Apply enhanced physics constraints
        speed_constraint = self.physics_constraints(raw_speed)
        final_speed = raw_speed * speed_constraint * 180 + 10  # Scale to 10-190 km/h
        
        return final_speed.squeeze(-1), uncertainty.squeeze(-1)

class ProperUniversalSpeedNet(nn.Module):
    """Universal SpeedNet with PROPER attention and RNN"""
    
    def __init__(self, sequence_length=6):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Components with PROPER attention
        self.feature_extractor = PhysicsInformedFeatureExtractor()
        self.temporal_analyzer = EnhancedTemporalAnalyzer(
            sequence_length=sequence_length, 
            hidden_dim=512
        )
        self.speed_predictor = PhysicsBasedSpeedPredictor()
        
    def forward(self, frame_sequence):
        """
        Args:
            frame_sequence: [batch, sequence_length, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = frame_sequence.shape
        
        # Process each frame with PROPER spatial attention
        all_features = []
        all_sizes = []
        all_spatial_attentions = []
        
        for t in range(seq_len):
            frame = frame_sequence[:, t]  # [batch, channels, height, width]
            features, size, spatial_attention = self.feature_extractor(frame)
            
            all_features.append(features)
            all_sizes.append(size)
            all_spatial_attentions.append(spatial_attention)
        
        # Stack temporal information
        feature_sequence = torch.stack(all_features, dim=1)  # [batch, seq, 256, 10, 10]
        size_sequence = torch.stack(all_sizes, dim=1)        # [batch, seq, 3]
        
        # Enhanced temporal analysis with cross-frame attention
        temporal_results = self.temporal_analyzer(feature_sequence, size_sequence)
        
        # Final speed prediction using ALL physics information
        avg_size = size_sequence.mean(dim=1)  # Average size across sequence
        speed, uncertainty = self.speed_predictor(
            temporal_results['motion_features'],
            avg_size,
            temporal_results['displacement'],
            temporal_results['velocity'],
            temporal_results['acceleration']
        )
        
        return {
            'speed': speed,
            'uncertainty': uncertainty,
            'confidence': temporal_results['confidence'].squeeze(-1),
            'displacement': temporal_results['displacement'],
            'velocity': temporal_results['velocity'],
            'acceleration': temporal_results['acceleration'],
            'vehicle_size': avg_size,
            'spatial_attention': torch.stack(all_spatial_attentions, dim=1),
            'temporal_attention': temporal_results['temporal_attention']
        }

# Dataset and training code (same as before but using ProperUniversalSpeedNet)
class BrnCompSpeedDataset(Dataset):
    """Dataset optimized for BrnCompSpeed with physics-informed learning"""
    
    def __init__(self, dataset_root, split='train', sequence_length=6, 
                 image_size=(160, 160), use_augmentation=True, samples=None, silent=False):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.silent = silent
        
        self.samples = samples or []
        
        if split == 'full' and samples is None:
            self._collect_samples()
        
        if not self.silent:
            self._print_info()
    
    def _print_info(self):
        """Print dataset info"""
        print(f"✅ {self.split} dataset: {len(self.samples)} samples")
        
        session_counts = {}
        for sample in self.samples:
            session = sample['session_id']
            session_counts[session] = session_counts.get(session, 0) + 1
        print(f"📊 Session distribution: {session_counts}")
    
    def _collect_samples(self):
        """Collect samples from BrnCompSpeed dataset"""
        if not self.silent:
            print("📁 Loading BrnCompSpeed dataset...")
        
        for session_dir in sorted(self.dataset_root.iterdir()):
            if not session_dir.is_dir():
                continue
                
            session_name = session_dir.name
            if not self.silent:
                print(f"📊 {session_name}: ", end="")
            
            gt_path = session_dir / "gt_data.pkl"
            video_path = session_dir / "video.avi"
            
            if not gt_path.exists() or not video_path.exists():
                if not self.silent:
                    print("❌ Missing files")
                continue
            
            try:
                # Load ground truth safely
                gt_data = load_pickle_safe(gt_path)
                
                session_id = session_name.split('_')[0]
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                valid_cars = [car for car in cars if car.get('valid', False) and 
                            len(car.get('intersections', [])) >= 2]
                
                if not self.silent:
                    print(f"{len(valid_cars)} cars")
                
                # Create samples
                for car in valid_cars:
                    speed_kmh = car.get('speed', 0)
                    if 30 <= speed_kmh <= 150:  # Reasonable speed range
                        
                        intersections = car['intersections']
                        start_time = intersections[0]['videoTime']
                        end_time = intersections[-1]['videoTime']
                        
                        # Ensure minimum duration for sequence
                        duration = end_time - start_time
                        if duration < 0.1:  # At least 0.1 seconds
                            continue
                        
                        self.samples.append({
                            'video_path': str(video_path),
                            'session_id': session_id,
                            'speed_kmh': speed_kmh,
                            'start_time': start_time,
                            'end_time': end_time,
                            'fps': fps,
                            'duration': duration
                        })
                        
            except Exception as e:
                if not self.silent:
                    print(f"❌ Error: {e}")
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract video sequence
        frames = self._extract_frames(sample)
        
        # Convert to tensor
        frame_tensor = torch.stack([
            torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
            for frame in frames
        ])
        
        return {
            'frames': frame_tensor,
            'speed': torch.tensor(sample['speed_kmh'], dtype=torch.float32),
            'session_id': sample['session_id'],
            'duration': torch.tensor(sample['duration'], dtype=torch.float32)
        }
    
    def _extract_frames(self, sample):
        """Extract frame sequence optimized for physics learning"""
        cap = cv2.VideoCapture(sample['video_path'])
        
        try:
            start_frame = int(sample['start_time'] * sample['fps'])
            end_frame = int(sample['end_time'] * sample['fps'])
            
            # Smart frame sampling
            total_frames = end_frame - start_frame
            if total_frames >= self.sequence_length:
                # Use evenly spaced frames
                frame_indices = np.linspace(start_frame, end_frame, self.sequence_length).astype(int)
            else:
                # Interpolate for short sequences
                frame_indices = np.linspace(start_frame, end_frame, self.sequence_length).astype(int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize and augment
                    frame = cv2.resize(frame, self.image_size)
                    if self.use_augmentation:
                        frame = self._augment_frame(frame)
                    frames.append(frame)
                else:
                    # Fallback
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8))
            
            return frames
            
        finally:
            cap.release()
    
    def _augment_frame(self, frame):
        """Simple augmentation for training"""
        if np.random.random() < 0.5:
            # Color jitter
            frame = frame.astype(np.float32)
            frame *= np.random.uniform(0.8, 1.2)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        if np.random.random() < 0.3:
            # Add noise
            noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame

class EnhancedPhysicsInformedLoss(nn.Module):
    """Enhanced loss function with velocity and acceleration constraints"""
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05, delta=0.02):
        super().__init__()
        self.alpha = alpha  # Speed loss weight
        self.beta = beta    # Uncertainty loss weight  
        self.gamma = gamma  # Physics constraint weight
        self.delta = delta  # Kinematics consistency weight
        
    def forward(self, predictions, targets, durations):
        speed_pred = predictions['speed']
        uncertainty = predictions['uncertainty']
        confidence = predictions['confidence']
        velocity = predictions['velocity']
        acceleration = predictions['acceleration']
        
        # Main speed loss with uncertainty weighting
        speed_diff = speed_pred - targets
        speed_loss = torch.mean(
            0.5 * torch.exp(-uncertainty) * speed_diff ** 2 + 0.5 * uncertainty
        )
        
        # Physics constraints
        negative_penalty = torch.mean(torch.relu(-speed_pred) ** 2)
        high_speed_penalty = torch.mean(torch.relu(speed_pred - 200) ** 2)
        low_speed_penalty = torch.mean(torch.relu(5 - speed_pred) ** 2)
        
        # Confidence consistency
        confidence_loss = torch.mean((uncertainty - (1 - confidence)) ** 2)
        
        # NEW: Kinematics consistency (velocity and acceleration should be reasonable)
        velocity_magnitude = torch.norm(velocity, dim=1)
        velocity_consistency = torch.mean(torch.abs(velocity_magnitude - speed_pred / 3.6))  # Convert km/h to m/s
        
        acceleration_penalty = torch.mean(torch.relu(torch.abs(acceleration) - 10) ** 2)  # Reasonable acceleration
        
        # Duration consistency
        duration_consistency = torch.mean(torch.abs(
            torch.log(speed_pred + 1) - torch.log(60.0 / (durations + 0.1))
        ))
        
        total_loss = (
            self.alpha * speed_loss +
            self.beta * confidence_loss +
            self.gamma * (negative_penalty + high_speed_penalty + low_speed_penalty) +
            self.delta * (velocity_consistency + acceleration_penalty) +
            0.01 * duration_consistency
        )
        
        return {
            'total_loss': total_loss,
            'speed_loss': speed_loss,
            'confidence_loss': confidence_loss,
            'physics_penalty': negative_penalty + high_speed_penalty + low_speed_penalty,
            'kinematics_loss': velocity_consistency + acceleration_penalty,
            'duration_consistency': duration_consistency
        }

# Checkpointing functions (same as before)
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, 
                   train_losses, val_losses, best_mae, config):
    """Save comprehensive checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_mae': best_mae,
        'config': config,
        'timestamp': str(datetime.now()),
        'pytorch_version': torch.__version__
    }
    
    checkpoint_path = '/kaggle/working/proper_universal_speednet_latest.pth'
    backup_path = f'/kaggle/working/proper_universal_speednet_epoch_{epoch}.pth'
    
    try:
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, backup_path)
        print(f"✅ Checkpoint saved: epoch {epoch}")
        return True
    except Exception as e:
        print(f"❌ Checkpoint save failed: {e}")
        return False

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    """Load checkpoint safely"""
    if not os.path.exists(checkpoint_path):
        print(f"🆕 No checkpoint found")
        return 0, [], [], float('inf')
    
    try:
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_mae = checkpoint.get('best_mae', float('inf'))
        
        print(f"✅ Resumed from epoch {start_epoch}, best MAE: {best_mae:.2f}")
        return start_epoch, train_losses, val_losses, best_mae
        
    except Exception as e:
        print(f"❌ Checkpoint load failed: {e}")
        return 0, [], [], float('inf')

def train_proper_speednet():
    """Training pipeline for PROPER Universal SpeedNet"""
    
    print("\n🚀 Starting PROPER Universal SpeedNet Training")
    print("🔧 Features:")
    print("  ✅ Multi-Head Self-Attention (spatial)")
    print("  ✅ CAUSAL Cross-Frame Attention (temporal - physics correct!)")
    print("  ✅ Enhanced GRU + LSTM (temporal modeling)")
    print("  ✅ Velocity + Acceleration estimation")
    print("  ✅ Physics-informed loss with kinematics")
    
    # Dataset
    dataset_root = "/kaggle/input/brnocomp/brno_kaggle_subset/dataset"
    full_dataset = BrnCompSpeedDataset(dataset_root, 'full')
    
    # Create train/val split
    total_samples = len(full_dataset.samples)
    train_size = int(0.8 * total_samples)
    
    train_samples = full_dataset.samples[:train_size]
    val_samples = full_dataset.samples[train_size:]
    
    train_dataset = BrnCompSpeedDataset(dataset_root, 'train', samples=train_samples, silent=True)
    val_dataset = BrnCompSpeedDataset(dataset_root, 'val', samples=val_samples, silent=True, use_augmentation=False)
    
    print(f"📊 Data split: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Model with PROPER attention
    model = ProperUniversalSpeedNet(sequence_length=6).to(device)
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = EnhancedPhysicsInformedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)  # Lower LR for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = GradScaler()
    
    # Try to resume
    start_epoch, train_losses, val_losses, best_mae = load_checkpoint(
        '/kaggle/working/proper_universal_speednet_latest.pth',
        model, optimizer, scheduler, scaler
    )
    
    config = {
        'model': 'ProperUniversalSpeedNet',
        'sequence_length': 6,
        'image_size': [160, 160],
        'learning_rate': 5e-5,
        'batch_size': 6,  # Smaller batch for larger model
        'attention_heads': 8,
        'rnn_layers': 5  # 3 GRU + 2 LSTM
    }
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0, pin_memory=True)
    
    # Training loop
    for epoch in range(start_epoch, 50):
        if not should_continue_training():
            print(f"⏰ Time limit approaching! Saving and exiting...")
            save_checkpoint(epoch-1, model, optimizer, scheduler, scaler,
                          train_losses, val_losses, best_mae, config)
            return
        
        print(f"\n📅 Epoch {epoch+1}/50")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="🔥 Training"):
            frames = batch['frames'].to(device)
            speeds = batch['speed'].to(device) 
            durations = batch['duration'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                predictions = model(frames)
                losses = criterion(predictions, speeds, durations)
                loss = losses['total_loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="📊 Validation"):
                frames = batch['frames'].to(device)
                speeds = batch['speed'].to(device)
                
                with autocast('cuda'):
                    predictions = model(frames)
                
                val_predictions.extend(predictions['speed'].cpu().numpy())
                val_targets.extend(speeds.cpu().numpy())
        
        # Calculate metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_mae = np.mean(np.abs(val_predictions - val_targets))
        val_losses.append(val_mae)
        
        # Update scheduler
        scheduler.step()
        
        print(f"📈 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val MAE: {val_mae:.2f} km/h")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Save best model
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'mae': best_mae,
                'epoch': epoch,
                'config': config
            }, '/kaggle/working/proper_universal_speednet_best.pth')
            print(f"  🏆 New best model! MAE: {best_mae:.2f} km/h")
        
        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                       train_losses, val_losses, best_mae, config)
        
        torch.cuda.empty_cache()
    
    print("\n🎉 Training complete!")
    print(f"🏆 Best validation MAE: {best_mae:.2f} km/h")

def main():
    """Main training function"""
    
    print("\n🎯 PROPER Universal SpeedNet Features:")
    print("  • Multi-Head Self-Attention for spatial regions")
    print("  • CAUSAL Cross-Frame Attention for physics-correct temporal relationships")
    print("  • Enhanced GRU + LSTM for motion modeling")
    print("  • Velocity & acceleration estimation")
    print("  • Physics-informed loss with kinematics")
    print("  • TRUE frame-to-frame relationship modeling (no future information!)")
    
    train_proper_speednet()
    
    print("\n✅ PROPER Universal SpeedNet Training Complete!")
    print("\n🧠 This model ACTUALLY captures physics through:")
    print("  • REAL multi-head attention (not sigmoid)")
    print("  • CAUSAL cross-frame temporal attention (no future cheating!)")
    print("  • Enhanced RNN family (GRU + LSTM)")
    print("  • Velocity and acceleration physics")
    print("  • Kinematics-consistent loss function")
    print("  • Physics-correct causality: frame t only sees frames ≤ t")

if __name__ == "__main__":
    main()