#!/usr/bin/env python3
"""
SpeedNet Training Pipeline for Kaggle
Train the 3D-aware vehicle speed estimation network on BrnCompSpeed dataset
"""

import os
import sys
import json
import pickle
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our models
from models.speednet import SpeedNet, SpeedNetLoss, VEHICLE_SIZE_PRIORS
from dataset_analyzer import BrnCompSpeedAnalyzer

class BrnCompSpeedDataset(Dataset):
    """
    PyTorch Dataset for BrnCompSpeed data
    Handles video sequences and ground truth speed annotations
    """
    
    def __init__(self, 
                 dataset_root,
                 sequence_length=8,
                 image_size=(640, 640),
                 transform=None,
                 mode='train',
                 test_split=0.2):
        
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.transform = transform
        self.mode = mode
        
        # Load and process dataset
        self.analyzer = BrnCompSpeedAnalyzer(dataset_root)
        self.sessions = self.analyzer.scan_dataset()
        
        # Prepare training samples
        self.samples = []
        self._prepare_samples()
        
        # Train/test split
        if mode in ['train', 'val']:
            train_samples, val_samples = train_test_split(
                self.samples, test_size=test_split, random_state=42
            )
            self.samples = train_samples if mode == 'train' else val_samples
        
        print(f"Dataset {mode}: {len(self.samples)} samples")
        
    def _prepare_samples(self):
        """Extract training samples from all sessions"""
        for session in self.sessions:
            try:
                gt_data = self.analyzer.load_ground_truth(session['name'])
                video_path = session['files']['video.avi']
                
                if not video_path or not video_path.exists():
                    continue
                    
                # Extract car trajectories
                cars = gt_data.get('cars', [])
                fps = gt_data.get('fps', 25.0)
                
                for car in cars:
                    if not car.get('valid', True):
                        continue
                        
                    intersections = car.get('intersections', [])
                    if len(intersections) < 2:
                        continue
                        
                    # Create training sample
                    sample = {
                        'session_name': session['name'],
                        'video_path': str(video_path),
                        'car_id': car.get('carId', -1),
                        'speed_kmh': car.get('speed', 0.0),
                        'lane_index': list(car.get('laneIndex', set())),
                        'intersections': intersections,
                        'fps': fps,
                        'measurement_lines': gt_data.get('measurementLines', []),
                        'distance_measurements': gt_data.get('distanceMeasurement', [])
                    }
                    
                    self.samples.append(sample)
                    
            except Exception as e:
                print(f"Error processing {session['name']}: {e}")
                continue
    
    def _extract_video_sequence(self, video_path, start_time, end_time, fps):
        """Extract video sequence for a car trajectory"""
        cap = cv2.VideoCapture(video_path)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Ensure we have enough frames for sequence_length
        total_frames = end_frame - start_frame + 1
        if total_frames < self.sequence_length:
            # Pad by extending the range
            extend = self.sequence_length - total_frames
            start_frame = max(0, start_frame - extend // 2)
            end_frame = start_frame + self.sequence_length - 1
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # Repeat last frame if video ends
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Resize frame
            frame = cv2.resize(frame, self.image_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return np.array(frames)  # [T, H, W, C]
    
    def _create_synthetic_bboxes(self, intersections, measurement_lines, image_size):
        """
        Create synthetic bounding boxes based on measurement line intersections
        This is a simplified approach - in practice you'd have actual detections
        """
        bboxes = []
        
        for intersection in intersections:
            line_id = intersection.get('measurementLineId', 0)
            video_time = intersection.get('videoTime', 0)
            
            if line_id < len(measurement_lines):
                # Get measurement line
                line = measurement_lines[line_id]
                a, b, c = line
                
                # Find intersection point on line (simplified)
                if abs(b) > 1e-6:
                    # Assume vehicle is at center of image horizontally
                    x_center = image_size[0] // 2
                    y_center = -(a * x_center + c) / b
                else:
                    x_center = -c / a if abs(a) > 1e-6 else image_size[0] // 2
                    y_center = image_size[1] // 2
                
                # Create bounding box (typical car size)
                bbox_w, bbox_h = 60, 120  # pixels
                x1 = max(0, x_center - bbox_w // 2)
                y1 = max(0, y_center - bbox_h)
                x2 = min(image_size[0], x_center + bbox_w // 2)
                y2 = min(image_size[1], y_center)
                
                bboxes.append([x1, y1, x2, y2])
        
        return np.array(bboxes)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get time range for this car
        intersections = sample['intersections']
        times = [i['videoTime'] for i in intersections]
        start_time = min(times) - 1.0  # Add some context
        end_time = max(times) + 1.0
        
        # Extract video sequence
        video_sequence = self._extract_video_sequence(
            sample['video_path'], start_time, end_time, sample['fps']
        )
        
        # Create synthetic bounding boxes
        bboxes = self._create_synthetic_bboxes(
            intersections, sample['measurement_lines'], self.image_size
        )
        
        # Convert to tensors
        video_tensor = torch.from_numpy(video_sequence).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        # Apply transforms
        if self.transform:
            transformed_frames = []
            for t in range(video_tensor.shape[0]):
                frame = self.transform(video_tensor[t])
                transformed_frames.append(frame)
            video_tensor = torch.stack(transformed_frames)
        
        # Prepare ground truth
        target = {
            'speed': sample['speed_kmh'],
            'car_id': sample['car_id'],
            'lane_index': sample['lane_index'],
            'bboxes': torch.from_numpy(bboxes).float(),
            'measurement_lines': sample['measurement_lines'],
            'distance_measurements': sample['distance_measurements']
        }
        
        return video_tensor, target

class SpeedNetTrainer:
    """
    Trainer class for SpeedNet
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=1e-4,
                 weight_decay=1e-5):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss function
        self.criterion = SpeedNetLoss(
            speed_weight=1.0,
            uncertainty_weight=0.1,
            camera_weight=0.5,
            depth_weight=0.3
        ).to(device)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (video_sequences, targets) in enumerate(pbar):
            video_sequences = video_sequences.to(self.device)
            
            # Prepare targets for GPU
            batch_targets = {
                'speeds': [{'speed': t['speed']} for t in targets],
                'bboxes': [t['bboxes'].to(self.device) for t in targets]
            }
            
            self.optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                predictions = self.model(video_sequences)
                
                # Compute loss
                loss_dict = self.criterion(predictions, batch_targets)
                loss = loss_dict['total']
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for video_sequences, targets in tqdm(self.val_loader, desc="Validation"):
                video_sequences = video_sequences.to(self.device)
                
                # Prepare targets
                batch_targets = {
                    'speeds': [{'speed': t['speed']} for t in targets],
                    'bboxes': [t['bboxes'].to(self.device) for t in targets]
                }
                
                with autocast():
                    predictions = self.model(video_sequences)
                    loss_dict = self.criterion(predictions, batch_targets)
                    loss = loss_dict['total']
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                if predictions['predictions']:
                    pred_speeds = [p['speed'].cpu().numpy() for p in predictions['predictions']]
                    target_speeds = [t['speed'] for t in targets]
                    
                    all_predictions.extend(pred_speeds)
                    all_targets.extend(target_speeds)
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Compute metrics
        if all_predictions and all_targets:
            all_predictions = np.array(all_predictions).flatten()
            all_targets = np.array(all_targets)
            
            mae = mean_absolute_error(all_targets, all_predictions)
            rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'loss': avg_loss
            }
            
            self.val_metrics.append(metrics)
            return metrics
        else:
            return {'mae': float('inf'), 'rmse': float('inf'), 'loss': avg_loss}
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        best_mae = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.2f} km/h")
            print(f"Val RMSE: {val_metrics['rmse']:.2f} km/h")
            
            # Save best model
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_mae': best_mae,
                    'metrics': val_metrics
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! MAE: {best_mae:.2f} km/h")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_metrics': self.val_metrics
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot training curves
        self.plot_training_curves(save_dir)
        
    def plot_training_curves(self, save_dir):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.train_losses, label='Train Loss')
        if self.val_losses:
            axes[0].plot(epochs, self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE curve
        if self.val_metrics:
            mae_values = [m['mae'] for m in self.val_metrics]
            axes[1].plot(epochs, mae_values, label='Val MAE', color='orange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE (km/h)')
            axes[1].set_title('Validation MAE')
            axes[1].legend()
            axes[1].grid(True)
        
        # RMSE curve
        if self.val_metrics:
            rmse_values = [m['rmse'] for m in self.val_metrics]
            axes[2].plot(epochs, rmse_values, label='Val RMSE', color='red')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('RMSE (km/h)')
            axes[2].set_title('Validation RMSE')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.show()

def create_data_transforms():
    """Create data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train SpeedNet on BrnCompSpeed dataset')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to BrnCompSpeed dataset root')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Number of frames in sequence')
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 640],
                        help='Image size (width height)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create data transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = BrnCompSpeedDataset(
        dataset_root=args.dataset_root,
        sequence_length=args.sequence_length,
        image_size=tuple(args.image_size),
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = BrnCompSpeedDataset(
        dataset_root=args.dataset_root,
        sequence_length=args.sequence_length,
        image_size=tuple(args.image_size),
        transform=val_transform,
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = SpeedNet(
        backbone='yolov8n',
        sequence_length=args.sequence_length,
        confidence_threshold=0.5
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = SpeedNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    # Train the model
    print("Starting training...")
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()