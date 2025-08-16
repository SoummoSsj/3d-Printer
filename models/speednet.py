#!/usr/bin/env python3
"""
SpeedNet: 3D-Aware Vehicle Speed Estimation Network
Combines vehicle detection with perspective-aware speed regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ultralytics import YOLO
import numpy as np
import cv2
import math

class CameraCalibrationModule(nn.Module):
    """
    Neural network module for automatic camera calibration
    Predicts vanishing point and camera parameters from image features
    """
    def __init__(self, input_dim=512):
        super().__init__()
        
        # Vanishing point prediction
        self.vp_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (vp_x, vp_y)
        )
        
        # Camera height and focal length prediction
        self.cam_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (height_m, focal_length_px)
        )
        
        # Road plane orientation (pitch angle)
        self.pitch_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # pitch angle in radians
        )
        
    def forward(self, features):
        """
        Args:
            features: Global image features [B, input_dim]
        Returns:
            dict with vanishing_point, camera_params, pitch_angle
        """
        vp = self.vp_head(features)  # [B, 2]
        cam_params = self.cam_head(features)  # [B, 2] 
        pitch = self.pitch_head(features)  # [B, 1]
        
        return {
            'vanishing_point': vp,
            'camera_height': cam_params[:, 0:1],
            'focal_length': cam_params[:, 1:2], 
            'pitch_angle': pitch
        }

class Vehicle3DModule(nn.Module):
    """
    Estimates 3D bounding box and depth for detected vehicles
    """
    def __init__(self, input_dim=512):
        super().__init__()
        
        # 3D box dimensions prediction (normalized by typical car size)
        self.dim_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # (length, width, height) relative to average car
        )
        
        # Rotation angle around Y-axis (yaw)
        self.rot_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (sin(yaw), cos(yaw))
        )
        
        # Depth estimation
        self.depth_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # depth in meters
        )
        
        # Vehicle type classification (affects size priors)
        self.type_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # car, truck, bus, motorcycle
        )
        
    def forward(self, vehicle_features):
        """
        Args:
            vehicle_features: Features from detected vehicles [N, input_dim]
        Returns:
            dict with 3D box parameters
        """
        dimensions = self.dim_head(vehicle_features)  # [N, 3]
        rotation = self.rot_head(vehicle_features)  # [N, 2]
        depth = self.depth_head(vehicle_features)  # [N, 1]
        vehicle_type = self.type_head(vehicle_features)  # [N, 4]
        
        # Normalize rotation to unit vector
        rotation = F.normalize(rotation, p=2, dim=1)
        
        return {
            'dimensions': dimensions,
            'rotation': rotation,
            'depth': depth,
            'vehicle_type': vehicle_type
        }

class TemporalFusionModule(nn.Module):
    """
    Fuses information across multiple frames for robust speed estimation
    """
    def __init__(self, feature_dim=256, sequence_length=8):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism for important frames
        self.attention = nn.MultiheadAttention(
            embed_dim=256,  # 128 * 2 (bidirectional)
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(256, feature_dim)
        
    def forward(self, sequence_features, mask=None):
        """
        Args:
            sequence_features: [B, T, feature_dim] features across time
            mask: [B, T] mask for valid frames
        Returns:
            Temporally fused features [B, feature_dim]
        """
        B, T, D = sequence_features.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(sequence_features)  # [B, T, 256]
        
        # Self-attention across time
        if mask is not None:
            # Convert mask for attention (True = masked)
            attn_mask = mask.float()
            attn_mask = attn_mask.masked_fill(mask, float('-inf'))
        else:
            attn_mask = None
            
        attended, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask if mask is not None else None
        )  # [B, T, 256]
        
        # Global temporal pooling
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(attended)
            attended_masked = attended.masked_fill(mask_expanded, 0.0)
            valid_counts = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1.0)
            pooled = attended_masked.sum(dim=1) / valid_counts
        else:
            pooled = attended.mean(dim=1)  # [B, 256]
            
        output = self.output_proj(pooled)  # [B, feature_dim]
        
        return output

class SpeedRegressionModule(nn.Module):
    """
    Final speed regression taking into account all geometric and temporal features
    """
    def __init__(self, input_dim=512):
        super().__init__()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + 32, 256),  # +32 for geometric features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Speed prediction head
        self.speed_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Speed in km/h
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Log variance
        )
        
    def forward(self, fused_features, geometric_features):
        """
        Args:
            fused_features: Temporally fused features [B, input_dim]
            geometric_features: Camera and 3D geometric features [B, 32]
        Returns:
            dict with speed and uncertainty
        """
        # Concatenate all features
        combined = torch.cat([fused_features, geometric_features], dim=1)
        
        # Fusion and prediction
        x = self.fusion(combined)
        speed = self.speed_head(x)
        log_var = self.uncertainty_head(x)
        
        # Ensure positive speed
        speed = F.softplus(speed)
        
        return {
            'speed': speed,  # [B, 1]
            'log_variance': log_var  # [B, 1]
        }

class SpeedNet(nn.Module):
    """
    Complete SpeedNet architecture for 3D-aware vehicle speed estimation
    """
    def __init__(self, 
                 backbone='yolov8n',
                 sequence_length=8,
                 confidence_threshold=0.5):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Vehicle detection backbone (frozen)
        self.detector = YOLO(f'{backbone}.pt')
        for param in self.detector.model.parameters():
            param.requires_grad = False
            
        # Feature extraction from detection backbone
        self.feature_extractor = self._build_feature_extractor()
        
        # Core modules
        self.camera_calib = CameraCalibrationModule(input_dim=512)
        self.vehicle_3d = Vehicle3DModule(input_dim=512)
        self.temporal_fusion = TemporalFusionModule(feature_dim=256, sequence_length=sequence_length)
        self.speed_regression = SpeedRegressionModule(input_dim=256)
        
        # Geometric feature processor
        self.geometric_processor = nn.Sequential(
            nn.Linear(16, 32),  # Camera params + 3D box params
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
    def _build_feature_extractor(self):
        """Build feature extraction layers on top of detection backbone"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7 * 7 * 256, 512),  # Adjust based on backbone
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
    def extract_vehicle_features(self, images, detections):
        """
        Extract features for each detected vehicle across the sequence
        
        Args:
            images: [B, T, C, H, W] sequence of images
            detections: List of detection results for each frame
            
        Returns:
            dict with vehicle features and metadata
        """
        B, T, C, H, W = images.shape
        device = images.device
        
        # Process each frame through detector backbone
        frame_features = []
        vehicle_sequences = {}  # track_id -> list of (frame_idx, bbox, features)
        
        for t in range(T):
            frame = images[:, t]  # [B, C, H, W]
            
            # Get detection features (this requires access to intermediate features)
            # For now, simulate this - in practice, you'd modify YOLO to return features
            with torch.no_grad():
                results = self.detector(frame)
                
            # Extract features per detection
            for b in range(B):
                if results[b].boxes is not None:
                    boxes = results[b].boxes.xyxy
                    track_ids = results[b].boxes.id if hasattr(results[b].boxes, 'id') else None
                    
                    for i, box in enumerate(boxes):
                        # Extract ROI features
                        x1, y1, x2, y2 = box.int()
                        roi = frame[b:b+1, :, y1:y2, x1:x2]
                        
                        if roi.numel() > 0:
                            roi_resized = F.interpolate(roi, size=(64, 64), mode='bilinear')
                            roi_features = self.feature_extractor(roi_resized)  # [1, 256]
                            
                            # Track vehicles across frames
                            track_id = int(track_ids[i]) if track_ids is not None else f"b{b}_f{t}_i{i}"
                            
                            if track_id not in vehicle_sequences:
                                vehicle_sequences[track_id] = []
                                
                            vehicle_sequences[track_id].append({
                                'frame': t,
                                'batch': b,
                                'bbox': box,
                                'features': roi_features.squeeze(0)  # [256]
                            })
            
        return vehicle_sequences
    
    def compute_geometric_features(self, camera_params, vehicle_3d_params, bboxes):
        """
        Compute geometric features for speed estimation
        
        Args:
            camera_params: Camera calibration parameters
            vehicle_3d_params: 3D vehicle parameters
            bboxes: 2D bounding boxes
            
        Returns:
            Geometric features [N, 16]
        """
        vp = camera_params['vanishing_point']  # [B, 2]
        cam_height = camera_params['camera_height']  # [B, 1]
        focal_length = camera_params['focal_length']  # [B, 1]
        pitch = camera_params['pitch_angle']  # [B, 1]
        
        depth = vehicle_3d_params['depth']  # [N, 1]
        dimensions = vehicle_3d_params['dimensions']  # [N, 3]
        rotation = vehicle_3d_params['rotation']  # [N, 2]
        
        # Compute perspective scale factor
        bbox_height = bboxes[:, 3] - bboxes[:, 1]  # [N]
        bbox_center_y = (bboxes[:, 1] + bboxes[:, 3]) / 2  # [N]
        
        # Distance to horizon line (vanishing point)
        horizon_dist = torch.abs(bbox_center_y - vp[:, 1].unsqueeze(0))  # [N]
        
        # Perspective scale (higher = closer to camera)
        perspective_scale = focal_length.squeeze() / (depth.squeeze() + 1e-6)  # [N]
        
        # Vehicle orientation relative to camera
        yaw_sin, yaw_cos = rotation[:, 0], rotation[:, 1]
        
        # Geometric features
        geometric_features = torch.stack([
            depth.squeeze(),           # 0: depth
            dimensions[:, 0],          # 1: length  
            dimensions[:, 1],          # 2: width
            dimensions[:, 2],          # 3: height
            bbox_height,               # 4: 2D bbox height
            horizon_dist,              # 5: distance to horizon
            perspective_scale,         # 6: perspective scale
            yaw_sin,                   # 7: vehicle yaw (sin)
            yaw_cos,                   # 8: vehicle yaw (cos)
            cam_height.squeeze().expand(depth.size(0)),  # 9: camera height
            focal_length.squeeze().expand(depth.size(0)), # 10: focal length
            pitch.squeeze().expand(depth.size(0)),        # 11: camera pitch
            bboxes[:, 0],              # 12: bbox x1
            bboxes[:, 1],              # 13: bbox y1  
            bboxes[:, 2],              # 14: bbox x2
            bboxes[:, 3],              # 15: bbox y2
        ], dim=1)  # [N, 16]
        
        return geometric_features
    
    def forward(self, image_sequence, track_data=None):
        """
        Forward pass for training
        
        Args:
            image_sequence: [B, T, C, H, W] sequence of images
            track_data: Optional ground truth tracking data
            
        Returns:
            dict with predictions and intermediate results
        """
        B, T, C, H, W = image_sequence.shape
        device = image_sequence.device
        
        # Extract global image features for camera calibration
        # Use first frame for camera calibration
        first_frame = image_sequence[:, 0]  # [B, C, H, W]
        global_features = self.feature_extractor(first_frame)  # [B, 256]
        
        # Predict camera parameters
        camera_params = self.camera_calib(global_features)
        
        # Extract vehicle features across sequence
        vehicle_sequences = self.extract_vehicle_features(image_sequence, None)
        
        all_predictions = []
        
        # Process each vehicle track
        for track_id, track_sequence in vehicle_sequences.items():
            if len(track_sequence) < 2:  # Need at least 2 frames for speed
                continue
                
            # Prepare sequence data
            sequence_features = []
            sequence_bboxes = []
            
            for frame_data in track_sequence[-self.sequence_length:]:  # Last N frames
                sequence_features.append(frame_data['features'])
                sequence_bboxes.append(frame_data['bbox'])
                
            # Pad sequence if too short
            while len(sequence_features) < self.sequence_length:
                sequence_features.insert(0, sequence_features[0])
                sequence_bboxes.insert(0, sequence_bboxes[0])
                
            # Stack into tensors
            seq_features = torch.stack(sequence_features)  # [T, 256]
            seq_bboxes = torch.stack(sequence_bboxes)  # [T, 4]
            
            # Predict 3D parameters for current vehicle
            current_features = seq_features[-1:].to(device)  # [1, 256]
            vehicle_3d_params = self.vehicle_3d(current_features)
            
            # Compute geometric features
            current_bbox = seq_bboxes[-1:].to(device)  # [1, 4]
            batch_idx = track_sequence[-1]['batch']
            
            geom_features = self.compute_geometric_features(
                {k: v[batch_idx:batch_idx+1] for k, v in camera_params.items()},
                vehicle_3d_params,
                current_bbox
            )  # [1, 16]
            
            # Process geometric features
            processed_geom = self.geometric_processor(geom_features)  # [1, 32]
            
            # Temporal fusion
            seq_features_input = seq_features.unsqueeze(0).to(device)  # [1, T, 256]
            fused_features = self.temporal_fusion(seq_features_input)  # [1, 256]
            
            # Speed prediction
            speed_output = self.speed_regression(fused_features, processed_geom)
            
            # Store prediction with metadata
            prediction = {
                'track_id': track_id,
                'batch_idx': batch_idx,
                'speed': speed_output['speed'],
                'uncertainty': speed_output['log_variance'],
                'camera_params': {k: v[batch_idx:batch_idx+1] for k, v in camera_params.items()},
                'vehicle_3d': vehicle_3d_params,
                'bbox': current_bbox
            }
            
            all_predictions.append(prediction)
        
        return {
            'predictions': all_predictions,
            'camera_params': camera_params,
            'num_vehicles': len(all_predictions)
        }

class SpeedNetLoss(nn.Module):
    """
    Multi-task loss for SpeedNet training
    """
    def __init__(self, 
                 speed_weight=1.0,
                 uncertainty_weight=0.1,
                 camera_weight=0.5,
                 depth_weight=0.3):
        super().__init__()
        
        self.speed_weight = speed_weight
        self.uncertainty_weight = uncertainty_weight
        self.camera_weight = camera_weight
        self.depth_weight = depth_weight
        
    def forward(self, predictions, targets):
        """
        Compute multi-task loss
        
        Args:
            predictions: Model output
            targets: Ground truth data
            
        Returns:
            dict with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        # Speed regression loss with uncertainty
        if predictions['predictions']:
            pred_speeds = torch.cat([p['speed'] for p in predictions['predictions']])
            pred_uncertainties = torch.cat([p['uncertainty'] for p in predictions['predictions']])
            target_speeds = torch.tensor([t['speed'] for t in targets['speeds']], 
                                       device=pred_speeds.device).unsqueeze(1)
            
            # Heteroscedastic loss (uncertainty-weighted)
            speed_diff = pred_speeds - target_speeds
            precision = torch.exp(-pred_uncertainties)
            speed_loss = torch.mean(precision * speed_diff**2 + pred_uncertainties)
            
            losses['speed'] = speed_loss
            total_loss += self.speed_weight * speed_loss
            
            # Uncertainty regularization
            uncertainty_reg = torch.mean(torch.exp(pred_uncertainties))
            losses['uncertainty'] = uncertainty_reg
            total_loss += self.uncertainty_weight * uncertainty_reg
        
        # Camera calibration loss (if ground truth available)
        if 'camera_calibration' in targets:
            cam_targets = targets['camera_calibration']
            cam_preds = predictions['camera_params']
            
            # Vanishing point loss
            vp_loss = F.mse_loss(cam_preds['vanishing_point'], cam_targets['vanishing_point'])
            losses['vanishing_point'] = vp_loss
            total_loss += self.camera_weight * vp_loss
            
            # Camera height loss
            height_loss = F.mse_loss(cam_preds['camera_height'], cam_targets['camera_height'])
            losses['camera_height'] = height_loss
            total_loss += self.camera_weight * height_loss
        
        losses['total'] = total_loss
        return losses

# Vehicle size priors (length, width, height in meters)
VEHICLE_SIZE_PRIORS = {
    'car': (4.5, 1.8, 1.5),
    'truck': (8.0, 2.5, 3.0),
    'bus': (12.0, 2.5, 3.5),
    'motorcycle': (2.0, 0.8, 1.2)
}