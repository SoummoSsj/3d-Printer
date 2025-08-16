#!/usr/bin/env python3
"""
Real-time Vehicle Speed Detection Inference Pipeline
Runs the trained SpeedNet model on live video feeds or video files
"""

import os
import sys
import time
import json
import argparse
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Import our models
from models.speednet import SpeedNet, VEHICLE_SIZE_PRIORS

class VehicleTracker:
    """
    Simple vehicle tracker using IoU-based association
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = {}  # track_id -> track_info
        self.next_id = 1
        self.frame_count = 0
        
    def compute_iou(self, box1, box2):
        """Compute IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: List of (bbox, confidence, class_id) tuples
            
        Returns:
            List of active tracks with track_ids
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, (bbox, conf, cls_id) in enumerate(detections):
            best_iou = 0.0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track['age'] > 0:  # Skip tracks that weren't updated last frame
                    continue
                    
                last_bbox = track['bboxes'][-1]
                iou = self.compute_iou(bbox, last_bbox)
                
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track['bboxes'].append(bbox)
                track['confidences'].append(conf)
                track['class_ids'].append(cls_id)
                track['frame_indices'].append(self.frame_count)
                track['age'] = 0
                track['hits'] += 1
                
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
            
        # Create new tracks for unmatched detections
        for det_idx, (bbox, conf, cls_id) in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bboxes': deque([bbox], maxlen=50),
                    'confidences': deque([conf], maxlen=50),
                    'class_ids': deque([cls_id], maxlen=50),
                    'frame_indices': deque([self.frame_count], maxlen=50),
                    'age': 0,
                    'hits': 1,
                    'speeds': deque(maxlen=10),  # Keep last 10 speed estimates
                    'created_frame': self.frame_count
                }
        
        # Age unmatched tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track['age'] += 1
                if track['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Return active tracks that have enough hits
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits and track['age'] <= 1:
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': track['bboxes'][-1],
                    'confidence': track['confidences'][-1],
                    'class_id': track['class_ids'][-1],
                    'track_length': len(track['bboxes']),
                    'speeds': list(track['speeds'])
                })
        
        return active_tracks
    
    def get_track_history(self, track_id, max_length=8):
        """Get historical data for a track"""
        if track_id not in self.tracks:
            return None
            
        track = self.tracks[track_id]
        length = min(len(track['bboxes']), max_length)
        
        return {
            'bboxes': list(track['bboxes'])[-length:],
            'frame_indices': list(track['frame_indices'])[-length:],
            'confidences': list(track['confidences'])[-length:],
            'class_ids': list(track['class_ids'])[-length:]
        }
    
    def update_speed(self, track_id, speed):
        """Update speed estimate for a track"""
        if track_id in self.tracks:
            self.tracks[track_id]['speeds'].append(speed)

class SpeedDetectionPipeline:
    """
    Complete pipeline for real-time vehicle speed detection
    """
    
    def __init__(self, 
                 model_path,
                 device='cuda',
                 confidence_threshold=0.5,
                 sequence_length=8,
                 fps_estimate=25.0):
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self.fps_estimate = fps_estimate
        
        # Load trained model
        self.model = self._load_model(model_path)
        
        # Vehicle detector for initial detection
        self.detector = YOLO('yolov8n.pt')
        
        # Vehicle tracker
        self.tracker = VehicleTracker()
        
        # Frame buffer for sequence processing
        self.frame_buffer = deque(maxlen=sequence_length)
        self.frame_times = deque(maxlen=sequence_length)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # Vehicle classes (COCO format)
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        
    def _load_model(self, model_path):
        """Load the trained SpeedNet model"""
        # Create model architecture
        model = SpeedNet(
            backbone='yolov8n',
            sequence_length=self.sequence_length,
            confidence_threshold=self.confidence_threshold
        )
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def detect_vehicles(self, frame):
        """Detect vehicles in a single frame"""
        results = self.detector(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                if cls in self.vehicle_classes:
                    detections.append((box, conf, cls))
        
        return detections
    
    def estimate_speeds(self, active_tracks, current_frame):
        """Estimate speeds for active tracks using the neural network"""
        if not active_tracks or len(self.frame_buffer) < self.sequence_length:
            return {}
        
        speed_estimates = {}
        
        # Prepare sequence for neural network
        sequence_tensor = self._prepare_sequence()
        
        if sequence_tensor is None:
            return {}
        
        try:
            with torch.no_grad():
                # Run inference
                predictions = self.model(sequence_tensor.unsqueeze(0))  # Add batch dimension
                
                # Extract predictions for each track
                if predictions['predictions']:
                    for i, pred in enumerate(predictions['predictions']):
                        if i < len(active_tracks):
                            track = active_tracks[i]
                            track_id = track['track_id']
                            
                            speed = float(pred['speed'].cpu().numpy().squeeze())
                            uncertainty = float(pred['uncertainty'].cpu().numpy().squeeze())
                            
                            # Apply post-processing filters
                            speed = self._post_process_speed(speed, track_id, uncertainty)
                            
                            speed_estimates[track_id] = {
                                'speed': speed,
                                'uncertainty': uncertainty
                            }
                            
                            # Update tracker with speed
                            self.tracker.update_speed(track_id, speed)
        
        except Exception as e:
            print(f"Speed estimation error: {e}")
        
        return speed_estimates
    
    def _prepare_sequence(self):
        """Prepare frame sequence for neural network input"""
        if len(self.frame_buffer) < self.sequence_length:
            return None
        
        # Convert frames to tensor
        frames = list(self.frame_buffer)[-self.sequence_length:]
        
        # Resize and normalize
        processed_frames = []
        for frame in frames:
            # Resize to model input size
            resized = cv2.resize(frame, (640, 640))
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert BGR to RGB and to tensor
            rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # HWC -> CHW
            processed_frames.append(tensor)
        
        # Stack into sequence tensor [T, C, H, W]
        sequence = torch.stack(processed_frames)
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        sequence = (sequence - mean) / std
        
        return sequence.to(self.device)
    
    def _post_process_speed(self, speed, track_id, uncertainty):
        """Apply post-processing filters to speed estimates"""
        # Get track history
        track_history = self.tracker.get_track_history(track_id)
        
        if track_history is None:
            return speed
        
        # Temporal smoothing using previous speeds
        if track_id in self.tracker.tracks:
            previous_speeds = list(self.tracker.tracks[track_id]['speeds'])
            if previous_speeds:
                # Weighted average with more weight on recent speeds
                weights = np.exp(np.linspace(-1, 0, len(previous_speeds)))
                smoothed_speed = np.average(previous_speeds + [speed], 
                                          weights=np.concatenate([weights, [1.0]]))
                speed = smoothed_speed
        
        # Physical constraints
        speed = max(0.0, min(speed, 200.0))  # 0-200 km/h range
        
        # Low-speed filtering (reduce noise for stationary vehicles)
        if speed < 5.0 and uncertainty > 0.5:
            speed = 0.0
        
        return speed
    
    def draw_results(self, frame, active_tracks, speed_estimates):
        """Draw detection and speed results on frame"""
        height, width = frame.shape[:2]
        
        # Draw tracks and speeds
        for track in active_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            conf = track['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0) if track_id in speed_estimates else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"ID: {track_id}"
            if track_id in speed_estimates:
                speed = speed_estimates[track_id]['speed']
                uncertainty = speed_estimates[track_id]['uncertainty']
                label += f" | {speed:.1f} km/h"
                
                # Add uncertainty indicator
                if uncertainty < 0.3:
                    conf_text = "HIGH"
                    conf_color = (0, 255, 0)
                elif uncertainty < 0.7:
                    conf_text = "MED"
                    conf_color = (0, 165, 255)
                else:
                    conf_text = "LOW"
                    conf_color = (0, 0, 255)
                
                label += f" ({conf_text})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw track history (trajectory)
            track_history = self.tracker.get_track_history(track_id, max_length=10)
            if track_history and len(track_history['bboxes']) > 1:
                points = []
                for bbox_hist in track_history['bboxes']:
                    center_x = int((bbox_hist[0] + bbox_hist[2]) / 2)
                    center_y = int((bbox_hist[1] + bbox_hist[3]) / 2)
                    points.append((center_x, center_y))
                
                # Draw trajectory
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)
        
        # Draw statistics
        stats_text = [
            f"Frame: {self.stats['total_frames']}",
            f"Vehicles: {len(active_tracks)}",
            f"Speed estimates: {len(speed_estimates)}"
        ]
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            fps = 1.0 / (avg_time + 1e-6)
            stats_text.append(f"FPS: {fps:.1f}")
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        return frame
    
    def process_frame(self, frame, timestamp=None):
        """Process a single frame and return results"""
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        self.frame_times.append(timestamp)
        
        # Detect vehicles
        detections = self.detect_vehicles(frame)
        
        # Update tracker
        active_tracks = self.tracker.update(detections)
        
        # Estimate speeds (only if we have enough frames)
        speed_estimates = {}
        if len(self.frame_buffer) >= self.sequence_length:
            speed_estimates = self.estimate_speeds(active_tracks, frame)
        
        # Update statistics
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        
        return {
            'active_tracks': active_tracks,
            'speed_estimates': speed_estimates,
            'detections': detections,
            'processing_time': processing_time
        }
    
    def run_on_video(self, video_path, output_path=None, display=True):
        """Run pipeline on a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_idx / fps
                
                # Process frame
                results = self.process_frame(frame, timestamp)
                
                # Draw results
                output_frame = self.draw_results(
                    frame, 
                    results['active_tracks'], 
                    results['speed_estimates']
                )
                
                # Write output video
                if writer:
                    writer.write(output_frame)
                
                # Display
                if display:
                    cv2.imshow('Vehicle Speed Detection', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"Processed {frame_idx} frames")
        print(f"Average processing time: {np.mean(self.stats['processing_times']):.3f}s")
    
    def run_on_webcam(self, camera_id=0, display=True):
        """Run pipeline on webcam feed"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        print(f"Starting webcam feed from camera {camera_id}")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                timestamp = time.time()
                
                # Process frame
                results = self.process_frame(frame, timestamp)
                
                # Draw results
                output_frame = self.draw_results(
                    frame, 
                    results['active_tracks'], 
                    results['speed_estimates']
                )
                
                # Display
                if display:
                    cv2.imshow('Vehicle Speed Detection - Live', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

def main():
    """Main function for running the inference pipeline"""
    parser = argparse.ArgumentParser(description='Real-time Vehicle Speed Detection')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained SpeedNet model')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: camera ID, video file path, or RTSP URL')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Number of frames for temporal analysis')
    parser.add_argument('--no_display', action='store_true',
                        help='Disable video display')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpeedDetectionPipeline(
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence,
        sequence_length=args.sequence_length
    )
    
    # Determine source type
    source = args.source
    display = not args.no_display
    
    if source.isdigit():
        # Webcam
        camera_id = int(source)
        pipeline.run_on_webcam(camera_id=camera_id, display=display)
    else:
        # Video file or RTSP stream
        pipeline.run_on_video(
            video_path=source, 
            output_path=args.output, 
            display=display
        )

if __name__ == "__main__":
    main()