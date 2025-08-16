#!/usr/bin/env python3
"""
SpeedNet Demo Script
Comprehensive demonstration of the 3D-aware vehicle speed estimation system
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# Import our modules
from dataset_analyzer import BrnCompSpeedAnalyzer
from models.speednet import SpeedNet, SpeedNetLoss
from train_speednet import SpeedNetTrainer, BrnCompSpeedDataset, create_data_transforms
from inference_pipeline import SpeedDetectionPipeline
from calibration import CameraCalibrator, VanishingPointDetector

class SpeedNetDemo:
    """
    Complete demonstration of SpeedNet capabilities
    """
    
    def __init__(self):
        self.analyzer = None
        self.model = None
        self.pipeline = None
        self.calibrator = CameraCalibrator()
        
    def demo_dataset_analysis(self, dataset_root):
        """Demonstrate dataset analysis capabilities"""
        print("=" * 60)
        print("SPEEDNET DEMO: Dataset Analysis")
        print("=" * 60)
        
        # Initialize analyzer
        self.analyzer = BrnCompSpeedAnalyzer(dataset_root)
        
        # Scan dataset
        sessions = self.analyzer.scan_dataset()
        print(f"Found {len(sessions)} sessions in dataset")
        
        # Analyze first available session
        if sessions:
            session_name = sessions[0]['name']
            print(f"\nAnalyzing session: {session_name}")
            
            try:
                # Load and analyze ground truth
                analysis = self.analyzer.analyze_ground_truth(session_name)
                
                print(f"  Total cars: {analysis['num_cars']}")
                print(f"  Valid cars: {analysis['speed_stats'].get('valid_cars', 0)}")
                print(f"  Speed range: {analysis['speed_stats'].get('min_speed', 0):.1f} - {analysis['speed_stats'].get('max_speed', 0):.1f} km/h")
                print(f"  Mean speed: {analysis['speed_stats'].get('mean_speed', 0):.1f} km/h")
                
                # Create visualization
                print("  Generating analysis visualization...")
                self.analyzer.visualize_analysis(session_name, f"demo_analysis_{session_name}.png")
                
                return analysis
                
            except Exception as e:
                print(f"  Error analyzing session: {e}")
                return None
        else:
            print("No sessions found in dataset")
            return None
    
    def demo_camera_calibration(self, image_path):
        """Demonstrate camera calibration"""
        print("\n" + "=" * 60)
        print("SPEEDNET DEMO: Camera Calibration")
        print("=" * 60)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # Load image
        image = cv2.imread(image_path)
        print(f"Loaded image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        try:
            # Perform automatic calibration
            print("Performing automatic vanishing point calibration...")
            calibration = self.calibrator.calibrate_from_vanishing_point(
                image, camera_height=1.5, lane_width=3.5
            )
            
            # Print calibration results
            vp = calibration['vanishing_point']
            print(f"  Vanishing point: ({vp[0]:.1f}, {vp[1]:.1f})")
            print(f"  Camera height: {calibration['camera_height']:.2f}m")
            print(f"  Focal length: fx={calibration['focal_length'][0]:.1f}, fy={calibration['focal_length'][1]:.1f}")
            print(f"  Pitch angle: {np.degrees(calibration['pitch_angle']):.1f}°")
            print(f"  MpP samples: {len(calibration['mpp_samples'])}")
            
            # Save calibration
            calib_path = "demo_calibration.json"
            self.calibrator.save_calibration(calibration, calib_path)
            print(f"  Calibration saved to: {calib_path}")
            
            # Visualize calibration
            print("  Generating calibration visualization...")
            self.calibrator.visualize_calibration(image, calibration, "demo_calibration.png")
            
            return calibration
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return None
    
    def demo_model_architecture(self):
        """Demonstrate model architecture"""
        print("\n" + "=" * 60)
        print("SPEEDNET DEMO: Model Architecture")  
        print("=" * 60)
        
        # Create model
        model = SpeedNet(
            backbone='yolov8n',
            sequence_length=8,
            confidence_threshold=0.5
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: SpeedNet")
        print(f"  Backbone: YOLOv8n")
        print(f"  Sequence length: 8 frames")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Show model components
        print(f"\nModel Components:")
        print(f"  • Vehicle Detection: YOLOv8n (frozen)")
        print(f"  • Camera Calibration: Neural network for vanishing point + parameters")
        print(f"  • Vehicle 3D Module: 3D bounding box + depth estimation")
        print(f"  • Temporal Fusion: Bidirectional LSTM + Multi-head attention")
        print(f"  • Speed Regression: MLP with uncertainty estimation")
        
        # Demo forward pass with dummy data
        print(f"\nTesting forward pass...")
        dummy_input = torch.randn(1, 8, 3, 640, 640)  # [B, T, C, H, W]
        
        try:
            with torch.no_grad():
                model.eval()
                output = model(dummy_input)
                print(f"  Forward pass successful!")
                print(f"  Predictions: {len(output['predictions'])} vehicles detected")
                print(f"  Camera parameters: {len(output['camera_params'])} parameters")
        except Exception as e:
            print(f"  Forward pass failed: {e}")
        
        self.model = model
        return model
    
    def demo_training_setup(self, dataset_root):
        """Demonstrate training setup (without actual training)"""
        print("\n" + "=" * 60)
        print("SPEEDNET DEMO: Training Setup")
        print("=" * 60)
        
        if not os.path.exists(dataset_root):
            print(f"Dataset not found: {dataset_root}")
            return None
        
        try:
            # Create data transforms
            train_transform, val_transform = create_data_transforms()
            print("Created data augmentation transforms")
            
            # Create datasets (small subset for demo)
            print("Creating demo datasets...")
            train_dataset = BrnCompSpeedDataset(
                dataset_root=dataset_root,
                sequence_length=8,
                image_size=(640, 640),
                transform=train_transform,
                mode='train'
            )
            
            val_dataset = BrnCompSpeedDataset(
                dataset_root=dataset_root,
                sequence_length=8,
                image_size=(640, 640),
                transform=val_transform,
                mode='val'
            )
            
            print(f"  Training samples: {len(train_dataset)}")
            print(f"  Validation samples: {len(val_dataset)}")
            
            # Demo data loading
            if len(train_dataset) > 0:
                print("Testing data loading...")
                sample_video, sample_target = train_dataset[0]
                print(f"  Video sequence shape: {sample_video.shape}")
                print(f"  Target speed: {sample_target['speed']:.1f} km/h")
                print(f"  Car ID: {sample_target['car_id']}")
                
            # Create loss function
            criterion = SpeedNetLoss(
                speed_weight=1.0,
                uncertainty_weight=0.1,
                camera_weight=0.5,
                depth_weight=0.3
            )
            print("Created multi-task loss function")
            
            return {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'criterion': criterion
            }
            
        except Exception as e:
            print(f"Training setup failed: {e}")
            return None
    
    def demo_inference(self, video_path, model_path=None):
        """Demonstrate real-time inference"""
        print("\n" + "=" * 60)
        print("SPEEDNET DEMO: Real-time Inference")
        print("=" * 60)
        
        # Create inference pipeline
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading trained model from: {model_path}")
            else:
                print("Using untrained model for demo (speeds will be random)")
                model_path = "untrained_model.pth"
                
            self.pipeline = SpeedDetectionPipeline(
                model_path=model_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                confidence_threshold=0.5,
                sequence_length=8
            )
            
            print(f"Pipeline initialized on device: {self.pipeline.device}")
            
            # Test inference
            if os.path.exists(video_path):
                print(f"Processing video: {video_path}")
                
                # Process a short segment for demo
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                frame_count = 0
                max_frames = 100  # Process only first 100 frames for demo
                
                results_summary = {
                    'total_frames': 0,
                    'total_detections': 0,
                    'unique_vehicles': set(),
                    'speed_estimates': []
                }
                
                print("Processing frames...")
                start_time = time.time()
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = frame_count / fps
                    results = self.pipeline.process_frame(frame, timestamp)
                    
                    # Collect statistics
                    results_summary['total_frames'] += 1
                    results_summary['total_detections'] += len(results['detections'])
                    
                    for track in results['active_tracks']:
                        results_summary['unique_vehicles'].add(track['track_id'])
                    
                    for track_id, speed_info in results['speed_estimates'].items():
                        results_summary['speed_estimates'].append(speed_info['speed'])
                    
                    frame_count += 1
                    
                    if frame_count % 25 == 0:  # Progress every second
                        print(f"  Processed {frame_count} frames...")
                
                cap.release()
                
                # Print results
                processing_time = time.time() - start_time
                avg_fps = frame_count / processing_time
                
                print(f"\nInference Results:")
                print(f"  Processed frames: {results_summary['total_frames']}")
                print(f"  Total detections: {results_summary['total_detections']}")
                print(f"  Unique vehicles: {len(results_summary['unique_vehicles'])}")
                print(f"  Speed estimates: {len(results_summary['speed_estimates'])}")
                print(f"  Processing time: {processing_time:.2f}s")
                print(f"  Average FPS: {avg_fps:.1f}")
                
                if results_summary['speed_estimates']:
                    speeds = results_summary['speed_estimates']
                    print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h")
                    print(f"  Mean speed: {np.mean(speeds):.1f} km/h")
                
                return results_summary
                
            else:
                print(f"Video not found: {video_path}")
                print("Demo will use webcam if available...")
                
                # Try webcam demo
                try:
                    print("Testing webcam access...")
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        print("Webcam available! Run with --live for webcam demo")
                    else:
                        print("No webcam available")
                        
                except Exception as e:
                    print(f"Webcam test failed: {e}")
                
                return None
                
        except Exception as e:
            print(f"Inference demo failed: {e}")
            return None
    
    def run_full_demo(self, 
                     dataset_root=None, 
                     image_path=None, 
                     video_path=None, 
                     model_path=None):
        """Run complete SpeedNet demonstration"""
        print("🚗 SPEEDNET COMPLETE DEMONSTRATION 🚗")
        print("3D-Aware Vehicle Speed Estimation from Video")
        print("=" * 60)
        
        results = {}
        
        # 1. Dataset Analysis
        if dataset_root and os.path.exists(dataset_root):
            results['dataset_analysis'] = self.demo_dataset_analysis(dataset_root)
        else:
            print("Skipping dataset analysis (dataset not provided)")
        
        # 2. Camera Calibration
        if image_path and os.path.exists(image_path):
            results['calibration'] = self.demo_camera_calibration(image_path)
        else:
            print("Skipping camera calibration (image not provided)")
        
        # 3. Model Architecture
        results['model'] = self.demo_model_architecture()
        
        # 4. Training Setup
        if dataset_root and os.path.exists(dataset_root):
            results['training_setup'] = self.demo_training_setup(dataset_root)
        else:
            print("Skipping training setup (dataset not provided)")
        
        # 5. Inference Demo
        if video_path:
            results['inference'] = self.demo_inference(video_path, model_path)
        else:
            print("Skipping inference demo (video not provided)")
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        
        completed_demos = [key for key, value in results.items() if value is not None]
        print(f"Completed demos: {len(completed_demos)}/5")
        
        for demo in completed_demos:
            print(f"  ✓ {demo.replace('_', ' ').title()}")
        
        failed_demos = [key for key, value in results.items() if value is None]
        for demo in failed_demos:
            print(f"  ✗ {demo.replace('_', ' ').title()}")
        
        print(f"\nGenerated files:")
        generated_files = [
            "demo_analysis_session0_center.png",
            "demo_calibration.json", 
            "demo_calibration.png"
        ]
        
        for file in generated_files:
            if os.path.exists(file):
                print(f"  • {file}")
        
        print(f"\nSpeedNet Demo Complete! 🎉")
        
        return results

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='SpeedNet Complete Demonstration')
    parser.add_argument('--dataset_root', type=str, 
                       help='Path to BrnCompSpeed dataset root')
    parser.add_argument('--image', type=str,
                       help='Path to road image for calibration demo')
    parser.add_argument('--video', type=str,
                       help='Path to video file for inference demo')
    parser.add_argument('--model', type=str,
                       help='Path to trained model (optional)')
    parser.add_argument('--live', action='store_true',
                       help='Run live webcam demo')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = SpeedNetDemo()
    
    # Run live webcam demo if requested
    if args.live:
        print("Starting live webcam demo...")
        try:
            pipeline = SpeedDetectionPipeline(
                model_path=args.model or "untrained_model.pth",
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            pipeline.run_on_webcam(camera_id=0, display=True)
        except Exception as e:
            print(f"Live demo failed: {e}")
        return
    
    # Run complete demo
    results = demo.run_full_demo(
        dataset_root=args.dataset_root,
        image_path=args.image,
        video_path=args.video,
        model_path=args.model
    )

if __name__ == "__main__":
    main()