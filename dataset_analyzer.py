#!/usr/bin/env python3
"""
BrnCompSpeed Dataset Analyzer
Analyzes the dataset structure and extracts key insights for training
"""

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

class BrnCompSpeedAnalyzer:
    def __init__(self, dataset_root):
        """
        Initialize analyzer with dataset root path
        Args:
            dataset_root: Path to brno_kaggle_subset/dataset/
        """
        self.dataset_root = Path(dataset_root)
        self.sessions = []
        self.ground_truth_data = {}
        
    def scan_dataset(self):
        """Scan dataset and collect all session information"""
        print("Scanning dataset structure...")
        
        for session_dir in self.dataset_root.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session'):
                session_info = {
                    'name': session_dir.name,
                    'path': session_dir,
                    'files': {}
                }
                
                # Check for required files
                required_files = ['gt_data.pkl', 'video.avi', 'video_mask.png', 'screen.png']
                for file_name in required_files:
                    file_path = session_dir / file_name
                    session_info['files'][file_name] = file_path if file_path.exists() else None
                
                self.sessions.append(session_info)
                
        print(f"Found {len(self.sessions)} sessions")
        return self.sessions
    
    def load_ground_truth(self, session_name):
        """Load and parse ground truth data for a session"""
        session = next((s for s in self.sessions if s['name'] == session_name), None)
        if not session:
            raise ValueError(f"Session {session_name} not found")
            
        gt_file = session['files']['gt_data.pkl']
        if not gt_file or not gt_file.exists():
            raise FileNotFoundError(f"Ground truth file not found for {session_name}")
            
        with open(gt_file, 'rb') as f:
            gt_data = pickle.load(f)
            
        self.ground_truth_data[session_name] = gt_data
        return gt_data
    
    def analyze_ground_truth(self, session_name):
        """Analyze ground truth data structure and extract insights"""
        if session_name not in self.ground_truth_data:
            self.load_ground_truth(session_name)
            
        gt = self.ground_truth_data[session_name]
        
        analysis = {
            'fps': gt.get('fps', 25.0),
            'num_cars': len(gt.get('cars', [])),
            'num_measurement_lines': len(gt.get('measurementLines', [])),
            'num_distance_measurements': len(gt.get('distanceMeasurement', [])),
            'lane_div_lines': len(gt.get('laneDivLines', [])),
            'invalid_lanes': len(gt.get('invalidLanes', set())),
            'speed_stats': {},
            'temporal_stats': {},
            'spatial_stats': {}
        }
        
        # Analyze car data
        cars = gt.get('cars', [])
        if cars:
            speeds = [car['speed'] for car in cars if car.get('valid', True)]
            analysis['speed_stats'] = {
                'min_speed': min(speeds) if speeds else 0,
                'max_speed': max(speeds) if speeds else 0,
                'mean_speed': np.mean(speeds) if speeds else 0,
                'std_speed': np.std(speeds) if speeds else 0,
                'valid_cars': len(speeds),
                'invalid_cars': len(cars) - len(speeds)
            }
            
            # Temporal analysis
            all_times = []
            for car in cars:
                intersections = car.get('intersections', [])
                for intersection in intersections:
                    all_times.append(intersection.get('videoTime', 0))
            
            if all_times:
                analysis['temporal_stats'] = {
                    'min_time': min(all_times),
                    'max_time': max(all_times),
                    'total_duration': max(all_times) - min(all_times) if all_times else 0
                }
        
        # Analyze measurement lines and spatial relationships
        measurement_lines = gt.get('measurementLines', [])
        distance_measurements = gt.get('distanceMeasurement', [])
        
        if distance_measurements:
            distances = [dm['distance'] for dm in distance_measurements]
            analysis['spatial_stats'] = {
                'unique_distances': list(set(distances)),
                'distance_counts': {d: distances.count(d) for d in set(distances)}
            }
        
        return analysis
    
    def extract_training_data(self, session_name, output_dir):
        """Extract training data from a session"""
        if session_name not in self.ground_truth_data:
            self.load_ground_truth(session_name)
            
        gt = self.ground_truth_data[session_name]
        session = next(s for s in self.sessions if s['name'] == session_name)
        
        output_path = Path(output_dir) / session_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract video info
        video_path = session['files']['video.avi']
        if video_path and video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            video_info = {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration_seconds': frame_count / fps if fps > 0 else 0
            }
        else:
            video_info = {}
        
        # Process car trajectories for training
        cars = gt.get('cars', [])
        training_samples = []
        
        for car in cars:
            if not car.get('valid', True):
                continue
                
            car_data = {
                'car_id': car.get('carId', -1),
                'lane_index': list(car.get('laneIndex', set())),
                'speed_kmh': car.get('speed', 0.0),
                'intersections': car.get('intersections', []),
                'time_last_shifted': car.get('timeIntersectionLastShifted', 0.0)
            }
            
            # Calculate trajectory features
            intersections = car.get('intersections', [])
            if len(intersections) >= 2:
                # Time between first and last measurement line
                times = [i['videoTime'] for i in intersections]
                car_data['trajectory_duration'] = max(times) - min(times)
                car_data['measurement_line_count'] = len(intersections)
                
                # Calculate distances between measurement lines
                measurement_lines = gt.get('measurementLines', [])
                if len(measurement_lines) >= 2:
                    # Use distance measurements to get real-world distances
                    distance_measurements = gt.get('distanceMeasurement', [])
                    distances = [dm['distance'] for dm in distance_measurements]
                    car_data['total_distance_m'] = sum(set(distances))  # Remove duplicates
            
            training_samples.append(car_data)
        
        # Save processed data
        output_data = {
            'session_name': session_name,
            'video_info': video_info,
            'measurement_lines': gt.get('measurementLines', []),
            'distance_measurements': gt.get('distanceMeasurement', []),
            'lane_div_lines': gt.get('laneDivLines', []),
            'training_samples': training_samples,
            'fps': gt.get('fps', 25.0)
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        output_data = convert_numpy(output_data)
        
        with open(output_path / 'processed_data.json', 'w') as f:
            json.dump(output_data, f, indent=2)
            
        return output_data
    
    def visualize_analysis(self, session_name, save_path=None):
        """Create visualizations of the analysis"""
        analysis = self.analyze_ground_truth(session_name)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Dataset Analysis: {session_name}', fontsize=16)
        
        # Speed distribution
        if session_name in self.ground_truth_data:
            gt = self.ground_truth_data[session_name]
            cars = gt.get('cars', [])
            speeds = [car['speed'] for car in cars if car.get('valid', True)]
            
            if speeds:
                axes[0, 0].hist(speeds, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Speed Distribution (km/h)')
                axes[0, 0].set_xlabel('Speed (km/h)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].axvline(np.mean(speeds), color='red', linestyle='--', 
                                 label=f'Mean: {np.mean(speeds):.1f} km/h')
                axes[0, 0].legend()
        
        # Lane distribution
        if session_name in self.ground_truth_data:
            lane_counts = {}
            for car in cars:
                if car.get('valid', True):
                    lanes = car.get('laneIndex', set())
                    for lane in lanes:
                        lane_counts[lane] = lane_counts.get(lane, 0) + 1
            
            if lane_counts:
                lanes = list(lane_counts.keys())
                counts = list(lane_counts.values())
                axes[0, 1].bar(lanes, counts)
                axes[0, 1].set_title('Vehicle Count by Lane')
                axes[0, 1].set_xlabel('Lane Index')
                axes[0, 1].set_ylabel('Vehicle Count')
        
        # Temporal distribution
        if session_name in self.ground_truth_data:
            all_times = []
            for car in cars:
                if car.get('valid', True):
                    intersections = car.get('intersections', [])
                    for intersection in intersections:
                        all_times.append(intersection.get('videoTime', 0))
            
            if all_times:
                axes[1, 0].hist(all_times, bins=30, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Temporal Distribution of Vehicle Detections')
                axes[1, 0].set_xlabel('Video Time (seconds)')
                axes[1, 0].set_ylabel('Detection Count')
        
        # Summary statistics
        stats_text = f"""
        Total Cars: {analysis['num_cars']}
        Valid Cars: {analysis['speed_stats'].get('valid_cars', 0)}
        Measurement Lines: {analysis['num_measurement_lines']}
        Distance Measurements: {analysis['num_distance_measurements']}
        
        Speed Statistics:
        Min: {analysis['speed_stats'].get('min_speed', 0):.1f} km/h
        Max: {analysis['speed_stats'].get('max_speed', 0):.1f} km/h
        Mean: {analysis['speed_stats'].get('mean_speed', 0):.1f} km/h
        Std: {analysis['speed_stats'].get('std_speed', 0):.1f} km/h
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return analysis

def main():
    """Example usage of the analyzer"""
    # Initialize analyzer
    dataset_root = "path/to/brno_kaggle_subset/dataset"  # Update this path
    analyzer = BrnCompSpeedAnalyzer(dataset_root)
    
    # Scan dataset
    sessions = analyzer.scan_dataset()
    
    # Analyze each session
    for session in sessions:
        print(f"\nAnalyzing {session['name']}...")
        try:
            analysis = analyzer.analyze_ground_truth(session['name'])
            print(f"Speed range: {analysis['speed_stats'].get('min_speed', 0):.1f} - {analysis['speed_stats'].get('max_speed', 0):.1f} km/h")
            print(f"Valid cars: {analysis['speed_stats'].get('valid_cars', 0)}")
            
            # Extract training data
            analyzer.extract_training_data(session['name'], "processed_data")
            
            # Create visualization
            analyzer.visualize_analysis(session['name'], f"analysis_{session['name']}.png")
            
        except Exception as e:
            print(f"Error analyzing {session['name']}: {e}")

if __name__ == "__main__":
    main()