#!/usr/bin/env python3
"""
Camera Calibration System for Vehicle Speed Estimation
Supports multiple calibration methods including automatic vanishing point detection
"""

import numpy as np
import cv2
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import scipy.optimize as opt

class VanishingPointDetector:
    """
    Automatic vanishing point detection using lane line analysis
    """
    
    def __init__(self, 
                 canny_low=50, 
                 canny_high=150,
                 hough_threshold=100,
                 min_line_length=50,
                 max_line_gap=10):
        
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        
    def detect_lines(self, image):
        """Detect lines in the image using HoughLinesP"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        return lines, edges
    
    def filter_lane_lines(self, lines, image_shape):
        """Filter lines to keep only potential lane lines"""
        if lines is None:
            return []
        
        height, width = image_shape[:2]
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / math.pi
            
            # Filter by angle (typical lane lines are not horizontal)
            if abs(angle) > 15 and abs(angle) < 85:
                # Filter by position (should be in the road area)
                if y1 > height * 0.3 and y2 > height * 0.3:
                    filtered_lines.append(line[0])
        
        return filtered_lines
    
    def compute_vanishing_point(self, lines):
        """Compute vanishing point from filtered lines"""
        if len(lines) < 2:
            return None
        
        # Convert lines to infinite line representations
        infinite_lines = []
        for x1, y1, x2, y2 in lines:
            # Line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            # Normalize
            norm = math.sqrt(a*a + b*b)
            if norm > 1e-6:
                infinite_lines.append([a/norm, b/norm, c/norm])
        
        # Find intersections between all pairs of lines
        intersections = []
        for i in range(len(infinite_lines)):
            for j in range(i+1, len(infinite_lines)):
                line1 = infinite_lines[i]
                line2 = infinite_lines[j]
                
                # Solve system of equations
                A = np.array([[line1[0], line1[1]], 
                             [line2[0], line2[1]]])
                b = np.array([-line1[2], -line2[2]])
                
                try:
                    intersection = np.linalg.solve(A, b)
                    intersections.append(intersection)
                except np.linalg.LinAlgError:
                    continue
        
        if not intersections:
            return None
        
        # Cluster intersections to find the most likely vanishing point
        intersections = np.array(intersections)
        
        # Remove outliers using DBSCAN
        if len(intersections) > 3:
            clustering = DBSCAN(eps=50, min_samples=2).fit(intersections)
            labels = clustering.labels_
            
            # Find the largest cluster
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # Remove noise points
            
            if unique_labels:
                largest_cluster = max(unique_labels, 
                                    key=lambda label: np.sum(labels == label))
                cluster_points = intersections[labels == largest_cluster]
                vp = np.mean(cluster_points, axis=0)
            else:
                vp = np.mean(intersections, axis=0)
        else:
            vp = np.mean(intersections, axis=0)
        
        return vp
    
    def detect_vanishing_point(self, image):
        """Main method to detect vanishing point"""
        # Detect lines
        lines, edges = self.detect_lines(image)
        
        # Filter lane lines
        lane_lines = self.filter_lane_lines(lines, image.shape)
        
        # Compute vanishing point
        vp = self.compute_vanishing_point(lane_lines)
        
        return {
            'vanishing_point': vp,
            'lane_lines': lane_lines,
            'all_lines': lines,
            'edges': edges
        }

class CameraCalibrator:
    """
    Comprehensive camera calibration system
    """
    
    def __init__(self):
        self.vp_detector = VanishingPointDetector()
        self.calibration_data = {}
        
    def calibrate_from_vanishing_point(self, 
                                     image, 
                                     camera_height=1.5, 
                                     lane_width=3.5):
        """
        Calibrate camera using vanishing point detection
        
        Args:
            image: Input image
            camera_height: Camera height above ground (meters)
            lane_width: Known lane width (meters)
            
        Returns:
            Calibration parameters
        """
        height, width = image.shape[:2]
        
        # Detect vanishing point
        vp_result = self.vp_detector.detect_vanishing_point(image)
        vp = vp_result['vanishing_point']
        
        if vp is None:
            raise ValueError("Could not detect vanishing point")
        
        vp_x, vp_y = vp
        
        # Estimate focal length from vanishing point
        # Assuming principal point is at image center
        cx, cy = width / 2, height / 2
        
        # For a typical road scene, the horizon is at the vanishing point
        horizon_y = vp_y
        
        # Camera pitch angle (angle between camera optical axis and ground plane)
        pitch_angle = math.atan2(horizon_y - cy, height)
        
        # Focal length estimation using camera height and pitch
        # This is a simplified model
        f_y = camera_height / math.tan(pitch_angle + math.pi/2) if abs(pitch_angle) > 1e-6 else height
        f_x = f_y  # Assume square pixels
        
        # Camera intrinsic matrix
        K = np.array([
            [f_x, 0, cx],
            [0, f_y, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Estimate meters per pixel at different y coordinates
        mpp_samples = self._estimate_mpp_from_lane_width(
            image, vp_result['lane_lines'], lane_width, K, camera_height
        )
        
        calibration = {
            'intrinsic_matrix': K,
            'vanishing_point': np.array([vp_x, vp_y]),
            'camera_height': camera_height,
            'pitch_angle': pitch_angle,
            'principal_point': np.array([cx, cy]),
            'focal_length': np.array([f_x, f_y]),
            'mpp_samples': mpp_samples,
            'image_size': (width, height)
        }
        
        return calibration
    
    def _estimate_mpp_from_lane_width(self, image, lane_lines, lane_width, K, camera_height):
        """Estimate meters per pixel using known lane width"""
        if not lane_lines:
            return []
        
        height, width = image.shape[:2]
        mpp_samples = []
        
        # Group lines by y-coordinate bands
        bands = 5
        band_height = height // bands
        
        for band_idx in range(bands):
            y_start = band_idx * band_height
            y_end = (band_idx + 1) * band_height
            y_center = (y_start + y_end) / 2
            
            # Find lines in this band
            band_lines = []
            for x1, y1, x2, y2 in lane_lines:
                line_y = (y1 + y2) / 2
                if y_start <= line_y <= y_end:
                    band_lines.append((x1, y1, x2, y2))
            
            if len(band_lines) >= 2:
                # Find lane spacing in pixels
                line_positions = []
                for x1, y1, x2, y2 in band_lines:
                    # Project line to y_center
                    if abs(y2 - y1) > 1e-6:
                        t = (y_center - y1) / (y2 - y1)
                        x_at_y = x1 + t * (x2 - x1)
                        line_positions.append(x_at_y)
                
                if len(line_positions) >= 2:
                    line_positions.sort()
                    # Assume adjacent lines represent lane boundaries
                    for i in range(len(line_positions) - 1):
                        lane_width_px = abs(line_positions[i+1] - line_positions[i])
                        if lane_width_px > 20:  # Minimum reasonable lane width in pixels
                            mpp = lane_width / lane_width_px
                            mpp_samples.append((y_center, mpp))
        
        return mpp_samples
    
    def calibrate_from_chessboard(self, images, chessboard_size=(9, 6)):
        """
        Traditional chessboard calibration
        
        Args:
            images: List of chessboard images
            chessboard_size: (width, height) of inner corners
            
        Returns:
            Calibration parameters
        """
        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if len(objpoints) == 0:
            raise ValueError("No chessboard patterns found")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if not ret:
            raise ValueError("Camera calibration failed")
        
        return {
            'intrinsic_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'image_size': gray.shape[::-1]
        }
    
    def calibrate_from_ground_plane(self, 
                                   image,
                                   ground_points_image,
                                   ground_points_world,
                                   camera_height=1.5):
        """
        Calibrate using known ground plane points
        
        Args:
            image: Input image
            ground_points_image: Points in image coordinates
            ground_points_world: Corresponding points in world coordinates
            camera_height: Camera height above ground
            
        Returns:
            Calibration parameters including homography
        """
        # Convert points to numpy arrays
        img_pts = np.array(ground_points_image, dtype=np.float32)
        world_pts = np.array(ground_points_world, dtype=np.float32)
        
        # Compute homography from ground plane to image
        H, _ = cv2.findHomography(world_pts, img_pts, cv2.RANSAC)
        
        if H is None:
            raise ValueError("Could not compute homography")
        
        # Estimate intrinsic parameters (simplified)
        height, width = image.shape[:2]
        cx, cy = width / 2, height / 2
        
        # Rough focal length estimation
        f = max(width, height)  # Initial guess
        
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return {
            'intrinsic_matrix': K,
            'homography': H,
            'camera_height': camera_height,
            'ground_points_image': img_pts,
            'ground_points_world': world_pts,
            'image_size': (width, height)
        }
    
    def world_to_image(self, world_points, calibration):
        """Convert world coordinates to image coordinates"""
        if 'homography' in calibration:
            # Use homography for ground plane points
            H = calibration['homography']
            world_pts = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)
            image_pts = cv2.perspectiveTransform(world_pts, H)
            return image_pts.reshape(-1, 2)
        else:
            # Use intrinsic matrix and estimated depth
            raise NotImplementedError("3D projection not implemented yet")
    
    def image_to_world(self, image_points, calibration, z=0):
        """Convert image coordinates to world coordinates (on ground plane)"""
        if 'homography' in calibration:
            # Use inverse homography
            H = calibration['homography']
            H_inv = np.linalg.inv(H)
            img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
            world_pts = cv2.perspectiveTransform(img_pts, H_inv)
            return world_pts.reshape(-1, 2)
        else:
            # Use intrinsic parameters and vanishing point geometry
            K = calibration['intrinsic_matrix']
            vp = calibration.get('vanishing_point')
            camera_height = calibration.get('camera_height', 1.5)
            
            world_points = []
            for img_pt in image_points:
                # This is a simplified ground plane projection
                # In practice, you'd use more sophisticated geometry
                u, v = img_pt
                
                # Normalize image coordinates
                x_norm = (u - K[0, 2]) / K[0, 0]
                y_norm = (v - K[1, 2]) / K[1, 1]
                
                # Simple ground plane intersection
                # Assuming camera looks down at angle
                if abs(y_norm) > 1e-6:
                    depth = camera_height / y_norm
                    world_x = depth * x_norm
                    world_y = depth
                    world_points.append([world_x, world_y])
                else:
                    world_points.append([0, 0])
            
            return np.array(world_points)
    
    def estimate_speed_from_displacement(self, 
                                       displacement_pixels,
                                       y_position,
                                       calibration,
                                       time_delta,
                                       direction_angle=0):
        """
        Estimate speed from pixel displacement
        
        Args:
            displacement_pixels: Movement in pixels between frames
            y_position: Y coordinate in image
            calibration: Camera calibration data
            time_delta: Time between frames (seconds)
            direction_angle: Direction of movement (radians)
            
        Returns:
            Speed in m/s
        """
        # Get meters per pixel at this y position
        if 'mpp_samples' in calibration and calibration['mpp_samples']:
            # Interpolate mpp based on y position
            mpp_samples = calibration['mpp_samples']
            if len(mpp_samples) == 1:
                mpp = mpp_samples[0][1]
            else:
                y_coords = [sample[0] for sample in mpp_samples]
                mpp_values = [sample[1] for sample in mpp_samples]
                mpp = np.interp(y_position, y_coords, mpp_values)
        else:
            # Fallback to homography or simple estimation
            if 'homography' in calibration:
                # Use homography to estimate scale
                test_points = np.array([[0, y_position], [1, y_position]])
                world_points = self.image_to_world(test_points, calibration)
                mpp = np.linalg.norm(world_points[1] - world_points[0])
            else:
                # Very rough estimation
                height = calibration['image_size'][1]
                camera_height = calibration.get('camera_height', 1.5)
                mpp = camera_height / (height - y_position + 1e-6)
        
        # Convert displacement to meters
        displacement_meters = displacement_pixels * mpp
        
        # Calculate speed
        speed_ms = displacement_meters / (time_delta + 1e-6)
        
        return speed_ms
    
    def save_calibration(self, calibration, filepath):
        """Save calibration to file"""
        # Convert numpy arrays to lists for JSON serialization
        calibration_json = {}
        for key, value in calibration.items():
            if isinstance(value, np.ndarray):
                calibration_json[key] = value.tolist()
            else:
                calibration_json[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(calibration_json, f, indent=2)
    
    def load_calibration(self, filepath):
        """Load calibration from file"""
        with open(filepath, 'r') as f:
            calibration_json = json.load(f)
        
        # Convert lists back to numpy arrays
        calibration = {}
        for key, value in calibration_json.items():
            if key in ['intrinsic_matrix', 'homography', 'vanishing_point', 
                      'principal_point', 'focal_length']:
                calibration[key] = np.array(value)
            else:
                calibration[key] = value
        
        return calibration
    
    def visualize_calibration(self, image, calibration, save_path=None):
        """Visualize calibration results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with vanishing point
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if 'vanishing_point' in calibration:
            vp = calibration['vanishing_point']
            axes[0].plot(vp[0], vp[1], 'ro', markersize=10, label='Vanishing Point')
            
            # Draw horizon line
            height, width = image.shape[:2]
            axes[0].axhline(y=vp[1], color='r', linestyle='--', alpha=0.7, label='Horizon')
        
        axes[0].set_title('Original Image with Calibration')
        axes[0].legend()
        axes[0].axis('off')
        
        # Calibration parameters text
        calib_text = []
        if 'intrinsic_matrix' in calibration:
            K = calibration['intrinsic_matrix']
            calib_text.append(f"Focal Length: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
            calib_text.append(f"Principal Point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        if 'camera_height' in calibration:
            calib_text.append(f"Camera Height: {calibration['camera_height']:.2f}m")
        
        if 'pitch_angle' in calibration:
            pitch_deg = math.degrees(calibration['pitch_angle'])
            calib_text.append(f"Pitch Angle: {pitch_deg:.1f}°")
        
        if 'mpp_samples' in calibration:
            calib_text.append(f"MpP Samples: {len(calibration['mpp_samples'])}")
        
        axes[1].text(0.1, 0.5, '\n'.join(calib_text), 
                    transform=axes[1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('Calibration Parameters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

def main():
    """Example usage of the calibration system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera calibration for speed estimation')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--method', type=str, choices=['vanishing_point', 'chessboard', 'ground_plane'],
                       default='vanishing_point', help='Calibration method')
    parser.add_argument('--camera_height', type=float, default=1.5, help='Camera height in meters')
    parser.add_argument('--lane_width', type=float, default=3.5, help='Lane width in meters')
    parser.add_argument('--output', type=str, default='calibration.json', help='Output calibration file')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not load image: {args.image}")
    
    # Initialize calibrator
    calibrator = CameraCalibrator()
    
    # Perform calibration
    if args.method == 'vanishing_point':
        calibration = calibrator.calibrate_from_vanishing_point(
            image, args.camera_height, args.lane_width
        )
    else:
        raise NotImplementedError(f"Method {args.method} not implemented in this example")
    
    # Save calibration
    calibrator.save_calibration(calibration, args.output)
    print(f"Calibration saved to {args.output}")
    
    # Visualize if requested
    if args.visualize:
        calibrator.visualize_calibration(image, calibration)

if __name__ == "__main__":
    main()