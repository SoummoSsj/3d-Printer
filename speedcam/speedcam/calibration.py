import cv2
import numpy as np
from skimage.transform import probabilistic_hough_line
from dataclasses import dataclass
from typing import Optional, Tuple, List


def _intersect_lines(l1, l2) -> Optional[np.ndarray]:
	p = np.cross(l1, l2)
	if abs(p[2]) < 1e-8:
		return None
	return p / p[2]


def _line_through_points(p1, p2) -> np.ndarray:
	return np.cross(p1, p2)


def _cluster_directions(lines: List[Tuple[Tuple[int,int],Tuple[int,int]]], k: int = 2) -> List[List[int]]:
	angles = []
	for (x1,y1),(x2,y2) in lines:
		dx, dy = x2-x1, y2-y1
		ang = np.arctan2(dy, dx)
		ang = (ang + np.pi) % np.pi
		angles.append([np.cos(2*ang), np.sin(2*ang)])
	angles = np.asarray(angles, dtype=float)
	if len(angles) < k:
		return [list(range(len(angles)))]
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
	_, labels, _ = cv2.kmeans(angles.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
	clusters = [[] for _ in range(k)]
	for i, lab in enumerate(labels.ravel()):
		clusters[lab].append(i)
	return clusters


def _vanishing_point_from_cluster(lines: List[Tuple[Tuple[int,int],Tuple[int,int]]], idxs: List[int], img_shape) -> Optional[np.ndarray]:
	intersections = []
	M = min(len(idxs), 100)
	for i in range(M):
		for j in range(i+1, M):
			(x1,y1),(x2,y2) = lines[idxs[i]]
			(x3,y3),(x4,y4) = lines[idxs[j]]
			p1 = np.array([x1,y1,1.0]); p2 = np.array([x2,y2,1.0])
			p3 = np.array([x3,y3,1.0]); p4 = np.array([x4,y4,1.0])
			l1 = _line_through_points(p1,p2)
			l2 = _line_through_points(p3,p4)
			p = _intersect_lines(l1,l2)
			if p is not None:
				intersections.append(p[:2])
	if len(intersections) < 10:
		return None
	pts = np.asarray(intersections, dtype=float)
	vp = np.median(pts, axis=0)
	return np.array([vp[0], vp[1], 1.0])


def detect_vanishing_points(frame_gray: np.ndarray) -> Optional[Tuple[np.ndarray,np.ndarray]]:
	edges = cv2.Canny(frame_gray, 100, 200)
	lines = probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=3)
	if not lines:
		return None
	clusters = _cluster_directions(lines, k=2)
	if len(clusters) < 2:
		return None
	vp1 = _vanishing_point_from_cluster(lines, clusters[0], frame_gray.shape)
	vp2 = _vanishing_point_from_cluster(lines, clusters[1], frame_gray.shape)
	if vp1 is None or vp2 is None:
		return None
	return vp1, vp2


def _estimate_f_from_orthogonal_vps(vp1: np.ndarray, vp2: np.ndarray, cx: float, cy: float) -> Optional[float]:
	dx1 = vp1[0] - cx; dy1 = vp1[1] - cy
	dx2 = vp2[0] - cx; dy2 = vp2[1] - cy
	val = - (dx1*dx2 + dy1*dy2)
	if val <= 1.0:
		return None
	return float(np.sqrt(val))


def _normalize(v):
	n = np.linalg.norm(v)
	if n < 1e-9:
		return v
	return v / n

@dataclass
class CalibrationState:
	vp1: Optional[np.ndarray] = None
	vp2: Optional[np.ndarray] = None
	K: Optional[np.ndarray] = None
	K_inv: Optional[np.ndarray] = None
	n_plane: Optional[np.ndarray] = None		# plane normal in camera frame (unit)
	plane_offset: float = 1.0				# n^T X + d = 0, with d>0 arbitrary (absorbed by scale)
	scale_m_per_unit: Optional[float] = None
	confidence: float = 0.0


class SelfCalibrator:
	def __init__(self):
		self.state = CalibrationState()
		self._vp_buffer: List[Tuple[np.ndarray,np.ndarray]] = []
		self._scale_buffer: List[float] = []

	def _update_intrinsics_and_plane(self, frame_shape):
		if self.state.vp1 is None or self.state.vp2 is None:
			return
		h, w = frame_shape[:2]
		cx, cy = w*0.5, h*0.5
		f = _estimate_f_from_orthogonal_vps(self.state.vp1, self.state.vp2, cx, cy)
		if f is None:
			f = 1.5 * max(w, h)
		K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=float)
		K_inv = np.linalg.inv(K)
		d1 = _normalize(K_inv @ self.state.vp1)
		d2 = _normalize(K_inv @ self.state.vp2)
		n = _normalize(np.cross(d1, d2))
		self.state.K = K
		self.state.K_inv = K_inv
		self.state.n_plane = n

	def _ray_plane_point(self, p_img: Tuple[float,float]) -> Optional[np.ndarray]:
		if self.state.K_inv is None or self.state.n_plane is None:
			return None
		r = self.state.K_inv @ np.array([p_img[0], p_img[1], 1.0])
		r = _normalize(r)
		n = self.state.n_plane
		d = self.state.plane_offset
		den = float(n.dot(r))
		if abs(den) < 1e-6:
			return None
		lam = -d / den
		X = lam * r
		return X

	def estimate_scale_from_lane_priors(self, frame: np.ndarray, lane_width_m: float = 3.5) -> Optional[float]:
		h, w = frame.shape[:2]
		row = int(h*0.75)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 50, 150)
		xs = np.where(edges[row] > 0)[0]
		if xs.size < 2:
			return None
		x_left, x_right = xs.min(), xs.max()
		P1 = self._ray_plane_point((x_left, row))
		P2 = self._ray_plane_point((x_right, row))
		if P1 is None or P2 is None:
			return None
		d_units = float(np.linalg.norm(P2 - P1))
		if d_units < 1e-4:
			return None
		return lane_width_m / d_units

	def estimate_scale_from_vehicle_priors(self, detections: List[Tuple[int,int,int,int]], avg_width_m: float = 1.85) -> Optional[float]:
		units = []
		for x1,y1,x2,y2 in detections:
			P1 = self._ray_plane_point((x1, y2))
			P2 = self._ray_plane_point((x2, y2))
			if P1 is None or P2 is None:
				continue
			w_units = float(np.linalg.norm(P2 - P1))
			if w_units > 1e-4:
				units.append(w_units)
		if not units:
			return None
		return avg_width_m / float(np.median(units))

	def _fuse_scales(self, s1: Optional[float], s2: Optional[float]) -> Optional[float]:
		cands = [s for s in [s1, s2] if s is not None and np.isfinite(s) and s > 0]
		if not cands:
			return None
		return float(np.median(cands))

	def update(self, frame_bgr: np.ndarray, current_detections: List[Tuple[int,int,int,int]], lane_width_m: float = 3.5, avg_vehicle_width_m: float = 1.85) -> CalibrationState:
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		vps = detect_vanishing_points(gray)
		if vps is not None:
			vp1, vp2 = vps
			self._vp_buffer.append((vp1, vp2))
			if len(self._vp_buffer) > 30:
				self._vp_buffer.pop(0)
			vp1_med = np.median(np.stack([v[0] for v in self._vp_buffer], axis=0), axis=0)
			vp2_med = np.median(np.stack([v[1] for v in self._vp_buffer], axis=0), axis=0)
			self.state.vp1 = vp1_med
			self.state.vp2 = vp2_med
			self._update_intrinsics_and_plane(frame_bgr.shape)
		# Update scale
		if self.state.K is not None and self.state.n_plane is not None:
			s_lane = self.estimate_scale_from_lane_priors(frame_bgr, lane_width_m=lane_width_m)
			s_veh = self.estimate_scale_from_vehicle_priors(current_detections, avg_width_m=avg_vehicle_width_m)
			s = self._fuse_scales(s_lane, s_veh)
			if s is not None:
				self._scale_buffer.append(s)
				if len(self._scale_buffer) > 50:
					self._scale_buffer.pop(0)
				self.state.scale_m_per_unit = float(np.median(self._scale_buffer))
		# Confidence
		conf = 0.0
		if self.state.vp1 is not None and self.state.vp2 is not None:
			conf += 0.4
		if self.state.K is not None and self.state.n_plane is not None:
			conf += 0.3
		if self.state.scale_m_per_unit is not None:
			conf += 0.3
		self.state.confidence = conf
		return self.state