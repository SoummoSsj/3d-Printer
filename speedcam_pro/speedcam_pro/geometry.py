import numpy as np
from typing import Optional, Tuple


def normalize(v):
	n = np.linalg.norm(v)
	return v / (n + 1e-9)


def pixel_ray_dir(K_inv: np.ndarray, x: float, y: float) -> np.ndarray:
	r = K_inv @ np.array([x, y, 1.0])
	return normalize(r)


def intersect_ray_plane(ray_dir_c: np.ndarray, n_c: np.ndarray, d: float = 1.0) -> Optional[np.ndarray]:
	den = float(n_c.dot(ray_dir_c))
	if abs(den) < 1e-6:
		return None
	s = -d / den
	if s <= 0:
		return None
	Xc = ray_dir_c * s
	return Xc


def displacement_speed(p0: np.ndarray, t0: float, p1: np.ndarray, t1: float) -> float:
	dt = max(1e-3, t1 - t0)
	return float(np.linalg.norm(p1 - p0) / dt)


def build_features(prev: Tuple[float,np.ndarray], curr: Tuple[float,np.ndarray], bbox, score: float) -> np.ndarray:
	# Simple 8-dim features: [dx, dy, dt, x, y, w, h, score]
	t0,p0 = prev
	t1,p1 = curr
	dt = max(1e-3, t1 - t0)
	d = (p1 - p0) / dt
	x1,y1,x2,y2 = bbox
	w = float(x2-x1); h = float(y2-y1)
	feat = np.array([d[0], d[1], dt, p1[0], p1[1], w, h, score], dtype=float)
	return feat