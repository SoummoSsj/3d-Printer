import numpy as np
from typing import Optional, Tuple


def intersect_ray_plane(ray_dir_c: np.ndarray, plane_n_w: np.ndarray, plane_p_w: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> Optional[np.ndarray]:
	# Ray in camera: X_c = s * ray_dir_c, s>0; Transform to world: X_w = R_wc X_c + t_wc
	d = plane_n_w.dot(R_wc @ ray_dir_c)
	if abs(d) < 1e-6:
		return None
	num = plane_n_w.dot(plane_p_w - t_wc)
	s = num / d
	if s <= 0:
		return None
	X_w = R_wc @ (ray_dir_c * s) + t_wc
	return X_w


def pixel_ray_dir(K: np.ndarray, x: float, y: float) -> np.ndarray:
	invK = np.linalg.inv(K)
	r = invK @ np.array([x, y, 1.0])
	r = r / max(1e-9, np.linalg.norm(r))
	return r


def ttc_to_plane(p_w: np.ndarray, v_w: np.ndarray, plane_p_w: np.ndarray, plane_n_w: np.ndarray) -> Optional[float]:
	den = plane_n_w.dot(v_w)
	if den >= -1e-6:
		return None
	num = plane_n_w.dot(plane_p_w - p_w)
	if num <= 0:
		return None
	return num / (-den)