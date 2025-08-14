import cv2
import numpy as np
from skimage.transform import probabilistic_hough_line
from dataclasses import dataclass
from typing import Optional, Tuple, List


def _line(p1, p2):
	return np.cross(p1, p2)

def _inter(l1, l2):
	p = np.cross(l1, l2)
	if abs(p[2]) < 1e-8:
		return None
	return p / p[2]

def _cluster(lines, k=2):
	angles = []
	for (x1,y1),(x2,y2) in lines:
		ang = (np.arctan2(y2-y1, x2-x1) + np.pi) % np.pi
		angles.append([np.cos(2*ang), np.sin(2*ang)])
	angles = np.asarray(angles, dtype=float)
	if len(angles) < k:
		return [list(range(len(angles)))]
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
	_, labels, _ = cv2.kmeans(angles.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
	cl = [[] for _ in range(k)]
	for i,lab in enumerate(labels.ravel()):
		cl[lab].append(i)
	return cl

def _vp(lines, idxs, shape):
	pts = []
	M = min(len(idxs), 100)
	for i in range(M):
		for j in range(i+1, M):
			(x1,y1),(x2,y2) = lines[idxs[i]]
			(x3,y3),(x4,y4) = lines[idxs[j]]
			p1,p2 = np.array([x1,y1,1.0]), np.array([x2,y2,1.0])
			p3,p4 = np.array([x3,y3,1.0]), np.array([x4,y4,1.0])
			l1,l2 = _line(p1,p2), _line(p3,p4)
			p = _inter(l1,l2)
			if p is not None:
				pts.append(p[:2])
	if len(pts) < 10:
		return None
	m = np.median(np.asarray(pts), axis=0)
	return np.array([m[0], m[1], 1.0])


def detect_vps(gray) -> Optional[Tuple[np.ndarray,np.ndarray]]:
	edges = cv2.Canny(gray, 100, 200)
	lines = probabilistic_hough_line(edges, threshold=10, line_length=30, line_gap=3)
	if not lines:
		return None
	cl = _cluster(lines, 2)
	if len(cl) < 2:
		return None
	vp1 = _vp(lines, cl[0], gray.shape)
	vp2 = _vp(lines, cl[1], gray.shape)
	if vp1 is None or vp2 is None:
		return None
	return vp1, vp2


def _estimate_f(vp1, vp2, cx, cy) -> Optional[float]:
	dx1 = vp1[0]-cx; dy1 = vp1[1]-cy
	dx2 = vp2[0]-cx; dy2 = vp2[1]-cy
	v = - (dx1*dx2 + dy1*dy2)
	if v <= 1.0:
		return None
	return float(np.sqrt(v))


def _norm(v):
	n = np.linalg.norm(v)
	if n < 1e-9:
		return v
	return v / n

@dataclass
class Calib:
	K: Optional[np.ndarray]
	K_inv: Optional[np.ndarray]
	n_plane_c: Optional[np.ndarray]
	conf: float


class SelfCalib:
	def __init__(self):
		self.vps = []
		self.calib = Calib(None, None, None, 0.0)

	def update(self, frame_bgr: np.ndarray) -> Calib:
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		vps = detect_vps(gray)
		if vps is not None:
			vp1, vp2 = vps
			self.vps.append((vp1, vp2))
			if len(self.vps) > 30:
				self.vps.pop(0)
			vp1m = np.median(np.stack([v[0] for v in self.vps]), axis=0)
			vp2m = np.median(np.stack([v[1] for v in self.vps]), axis=0)
			h, w = frame_bgr.shape[:2]
			cx, cy = w*0.5, h*0.5
			f = _estimate_f(vp1m, vp2m, cx, cy)
			if f is None:
				f = 1.5*max(w,h)
			K = np.array([[f,0,cx],[0,f,cy],[0,0,1.0]], dtype=float)
			K_inv = np.linalg.inv(K)
			d1 = _norm(K_inv @ vp1m); d2 = _norm(K_inv @ vp2m)
			n = _norm(np.cross(d1, d2))
			self.calib = Calib(K, K_inv, n, conf=0.8)
		return self.calib