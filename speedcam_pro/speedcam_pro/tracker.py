import numpy as np
from typing import List, Tuple, Optional
from filterpy.kalman import KalmanFilter


class Track:
	def __init__(self, tid: int, bbox: Tuple[int,int,int,int], timestamp: float, emb: Optional[np.ndarray] = None):
		self.id = tid
		self.kf = self._init_kf(bbox)
		self.bbox = bbox
		self.last_timestamp = timestamp
		self.hits = 1
		self.misses = 0
		self.emb = emb

	def _init_kf(self, bbox):
		cx = (bbox[0]+bbox[2]) * 0.5
		cy = (bbox[1]+bbox[3]) * 0.5
		w = bbox[2]-bbox[0]
		h = bbox[3]-bbox[1]
		kf = KalmanFilter(dim_x=8, dim_z=4)
		kf.F = np.eye(8)
		for i in range(4):
			kf.F[i, i+4] = 1.0
		kf.H = np.zeros((4,8)); kf.H[0,0]=kf.H[1,1]=kf.H[2,2]=kf.H[3,3]=1.0
		kf.P *= 10.0; kf.R *= 5.0; kf.Q = np.eye(8) * 0.01
		kf.x[:4,0] = np.array([cx, cy, w, h])
		return kf

	def predict(self, dt: float):
		for i in range(4):
			self.kf.F[i, i+4] = dt
		self.kf.predict()

	def update(self, bbox: Tuple[int,int,int,int], emb: Optional[np.ndarray] = None):
		cx = (bbox[0]+bbox[2]) * 0.5
		cy = (bbox[1]+bbox[3]) * 0.5
		w = bbox[2]-bbox[0]
		h = bbox[3]-bbox[1]
		self.kf.update(np.array([cx,cy,w,h]))
		x = self.kf.x[:,0]
		cx, cy, w, h = x[0], x[1], x[2], x[3]
		x1 = int(cx - w*0.5); y1 = int(cy - h*0.5)
		x2 = int(cx + w*0.5); y2 = int(cy + h*0.5)
		self.bbox = (x1,y1,x2,y2)
		self.hits += 1; self.misses = 0
		if emb is not None:
			self.emb = emb

	def mark_missed(self):
		self.misses += 1

	def bottom_center(self) -> Tuple[int,int]:
		x1,y1,x2,y2 = self.bbox
		return int((x1+x2)*0.5), int(y2)


def iou(a, b) -> float:
	x1,y1,x2,y2 = a
	x1b,y1b,x2b,y2b = b
	ix1 = max(x1, x1b); iy1 = max(y1, y1b)
	ix2 = min(x2, x2b); iy2 = min(y2, y2b)
	w = max(0, ix2-ix1+1); h = max(0, iy2-iy1+1)
	inter = w*h
	if inter == 0:
		return 0.0
	area_a = (x2-x1+1)*(y2-y1+1)
	area_b = (x2b-x1b+1)*(y2b-y1b+1)
	return inter / float(area_a + area_b - inter)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
	if a is None or b is None:
		return 0.0
	a = a / (np.linalg.norm(a) + 1e-9)
	b = b / (np.linalg.norm(b) + 1e-9)
	return float(np.dot(a, b))


class ReIDExtractor:
	def __init__(self):
		self.enabled = False
	def embed(self, frame, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
		return None


class ByteTracker:
	def __init__(self, max_age: int = 30, iou_thresh: float = 0.3, emb_w: float = 0.25):
		self.tracks: List[Track] = []
		self.next_id = 1
		self.max_age = max_age
		self.iou_thresh = iou_thresh
		self.emb_w = emb_w
		self.reid = ReIDExtractor()

	def step(self, detections: List[Tuple[int,int,int,int,float,int]], timestamp: float, frame=None) -> List[Track]:
		for tr in self.tracks:
			dt = max(1e-3, timestamp - tr.last_timestamp)
			tr.predict(dt)
			tr.last_timestamp = timestamp
		cand_embs = []
		if self.reid.enabled and frame is not None:
			for det in detections:
				cand_embs.append(self.reid.embed(frame, det[:4]))
		else:
			cand_embs = [None] * len(detections)
		assigned = set()
		for idx, det in enumerate(sorted(range(len(detections)), key=lambda i: detections[i][4], reverse=True)):
			bbox = detections[det][:4]
			emb = cand_embs[det]
			best_score = -1.0; best_tr = None
			for tr in self.tracks:
				if tr in assigned:
					continue
				IoU = iou(tr.bbox, bbox)
				if IoU < self.iou_thresh:
					continue
				E = self.emb_w * cos_sim(tr.emb, emb) if emb is not None and tr.emb is not None else 0.0
				score = IoU + E
				if score > best_score:
					best_score = score; best_tr = tr
			if best_tr is not None:
				best_tr.update(bbox, emb)
				assigned.add(best_tr)
			else:
				tr = Track(self.next_id, bbox, timestamp, emb)
				self.next_id += 1
				self.tracks.append(tr)
		alive = []
		for tr in self.tracks:
			if tr in assigned:
				alive.append(tr)
			else:
				tr.mark_missed()
				if tr.misses <= self.max_age:
					alive.append(tr)
		self.tracks = alive
		return self.tracks