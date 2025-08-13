import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO

VEHICLE_CLASSES = {"car","truck","bus","motorcycle","bicycle","train","boat"}

class Detector:
	def __init__(self, weights_path: str, device: str = ""):
		self.model = YOLO(weights_path)
		if device:
			self.model.to(device)
		self.class_names = {i: name for i, name in enumerate(self.model.names)}

	def detect(self, frame_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> List[Tuple[int,int,int,int,float,int]]:
		res = self.model.predict(source=frame_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
		out: List[Tuple[int,int,int,int,float,int]] = []
		if res.boxes is None or len(res.boxes) == 0:
			return out
		boxes = res.boxes.xyxy.cpu().numpy().astype(float)
		scores = res.boxes.conf.cpu().numpy().astype(float)
		classes = res.boxes.cls.cpu().numpy().astype(int)
		for (x1,y1,x2,y2), sc, cl in zip(boxes, scores, classes):
			name = self.class_names.get(int(cl), "")
			if name in VEHICLE_CLASSES:
				out.append((int(x1), int(y1), int(x2), int(y2), float(sc), int(cl)))
		return out