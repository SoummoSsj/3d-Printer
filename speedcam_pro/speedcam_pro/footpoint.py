import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class FootpointHead(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = nn.Sequential(
			nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
			nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
			nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
		)
		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(64, 64), nn.ReLU(),
			nn.Linear(64, 2),
		)

	def forward(self, x):
		f = self.backbone(x)
		xy = self.head(f)
		return xy


class FootpointPredictor:
	def __init__(self, weights: str = ""):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = FootpointHead().to(self.device)
		self.ok = False
		if weights and os.path.exists(weights):
			ckpt = torch.load(weights, map_location=self.device)
			self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
			self.model.eval()
			self.ok = True

	@torch.no_grad()
	def predict(self, frame_bgr: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[int,int]:
		x1,y1,x2,y2 = bbox
		if self.ok:
			crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2), :]
			if crop.size == 0:
				return int((x1+x2)*0.5), int(y2)
			img = torch.from_numpy(crop[:, :, ::-1].copy()).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
			xy = self.model(img)[0].detach().cpu().numpy()
			# model predicts normalized offset in [0,1] relative to crop size
			px = int(x1 + np.clip(xy[0], 0, 1) * max(1, x2-x1))
			py = int(y1 + np.clip(xy[1], 0, 1) * max(1, y2-y1))
			return px, py
		else:
			return int((x1+x2)*0.5), int(y2)