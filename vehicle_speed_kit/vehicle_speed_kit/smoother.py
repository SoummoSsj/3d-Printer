import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class SeqSmoother(nn.Module):
	def __init__(self, in_dim: int = 8, hid: int = 64, layers: int = 1):
		super().__init__()
		self.rnn = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
		self.fc = nn.Linear(hid, 1)
	def forward(self, x):
		y,_ = self.rnn(x)
		return self.fc(y)


class SmootherPredictor:
	def __init__(self, weights: str = "", in_dim: int = 8):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = SeqSmoother(in_dim=in_dim).to(self.device)
		self.ok = False
		if weights and os.path.exists(weights):
			ckpt = torch.load(weights, map_location=self.device)
			self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
			self.model.eval(); self.ok = True
	@torch.no_grad()
	def smooth(self, feat_seq: np.ndarray) -> Optional[float]:
		if not self.ok or feat_seq.shape[0] == 0:
			return None
		x = torch.from_numpy(feat_seq[None, :, :]).float().to(self.device)
		return float(self.model(x)[0, -1, 0].item())