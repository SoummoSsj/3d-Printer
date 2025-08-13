from typing import Dict, Tuple, Optional
import numpy as np


class WorldTrack:
	def __init__(self):
		self.prev: Dict[int, Tuple[float, np.ndarray]] = {}

	def step(self, tid: int, t: float, p_w: np.ndarray) -> Optional[np.ndarray]:
		if tid in self.prev:
			t0, p0 = self.prev[tid]
			dt = max(1e-3, t - t0)
			v = (p_w - p0) / dt
			self.prev[tid] = (t, p_w)
			return v
		else:
			self.prev[tid] = (t, p_w)
			return None


def mps_to_kph(v: float) -> float:
	return v * 3.6