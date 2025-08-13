import numpy as np
from typing import Optional, Tuple, Dict


def warp_to_road_via(calibrator, x: float, y: float) -> Optional[Tuple[float,float]]:
	P = calibrator._ray_plane_point((x, y))
	if P is None:
		return None
	return float(P[0]), float(P[1])


class SpeedSmoother:
	def __init__(self, alpha: float = 0.5):
		self.alpha = alpha
		self.prev: Dict[int, Tuple[float,float,float]] = {}  # id -> (t, vx, vy)

	def step(self, tid: int, t: float, vx: float, vy: float) -> Tuple[float,float]:
		if tid not in self.prev:
			self.prev[tid] = (t, vx, vy)
			return vx, vy
		_, pvx, pvy = self.prev[tid]
		vx_s = self.alpha*vx + (1-self.alpha)*pvx
		vy_s = self.alpha*vy + (1-self.alpha)*pvy
		self.prev[tid] = (t, vx_s, vy_s)
		return vx_s, vy_s