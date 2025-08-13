from typing import Dict, Tuple, Optional


class TrackWorldState:
	def __init__(self):
		self.history: Dict[int, Tuple[float,float,float]] = {}  # id -> (t, x, y)

	def update(self, tid: int, t: float, x: float, y: float) -> Optional[float]:
		if tid in self.history:
			pt = self.history[tid]
			dt = max(1e-3, t - pt[0])
			dx = x - pt[1]
			dy = y - pt[2]
			v = (dx*dx + dy*dy) ** 0.5 / dt
			self.history[tid] = (t, x, y)
			return v
		else:
			self.history[tid] = (t, x, y)
			return None


def mps_to_kph(v: float, scale_m_per_unit: float) -> float:
	# v is in homography units/sec; convert to m/s via scale, then to km/h
	vmps = v * scale_m_per_unit
	return vmps * 3.6