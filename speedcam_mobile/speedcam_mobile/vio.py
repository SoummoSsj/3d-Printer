import csv
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


def _quat_to_R(qx, qy, qz, qw) -> np.ndarray:
	q = np.array([qw, qx, qy, qz], dtype=float)
	n = np.linalg.norm(q)
	if n < 1e-9:
		return np.eye(3)
	q /= n
	w, x, y, z = q
	R = np.array([
		[1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
		[2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
		[2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
	], dtype=float)
	return R


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
	dot = np.dot(q0, q1)
	if dot < 0.0:
		q1 = -q1; dot = -dot
	if dot > 0.9995:
		return (q0 + u*(q1-q0)) / np.linalg.norm(q0 + u*(q1-q0))
	theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
	sin0 = math.sin(theta_0)
	theta = theta_0 * u
	sin = math.sin(theta)
	return (math.sin(theta_0-theta)/sin0)*q0 + (sin/sin0)*q1


def _R_to_quat(R: np.ndarray) -> np.ndarray:
	# returns [w,x,y,z]
	K = np.array([
		[R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
		[R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
		[R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
		[R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]]
	], dtype=float) / 3.0
	w, V = np.linalg.eigh(K)
	q = V[:, np.argmax(w)]
	if q[0] < 0:
		q = -q
	return q

@dataclass
class Pose:
	t: float
	R: np.ndarray
	tvec: np.ndarray

	def T(self) -> np.ndarray:
		T = np.eye(4)
		T[:3,:3] = self.R
		T[:3,3] = self.tvec
		return T

class PoseSequence:
	def __init__(self, poses: List[Pose]):
		self.poses = sorted(poses, key=lambda p: p.t)
		self.ts = np.array([p.t for p in self.poses], dtype=float)

	@staticmethod
	def from_csv(path: str) -> "PoseSequence":
		poses: List[Pose] = []
		with open(path, "r") as f:
			reader = csv.DictReader(f)
			for row in reader:
				t = float(row["time_s"]) 
				tx,ty,tz = float(row["tx"]), float(row["ty"]), float(row["tz"])
				qx,qy,qz,qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
				R = _quat_to_R(qx,qy,qz,qw)
				poses.append(Pose(t=t, R=R, tvec=np.array([tx,ty,tz], dtype=float)))
		return PoseSequence(poses)

	def at(self, t: float) -> Optional[Pose]:
		if len(self.poses) == 0:
			return None
		if t <= self.poses[0].t:
			return self.poses[0]
		if t >= self.poses[-1].t:
			return self.poses[-1]
		idx = int(np.searchsorted(self.ts, t))
		p0, p1 = self.poses[idx-1], self.poses[idx]
		u = (t - p0.t) / max(1e-6, p1.t - p0.t)
		q0 = _R_to_quat(p0.R)
		q1 = _R_to_quat(p1.R)
		q = _slerp(q0, q1, u)
		R = _quat_to_R(q[1], q[2], q[3], q[0])
		tvec = (1-u)*p0.tvec + u*p1.tvec
		return Pose(t=t, R=R, tvec=tvec)