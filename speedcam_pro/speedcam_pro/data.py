import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Optional
import numpy as np


@dataclass
class BrnoGTCar:
	carId: int
	speed_kph: float
	intersections: List[Dict[str, Any]]  # each has measurementLineId and videoTime


@dataclass
class BrnoGT:
	fps: float
	measurement_lines: List[np.ndarray]
	cars: List[BrnoGTCar]


@dataclass
class BrnoSession:
	name: str
	video_path: str
	mask_path: Optional[str]
	gt_path: str
	gt: Optional[BrnoGT]


def load_gt(gt_pkl_path: str) -> BrnoGT:
	with open(gt_pkl_path, "rb") as f:
		obj = pickle.load(f)
	fps = float(obj.get("fps", 25.0))
	ml = [np.asarray(l, dtype=float) for l in obj.get("measurementLines", [])]
	cars = []
	for c in obj.get("cars", []):
		cid = int(c.get("carId"))
		spd = float(c.get("speed"))
		inters = []
		for it in c.get("intersections", []):
			inters.append({
				"measurementLineId": int(it.get("measurementLineId")),
				"videoTime": float(it.get("videoTime")),
			})
		cars.append(BrnoGTCar(carId=cid, speed_kph=spd, intersections=inters))
	return BrnoGT(fps=fps, measurement_lines=ml, cars=cars)


def discover_sessions(root: str) -> Iterator[BrnoSession]:
	# expects root/brno_kaggle_subset/dataset/sessionX_{center|left|right}/...
	base = os.path.join(root, "brno_kaggle_subset", "dataset")
	if not os.path.isdir(base):
		return
	for sess in sorted(os.listdir(base)):
		sp = os.path.join(base, sess)
		if not os.path.isdir(sp):
			continue
		gt = os.path.join(sp, "gt_data.pkl")
		vid = os.path.join(sp, "video.avi")
		mask = os.path.join(sp, "video_mask.png")
		if os.path.isfile(gt) and os.path.isfile(vid):
			yield BrnoSession(name=sess, video_path=vid, mask_path=mask if os.path.isfile(mask) else None, gt_path=gt, gt=None)