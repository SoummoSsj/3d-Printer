import cv2
from typing import Tuple, Optional


def draw_bbox(frame, bbox: Tuple[int,int,int,int], tid: int, label: str, color=(0,255,0)):
	x1,y1,x2,y2 = bbox
	cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
	cv2.putText(frame, f"ID {tid} {label}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_speed_ttc(frame, x: int, y: int, speed_kph: Optional[float], ttc_s: Optional[float]):
	parts = []
	if speed_kph is not None:
		parts.append(f"{speed_kph:.1f} km/h")
	if ttc_s is not None:
		parts.append(f"TTC {ttc_s:.1f}s")
	if not parts:
		return
	cv2.putText(frame, " | ".join(parts), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)