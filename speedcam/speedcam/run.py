import argparse
import os
import time
import csv
import cv2

from .detection import Detector
from .tracking import IOUTracker
from .calibration import SelfCalibrator
from .geometry import warp_to_road_via, SpeedSmoother
from .speed import mps_to_kph
from .viz import draw_bbox, draw_speed


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, required=True)
	parser.add_argument("--model", type=str, default="yolov8n.pt")
	parser.add_argument("--device", type=str, default="")
	parser.add_argument("--display", type=int, default=1)
	parser.add_argument("--save", type=int, default=0)
	parser.add_argument("--out", type=str, default="")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--lane-width-m", type=float, default=3.5)
	parser.add_argument("--lock-scale-m-per-unit", type=float, default=-1.0)
	parser.add_argument("--csv-out", type=str, default="")
	parser.add_argument("--min-hits", type=int, default=3)
	parser.add_argument("--min-conf", type=float, default=0.7)
	parser.add_argument("--max-kph", type=float, default=300.0)
	args = parser.parse_args()

	cap = cv2.VideoCapture(args.source)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open source {args.source}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	detector = Detector(args.model, device=args.device)
	tracker = IOUTracker(max_age=int(fps))
	calib = SelfCalibrator()
	smoother = SpeedSmoother(alpha=0.6)

	writer = None
	if args.save:
		out_path = args.out or os.path.splitext(args.source)[0] + "_annot.mp4"
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

	csv_file = None
	csv_writer = None
	if args.csv_out:
		csv_file = open(args.csv_out, "w", newline="")
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["time_s","track_id","x_units","y_units","speed_kph","scale_m_per_unit","conf"])

	frame_idx = 0
	prev_world_pos = {}  # id -> (t,x,y)
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		t = frame_idx / fps
		dets = detector.detect(frame, conf=args.conf)
		tracks = tracker.step(dets, t)
		state = calib.update(frame, [tr.bbox for tr in tracks], lane_width_m=args.lane_width_m)
		if args.lock_scale_m_per_unit > 0:
			state.scale_m_per_unit = args.lock_scale_m_per_unit
			state.confidence = max(state.confidence, 0.9)
		for tr in tracks:
			xb, yb = tr.bottom_center()
			label = f"conf {state.confidence:.2f}"
			draw_bbox(frame, tr.bbox, tr.id, label)
			if state.K is None or state.n_plane is None or state.scale_m_per_unit is None:
				continue
			pw = warp_to_road_via(calib, xb, yb)
			if pw is None:
				continue
			xw, yw = pw
			if tr.id in prev_world_pos and tr.hits >= args.min_hits and state.confidence >= args.min_conf:
				pt = prev_world_pos[tr.id]
				dt = max(1e-3, t - pt[0])
				vx = (xw - pt[1]) / dt
				vy = (yw - pt[2]) / dt
				vx_s, vy_s = smoother.step(tr.id, t, vx, vy)
				v_kph = mps_to_kph((vx_s*vx_s + vy_s*vy_s) ** 0.5, state.scale_m_per_unit)
				if v_kph <= args.max_kph:
					draw_speed(frame, xb, yb, v_kph)
					if csv_writer is not None:
						csv_writer.writerow([f"{t:.3f}", tr.id, f"{xw:.3f}", f"{yw:.3f}", f"{v_kph:.3f}", f"{state.scale_m_per_unit:.6f}", f"{state.confidence:.2f}"])
			prev_world_pos[tr.id] = (t, xw, yw)
		if args.display:
			cv2.imshow("speedcam", frame)
			if cv2.waitKey(1) & 0xFF == 27:
				break
		if writer is not None:
			writer.write(frame)
		frame_idx += 1

	cap.release()
	if writer is not None:
		writer.release()
	if csv_file is not None:
		csv_file.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()