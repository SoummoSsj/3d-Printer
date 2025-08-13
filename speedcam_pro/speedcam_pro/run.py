import argparse
import os
import time
import csv
import cv2
import numpy as np

from .detector import Detector
from .tracker import ByteTracker
from .footpoint import FootpointPredictor
from .calibration import SelfCalib
from .geometry import pixel_ray_dir, intersect_ray_plane, displacement_speed, build_features
from .smoother import SmootherPredictor


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, required=True)
	parser.add_argument("--det", type=str, default="yolov8x.pt")
	parser.add_argument("--device", type=str, default="")
	parser.add_argument("--display", type=int, default=1)
	parser.add_argument("--save", type=int, default=0)
	parser.add_argument("--out", type=str, default="")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--fp-weights", type=str, default="")
	parser.add_argument("--smooth-weights", type=str, default="")
	parser.add_argument("--csv-out", type=str, default="")
	parser.add_argument("--max-kph", type=float, default=300.0)
	args = parser.parse_args()

	cap = cv2.VideoCapture(args.source)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open source {args.source}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	detector = Detector(args.det, device=args.device)
	tracker = ByteTracker(max_age=int(fps))
	foot = FootpointPredictor(args.fp_weights)
	smoother = SmootherPredictor(args.smooth_weights, in_dim=8)
	calib = SelfCalib()

	writer = None
	if args.save:
		out_path = args.out or os.path.splitext(args.source)[0] + "_pro_annot.mp4"
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
	csv_file = None
	csv_writer = None
	if args.csv_out:
		csv_file = open(args.csv_out, "w", newline="")
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["time_s","track_id","x_units","y_units","speed_kph","smoothed_kph","conf"])

	K_inv = None
	prev_world = {}
	feat_buffers = {}
	frame_idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		t = frame_idx / fps
		state = calib.update(frame)
		if state.K_inv is None or state.n_plane_c is None:
			if args.display:
				cv2.imshow("speedcam-pro", frame)
				if cv2.waitKey(1) & 0xFF == 27:
					break
			frame_idx += 1
			continue
		K_inv = state.K_inv
		dets = detector.detect(frame, conf=args.conf)
		tracks = tracker.step(dets, t, frame=frame)
		for tr, det in zip(tracks, dets[:len(tracks)]):
			xb, yb = foot.predict(frame, tr.bbox)
			ray_c = pixel_ray_dir(K_inv, xb, yb)
			p_c = intersect_ray_plane(ray_c, state.n_plane_c, d=1.0)
			if p_c is None:
				continue
			if tr.id in prev_world:
				(t0, p0) = prev_world[tr.id]
				v_units = displacement_speed(p0, t0, p_c, t)
				kph = v_units * 3.6
				# build features and smooth if available
				feat = build_features(prev_world[tr.id], (t, p_c), tr.bbox, det[4])
				buf = feat_buffers.get(tr.id, [])
				buf.append(feat)
				if len(buf) > 20:
					buf.pop(0)
				feat_buffers[tr.id] = buf
				kph_sm = None
				if len(buf) >= 5:
					pred = smoother.smooth(np.stack(buf, axis=0))
					if pred is not None:
						kph_sm = float(pred)
				# draw
				label = f"{(kph_sm if kph_sm is not None else kph):.1f} km/h"
				cv2.circle(frame, (xb, yb), 3, (0,255,255), -1)
				cv2.putText(frame, label, (xb+6, yb-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
				if csv_writer is not None:
					csv_writer.writerow([f"{t:.3f}", tr.id, f"{p_c[0]:.3f}", f"{p_c[1]:.3f}", f"{kph:.3f}", f"{(kph_sm if kph_sm is not None else -1):.3f}", f"{state.conf:.2f}"])
			prev_world[tr.id] = (t, p_c)
		if args.display:
			cv2.imshow("speedcam-pro", frame)
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