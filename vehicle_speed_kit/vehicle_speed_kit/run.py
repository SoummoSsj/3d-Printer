import argparse
import os
import cv2
import numpy as np

from .detector import Detector
from .tracker import ByteTracker
from .calibration import SelfCalib
from .geometry import pixel_ray_dir, intersect_ray_plane, displacement_speed, build_features
from .footpoint import FootpointPredictor
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
	parser.add_argument("--smooth-weights", type=str, default="")
	parser.add_argument("--fp-weights", type=str, default="")
	parser.add_argument("--csv-out", type=str, default="")
	args = parser.parse_args()

	cap = cv2.VideoCapture(args.source)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open source {args.source}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	detector = Detector(args.det, device=args.device)
	tracker = ByteTracker(max_age=int(fps))
	calib = SelfCalib()
	foot = FootpointPredictor(args.fp_weights)
	smoother = SmootherPredictor(args.smooth_weights, in_dim=8)

	writer = None
	if args.save:
		out_path = args.out or os.path.splitext(args.source)[0] + "_kit_annot.mp4"
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
	csv_file = None
	if args.csv_out:
		csv_file = open(args.csv_out, "w")
		csv_file.write("time_s,track_id,x_units,y_units,speed_kph,smoothed_kph\n")

	prev_world = {}
	feat_buffers = {}
	frame_idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		t = frame_idx / fps
		state = calib.update(frame)
		calib_ready = (state.K_inv is not None and state.n_plane_c is not None)
		status = f"calib: {'ready' if calib_ready else 'estimating'}"

		dets = detector.detect(frame, conf=args.conf)
		tracks = tracker.step(dets, t, frame=frame)

		# Always draw tracked boxes and IDs so user sees detections even before calibration
		for tr in tracks:
			x1,y1,x2,y2 = tr.bbox
			cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
			cv2.putText(frame, f"ID {tr.id}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

		# Compute speeds only when calibration is ready
		if calib_ready:
			for i, tr in enumerate(tracks):
				score = dets[i][4] if i < len(dets) else 0.5
				xb, yb = foot.predict(frame, tr.bbox)
				ray_c = pixel_ray_dir(state.K_inv, xb, yb)
				p_c = intersect_ray_plane(ray_c, state.n_plane_c, d=1.0)
				if p_c is None:
					continue
				if tr.id in prev_world:
					v_units = displacement_speed(prev_world[tr.id][1], prev_world[tr.id][0], p_c, t)
					kph = v_units * 3.6
					feat = build_features(prev_world[tr.id], (t, p_c), tr.bbox, score)
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
					label = f"{(kph_sm if kph_sm is not None else kph):.1f} km/h"
					cv2.circle(frame, (xb, yb), 3, (0,255,255), -1)
					cv2.putText(frame, label, (xb+6, yb-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
					if csv_file:
						csv_file.write(f"{t:.3f},{tr.id},{p_c[0]:.3f},{p_c[1]:.3f},{kph:.3f},{(-1 if kph_sm is None else kph_sm):.3f}\n")
				prev_world[tr.id] = (t, p_c)

		# Draw calibration status
		cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
		cv2.putText(frame, f"tracks: {len(tracks)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)

		if args.display:
			cv2.imshow("vehicle-speed-kit", frame)
			if cv2.waitKey(1) & 0xFF == 27:
				break
		if writer is not None:
			writer.write(frame)
		frame_idx += 1

	cap.release()
	if writer is not None:
		writer.release()
	if csv_file:
		csv_file.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()