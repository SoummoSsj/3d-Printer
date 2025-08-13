import argparse
import os
import time
import json
import csv
import cv2
import numpy as np

from .vio import PoseSequence
from .detection import Detector
from .tracking import IOUTracker
from .geometry import pixel_ray_dir, intersect_ray_plane, ttc_to_plane
from .ttc import WorldTrack, mps_to_kph
from .viz import draw_bbox, draw_speed_ttc


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--source", type=str, required=True)
	parser.add_argument("--poses", type=str, required=True)
	parser.add_argument("--crossing", type=str, required=True)
	parser.add_argument("--model", type=str, default="yolov8n.pt")
	parser.add_argument("--device", type=str, default="")
	parser.add_argument("--display", type=int, default=1)
	parser.add_argument("--save", type=int, default=0)
	parser.add_argument("--out", type=str, default="")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--fx", type=float, default=0.0)
	parser.add_argument("--fy", type=float, default=0.0)
	parser.add_argument("--cx", type=float, default=0.0)
	parser.add_argument("--cy", type=float, default=0.0)
	parser.add_argument("--csv-out", type=str, default="")
	parser.add_argument("--min-hits", type=int, default=3)
	parser.add_argument("--max-kph", type=float, default=300.0)
	args = parser.parse_args()

	cap = cv2.VideoCapture(args.source)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open source {args.source}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cx = args.cx if args.cx > 0 else W*0.5
	cy = args.cy if args.cy > 0 else H*0.5
	fx = args.fx if args.fx > 0 else max(W,H)*1.5
	fy = args.fy if args.fy > 0 else max(W,H)*1.5
	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)

	poses = PoseSequence.from_csv(args.poses)
	cross = json.load(open(args.crossing, "r"))
	plane_p_w = np.array(cross["point"], dtype=float)
	plane_n_w = np.array(cross["normal"], dtype=float)
	plane_n_w = plane_n_w / max(1e-9, np.linalg.norm(plane_n_w))

	detector = Detector(args.model, device=args.device)
	tracker = IOUTracker(max_age=int(fps))
	world = WorldTrack()

	writer = None
	if args.save:
		out_path = args.out or os.path.splitext(args.source)[0] + "_mobile_annot.mp4"
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
	csv_file = None
	csv_writer = None
	if args.csv_out:
		csv_file = open(args.csv_out, "w", newline="")
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["time_s","track_id","x","y","z","speed_kph","ttc_s"])

	frame_idx = 0
	prev_world_pos = {}
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		t = frame_idx / fps
		pose = poses.at(t)
		if pose is None:
			break
		R_wc = pose.R
		t_wc = pose.tvec
		dets = detector.detect(frame, conf=args.conf)
		tracks = tracker.step(dets, t)
		for tr in tracks:
			xb, yb = tr.bottom_center()
			draw_bbox(frame, tr.bbox, tr.id, "")
			ray_c = pixel_ray_dir(K, xb, yb)
			p_w = intersect_ray_plane(ray_c, plane_n_w, plane_p_w, R_wc, t_wc)
			if p_w is None:
				continue
			v_w = world.step(tr.id, t, p_w)
			speed_kph = None; ttc_s = None
			if v_w is not None and tr.hits >= args.min_hits:
				speed = float(np.linalg.norm(v_w))
				speed_kph = mps_to_kph(speed)
				if speed_kph <= args.max_kph:
					ttc_s = ttc_to_plane(p_w, v_w, plane_p_w, plane_n_w)
					draw_speed_ttc(frame, xb, yb, speed_kph, ttc_s)
					if csv_writer is not None:
						csv_writer.writerow([f"{t:.3f}", tr.id, f"{p_w[0]:.3f}", f"{p_w[1]:.3f}", f"{p_w[2]:.3f}", f"{speed_kph:.3f}", f"{(ttc_s if ttc_s is not None else -1):.3f}"])
		if args.display:
			cv2.imshow("speedcam-mobile", frame)
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