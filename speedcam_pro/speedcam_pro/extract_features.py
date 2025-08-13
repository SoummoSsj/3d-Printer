import os
import argparse
import random
import cv2
import numpy as np
from tqdm import tqdm

from .data import discover_sessions, load_gt
from .detector import Detector
from .tracker import ByteTracker
from .footpoint import FootpointPredictor
from .calibration import SelfCalib
from .geometry import pixel_ray_dir, intersect_ray_plane, build_features


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--root", type=str, required=True, help="Dataset root containing brno_kaggle_subset/dataset/")
	parser.add_argument("--det", type=str, default="yolov8n.pt")
	parser.add_argument("--device", type=str, default="")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--out-npz", type=str, required=True)
	parser.add_argument("--save-crops", type=str, default="", help="Optional folder to save footpoint training crops")
	parser.add_argument("--max-frames", type=int, default=0)
	args = parser.parse_args()

	detector = Detector(args.det, device=args.device)
	foot = FootpointPredictor("")
	feat_list = []
	y_list = []

	for sess in discover_sessions(args.root):
		gt = load_gt(sess.gt_path)
		cap = cv2.VideoCapture(sess.video_path)
		if not cap.isOpened():
			continue
		fps = cap.get(cv2.CAP_PROP_FPS) or gt.fps
		tracker = ByteTracker(max_age=int(fps))
		calib = SelfCalib()
		prev_world = {}
		feat_buffers = {}
		frame_idx = 0
		with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc=sess.name) as pbar:
			while True:
				ret, frame = cap.read()
				if not ret:
					break
				t = frame_idx / fps
				state = calib.update(frame)
				if state.K_inv is None or state.n_plane_c is None:
					frame_idx += 1; pbar.update(1)
					if args.max_frames and frame_idx >= args.max_frames:
						break
					continue
				dets = detector.detect(frame, conf=args.conf)
				tracks = tracker.step(dets, t, frame=frame)
				for i, tr in enumerate(tracks):
					# associate det for score
					if i < len(dets):
						det = dets[i]
						score = det[4]
					else:
						score = 0.5
					xb, yb = foot.predict(frame, tr.bbox)
					ray_c = pixel_ray_dir(state.K_inv, xb, yb)
					p_c = intersect_ray_plane(ray_c, state.n_plane_c, d=1.0)
					if p_c is None:
						continue
					if tr.id in prev_world:
						prev = prev_world[tr.id]
						feat = build_features(prev, (t, p_c), tr.bbox, score)
						buf = feat_buffers.get(tr.id, [])
						buf.append(feat)
						if len(buf) > 30:
							buf.pop(0)
						feat_buffers[tr.id] = buf
						# pseudo target: use last instantaneous speed magnitude (first two features are vx, vy)
						vx, vy = feat[0], feat[1]
						y = float((vx*vx + vy*vy) ** 0.5)
						if len(buf) >= 8:
							feat_list.append(np.stack(buf[-8:], axis=0))
							y_list.append(y)
						# optionally save crops for footpoint training
						if args.save_crops:
							x1,y1,x2,y2 = tr.bbox
							crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2), :]
							if crop.size > 0:
								os.makedirs(args.save_crops, exist_ok=True)
								fn = os.path.join(args.save_crops, f"{sess.name}_{frame_idx}_{tr.id}.jpg")
								cv2.imwrite(fn, crop)
					prev_world[tr.id] = (t, p_c)
				frame_idx += 1; pbar.update(1)
				if args.max_frames and frame_idx >= args.max_frames:
					break
		cap.release()

	if len(feat_list) == 0:
		print("No features extracted.")
		return
	X = np.stack(feat_list, axis=0)
	Y = np.asarray(y_list, dtype=float)
	npz_path = args.out_npz
	np.savez_compressed(npz_path, X=X, Y=Y)
	print(f"Saved features to {npz_path}: X {X.shape}, Y {Y.shape}")

if __name__ == "__main__":
	main()