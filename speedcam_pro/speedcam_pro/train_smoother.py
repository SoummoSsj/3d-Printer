import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .smoother import SeqSmoother


class FeatDataset(Dataset):
	def __init__(self, X: np.ndarray, Y: np.ndarray):
		self.X = X.astype(np.float32)
		self.Y = Y.astype(np.float32)
	def __len__(self):
		return len(self.X)
	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--npz", type=str, required=True)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch", type=int, default=64)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--out", type=str, required=True)
	args = parser.parse_args()

	data = np.load(args.npz)
	X = data["X"]; Y = data["Y"]
	# shuffle/split
	idx = np.arange(len(X))
	np.random.shuffle(idx)
	tr = int(0.9*len(idx))
	tr_idx, va_idx = idx[:tr], idx[tr:]
	tr_ds = FeatDataset(X[tr_idx], Y[tr_idx])
	va_ds = FeatDataset(X[va_idx], Y[va_idx])
	tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0)
	va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SeqSmoother(in_dim=X.shape[-1]).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
	crit = nn.SmoothL1Loss(beta=0.1)

	best = 1e9
	for ep in range(args.epochs):
		model.train(); tl=0.0; n=0
		for xb, yb in tr_ld:
			xb = xb.to(device); yb = yb.to(device)
			pred = model(xb)[:, -1, 0]
			loss = crit(pred, yb)
			opt.zero_grad(); loss.backward(); opt.step()
			tl += loss.item()*len(xb); n += len(xb)
		model.eval(); vl=0.0; m=0
		with torch.no_grad():
			for xb, yb in va_ld:
				xb = xb.to(device); yb = yb.to(device)
				pred = model(xb)[:, -1, 0]
				loss = crit(pred, yb)
				vl += loss.item()*len(xb); m += len(xb)
		print(f"epoch {ep+1}/{args.epochs} train {tl/n:.4f} val {vl/m:.4f}")
		if vl/m < best:
			best = vl/m
			os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
			torch.save({"model": model.state_dict()}, args.out)
	print(f"Saved best to {args.out}")

if __name__ == "__main__":
	main()