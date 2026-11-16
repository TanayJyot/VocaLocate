"""Digitize Karpathy's d20 RL W&B screenshot (nanochat discussion #1) into CSVs.

Source image: https://github.com/user-attachments/assets/4ac908e8-a018-4381-b232-735c746d102b
Axes are calibrated from the gridlines / tick marks detected in the image (pixel constants below);
the line is the W&B series colour (128,96,207). Reading error is about +/- 1-2 px, i.e.
  reward  +/- 0.001,  pass@1 +/- 0.0002,  pass@8 +/- 0.0003   (from px-per-unit scales below),
plus whatever W&B's smoothing did to the reward panel (the dark line is W&B's smoothed series;
the faint raw series is not recoverable from the screenshot).

Usage: python digitize_karpathy.py <screenshot.png> <out_dir>
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LINE = np.array([128, 96, 207])

# (x_px for value a, a), (x_px for value b, b), (y_px for value c, c), (y_px for value d, d)
PANELS = {
    "reward": dict(x=((105.5, 0), (620.5, 400)), y=((199, 0.15), (312, 0.10)), xr=(100, 712), yr=(160, 540)),
    "pass1":  dict(x=((871.5, 0), (1443, 400)), y=((191, 0.06), (468, 0.03)), xr=(866, 1480), yr=(160, 540)),
    "pass8":  dict(x=((1764, 20), (2192, 80)), y=((227, 0.20), (523, 0.15)), xr=(1630, 2245), yr=(160, 540)),
}


def lin(p, q):
    (p0, v0), (p1, v1) = p, q
    s = (v1 - v0) / (p1 - p0)
    return lambda px: v0 + (px - p0) * s


def trace(im, cfg):
    fx, fy = lin(*cfg["x"]), lin(*cfg["y"])
    x0, x1 = cfg["xr"]
    y0, y1 = cfg["yr"]
    sub = im[y0:y1, x0:x1].astype(int)
    mask = (np.abs(sub - LINE).sum(-1) < 40)
    xs, ys = [], []
    for c in range(mask.shape[1]):
        rows = np.nonzero(mask[:, c])[0]
        if len(rows):
            xs.append(x0 + c)
            ys.append(y0 + float(np.median(rows)))
    xs, ys = np.array(xs), np.array(ys)
    return xs, ys, fx, fy


def vertices(xs, ys, n):
    """Pick n vertices of a piecewise-linear polyline by greedy Douglas-Peucker splitting:
    repeatedly add the point farthest from the current piecewise-linear approximation."""
    idx = [0, len(xs) - 1]
    while len(idx) < n:
        approx = np.interp(xs, xs[idx], ys[idx])
        err = np.abs(ys - approx)
        for j in idx:  # do not re-pick points next to existing vertices
            err[max(0, j - 10): j + 11] = 0
        idx = sorted(idx + [int(err.argmax())])
    return [(xs[i], ys[i]) for i in idx]


def main(img, out):
    im = np.array(Image.open(img).convert("RGB"))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Reward: smoothed W&B line, one sample per pixel column, then resampled to integer steps.
    xs, ys, fx, fy = trace(im, PANELS["reward"])
    steps, vals = fx(xs), fy(ys)
    grid = np.arange(0, int(steps.max()) + 1)
    rew = np.interp(grid, steps, vals)
    with open(out / "karpathy_d20_reward_smoothed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "reward_wandb_smoothed"])
        w.writerows((int(s), round(float(v), 4)) for s, v in zip(grid, rew))

    # pass@1 vs step: 8 evals at steps 0, 60, ..., 420. Read the line at those x positions.
    xs, ys, fx, fy = trace(im, PANELS["pass1"])
    p1 = []
    for s in range(0, 421, 60):
        px = cfg_x = (s - 0) / (400 - 0) * (1443 - 871.5) + 871.5
        y = np.interp(px, xs, ys)
        p1.append((s, round(float(fy(y)), 4)))
    with open(out / "karpathy_d20_pass1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "pass@1"])
        w.writerows(p1)

    # pass@8 vs wall-clock minutes: 8 vertices of the polyline.
    xs, ys, fx, fy = trace(im, PANELS["pass8"])
    v = vertices(xs, ys, 8)
    with open(out / "karpathy_d20_pass8.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_index", "assumed_step", "minutes", "pass@8"])
        for k, (x, y) in enumerate(v):
            w.writerow([k, k * 60, round(float(fx(x)), 1), round(float(fy(y)), 4)])

    print("reward: start %.4f  max %.4f @ step %d  end %.4f (step %d)" % (
        rew[0], rew.max(), grid[rew.argmax()], rew[-1], grid[-1]))
    print("pass@1:", p1)
    print("pass@8:", [(round(float(fx(x)), 1), round(float(fy(y)), 4)) for x, y in v])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
