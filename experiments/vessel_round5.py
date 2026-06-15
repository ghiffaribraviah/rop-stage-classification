"""
Round 5: refine the round-4 breakthrough `blend_sato_50` (TEST 0.4655).
  blend = beta*sato_ridge(sigmas) + (1-beta)*gabor_tophat_soft

Levers, all at the locked downstream (P0.16 + top-3 CC + close 3x3):
  (a) sato sigma-range: finer / denser / extended multiscale sets
  (b) beta fine grid around 0.5: 0.40, 0.45, 0.50, 0.55, 0.60
  (c) normalize the sato map with per-image FOV percentile clip before blending
      (ridge maps are heavy-tailed; clipping may balance the fusion)
Tune on TRAIN(60); validate top-5 on TEST(40).
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import cv2
from skimage.filters import sato

sys.path.insert(0, str(Path(__file__).parent))
from vessel_eval import load_dataset, split_dataset, OUTPUT_DIR  # noqa: E402
from vessel_pipeline import (  # noqa: E402
    normalize01, threshold_response_map, segmentation_metrics,
)
from vessel_round3 import build_soft as build_gabor_soft, FULL_KS  # noqa: E402


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def _inv_green(rgb, fov):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, 6.0, 16); enh[~fov] = 0
    inv = (255 - enh).astype(np.float32) / 255.0
    inv[~fov] = 0
    return inv


def sato_map(rgb, fov, sigmas, clip_pct):
    inv = _inv_green(rgb, fov)
    r = sato(inv, sigmas=tuple(sigmas), black_ridges=False).astype(np.float32)
    r[~fov] = 0
    if clip_pct is not None and fov.any():
        hi = np.percentile(r[fov], clip_pct)
        if hi > 0:
            r = np.clip(r, 0, hi)
    return normalize01(r, fov)


# cache the gabor soft per-image: it is identical across all variants
_GAB_CACHE = {}


def gabor_soft_cached(d):
    key = id(d)
    if key not in _GAB_CACHE:
        _GAB_CACHE[key] = build_gabor_soft(d["rgb"], d["fov"], 0.5, FULL_KS, 15, 7, 12.0)
    return _GAB_CACHE[key]


def blend(d, sigmas, beta, clip_pct):
    fov = d["fov"]
    ridge = sato_map(d["rgb"], fov, sigmas, clip_pct)
    gab = gabor_soft_cached(d)
    return normalize01(beta * ridge + (1 - beta) * gab, fov)


def predict(soft, fov):
    th = threshold_response_map(soft, fov, method="percentile", target_density=0.16) > 0
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    if nl > 1:
        areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
        keep = {idx for _, idx in areas[:3]}
        th = np.isin(labels, list(keep))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & fov


def score(subset, params):
    out = []
    for d in subset:
        soft = blend(d, *params)
        out.append(segmentation_metrics(predict(soft, d["fov"]), d["gt"])["dice"])
    return float(np.mean(out))


S_MID = (1, 2, 3, 4, 5)
S_FINE = (0.8, 1.4, 2.0, 2.8, 3.6, 4.5)
S_DENSE = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5)
S_LOW = (0.8, 1.2, 1.8, 2.5, 3.2)


VARIANTS = {
    # name: (sigmas, beta, clip_pct)
    "r4_winner":     (S_MID, 0.50, None),     # reproduce round-4 best
    "beta_045":      (S_MID, 0.45, None),
    "beta_055":      (S_MID, 0.55, None),
    "beta_040":      (S_MID, 0.40, None),
    "beta_060":      (S_MID, 0.60, None),
    "fine_050":      (S_FINE, 0.50, None),
    "dense_050":     (S_DENSE, 0.50, None),
    "low_050":       (S_LOW, 0.50, None),
    "fine_055":      (S_FINE, 0.55, None),
    "mid_clip99":    (S_MID, 0.50, 99.0),
    "mid_clip995":   (S_MID, 0.50, 99.5),
    "fine_clip99":   (S_FINE, 0.50, 99.0),
}


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}", flush=True)
    print("BASELINE TEST=0.4469 | round-4 best TEST=0.4655 (blend_sato_50)\n", flush=True)

    results = []
    for name, params in VARIANTS.items():
        t = time.time()
        d = score(train, params)
        results.append((d, name, params))
        print(f"  TRAIN {d:.4f}  {name:14s} ({time.time()-t:.0f}s)", flush=True)
    results.sort(reverse=True)

    print("\n=== Validate top-5 on TEST ===", flush=True)
    rows = []
    for d, name, params in results[:5]:
        td = score(test, params)
        rows.append({"variant": name, "train_dice": round(d, 4), "test_dice": round(td, 4),
                     "sigmas": str(params[0]), "beta": params[1], "clip_pct": params[2]})
        print(f"  TEST {td:.4f}  TRAIN {d:.4f}  {name}", flush=True)
    rows.sort(key=lambda r: r["test_dice"], reverse=True)

    out = OUTPUT_DIR / "vessel_round5.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = rows[0]
    print(f"\nBest TEST = {best['test_dice']:.4f} ({best['variant']})", flush=True)
    print("vs round-4 best 0.4655 | baseline 0.4469", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
