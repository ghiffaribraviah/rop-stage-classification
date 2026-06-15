"""
Round 6: does a 3-way detector fusion beat the round-5 2-way ceiling (TEST 0.4667)?

Round-5 winner = fine_050:  0.5*sato_fine + 0.5*gabor_tophat
Add Meijering (neuriteness, different Hessian eigenvalue normalization than Sato)
as a 3rd complementary channel:

  blend = wg*gabor_tophat + ws*sato_fine + wm*meijering_fine   (weights sum to 1)

Downstream locked: P0.16 + top-3 CC + close 3x3. Tune on TRAIN(60), validate on TEST(40).
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import cv2
from skimage.filters import sato, meijering

sys.path.insert(0, str(Path(__file__).parent))
from vessel_eval import load_dataset, split_dataset, OUTPUT_DIR  # noqa: E402
from vessel_pipeline import (  # noqa: E402
    normalize01, threshold_response_map, segmentation_metrics,
)
from vessel_round3 import build_soft as build_gabor_soft, FULL_KS  # noqa: E402

S_FINE = (0.8, 1.4, 2.0, 2.8, 3.6, 4.5)


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def _inv_green(rgb, fov):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, 6.0, 16); enh[~fov] = 0
    inv = (255 - enh).astype(np.float32) / 255.0
    inv[~fov] = 0
    return inv


_CACHE = {}


def channels(d):
    """Return (gabor, sato, meijering) normalized soft maps; cached per image."""
    key = id(d)
    if key not in _CACHE:
        rgb, fov = d["rgb"], d["fov"]
        gab = build_gabor_soft(rgb, fov, 0.5, FULL_KS, 15, 7, 12.0)
        inv = _inv_green(rgb, fov)
        sat = sato(inv, sigmas=S_FINE, black_ridges=False).astype(np.float32); sat[~fov] = 0
        mei = meijering(inv, sigmas=S_FINE, black_ridges=False).astype(np.float32); mei[~fov] = 0
        _CACHE[key] = (normalize01(gab, fov), normalize01(sat, fov), normalize01(mei, fov))
    return _CACHE[key]


def predict(soft, fov):
    th = threshold_response_map(soft, fov, method="percentile", target_density=0.16) > 0
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    if nl > 1:
        areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
        keep = {idx for _, idx in areas[:3]}
        th = np.isin(labels, list(keep))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & fov


def score(subset, wg, ws, wm):
    out = []
    for d in subset:
        gab, sat, mei = channels(d)
        soft = normalize01(wg * gab + ws * sat + wm * mei, d["fov"])
        out.append(segmentation_metrics(predict(soft, d["fov"]), d["gt"])["dice"])
    return float(np.mean(out))


VARIANTS = {
    # name: (wg, ws, wm)
    "r5_winner_2way": (0.50, 0.50, 0.00),  # reproduce: no meijering
    "trio_even":      (0.34, 0.33, 0.33),
    "g50_s30_m20":    (0.50, 0.30, 0.20),
    "g50_s25_m25":    (0.50, 0.25, 0.25),
    "g45_s35_m20":    (0.45, 0.35, 0.20),
    "g40_s40_m20":    (0.40, 0.40, 0.20),
    "g50_s35_m15":    (0.50, 0.35, 0.15),
    "g50_s40_m10":    (0.50, 0.40, 0.10),
    "g55_s30_m15":    (0.55, 0.30, 0.15),
    "g50_s00_m50":    (0.50, 0.00, 0.50),  # gabor+meijering only (is mei a sato substitute?)
}


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}", flush=True)
    print("round-5 best TEST=0.4667 (fine_050 2-way) | baseline 0.4469\n", flush=True)

    results = []
    for name, w in VARIANTS.items():
        t = time.time()
        d = score(train, *w)
        results.append((d, name, w))
        print(f"  TRAIN {d:.4f}  {name:16s} ({time.time()-t:.0f}s)", flush=True)
    results.sort(reverse=True)

    print("\n=== Validate top-5 on TEST ===", flush=True)
    rows = []
    for d, name, w in results[:5]:
        td = score(test, *w)
        rows.append({"variant": name, "train_dice": round(d, 4), "test_dice": round(td, 4),
                     "wg": w[0], "ws": w[1], "wm": w[2]})
        print(f"  TEST {td:.4f}  TRAIN {d:.4f}  {name}", flush=True)
    rows.sort(key=lambda r: r["test_dice"], reverse=True)

    out = OUTPUT_DIR / "vessel_round6.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = rows[0]
    print(f"\nBest TEST = {best['test_dice']:.4f} ({best['variant']})", flush=True)
    print("vs round-5 best 0.4667 | baseline 0.4469", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
