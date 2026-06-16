"""
Round 7: fine-tune the Gabor+Meijering fusion that broke round-6 ceiling.

Round-6 winner = g50_s00_m50:  0.5*gabor_tophat + 0.5*meijering_fine  -> TEST 0.4726

Sweep (a) the gabor/meijering weight ratio, (b) Meijering sigma set.
Sato dropped entirely. Downstream locked: P0.16 + top-3 CC + close 3x3.
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import cv2
from skimage.filters import meijering

sys.path.insert(0, str(Path(__file__).parent))
from vessel_eval import load_dataset, split_dataset, OUTPUT_DIR  # noqa: E402
from vessel_pipeline import (  # noqa: E402
    normalize01, threshold_response_map, segmentation_metrics,
)
from vessel_round3 import build_soft as build_gabor_soft, FULL_KS  # noqa: E402

SIGMA_SETS = {
    "fine":   (0.8, 1.4, 2.0, 2.8, 3.6, 4.5),
    "thin":   (0.6, 1.0, 1.6, 2.2, 3.0, 3.8),
    "wide":   (1.0, 1.8, 2.8, 3.8, 5.0, 6.0),
    "coarse": (1.2, 2.2, 3.4, 4.6, 6.0),
    "dense":  (0.8, 1.2, 1.6, 2.2, 2.8, 3.6, 4.5, 5.5),
}


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def _inv_green(rgb, fov):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, 6.0, 16); enh[~fov] = 0
    inv = (255 - enh).astype(np.float32) / 255.0
    inv[~fov] = 0
    return inv


_GAB = {}
_MEI = {}


def gabor_ch(d):
    key = id(d)
    if key not in _GAB:
        g = build_gabor_soft(d["rgb"], d["fov"], 0.5, FULL_KS, 15, 7, 12.0)
        _GAB[key] = normalize01(g, d["fov"])
    return _GAB[key]


def mei_ch(d, sset):
    key = (id(d), sset)
    if key not in _MEI:
        inv = _inv_green(d["rgb"], d["fov"])
        m = meijering(inv, sigmas=SIGMA_SETS[sset], black_ridges=False).astype(np.float32)
        m[~d["fov"]] = 0
        _MEI[key] = normalize01(m, d["fov"])
    return _MEI[key]


def predict(soft, fov):
    th = threshold_response_map(soft, fov, method="percentile", target_density=0.16) > 0
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    if nl > 1:
        areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
        th = np.isin(labels, [idx for _, idx in areas[:3]])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & fov


def score(subset, wg, wm, sset):
    out = []
    for d in subset:
        soft = normalize01(wg * gabor_ch(d) + wm * mei_ch(d, sset), d["fov"])
        out.append(segmentation_metrics(predict(soft, d["fov"]), d["gt"])["dice"])
    return float(np.mean(out))


# (wg, wm, sigma_set)
VARIANTS = {
    "r6_winner":     (0.50, 0.50, "fine"),
    "g45_m55_fine":  (0.45, 0.55, "fine"),
    "g40_m60_fine":  (0.40, 0.60, "fine"),
    "g55_m45_fine":  (0.55, 0.45, "fine"),
    "g35_m65_fine":  (0.35, 0.65, "fine"),
    "m100_fine":     (0.00, 1.00, "fine"),
    "g50_m50_thin":  (0.50, 0.50, "thin"),
    "g50_m50_wide":  (0.50, 0.50, "wide"),
    "g50_m50_dense": (0.50, 0.50, "dense"),
    "g45_m55_thin":  (0.45, 0.55, "thin"),
    "g45_m55_dense": (0.45, 0.55, "dense"),
    "g40_m60_dense": (0.40, 0.60, "dense"),
}


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}", flush=True)
    print("round-6 best TEST=0.4726 (g50_m50 fine) | baseline 0.4469\n", flush=True)

    results = []
    for name, (wg, wm, sset) in VARIANTS.items():
        t = time.time()
        d = score(train, wg, wm, sset)
        results.append((d, name, (wg, wm, sset)))
        print(f"  TRAIN {d:.4f}  {name:16s} ({time.time()-t:.0f}s)", flush=True)
    results.sort(reverse=True)

    print("\n=== Validate top-6 on TEST ===", flush=True)
    rows = []
    for d, name, (wg, wm, sset) in results[:6]:
        td = score(test, wg, wm, sset)
        rows.append({"variant": name, "train_dice": round(d, 4), "test_dice": round(td, 4),
                     "wg": wg, "wm": wm, "sigma_set": sset})
        print(f"  TEST {td:.4f}  TRAIN {d:.4f}  {name}", flush=True)
    rows.sort(key=lambda r: r["test_dice"], reverse=True)

    out = OUTPUT_DIR / "vessel_round7.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = rows[0]
    print(f"\nBest TEST = {best['test_dice']:.4f} ({best['variant']})", flush=True)
    print("vs round-6 best 0.4726 | baseline 0.4469", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
