"""
Round 2: tune the Gabor filter bank + preprocessing (the soft-map lever).
Fixed downstream: percentile P0.16 + top-3 components + close 3x3 (round-1 winner).
Tune on TRAIN(60), validate best on TEST(40).
"""
import sys
import csv
import itertools
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_eval import load_dataset, split_dataset, OUTPUT_DIR  # noqa: E402
from vessel_pipeline import (  # noqa: E402
    normalize01, threshold_response_map, segmentation_metrics, modified_tophat,
)
from advanced_pipeline import gabor_kernel_2d  # noqa: E402


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def gabor_bank(inv_f, fov, sigmas, lambdas, angle_step):
    response = np.zeros(inv_f.shape, dtype=np.float32)
    angles = range(0, 180, angle_step)
    for sigma, lambd in zip(sigmas, lambdas):
        size = int(6 * sigma)
        size = max(3, size + 1 if size % 2 == 0 else size)
        for angle in angles:
            theta = np.deg2rad(angle)
            kernel = gabor_kernel_2d(size, theta, sigma, lambd)
            filtered = cv2.filter2D(inv_f, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)
    response[~fov] = 0
    return normalize01(response, fov)


def build_soft(rgb, fov, sigmas, lambdas, angle_step, use_tophat, median, clahe2_clip):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, 6.0, 16); enh[~fov] = 0
    inv = 255 - enh
    if use_tophat:
        th = modified_tophat(inv, fov)
        inv_f = normalize01((normalize01(inv.astype(np.float32), fov) + th) * 0.5, fov)
    else:
        inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_bank(inv_f, fov, sigmas, lambdas, angle_step)
    if median > 0:
        r = cv2.medianBlur((gab * 255).astype(np.uint8), median); r[~fov] = 0
        gab = normalize01(r.astype(np.float32), fov)
    u8 = np.clip(gab * 255, 0, 255).astype(np.uint8); u8[~fov] = 0
    enh_s = _clahe(u8, clahe2_clip, 12); enh_s[~fov] = 0
    return normalize01(enh_s.astype(np.float32), fov)


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
    ds = []
    for d in subset:
        soft = build_soft(d["rgb"], d["fov"], *params)
        ds.append(segmentation_metrics(predict(soft, d["fov"]), d["gt"])["dice"])
    return float(np.mean(ds))


VARIANTS = {
    # name: (sigmas, lambdas, angle_step, use_tophat, median, clahe2_clip)
    "baseline":        ([1.5, 2.5, 3.5, 5.0], [3, 5, 7, 10], 15, False, 7, 12.0),
    "finer_angles":    ([1.5, 2.5, 3.5, 5.0], [3, 5, 7, 10], 10, False, 7, 12.0),
    "coarse_angles":   ([1.5, 2.5, 3.5, 5.0], [3, 5, 7, 10], 30, False, 7, 12.0),
    "thin_scales":     ([1.0, 1.5, 2.5, 3.5], [2, 3, 5, 7], 15, False, 7, 12.0),
    "thick_scales":    ([2.0, 3.5, 5.0, 7.0], [4, 7, 10, 14], 15, False, 7, 12.0),
    "more_scales":     ([1.0, 1.5, 2.5, 3.5, 5.0, 7.0], [2, 3, 5, 7, 10, 14], 15, False, 7, 12.0),
    "tophat_pre":      ([1.5, 2.5, 3.5, 5.0], [3, 5, 7, 10], 15, True, 7, 12.0),
    "tophat_thin":     ([1.0, 1.5, 2.5, 3.5], [2, 3, 5, 7], 15, True, 7, 12.0),
    "median5":         ([1.5, 2.5, 3.5, 5.0], [3, 5, 7, 10], 15, False, 5, 12.0),
    "no_median":       ([1.5, 2.5, 3.5, 5.0], [3, 5, 7, 10], 15, False, 0, 12.0),
    "thin_finer":      ([1.0, 1.5, 2.5, 3.5], [2, 3, 5, 7], 10, False, 7, 12.0),
}


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}\nBASELINE TEST to beat = 0.4469; round-1 best TEST = 0.4505\n", flush=True)

    results = []
    for name, params in VARIANTS.items():
        t = time.time()
        d = score(train, params)
        results.append((d, name, params))
        print(f"  TRAIN {d:.4f}  {name:16s} ({time.time()-t:.0f}s)", flush=True)
    results.sort(reverse=True)

    print("\n=== Validate top-4 on TEST ===", flush=True)
    rows = []
    for d, name, params in results[:4]:
        td = score(test, params)
        rows.append({"variant": name, "train_dice": round(d, 4), "test_dice": round(td, 4),
                     "sigmas": params[0], "lambdas": params[1], "angle_step": params[2],
                     "tophat": params[3], "median": params[4], "clahe2": params[5]})
        print(f"  TEST {td:.4f}  TRAIN {d:.4f}  {name}", flush=True)
    rows.sort(key=lambda r: r["test_dice"], reverse=True)

    out = OUTPUT_DIR / "vessel_gabor_tune.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nBest TEST = {rows[0]['test_dice']:.4f} ({rows[0]['variant']})")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
