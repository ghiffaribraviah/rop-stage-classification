"""
Round 3: tune AROUND the round-2 winner `tophat_pre` (TEST 0.4600).
Levers explored:
  (a) blend weight alpha:  inv_f = alpha*norm(inv) + (1-alpha)*tophat
  (b) top-hat kernel-size set (thin vs full vs thick structuring elements)
  (c) stack tophat with finer 10-deg angles (neutral-but-free lever from round 2)
  (d) tophat + lighter median (median5) since tophat already denoises lines
Fixed downstream: percentile P0.16 + top-3 components + close 3x3.
Tune on TRAIN(60), validate top-4 on TEST(40).
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_eval import load_dataset, split_dataset, OUTPUT_DIR  # noqa: E402
from vessel_pipeline import (  # noqa: E402
    normalize01, threshold_response_map, segmentation_metrics,
)
from advanced_pipeline import gabor_kernel_2d  # noqa: E402


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def tophat_custom(inverted, fov, kernel_sizes):
    maps = []
    for ks in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        top_hat = cv2.subtract(inverted, opened).astype(np.float32)
        maps.append(top_hat)
    response = np.max(np.stack(maps, axis=0), axis=0)
    response = cv2.GaussianBlur(response, (0, 0), 0.8)
    return normalize01(response, fov)


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


SIGMAS = [1.5, 2.5, 3.5, 5.0]
LAMBDAS = [3, 5, 7, 10]
FULL_KS = (5, 7, 9, 13, 17, 23, 31)


def build_soft(rgb, fov, alpha, kernel_sizes, angle_step, median, clahe2_clip):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, 6.0, 16); enh[~fov] = 0
    inv = 255 - enh
    th = tophat_custom(inv, fov, kernel_sizes)
    inv_f = normalize01(alpha * normalize01(inv.astype(np.float32), fov) + (1 - alpha) * th, fov)
    gab = gabor_bank(inv_f, fov, SIGMAS, LAMBDAS, angle_step)
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
    out = []
    for d in subset:
        soft = build_soft(d["rgb"], d["fov"], *params)
        out.append(segmentation_metrics(predict(soft, d["fov"]), d["gt"])["dice"])
    return float(np.mean(out))


VARIANTS = {
    # name: (alpha, kernel_sizes, angle_step, median, clahe2_clip)
    "r2_winner":      (0.5, FULL_KS, 15, 7, 12.0),         # reproduce round-2 best
    "alpha_065":      (0.65, FULL_KS, 15, 7, 12.0),
    "alpha_035":      (0.35, FULL_KS, 15, 7, 12.0),
    "alpha_020":      (0.20, FULL_KS, 15, 7, 12.0),
    "alpha_080":      (0.80, FULL_KS, 15, 7, 12.0),
    "ks_thin":        (0.5, (3, 5, 7, 9, 13), 15, 7, 12.0),
    "ks_thick":       (0.5, (9, 13, 17, 23, 31, 41), 15, 7, 12.0),
    "ks_mid":         (0.5, (5, 9, 13, 19), 15, 7, 12.0),
    "alpha035_finer": (0.35, FULL_KS, 10, 7, 12.0),
    "alpha035_med5":  (0.35, FULL_KS, 15, 5, 12.0),
    "alpha035_thin":  (0.35, (3, 5, 7, 9, 13), 15, 7, 12.0),
}


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}", flush=True)
    print("BASELINE TEST=0.4469 | round-2 best TEST=0.4600 (tophat_pre)\n", flush=True)

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
                     "alpha": params[0], "kernel_sizes": params[1], "angle_step": params[2],
                     "median": params[3], "clahe2": params[4]})
        print(f"  TEST {td:.4f}  TRAIN {d:.4f}  {name}", flush=True)
    rows.sort(key=lambda r: r["test_dice"], reverse=True)

    out = OUTPUT_DIR / "vessel_round3.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nBest TEST = {rows[0]['test_dice']:.4f} ({rows[0]['variant']})", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
