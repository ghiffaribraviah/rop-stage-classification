"""
Round 4: change the SOFT-MAP BUILDER itself (the real ceiling).
Threshold/cleanup/preprocessing all saturated at TEST~0.460 (rounds 1-3).

New lever: multiscale Hessian ridge filters (Frangi / Sato / Meijering) on the
CLAHE-enhanced inverted green channel. These detect tubular structures via the
local Hessian eigenstructure -- fundamentally different from oriented Gabor
energy. We test:
  (a) pure ridge filters as the soft map
  (b) ridge filter BLENDED with the round-2 Gabor+tophat soft map (best of both)
  (c) sigma-range sweeps for the multiscale set

Downstream fixed at the rounds 1-3 champion: percentile P0.16 + top-3 CCs + close 3x3.
Tune on TRAIN(60); validate the best few on TEST(40).
"""
import sys
import csv
import time
from pathlib import Path

import numpy as np
import cv2
from skimage.filters import frangi, sato, meijering

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


def ridge_map(rgb, fov, kind, sigmas):
    inv = _inv_green(rgb, fov)
    sig = tuple(sigmas)
    if kind == "frangi":
        r = frangi(inv, sigmas=sig, black_ridges=False)
    elif kind == "sato":
        r = sato(inv, sigmas=sig, black_ridges=False)
    elif kind == "meijering":
        r = meijering(inv, sigmas=sig, black_ridges=False)
    else:
        raise ValueError(kind)
    r = r.astype(np.float32)
    r[~fov] = 0
    return normalize01(r, fov)


def blend_map(rgb, fov, kind, sigmas, beta):
    """beta*ridge + (1-beta)*gabor_soft (round-2 winner config)."""
    ridge = ridge_map(rgb, fov, kind, sigmas)
    gab = build_gabor_soft(rgb, fov, 0.5, FULL_KS, 15, 7, 12.0)
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


def score(subset, builder):
    out = []
    for d in subset:
        soft = builder(d["rgb"], d["fov"])
        out.append(segmentation_metrics(predict(soft, d["fov"]), d["gt"])["dice"])
    return float(np.mean(out))


S_NARROW = (1, 2, 3)
S_MID = (1, 2, 3, 4, 5)
S_WIDE = (1, 2, 3, 5, 7)


def make_builders():
    b = {}
    # (a) pure ridge filters, sigma sweeps
    for kind in ("frangi", "sato", "meijering"):
        for sname, sig in (("narrow", S_NARROW), ("mid", S_MID), ("wide", S_WIDE)):
            b[f"{kind}_{sname}"] = lambda rgb, fov, k=kind, s=sig: ridge_map(rgb, fov, k, s)
    # (b) blends with round-2 gabor soft (only the most promising kind picked after pure run,
    #     but we pre-stage a few common blends here)
    for kind in ("frangi", "sato"):
        for beta in (0.3, 0.5, 0.7):
            b[f"blend_{kind}_{int(beta*100)}"] = (
                lambda rgb, fov, k=kind, be=beta: blend_map(rgb, fov, k, S_MID, be)
            )
    return b


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}", flush=True)
    print("BASELINE TEST=0.4469 | rounds1-3 best TEST=0.4600 (gabor+tophat)\n", flush=True)

    builders = make_builders()
    results = []
    for name, fn in builders.items():
        t = time.time()
        try:
            d = score(train, fn)
        except Exception as e:  # noqa: BLE001
            print(f"  TRAIN  ----  {name:18s} FAILED: {e}", flush=True)
            continue
        results.append((d, name, fn))
        print(f"  TRAIN {d:.4f}  {name:18s} ({time.time()-t:.0f}s)", flush=True)
    results.sort(reverse=True)

    print("\n=== Validate top-5 on TEST ===", flush=True)
    rows = []
    for d, name, fn in results[:5]:
        td = score(test, fn)
        rows.append({"variant": name, "train_dice": round(d, 4), "test_dice": round(td, 4)})
        print(f"  TEST {td:.4f}  TRAIN {d:.4f}  {name}", flush=True)
    rows.sort(key=lambda r: r["test_dice"], reverse=True)

    out = OUTPUT_DIR / "vessel_round4.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = rows[0]
    print(f"\nBest TEST = {best['test_dice']:.4f} ({best['variant']})", flush=True)
    print("vs rounds1-3 best 0.4600 | baseline 0.4469", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
