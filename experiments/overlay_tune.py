"""
Sweep variations of the overlay_results vessel pipeline to beat mean Dice 0.466
on the SAME 8-image subset (first 4 RetCam + first 4 Neo from Agrawal2021).

Control = exact overlay_results config (mean Dice ~0.4662).
We vary: target_density, keep_top_components, median ksize, clahe2 params,
and optional morphological close, then report mean Dice per variant.
"""
import sys
import csv
import itertools
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import (  # noqa: E402
    VesselPipelineConfig,
    find_agrawal_pairs,
    read_rgb,
    resize_max_side,
    estimate_fov_mask,
    read_binary_mask,
    normalize01,
    threshold_response_map,
    segmentation_metrics,
)
from advanced_pipeline import gabor_filter_response  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output"
AGRAWAL_ROOT = PROJECT_ROOT / "data" / "Agrawal2021"
CONFIG = VesselPipelineConfig()


def build_soft(rgb, fov, clahe1_clip, clahe1_tile, median_ksize, clahe2_clip, clahe2_tile):
    """Produce the sharpened soft vesselness map (everything before thresholding)."""
    green = rgb[:, :, 1].copy()
    green[~fov] = 0
    c1 = cv2.createCLAHE(clipLimit=clahe1_clip, tileGridSize=(clahe1_tile, clahe1_tile))
    enh = c1.apply(green)
    enh[~fov] = 0
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    r = cv2.medianBlur((gab * 255).astype(np.uint8), median_ksize)
    r[~fov] = 0
    soft = normalize01(r.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8)
    u8[~fov] = 0
    c2 = cv2.createCLAHE(clipLimit=clahe2_clip, tileGridSize=(clahe2_tile, clahe2_tile))
    enh_s = c2.apply(u8)
    enh_s[~fov] = 0
    return normalize01(enh_s.astype(np.float32), fov)


def predict(soft, fov, density, top, close_ksize=0):
    th = threshold_response_map(soft, fov, method="percentile", target_density=density) > 0
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    if nl > 1 and top > 0:
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)]
        areas.sort(reverse=True)
        keep = {idx for _, idx in areas[: min(top, len(areas))]}
        th = np.isin(labels, list(keep))
    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        th = cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        th &= fov
    return th


def main():
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    selected = (
        [p for p in pairs if p["source"] == "RetCam"][:4]
        + [p for p in pairs if p["source"] == "Neo"][:4]
    )

    # Preload images + precompute soft maps keyed by (clahe1,median,clahe2) to avoid recompute
    loaded = []
    for pair in selected:
        rgb = read_rgb(pair["image_path"])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair["mask_path"], fov.shape)
        loaded.append((pair, working, fov, gt))

    # ── Variant grid ──
    # Baseline soft params held near the original; vary threshold/cleanup most.
    soft_param_sets = [
        # (clahe1_clip, clahe1_tile, median, clahe2_clip, clahe2_tile)
        (6.0, 16, 7, 8.0, 12),    # softer 2nd clahe (winner family)
        (6.0, 16, 7, 9.0, 12),
        (6.0, 16, 7, 10.0, 12),
        (6.0, 16, 7, 7.0, 12),
        (6.0, 16, 5, 8.0, 12),
        (7.0, 16, 7, 8.0, 12),
    ]
    densities = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    tops = [3]  # winner; keep fixed
    closes = [0, 3, 5]

    soft_cache = {}
    results = []
    for sp in soft_param_sets:
        for density, top, close in itertools.product(densities, tops, closes):
            dices = []
            rows = []
            for (pair, working, fov, gt) in loaded:
                key = (id(pair), sp)
                if key not in soft_cache:
                    soft_cache[key] = build_soft(working, fov, *sp)
                soft = soft_cache[key]
                pred = predict(soft, fov, density, top, close)
                m = segmentation_metrics(pred, gt)
                dices.append(m["dice"])
                rows.append(m)
            mean_dice = float(np.mean(dices))
            results.append({
                "clahe1_clip": sp[0], "clahe1_tile": sp[1], "median": sp[2],
                "clahe2_clip": sp[3], "clahe2_tile": sp[4],
                "target_density": density, "keep_top": top, "close_ksize": close,
                "mean_dice": mean_dice,
                "mean_iou": float(np.mean([r["iou"] for r in rows])),
                "mean_precision": float(np.mean([r["precision"] for r in rows])),
                "mean_sensitivity": float(np.mean([r["sensitivity"] for r in rows])),
            })

    results.sort(key=lambda r: r["mean_dice"], reverse=True)

    sweep_path = OUTPUT_DIR / "overlay_sweep_summary.csv"
    with open(sweep_path, "w", newline="") as f:
        cols = ["clahe1_clip", "clahe1_tile", "median", "clahe2_clip", "clahe2_tile",
                "target_density", "keep_top", "close_ksize",
                "mean_dice", "mean_iou", "mean_precision", "mean_sensitivity"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    BASELINE = 0.4662
    print(f"Control baseline (overlay_results) mean Dice = {BASELINE:.4f}\n")
    print("Top 12 variants:")
    print(f"{'dice':>7s} {'iou':>6s} {'prec':>6s} {'sens':>6s} | "
          f"{'dens':>5s} {'top':>3s} {'cls':>3s} {'c1':>4s} {'tl1':>3s} {'med':>3s} {'c2':>4s} {'tl2':>3s}")
    for r in results[:12]:
        flag = "  <-- beats baseline" if r["mean_dice"] > BASELINE else ""
        print(f"{r['mean_dice']:7.4f} {r['mean_iou']:6.3f} {r['mean_precision']:6.3f} "
              f"{r['mean_sensitivity']:6.3f} | {r['target_density']:5.2f} {r['keep_top']:3d} "
              f"{r['close_ksize']:3d} {r['clahe1_clip']:4.1f} {r['clahe1_tile']:3d} {r['median']:3d} "
              f"{r['clahe2_clip']:4.1f} {r['clahe2_tile']:3d}{flag}")

    best = results[0]
    print(f"\nBest mean Dice = {best['mean_dice']:.4f} "
          f"({'+' if best['mean_dice'] >= BASELINE else ''}{best['mean_dice'] - BASELINE:.4f} vs baseline)")
    print(f"Wrote {sweep_path}")
    return best, BASELINE


if __name__ == "__main__":
    main()
