"""
Reproduce the exact pipeline used to generate experiments/output/overlay_results.jpg
and dump its configuration + per-image and summary metrics to CSV.

Pipeline (from the inline script that produced overlay_results.jpg):
  green channel -> CLAHE(clip=6.0, tile=16x16) -> invert -> normalize
  -> Gabor response -> median7 -> renormalize
  -> second CLAHE(clip=12, tile=12x12) sharpen -> renormalize
  -> percentile threshold @ target_density=0.16
  -> keep 2 largest connected components
Evaluated on Agrawal2021: first 4 RetCam + first 4 Neo image/mask pairs.
"""
import sys
import csv
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / "data" / "Agrawal2021"
CONFIG = VesselPipelineConfig()

# ── Exact config that produced overlay_results.jpg ──
CFG = {
    "method": "overlay_results",
    "channel": "green",
    "clahe1_clip": 6.0,
    "clahe1_tile": 16,
    "invert": True,
    "filter": "gabor",
    "median_blur": 7,
    "clahe2_clip": 12.0,
    "clahe2_tile": 12,
    "threshold_method": "percentile",
    "target_density": 0.16,
    "keep_top_components": 2,
}


def get_pred(rgb, fov, density=0.16, top=2):
    green = rgb[:, :, 1].copy()
    green[~fov] = 0
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
    enh = clahe.apply(green)
    enh[~fov] = 0
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    r7 = cv2.medianBlur((gab * 255).astype(np.uint8), 7)
    r7[~fov] = 0
    soft = normalize01(r7.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8)
    u8[~fov] = 0
    c = cv2.createCLAHE(clipLimit=12, tileGridSize=(12, 12))
    enh_s = c.apply(u8)
    enh_s[~fov] = 0
    sharp = normalize01(enh_s.astype(np.float32), fov)
    th = threshold_response_map(sharp, fov, method="percentile", target_density=density) > 0
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    if nl > 1:
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)]
        areas.sort(reverse=True)
        keep = {idx for _, idx in areas[: min(top, len(areas))]}
        th = np.isin(labels, list(keep))
    return th


def main():
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    selected = (
        [p for p in pairs if p["source"] == "RetCam"][:4]
        + [p for p in pairs if p["source"] == "Neo"][:4]
    )

    metric_keys = [
        "dice", "iou", "precision", "sensitivity",
        "specificity", "accuracy", "pred_density", "gt_density",
    ]
    rows = []
    for pair in selected:
        rgb = read_rgb(pair["image_path"])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair["mask_path"], fov.shape)
        pred = get_pred(working, fov, CFG["target_density"], CFG["keep_top_components"])
        m = segmentation_metrics(pred, gt)
        row = {"source": pair["source"], "name": pair["name"]}
        row.update({k: m[k] for k in metric_keys})
        rows.append(row)

    # ── per-image metrics CSV ──
    metrics_path = OUTPUT_DIR / "overlay_results_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "name"] + metric_keys)
        w.writeheader()
        w.writerows(rows)

    # ── summary CSV (mean over images) ──
    summary = {"method": CFG["method"], "n_images": len(rows)}
    for k in metric_keys:
        summary[k] = float(np.mean([r[k] for r in rows]))
    summary_path = OUTPUT_DIR / "overlay_results_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_images"] + metric_keys)
        w.writeheader()
        w.writerow(summary)

    # ── config CSV ──
    config_path = OUTPUT_DIR / "overlay_results_config.csv"
    with open(config_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in CFG.items():
            w.writerow([k, v])

    # ── console report ──
    print("Per-image metrics:")
    print(f"{'source':8s} {'name':28s} {'dice':>7s} {'iou':>7s} {'prec':>7s} {'sens':>7s}")
    for r in rows:
        print(f"{r['source']:8s} {r['name'][:28]:28s} {r['dice']:7.4f} {r['iou']:7.4f} "
              f"{r['precision']:7.4f} {r['sensitivity']:7.4f}")
    print("\nMean over {} images:".format(len(rows)))
    print(f"  dice={summary['dice']:.4f}  iou={summary['iou']:.4f}  "
          f"precision={summary['precision']:.4f}  sensitivity={summary['sensitivity']:.4f}  "
          f"specificity={summary['specificity']:.4f}")
    print("\nWrote:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
