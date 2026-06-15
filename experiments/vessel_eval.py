"""
Full-dataset vessel evaluation harness for Agrawal2021 (100 pairs).

Provides:
  - load_dataset(): cache all 100 image/fov/gt triples
  - split: deterministic train/test (60/40) so tuning never sees test
  - eval_config(): run a named pipeline fn over a subset, return mean metrics
  - registered pipeline builders shared across experiments

Run directly to print honest baselines (overlay_results + overlay_best) on
TRAIN, TEST, and ALL.
"""
import sys
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

_CACHE = None


def load_dataset():
    """Return list of dicts: source,name,rgb(working),fov,gt. Cached in-process."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    data = []
    for pair in pairs:
        try:
            rgb = read_rgb(pair["image_path"])
            working = resize_max_side(rgb, CONFIG.process_max_side)
            fov = estimate_fov_mask(working)
            gt = read_binary_mask(pair["mask_path"], fov.shape)
            data.append({
                "source": pair["source"], "name": pair["name"],
                "rgb": working, "fov": fov, "gt": gt,
            })
        except Exception as e:
            print(f"skip {pair.get('name')}: {e}")
    _CACHE = data
    return data


def split_dataset(data, test_frac=0.4, seed=42):
    """Deterministic stratified split by source. Returns (train, test)."""
    rng = np.random.RandomState(seed)
    train, test = [], []
    for src in sorted(set(d["source"] for d in data)):
        items = [d for d in data if d["source"] == src]
        idx = rng.permutation(len(items))
        n_test = int(round(len(items) * test_frac))
        test_idx = set(idx[:n_test].tolist())
        for i, it in enumerate(items):
            (test if i in test_idx else train).append(it)
    return train, test


def eval_config(pipeline_fn, subset):
    """Run pipeline_fn(rgb,fov)->bool mask over subset, return (mean_metrics, per_image)."""
    keys = ["dice", "iou", "precision", "sensitivity", "specificity", "accuracy"]
    per = []
    for d in subset:
        pred = pipeline_fn(d["rgb"], d["fov"])
        m = segmentation_metrics(pred, d["gt"])
        per.append({"source": d["source"], "name": d["name"], **{k: m[k] for k in keys}})
    mean = {k: float(np.mean([p[k] for p in per])) for k in keys}
    return mean, per


# ── Shared building blocks ──

def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def soft_overlay(rgb, fov, clahe1_clip=6.0, clahe1_tile=16, median=7,
                 clahe2_clip=12.0, clahe2_tile=12):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, clahe1_clip, clahe1_tile); enh[~fov] = 0
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    r = cv2.medianBlur((gab * 255).astype(np.uint8), median); r[~fov] = 0
    soft = normalize01(r.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8); u8[~fov] = 0
    enh_s = _clahe(u8, clahe2_clip, clahe2_tile); enh_s[~fov] = 0
    return normalize01(enh_s.astype(np.float32), fov)


def threshold_and_clean(soft, fov, density, top, close_ksize=0):
    th = threshold_response_map(soft, fov, method="percentile", target_density=density) > 0
    if top > 0:
        nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
        if nl > 1:
            areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
            keep = {idx for _, idx in areas[: min(top, len(areas))]}
            th = np.isin(labels, list(keep))
    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        th = cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & fov
    return th


def make_overlay_results():
    def fn(rgb, fov):
        soft = soft_overlay(rgb, fov, 6.0, 16, 7, 12.0, 12)
        return threshold_and_clean(soft, fov, 0.16, 2, 0)
    return fn


def make_overlay_best():
    def fn(rgb, fov):
        soft = soft_overlay(rgb, fov, 6.0, 16, 7, 9.0, 12)
        return threshold_and_clean(soft, fov, 0.16, 3, 5)
    return fn


if __name__ == "__main__":
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"Dataset: {len(data)} pairs  |  train={len(train)}  test={len(test)}\n")

    for name, fn in [("overlay_results", make_overlay_results()),
                     ("overlay_best", make_overlay_best())]:
        mtr, _ = eval_config(fn, train)
        mte, _ = eval_config(fn, test)
        mall, _ = eval_config(fn, data)
        print(f"{name:18s} ALL dice={mall['dice']:.4f}  "
              f"TRAIN={mtr['dice']:.4f}  TEST={mte['dice']:.4f}  "
              f"(prec={mall['precision']:.3f} sens={mall['sensitivity']:.3f})")
