"""
Fast variant search: precompute soft maps ONCE per (builder, image), then sweep
thresholds + cleanup cheaply. Tune on TRAIN, validate on TEST.
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
from vessel_variants_v1 import (  # noqa: E402
    SOFT_BUILDERS as _ALL_BUILDERS, thr_percentile, thr_otsu, thr_triangle,
    post_topk, post_minarea, post_close, ls_cf,
)
from vessel_pipeline import segmentation_metrics  # noqa: E402

SOFT_BUILDERS = {
    "gabor_c12": _ALL_BUILDERS["gabor_c12"],
    "gabor_c9": _ALL_BUILDERS["gabor_c9"],
    "gabor_nomed": _ALL_BUILDERS["gabor_nomed"],
    "fusion_balanced": _ALL_BUILDERS["fusion_balanced"],
}


def precompute_soft(builders, subset):
    """Return {builder_key: [soft_map per image]} computed once."""
    cache = {}
    for sk, fn in builders.items():
        t = time.time()
        cache[sk] = [fn(d["rgb"], d["fov"]) for d in subset]
        print(f"  precomputed {sk:18s} {len(subset)} imgs in {time.time()-t:.1f}s", flush=True)
    return cache


THRESHOLDS = {
    "P0.12": lambda s, f: thr_percentile(s, f, 0.12),
    "P0.14": lambda s, f: thr_percentile(s, f, 0.14),
    "P0.16": lambda s, f: thr_percentile(s, f, 0.16),
    "P0.18": lambda s, f: thr_percentile(s, f, 0.18),
    "P0.20": lambda s, f: thr_percentile(s, f, 0.20),
    "otsu": thr_otsu,
    "triangle": thr_triangle,
}


def eval_cached(softs, subset, thr_fn, top, minarea, close, lscf):
    dices, precs, senss = [], [], []
    for soft, d in zip(softs, subset):
        fov = d["fov"]
        mask = thr_fn(soft, fov)
        if lscf:
            mask = ls_cf(soft, mask, fov)
        mask = post_topk(mask, fov, top)
        mask = post_minarea(mask, fov, minarea)
        mask = post_close(mask, fov, close)
        m = segmentation_metrics(mask, d["gt"])
        dices.append(m["dice"]); precs.append(m["precision"]); senss.append(m["sensitivity"])
    return float(np.mean(dices)), float(np.mean(precs)), float(np.mean(senss))


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}\n")

    print("Precomputing soft maps (TRAIN)...", flush=True)
    soft_train = precompute_soft(SOFT_BUILDERS, train)

    # Full grid on TRAIN (cheap now)
    print("\n=== Full grid on TRAIN ===", flush=True)
    grid = []
    combos = list(itertools.product(
        [0, 3, 5], [0, 30], [0, 3], [False, True]
    ))
    for sk in SOFT_BUILDERS:
        softs = soft_train[sk]
        for tk, tfn in THRESHOLDS.items():
            for top, minarea, close, lscf in combos:
                d, p, s = eval_cached(softs, train, tfn, top, minarea, close, lscf)
                grid.append((d, sk, tk, top, minarea, close, lscf, p, s))
    grid.sort(reverse=True)
    print("Top 15 TRAIN configs:")
    for d, sk, tk, top, minarea, close, lscf, p, s in grid[:15]:
        print(f"  {d:.4f} | {sk:18s} {tk:8s} top={top} minA={minarea} close={close} "
              f"lscf={int(lscf)} | prec={p:.3f} sens={s:.3f}", flush=True)

    # Validate top-8 distinct TRAIN configs on TEST
    print("\n=== Validate top-8 on TEST ===", flush=True)
    soft_test = precompute_soft(SOFT_BUILDERS, test)
    rows = []
    seen = set()
    for d, sk, tk, top, minarea, close, lscf, p, s in grid:
        key = (sk, tk, top, minarea, close, lscf)
        if key in seen:
            continue
        seen.add(key)
        td, tp, ts = eval_cached(soft_test[sk], test, THRESHOLDS[tk], top, minarea, close, lscf)
        rows.append({
            "soft": sk, "threshold": tk, "top": top, "minarea": minarea,
            "close": close, "lscf": int(lscf),
            "train_dice": round(d, 4), "test_dice": round(td, 4),
            "test_prec": round(tp, 4), "test_sens": round(ts, 4),
        })
        if len(seen) >= 8:
            break
    rows.sort(key=lambda r: r["test_dice"], reverse=True)
    print(f"{'TEST':>7s} {'TRAIN':>7s} | config")
    for r in rows:
        print(f"{r['test_dice']:7.4f} {r['train_dice']:7.4f} | {r['soft']} {r['threshold']} "
              f"top={r['top']} minA={r['minarea']} close={r['close']} lscf={r['lscf']}", flush=True)

    out = OUTPUT_DIR / "vessel_variants_fast.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nBASELINE to beat: TEST=0.4469")
    print(f"Best here: TEST={rows[0]['test_dice']:.4f}  (TRAIN={rows[0]['train_dice']:.4f})")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
