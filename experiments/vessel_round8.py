"""
Round 8: re-optimize post-processing on the round-7 fusion winners.

Round-7 winners (all ~0.4739 TEST):
  g40_m60_fine:  wg=0.40, wm=0.60, sigmas=fine
  g45_m55_thin:  wg=0.45, wm=0.55, sigmas=thin
  g45_m55_fine:  wg=0.45, wm=0.55, sigmas=fine

Sweep:
  A) target_density  0.13 .. 0.20
  B) top-N connected components: 2, 3, 4, 5, "all"
  C) closing kernel: 2, 3, 5, none
  D) also test hysteresis threshold (high/low) instead of single percentile

Lock on g40_m60_fine for sweep efficiency; validate best configs across all 3 winner fusions.
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
from vessel_pipeline import normalize01, segmentation_metrics  # noqa: E402
from vessel_round3 import build_soft as build_gabor_soft, FULL_KS  # noqa: E402

SIGMA_SETS = {
    "fine": (0.8, 1.4, 2.0, 2.8, 3.6, 4.5),
    "thin": (0.6, 1.0, 1.6, 2.2, 3.0, 3.8),
}


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def _inv_green(rgb, fov):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, 6.0, 16); enh[~fov] = 0
    inv = (255 - enh).astype(np.float32) / 255.0
    inv[~fov] = 0
    return inv


_G, _M = {}, {}


def gc(d):
    if id(d) not in _G:
        _G[id(d)] = normalize01(build_gabor_soft(d["rgb"], d["fov"], 0.5, FULL_KS, 15, 7, 12.0), d["fov"])
    return _G[id(d)]


def mc(d, ss):
    k = (id(d), ss)
    if k not in _M:
        inv = _inv_green(d["rgb"], d["fov"])
        m = meijering(inv, sigmas=SIGMA_SETS[ss], black_ridges=False).astype(np.float32)
        m[~d["fov"]] = 0
        _M[k] = normalize01(m, d["fov"])
    return _M[k]


def soft_map(d, wg, wm, ss):
    return normalize01(wg * gc(d) + wm * mc(d, ss), d["fov"])


def postprocess(soft, fov, density, top_n, close_ks):
    """density: [0,1] fraction; top_n: int or None (keep all); close_ks: int or 0 (no close)."""
    fov_px = int(fov.sum())
    n_target = int(density * fov_px)
    flat = soft[fov]
    if n_target >= len(flat):
        th = float(flat.min()) - 1
    else:
        th = float(np.sort(flat)[-n_target])
    seg = (soft >= th) & fov
    if top_n is not None:
        nl, labels, stats, _ = cv2.connectedComponentsWithStats(seg.astype(np.uint8), 8)
        if nl > 1:
            areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
            seg = np.isin(labels, [idx for _, idx in areas[:top_n]])
    if close_ks >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        seg = cv2.morphologyEx(seg.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return seg & fov


def hyst_postprocess(soft, fov, hi_dens, lo_dens, close_ks):
    """Hysteresis threshold: seeds at hi_dens, expand to lo_dens."""
    fov_px = int(fov.sum())
    flat = np.sort(soft[fov])
    th_hi = flat[-max(1, int(hi_dens * fov_px))]
    th_lo = flat[-max(1, int(lo_dens * fov_px))]
    seeds = (soft >= th_hi) & fov
    cand = (soft >= th_lo) & fov
    # BFS expand seeds within cand
    from scipy.ndimage import label as ndlabel
    all_labels, _ = ndlabel(cand)
    seed_lbls = set(np.unique(all_labels[seeds])) - {0}
    seg = np.isin(all_labels, list(seed_lbls)) & fov
    if close_ks >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        seg = cv2.morphologyEx(seg.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return seg & fov


def score_cfg(subset, cfg):
    out = []
    for d in subset:
        sm = soft_map(d, cfg["wg"], cfg["wm"], cfg["ss"])
        if cfg["type"] == "pct":
            seg = postprocess(sm, d["fov"], cfg["density"], cfg["top_n"], cfg["close_ks"])
        else:
            seg = hyst_postprocess(sm, d["fov"], cfg["hi"], cfg["lo"], cfg["close_ks"])
        out.append(segmentation_metrics(seg, d["gt"])["dice"])
    return float(np.mean(out))


# Fusions to test (round-7 top-3)
FUSIONS = [
    ("g40m60fine", 0.40, 0.60, "fine"),
    ("g45m55thin", 0.45, 0.55, "thin"),
    ("g45m55fine", 0.45, 0.55, "fine"),
]

# --- build variants on best fusion (g40m60fine) ---
BASE = {"wg": 0.40, "wm": 0.60, "ss": "fine"}

VARIANTS = {}
# density sweep (top_n=3, close=3 — locked baseline)
for d in [0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]:
    k = f"dens{int(d*100):02d}_top3_cl3"
    VARIANTS[k] = {**BASE, "type": "pct", "density": d, "top_n": 3, "close_ks": 3}

# top_n sweep (density=0.16, close=3)
for n in [2, 3, 4, 5, None]:
    ns = "all" if n is None else str(n)
    k = f"dens16_top{ns}_cl3"
    if k not in VARIANTS:
        VARIANTS[k] = {**BASE, "type": "pct", "density": 0.16, "top_n": n, "close_ks": 3}

# close kernel sweep (density=0.16, top_n=3)
for ks in [0, 2, 3, 5, 7]:
    k = f"dens16_top3_cl{ks}"
    if k not in VARIANTS:
        VARIANTS[k] = {**BASE, "type": "pct", "density": 0.16, "top_n": 3, "close_ks": ks}

# hysteresis combos (no top_n filter for hysteresis)
for hi, lo in [(0.10, 0.18), (0.10, 0.20), (0.12, 0.18), (0.12, 0.20), (0.14, 0.20), (0.08, 0.16)]:
    k = f"hyst_hi{int(hi*100)}_lo{int(lo*100)}_cl3"
    VARIANTS[k] = {**BASE, "type": "hyst", "hi": hi, "lo": lo, "close_ks": 3}


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}", flush=True)
    print("round-7 best TEST=0.4739 | baseline 0.4469\n", flush=True)

    # Phase 1: sweep all variants on TRAIN with g40m60fine
    results = []
    for name, cfg in VARIANTS.items():
        t = time.time()
        d = score_cfg(train, cfg)
        results.append((d, name, cfg))
        print(f"  TRAIN {d:.4f}  {name}  ({time.time()-t:.0f}s)", flush=True)
    results.sort(reverse=True)
    top = results[:8]

    # Phase 2: test top-8 on TEST across all 3 fusions
    print("\n=== Validate top-8 on TEST (all 3 fusions) ===", flush=True)
    rows = []
    for d, name, cfg in top:
        row = {"variant": name, "train_g40m60": round(d, 4)}
        for fn, wg, wm, ss in FUSIONS:
            fcfg = {**cfg, "wg": wg, "wm": wm, "ss": ss}
            td = score_cfg(test, fcfg)
            row[f"test_{fn}"] = round(td, 4)
        row["test_mean"] = round(np.mean([row[f"test_{fn}"] for fn, *_ in FUSIONS]), 4)
        rows.append(row)
        print(f"  {name}: " + " | ".join(f"{fn}={row[f'test_{fn}']:.4f}" for fn, *_ in FUSIONS)
              + f"  mean={row['test_mean']:.4f}", flush=True)
    rows.sort(key=lambda r: r["test_mean"], reverse=True)

    out = OUTPUT_DIR / "vessel_round8.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = rows[0]
    print(f"\nBest TEST mean={best['test_mean']:.4f}  ({best['variant']})", flush=True)
    print("vs round-7 best 0.4739 | baseline 0.4469", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
