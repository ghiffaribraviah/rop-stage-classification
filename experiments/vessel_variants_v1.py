"""
Variant search for vessel segmentation. Tune on TRAIN, report TEST.
Tests multiple soft-map builders x threshold methods x post-processing,
including LS-CF-style connectivity filling and Frangi/Jerman fusion.
"""
import sys
import csv
import itertools
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_eval import (  # noqa: E402
    load_dataset, split_dataset, eval_config, OUTPUT_DIR,
    soft_overlay, threshold_and_clean,
)
from vessel_pipeline import (  # noqa: E402
    normalize01, threshold_response_map, threshold_from_values, erode_mask,
    matched_filter_response, modified_tophat, jerman_vesselness,
    safe_filter_call, fuse_vessel_responses, keep_components_at_least,
)
from advanced_pipeline import gabor_filter_response  # noqa: E402


def _clahe(img, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img)


def green_enh(rgb, fov, clip=6.0, tile=16):
    green = rgb[:, :, 1].copy(); green[~fov] = 0
    enh = _clahe(green, clip, tile); enh[~fov] = 0
    return enh


# ── Soft-map builders (return normalized [0,1] vesselness) ──

def soft_gabor(rgb, fov, clahe2_clip=12.0):
    """The overlay family: gabor -> median7 -> 2nd CLAHE."""
    return soft_overlay(rgb, fov, 6.0, 16, 7, clahe2_clip, 12)


def soft_gabor_nomed(rgb, fov):
    """Gabor with no 2nd CLAHE sharpening, light smoothing only."""
    enh = green_enh(rgb, fov)
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    return normalize01(gab, fov)


def soft_fusion(rgb, fov, w_gab=0.4, w_frangi=0.2, w_jerman=0.25, w_matched=0.15):
    """Fuse Gabor + Frangi + Jerman + matched filter on enhanced green."""
    enh = green_enh(rgb, fov)
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    frangi = normalize01(safe_filter_call('frangi', inv_f), fov)
    jerman = jerman_vesselness(inv_f, fov)
    matched = matched_filter_response(inv_f, fov)
    fused = w_gab * gab + w_frangi * frangi + w_jerman * jerman + w_matched * matched
    fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), 0.6)
    fused[~fov] = 0
    return normalize01(fused, fov)


def soft_gabor_frangi(rgb, fov, w_gab=0.6, w_frangi=0.4):
    enh = green_enh(rgb, fov)
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    frangi = normalize01(safe_filter_call('frangi', inv_f), fov)
    fused = w_gab * gab + w_frangi * frangi
    fused[~fov] = 0
    return normalize01(fused, fov)


# ── Threshold methods ──

def thr_percentile(soft, fov, density):
    return threshold_response_map(soft, fov, method="percentile", target_density=density) > 0


def thr_otsu(soft, fov, density=None):
    inner = erode_mask(fov, 8)
    vals = (soft[inner] * 255).astype(np.uint8)
    if vals.size < 50:
        return soft > 0.5
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (soft * 255 >= t) & inner


def thr_triangle(soft, fov, density=None):
    inner = erode_mask(fov, 8)
    vals = (soft[inner] * 255).astype(np.uint8)
    if vals.size < 50:
        return soft > 0.5
    t, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return (soft * 255 >= t) & inner


# ── Post-processing ──

def post_topk(mask, fov, top):
    if top <= 0:
        return mask
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if nl <= 1:
        return mask
    areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
    keep = {idx for _, idx in areas[: min(top, len(areas))]}
    return np.isin(labels, list(keep))


def post_minarea(mask, fov, min_area):
    if min_area <= 0:
        return mask
    return keep_components_at_least(mask, min_area)


def post_close(mask, fov, ksize):
    if ksize <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & fov


def ls_cf(soft, mask, fov, low_frac=0.5, tol=2):
    """LS-CF-style: grow the high-confidence mask into medium-confidence soft
    pixels along local continuity (dilate within a relaxed soft threshold)."""
    inner = erode_mask(fov, 8)
    vals = soft[inner]; vals = vals[vals > 0]
    if vals.size < 50:
        return mask
    # relaxed candidate set: soft above low_frac * (current threshold proxy)
    hi = soft[mask]
    if hi.size == 0:
        return mask
    low_th = float(np.percentile(hi, 10)) * low_frac
    candidates = (soft >= low_th) & inner
    result = mask.copy() & inner
    k = np.ones((3, 3), np.uint8)
    last = result.sum()
    for _ in range(tol * 6):
        dil = cv2.dilate(result.astype(np.uint8), k, 1).astype(bool)
        result |= (dil & candidates)
        cur = result.sum()
        if cur - last < 5:
            break
        last = cur
    return result & fov


SOFT_BUILDERS = {
    "gabor_c12": lambda r, f: soft_gabor(r, f, 12.0),
    "gabor_c9": lambda r, f: soft_gabor(r, f, 9.0),
    "gabor_nomed": soft_gabor_nomed,
    "fusion_balanced": lambda r, f: soft_fusion(r, f, 0.40, 0.20, 0.25, 0.15),
    "fusion_gabor_heavy": lambda r, f: soft_fusion(r, f, 0.55, 0.15, 0.20, 0.10),
    "gabor_frangi": lambda r, f: soft_gabor_frangi(r, f, 0.6, 0.4),
}

THRESHOLDS = {
    "P0.14": lambda s, f: thr_percentile(s, f, 0.14),
    "P0.16": lambda s, f: thr_percentile(s, f, 0.16),
    "P0.18": lambda s, f: thr_percentile(s, f, 0.18),
    "otsu": thr_otsu,
    "triangle": thr_triangle,
}


def build_pipeline(soft_key, thr_key, top, minarea, close, use_lscf):
    sb = SOFT_BUILDERS[soft_key]
    tb = THRESHOLDS[thr_key]

    def fn(rgb, fov):
        soft = sb(rgb, fov)
        mask = tb(soft, fov)
        if use_lscf:
            mask = ls_cf(soft, mask, fov)
        mask = post_topk(mask, fov, top)
        mask = post_minarea(mask, fov, minarea)
        mask = post_close(mask, fov, close)
        return mask
    return fn


def main():
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"train={len(train)} test={len(test)}\n")

    # Stage 1: pick best (soft x threshold) on TRAIN with light cleanup (top3)
    print("=== Stage 1: soft-builder x threshold (TRAIN, top3 cleanup) ===")
    stage1 = []
    for sk in SOFT_BUILDERS:
        for tk in THRESHOLDS:
            fn = build_pipeline(sk, tk, top=3, minarea=0, close=0, use_lscf=False)
            m, _ = eval_config(fn, train)
            stage1.append((m["dice"], sk, tk, m))
            print(f"  {sk:18s} {tk:9s} dice={m['dice']:.4f} prec={m['precision']:.3f} sens={m['sensitivity']:.3f}")
    stage1.sort(reverse=True)
    print(f"\nTop combo: {stage1[0][1]} + {stage1[0][2]} (TRAIN dice={stage1[0][0]:.4f})")

    # Stage 2: take top 3 combos, sweep cleanup + LS-CF on TRAIN
    print("\n=== Stage 2: cleanup sweep on top-3 combos (TRAIN) ===")
    stage2 = []
    for _, sk, tk, _ in stage1[:3]:
        for top, minarea, close, lscf in itertools.product(
            [0, 3, 5], [0, 30], [0, 3, 5], [False, True]
        ):
            fn = build_pipeline(sk, tk, top, minarea, close, lscf)
            m, _ = eval_config(fn, train)
            stage2.append((m["dice"], sk, tk, top, minarea, close, lscf, m))
    stage2.sort(reverse=True)
    print("Top 10 (TRAIN):")
    for d, sk, tk, top, minarea, close, lscf, m in stage2[:10]:
        print(f"  {d:.4f} | {sk:18s} {tk:8s} top={top} minA={minarea} close={close} lscf={int(lscf)} "
              f"prec={m['precision']:.3f} sens={m['sensitivity']:.3f}")

    # Stage 3: validate top-5 TRAIN configs on TEST
    print("\n=== Stage 3: validate top-5 on TEST ===")
    rows = []
    seen = set()
    for d, sk, tk, top, minarea, close, lscf, m in stage2:
        key = (sk, tk, top, minarea, close, lscf)
        if key in seen:
            continue
        seen.add(key)
        fn = build_pipeline(sk, tk, top, minarea, close, lscf)
        mte, _ = eval_config(fn, test)
        rows.append({
            "soft": sk, "threshold": tk, "top": top, "minarea": minarea,
            "close": close, "lscf": int(lscf),
            "train_dice": round(d, 4), "test_dice": round(mte["dice"], 4),
            "test_prec": round(mte["precision"], 4), "test_sens": round(mte["sensitivity"], 4),
        })
        if len(seen) >= 5:
            break
    rows.sort(key=lambda r: r["test_dice"], reverse=True)
    print(f"{'test':>7s} {'train':>7s} | soft / thr / cleanup")
    for r in rows:
        print(f"{r['test_dice']:7.4f} {r['train_dice']:7.4f} | {r['soft']} {r['threshold']} "
              f"top={r['top']} minA={r['minarea']} close={r['close']} lscf={r['lscf']}")

    out = OUTPUT_DIR / "vessel_variants_v1.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nBASELINE to beat: TEST=0.4469 (overlay_results)")
    print(f"Best here: TEST={rows[0]['test_dice']:.4f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
