"""
Replication of a friend's "Gabor + CLAHE P10" vessel-segmentation pipeline.

Implemented faithfully from the provided implementation spec (not from any repo).
Reuses only the project's I/O + metric helpers for a fair, comparable evaluation
on the Agrawal2021 HVDROPDB-BV pairs (same loader/split as vessel_eval.py).

Friend's claimed tuning-set scores (best-tuned config):
    clDice 0.4888 | Dice 0.4624 | Precision 0.5094 | Recall 0.4234 | Acc 0.8751

Output: console report + experiments/output/friend_gabor_p10_summary.csv
"""
import sys
import csv
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import normalize01, segmentation_metrics  # noqa: E402
from vessel_eval import load_dataset, split_dataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Best tuned configuration (from the spec) ───────────────────────────
NORM_PCT = False
BEST_CONFIG = {
    "target_density": 0.115,
    "main_low_mult": 1.25,
    "main_high_mult": 0.55,
    "residual_enabled": True,
    "residual_low_mult": 1.22,
    "recovery_axis_ratio": 2.6,
    "recovery_skeleton_length": 8,
    "recovery_branch_density": 0.10,
}

RECALL_CONFIG = {
    "target_density": 0.115,
    "main_low_mult": 1.45,
    "main_high_mult": 0.60,
    "residual_enabled": True,
    "residual_low_mult": 1.22,
    "recovery_axis_ratio": 2.6,
    "recovery_skeleton_length": 8,
    "recovery_branch_density": 0.10,
}


# ── Small helpers ──────────────────────────────────────────────────────

def _norm01_in(img, mask):
    img = img.astype(np.float32)
    vals = img[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(img)
    if NORM_PCT:
        lo, hi = np.percentile(vals, [1, 99])
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
    else:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros_like(img)
    out = (img - float(lo)) / (float(hi) - float(lo))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fill_outside_fov_nearest(green, fov):
    """Nearest-FOV-pixel fill for pixels outside the FOV (spec Step 2)."""
    green = green.astype(np.float32)
    if fov.all():
        return green.copy()
    # distance transform indices of nearest in-FOV pixel
    inv = ~fov
    _, (iy, ix) = ndi.distance_transform_edt(inv, return_indices=True)
    filled = green.copy()
    filled[inv] = green[iy[inv], ix[inv]]
    return filled


def percentile_stretch(img, mask, low_pct, high_pct):
    """Stretch intensities so [low_pct, high_pct] (in-mask) map to [0, 255]."""
    img = img.astype(np.float32)
    vals = img[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(img)
    lo = np.percentile(vals, low_pct)
    hi = np.percentile(vals, high_pct)
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return np.zeros_like(img)
    out = (img - lo) / (hi - lo) * 255.0
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def cl_dice(pred, gt):
    """clDice = harmonic mean of topology precision/sensitivity (Shit et al.)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    s_pred = skeletonize(pred)
    s_gt = skeletonize(gt)
    eps = 1e-8
    # topology precision: skeleton of pred inside gt
    tprec = (np.logical_and(s_pred, gt).sum()) / (s_pred.sum() + eps)
    # topology sensitivity: skeleton of gt inside pred
    tsens = (np.logical_and(s_gt, pred).sum()) / (s_gt.sum() + eps)
    if (tprec + tsens) <= eps:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def fov_outline_subtraction_mask(fov):
    """Thin FOV outline mask to subtract later (spec Step 1)."""
    h, w = fov.shape
    min_side = min(h, w)
    thickness = int(np.clip(round(min_side * 0.034), 12, 26))
    outline = np.zeros_like(fov, dtype=np.uint8)
    cnts, _ = cv2.findContours(fov.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outline, cnts, -1, 1, thickness=thickness)
    return (outline.astype(bool) & fov)


# ── Soft response (spec Steps 2-7) ─────────────────────────────────────

def build_soft_response(rgb, fov, fov_outline):
    """Steps 2-7: green -> flatten -> dark boost -> CLAHE+invert -> Gabor
    -> median -> CLAHE2 -> soft_response. Returns (soft_response, debug)."""
    h, w = fov.shape
    min_side = min(h, w)

    # Step 2: green + fill outside FOV
    green = rgb[:, :, 1].astype(np.float32)
    green_filled = fill_outside_fov_nearest(green, fov)

    # Step 3: illumination flattening
    median_value = float(np.median(green_filled[fov]))
    background_sigma = float(np.clip(round(min_side * 0.035), 12, 32))
    background = ndi.gaussian_filter(green_filled, sigma=background_sigma)
    background = np.maximum(background, 1.0)
    flattened = green_filled * median_value / background

    # Step 4: multi-scale dark-vessel boost
    local_background = ndi.gaussian_filter(flattened, sigma=2.0)
    local_dark = np.maximum(local_background - flattened, 0.0)
    local_dark = ndi.gaussian_filter(local_dark, sigma=0.35)
    local_dark_norm = _norm01_in(local_dark, fov)

    scale_sigmas = (1.4, 2.2, 3.6, 5.5, 8.0)
    scale_maps = []
    for s in scale_sigmas:
        sb = ndi.gaussian_filter(flattened, sigma=s)
        sd = np.maximum(sb - flattened, 0.0)
        sd = ndi.gaussian_filter(sd, sigma=0.35)
        scale_maps.append(_norm01_in(sd, fov))
    multiscale_dark = np.maximum.reduce(scale_maps)
    multiscale_dark = cv2.medianBlur(
        (np.clip(multiscale_dark, 0, 1) * 255).astype(np.uint8), 3
    ).astype(np.float32) / 255.0

    vessel_boost = _norm01_in(0.35 * local_dark_norm + 0.65 * multiscale_dark, fov)
    boosted_green = flattened - 42.0 * vessel_boost
    boosted_green = ndi.gaussian_filter(boosted_green, sigma=0.25)
    boosted_green = percentile_stretch(boosted_green, fov, 0.7, 99.3)
    boosted_green[~fov] = 0

    # Step 5: first CLAHE + inversion
    bg_u8 = np.clip(boosted_green, 0, 255).astype(np.uint8)
    clahe1 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16)).apply(bg_u8)
    inverted = (255 - clahe1).astype(np.float32)
    inverted[~fov] = float(np.median(inverted[fov]))

    # Step 6: Gabor filter bank
    wavelengths = (8.0, 12.0, 16.0)
    sigma_g = 4.0
    gamma = 0.50
    angles = range(0, 180, 15)
    inv_f = inverted / 255.0
    response = np.zeros_like(inv_f, dtype=np.float32)
    for lam in wavelengths:
        ksize = int(max(17, round(lam * 2.6)))
        if ksize % 2 == 0:
            ksize += 1
        for deg in angles:
            theta = np.deg2rad(deg)
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma_g, theta, lam, gamma, 0.0,
                ktype=cv2.CV_32F,
            )
            kern -= kern.mean()
            denom = np.abs(kern).sum()
            if denom > 0:
                kern /= denom
            filt = cv2.filter2D(inv_f, cv2.CV_32F, kern)
            response = np.maximum(response, filt)
    response = np.maximum(response, 0.0)
    gabor_norm = _norm01_in(response, fov)
    gabor_norm[~fov] = 0

    # Step 7: median, second CLAHE, soft response
    median7 = cv2.medianBlur((gabor_norm * 255).astype(np.uint8), 7).astype(np.float32) / 255.0
    m7_u8 = np.clip(median7 * 255, 0, 255).astype(np.uint8)
    clahe2 = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(12, 12)).apply(m7_u8).astype(np.float32)

    median_norm = _norm01_in(median7, fov)
    clahe2_norm = _norm01_in(clahe2, fov)
    soft_response = _norm01_in(0.65 * median_norm + 0.35 * clahe2_norm, fov)
    soft_response[~fov] = 0
    soft_response[fov_outline] = 0

    debug = {
        "green_filled": green_filled, "flattened": flattened,
        "boosted_green": boosted_green, "clahe1": clahe1,
        "inverted": inverted, "gabor_norm": gabor_norm,
        "median7": median7, "clahe2": clahe2,
    }
    return soft_response, debug


# ── Hysteresis density threshold (spec Step 8) ─────────────────────────

def hysteresis_density(response, fov, target_density, low_mult, high_mult):
    low_density = float(np.clip(target_density * low_mult, 0.015, 0.20))
    high_density = float(np.clip(target_density * high_mult, 0.008, low_density * 0.85))

    vals = response[fov]
    if vals.size == 0:
        return np.zeros_like(fov, dtype=bool)
    low_threshold = np.percentile(vals, 100 * (1 - low_density))
    high_threshold = np.percentile(vals, 100 * (1 - high_density))

    low_mask = (response >= low_threshold) & fov
    high_mask = (response >= high_threshold) & fov

    nl, labels = cv2.connectedComponents(low_mask.astype(np.uint8), 8)
    seed_labels = set(np.unique(labels[high_mask]).tolist()) - {0}

    h, w = fov.shape
    min_side = min(h, w)
    min_area = int(np.clip(round(min_side * min_side * 0.000035), 8, 28))

    out = np.zeros_like(fov, dtype=bool)
    if not seed_labels:
        return out
    for lab in seed_labels:
        comp = labels == lab
        if comp.sum() >= min_area:
            out |= comp
    return out


def keep_largest_components(mask, count):
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if nl <= 1:
        return mask.astype(bool)
    areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)], reverse=True)
    keep = [idx for _, idx in areas[:count]]
    return np.isin(labels, keep)


def _close3(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1).astype(bool)


# ── Threshold pass 1 (spec Step 9) ─────────────────────────────────────

def threshold_pass1(soft_response, fov, fov_outline, cfg):
    raw = hysteresis_density(
        soft_response, fov, cfg["target_density"],
        cfg["main_low_mult"], cfg["main_high_mult"],
    )
    raw &= ~fov_outline
    top2 = keep_largest_components(raw, 2)
    final = _close3(top2) & fov & ~fov_outline
    return final, raw


# ── Threshold pass 2: residual recovery (spec Steps 10-11) ─────────────

def _component_is_vessel_like(comp, residual_soft, residual_fov, cfg, min_side):
    min_area = int(np.clip(round(min_side * min_side * 0.000025), 8, 24))
    area = int(comp.sum())
    if area < min_area:
        return False

    coords = np.column_stack(np.nonzero(comp)).astype(np.float64)
    if coords.shape[0] < 2:
        return False
    cov = np.cov(coords.T)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-9, None)
    axis_ratio = float(np.sqrt(eig[-1] / eig[0]))
    if axis_ratio < cfg["recovery_axis_ratio"]:
        return False

    skel = skeletonize(comp)
    skel_len = int(np.count_nonzero(skel))
    if skel_len < cfg["recovery_skeleton_length"]:
        return False

    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    neighbor_count = ndi.convolve(skel.astype(np.uint8), neighbor_kernel, mode="constant")
    branch_points = int(np.count_nonzero((neighbor_count >= 5) & skel))
    branch_density = branch_points / max(skel_len, 1)
    if branch_density > cfg["recovery_branch_density"]:
        return False

    rvals = residual_soft[residual_fov]
    p45, p82, p93 = np.percentile(rvals, [45, 82, 93])
    comp_vals = residual_soft[comp]
    mean_response = float(comp_vals.mean())
    max_response = float(comp_vals.max())
    strong_enough = (mean_response >= p45 and max_response >= p82) or (max_response >= p93)
    return bool(strong_enough)


def threshold_pass2(soft_response, fov, fov_outline, threshold1_final, cfg):
    block = cv2.dilate(
        threshold1_final.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
    ).astype(bool)
    residual_fov = fov & ~block & ~fov_outline

    residual_soft = soft_response.copy()
    residual_soft[~residual_fov] = 0
    residual_soft = _norm01_in(residual_soft, residual_fov)

    target_density_2 = cfg["target_density"] * cfg["residual_low_mult"]
    candidates = hysteresis_density(
        residual_soft, residual_fov, target_density_2,
        cfg["main_low_mult"], cfg["main_high_mult"],
    )

    h, w = fov.shape
    min_side = min(h, w)
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    recovered = np.zeros_like(fov, dtype=bool)
    for i in range(1, nl):
        comp = labels == i
        if _component_is_vessel_like(comp, residual_soft, residual_fov, cfg, min_side):
            recovered |= comp

    top2 = keep_largest_components(recovered, 2) if recovered.any() else recovered
    top2 = top2 & fov & ~fov_outline
    return top2, residual_soft, candidates


# ── Final mask: union (spec Step 12) ───────────────────────────────────

def segment(rgb, fov, cfg):
    fov_outline = fov_outline_subtraction_mask(fov)
    soft_response, debug = build_soft_response(rgb, fov, fov_outline)
    t1_final, t1_raw = threshold_pass1(soft_response, fov, fov_outline, cfg)

    if cfg["residual_enabled"]:
        t2_top2, residual_soft, t2_cand = threshold_pass2(
            soft_response, fov, fov_outline, t1_final, cfg,
        )
    else:
        t2_top2 = np.zeros_like(fov, dtype=bool)
        residual_soft = np.zeros_like(soft_response)
        t2_cand = t2_top2

    final_candidates = t1_final | t2_top2
    mask_final = _close3(final_candidates) & fov & ~fov_outline

    return {
        "soft_response": soft_response,
        "threshold1_final": t1_final,
        "residual_soft": residual_soft,
        "residual_candidates": t2_cand,
        "threshold2_largest2": t2_top2,
        "mask_final": mask_final,
        "fov_outline_subtraction": fov_outline,
        **debug,
    }


# ── Eval harness ───────────────────────────────────────────────────────

def evaluate(subset, cfg):
    rows = []
    for d in subset:
        out = segment(d["rgb"], d["fov"], cfg)
        pred = out["mask_final"]
        gt = d["gt"]
        m = segmentation_metrics(pred, gt)
        m["cldice"] = cl_dice(pred, gt)
        rows.append(m)
    keys = ["dice", "cldice", "precision", "sensitivity", "specificity",
            "accuracy", "pred_density", "gt_density"]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}, len(rows)


def _report(name, cfg):
    data = load_dataset()
    train, test = split_dataset(data)
    print(f"\n{'='*64}\n{name}\n{'='*64}", flush=True)
    print(f"config: {cfg}", flush=True)
    results = {}
    for split_name, subset in [("train", train), ("test", test), ("all", data)]:
        means, n = evaluate(subset, cfg)
        results[split_name] = means
        print(
            f"  [{split_name:5s} n={n:3d}] "
            f"Dice={means['dice']:.4f}  clDice={means['cldice']:.4f}  "
            f"Prec={means['precision']:.4f}  Rec={means['sensitivity']:.4f}  "
            f"Acc={means['accuracy']:.4f}",
            flush=True,
        )
    return results


def main():
    global NORM_PCT
    print("Friend's claimed (best-tuned): "
          "clDice=0.4888 Dice=0.4624 Prec=0.5094 Rec=0.4234 Acc=0.8751", flush=True)
    print("Friend's claimed (recall-cfg): "
          "clDice=0.4807 Dice=0.4615 Prec=0.4691 Rec=0.4541", flush=True)

    for norm_pct in (False, True):
        NORM_PCT = norm_pct
        tag = "PERCENTILE-CLIP norm01" if norm_pct else "MIN-MAX norm01"
        best = _report(f"BEST TUNED CONFIG  [{tag}]", BEST_CONFIG)

        out_path = OUTPUT_DIR / f"friend_gabor_p10_{'pct' if norm_pct else 'minmax'}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "split", "dice", "cldice", "precision",
                        "sensitivity", "specificity", "accuracy",
                        "pred_density", "gt_density"])
            for split_name, m in best.items():
                w.writerow(["best_tuned", split_name, m["dice"], m["cldice"],
                            m["precision"], m["sensitivity"], m["specificity"],
                            m["accuracy"], m["pred_density"], m["gt_density"]])
        print(f"  wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
