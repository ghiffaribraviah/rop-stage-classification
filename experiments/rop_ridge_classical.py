"""Classical ridge / demarcation-line enhancement for ROP (no learned model).

The ridge is a bright, elongated, curved arc. We enhance it with Hessian-based
ridge filters (Frangi / Sato / Meijering) computed on a CLAHE-normalized green
channel, optionally combined with an oriented black/top-hat response. Everything
is deterministic hand-written image processing - no training, no weights.

This module also provides a tuning harness that measures the ridge response
against the Agrawal2021 HVDROPDB-RIDGE ground-truth masks. Because the ridge is
~1% of pixels, raw Dice is uninformative; we report:
  - recall@density: of the true-ridge pixels, how many fall inside the top-k%
    strongest response pixels (the signal we actually feed downstream)
  - separation: mean response on ridge pixels vs non-ridge pixels (ratio)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi, sato, meijering

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vessel_pipeline import (  # noqa: E402
    read_rgb, resize_max_side, estimate_fov_mask, read_binary_mask,
    normalize01, find_agrawal_pairs,
)

PROCESS_SIDE = 768
RIDGE_SCALES = (3, 5, 7, 9, 11)


def ridge_source_channel(rgb: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """CLAHE-normalized green channel - the ridge is brightest/most coherent here."""
    green = rgb[:, :, 1].copy()
    green[~fov] = 0
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enh = clahe.apply(green)
    enh[~fov] = 0
    return normalize01(enh.astype(np.float32), fov)


def hessian_ridge_response(channel: np.ndarray, fov: np.ndarray, method: str = "frangi") -> np.ndarray:
    """Bright-ridge Hessian response. Ridge is bright, so black_ridges=False."""
    img = channel.astype(np.float64)
    sigmas = list(RIDGE_SCALES)
    if method == "frangi":
        resp = frangi(img, sigmas=sigmas, black_ridges=False)
    elif method == "sato":
        resp = sato(img, sigmas=sigmas, black_ridges=False)
    elif method == "meijering":
        resp = meijering(img, sigmas=sigmas, black_ridges=False)
    else:
        raise ValueError(method)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    resp[~fov] = 0.0
    return normalize01(resp, fov)


def oriented_tophat_response(channel: np.ndarray, fov: np.ndarray,
                             length: int = 21, n_angles: int = 12) -> np.ndarray:
    """Max white-top-hat over oriented line structuring elements (bright arcs)."""
    u8 = np.clip(channel * 255, 0, 255).astype(np.uint8)
    best = np.zeros_like(channel, dtype=np.float32)
    for k in range(n_angles):
        ang = 180.0 * k / n_angles
        base = np.zeros((length, length), np.uint8)
        cv2.line(base, (0, length // 2), (length - 1, length // 2), 255, 1)
        M = cv2.getRotationMatrix2D((length / 2, length / 2), ang, 1.0)
        se = cv2.warpAffine(base, M, (length, length)) > 0
        se = se.astype(np.uint8)
        th = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, se)
        best = np.maximum(best, th.astype(np.float32))
    best[~fov] = 0.0
    return normalize01(best, fov)


def ridge_response_map(rgb: np.ndarray, fov: np.ndarray,
                       method: str = "frangi", use_tophat: bool = True,
                       tophat_weight: float = 0.4) -> np.ndarray:
    channel = ridge_source_channel(rgb, fov)
    hess = hessian_ridge_response(channel, fov, method=method)
    if use_tophat:
        top = oriented_tophat_response(channel, fov)
        resp = (1 - tophat_weight) * hess + tophat_weight * top
    else:
        resp = hess
    return normalize01(resp.astype(np.float32), fov)


def recall_at_density(resp: np.ndarray, gt: np.ndarray, fov: np.ndarray, density: float) -> float:
    vals = resp[fov > 0]
    if vals.size == 0 or gt.sum() == 0:
        return 0.0
    thr = np.percentile(vals, 100 * (1 - density))
    pred = (resp >= thr) & (fov > 0)
    gt_b = gt > 0
    inter = float((pred & gt_b).sum())
    return inter / float(gt_b.sum())


def separation_ratio(resp: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> float:
    gt_b = (gt > 0) & (fov > 0)
    bg = (gt == 0) & (fov > 0)
    if gt_b.sum() == 0 or bg.sum() == 0:
        return 0.0
    return float(resp[gt_b].mean() / (resp[bg].mean() + 1e-6))


def tune() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "Agrawal2021"
    pairs = find_agrawal_pairs(root / "HVDROPDB-RIDGE") if (root / "HVDROPDB-RIDGE").exists() else None
    if not pairs:
        pairs = []
        rd = root / "HVDROPDB-RIDGE"
        for src in ("RetCam", "Neo"):
            idir, mdir = rd / f"{src}_Ridge_images", rd / f"{src}_Ridge_masks"
            for ip in sorted(idir.iterdir()):
                mp = mdir / ip.name
                if mp.exists():
                    pairs.append({"image_path": str(ip), "mask_path": str(mp), "source": src})

    print(f"Tuning ridge filter on {len(pairs)} Agrawal RIDGE pairs\n")
    configs = [
        ("frangi", True, 0.4),
        ("frangi", False, 0.0),
        ("sato", True, 0.4),
        ("meijering", True, 0.4),
        ("frangi", True, 0.6),
    ]
    for method, use_th, tw in configs:
        r10, r16, seps = [], [], []
        for p in pairs:
            rgb = resize_max_side(read_rgb(p["image_path"]), PROCESS_SIDE)
            fov = estimate_fov_mask(rgb)
            gt = read_binary_mask(p["mask_path"], fov.shape)
            resp = ridge_response_map(rgb, fov, method=method, use_tophat=use_th, tophat_weight=tw)
            r10.append(recall_at_density(resp, gt, fov, 0.10))
            r16.append(recall_at_density(resp, gt, fov, 0.16))
            seps.append(separation_ratio(resp, gt, fov))
        tag = f"{method}+top{tw}" if use_th else f"{method}-only"
        print(f"  {tag:18s} recall@10%={np.mean(r10):.3f} recall@16%={np.mean(r16):.3f} "
              f"separation={np.mean(seps):.2f}")


if __name__ == "__main__":
    tune()
