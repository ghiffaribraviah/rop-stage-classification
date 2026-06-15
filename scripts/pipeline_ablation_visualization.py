from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi

from almeida_workflow_visualization import (
    DEFAULT_OUTPUT,
    add_label,
    agrawal_rows,
    estimate_fov_mask,
    jerman_vesselness_with_scales,
    normalize01,
    read_binary_mask,
    read_rgb,
    resize_max_side,
)
from threshold_workflow_visualization import (
    ZHAO_ROOT,
    clean_balanced_noise,
    coherence_gate,
    connected_hysteresis_mask,
    connected_support_mask,
    connected_support_response,
    matched_filter_response,
    response_percentile_floor,
    smooth_response_preserve_edges,
    suppress_fov_border,
    threshold_response_maps,
    zhao_rows,
)


DEFAULT_ABLATION_OUTPUT = DEFAULT_OUTPUT.with_name("debug_pipeline_ablation_15_samples.jpg")


def dark_source_base(source: np.ndarray, fov: np.ndarray) -> np.ndarray:
    dark = 255.0 - source.astype(np.float32)
    dark[~fov] = 0
    dark = response_percentile_floor(dark, fov, floor_pct=35.0, high_pct=99.2, gamma=0.95)
    dark = cv2.GaussianBlur(dark, (0, 0), sigmaX=0.35, sigmaY=0.35, borderType=cv2.BORDER_REFLECT)
    dark[~fov] = 0
    return normalize01(dark, fov)


def mask_from_response(
    maps: dict[str, np.ndarray],
    fov: np.ndarray,
    high_pct: float = 97.5,
    low_pct: float = 90.0,
    support_floor: float = 0.14,
    support_pct_floor: float = 42.0,
    min_area: int = 16,
) -> np.ndarray:
    mask = connected_hysteresis_mask(
        maps["response"],
        fov,
        high_pct=high_pct,
        low_pct=low_pct,
        min_area=min_area,
        support=maps["support"],
        support_floor=support_floor,
        support_pct_floor=support_pct_floor,
    )
    return clean_balanced_noise(mask, maps["response"], maps["support"], maps["coherence"], fov)


def no_zscore_maps(source: np.ndarray, fov: np.ndarray, simple: bool) -> dict[str, np.ndarray]:
    base = dark_source_base(source, fov)
    matched = matched_filter_response(base, fov, size=15, sigma=2.0)
    matched_small = matched_filter_response(base, fov, size=9, sigma=1.2)
    jerman = jerman_vesselness_with_scales(base, fov, sigmas=(1.0, 2.0, 3.0, 5.0), tau=0.90)
    jerman_small = jerman_vesselness_with_scales(base, fov, sigmas=(0.6, 0.9, 1.2, 1.8), tau=0.90)
    frangi_map = normalize01(
        frangi(base, sigmas=(0.7, 1.0, 1.5, 2.5, 4.0), alpha=0.5, beta=15.0, black_ridges=False),
        fov,
    )
    coherence = coherence_gate(source, fov)
    coherence = response_percentile_floor(coherence, fov, floor_pct=42.0, high_pct=98.0)

    if simple:
        support_raw = normalize01(0.45 * matched + 0.25 * matched_small + 0.22 * jerman + 0.08 * jerman_small, fov)
        raw = normalize01(0.46 * matched + 0.26 * matched_small + 0.20 * jerman + 0.08 * jerman_small, fov)
        support_floor_pct = 46.0
        support_gamma = 0.9
        support_strong_pct = 88.0
        support_weak_pct = 58.0
        response_floor_pct = 50.0
    else:
        support_raw = normalize01(0.43 * matched + 0.25 * matched_small + 0.24 * jerman + 0.08 * jerman_small, fov)
        raw = normalize01(
            0.44 * matched + 0.26 * matched_small + 0.22 * jerman + 0.08 * jerman_small,
            fov,
        )
        support_floor_pct = 38.0
        support_gamma = 0.8
        support_strong_pct = 86.0
        support_weak_pct = 54.0
        response_floor_pct = 42.0

    support_raw = response_percentile_floor(
        support_raw,
        fov,
        floor_pct=support_floor_pct,
        high_pct=99.0,
        gamma=support_gamma,
    )
    support_mask = connected_support_mask(
        support_raw,
        coherence,
        fov,
        strong_pct=support_strong_pct,
        weak_pct=support_weak_pct,
    )
    softened = cv2.GaussianBlur(support_mask.astype(np.float32), (0, 0), sigmaX=0.75, sigmaY=0.75)
    support = response_percentile_floor(
        support_raw * np.clip(0.08 + 0.92 * softened, 0.0, 1.0),
        fov,
        floor_pct=36.0,
        high_pct=99.1,
        gamma=0.95,
    )
    gated = raw * (0.16 + 0.84 * support) * (0.62 + 0.38 * coherence)
    response = smooth_response_preserve_edges(gated, fov)
    response = response_percentile_floor(response, fov, floor_pct=response_floor_pct, high_pct=99.4, gamma=1.02)
    return {
        "base": base,
        "matched": matched,
        "jerman": jerman,
        "frangi": frangi_map,
        "support_raw": support_raw,
        "support": support,
        "support_mask": support_mask,
        "coherence": coherence,
        "response": response,
    }


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, fov: np.ndarray) -> np.ndarray:
    base = rgb.copy()
    base[~fov] = 0
    overlay = base.copy()
    overlay[mask] = (0, 220, 120)
    return cv2.addWeighted(base, 0.62, overlay, 0.38, 0)


def binary_metrics(pred: np.ndarray, gt: np.ndarray | None, fov: np.ndarray) -> dict[str, float]:
    if gt is None:
        return {}
    pred = pred.astype(bool) & fov
    gt = gt.astype(bool) & fov
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    return {"precision": precision, "recall": recall, "dice": dice}


def workflow_tiles(rgb: np.ndarray, fov: np.ndarray, gt: np.ndarray | None) -> tuple[list[tuple[str, np.ndarray]], dict[str, dict[str, float]]]:
    current = threshold_response_maps(rgb, fov)
    valid_fov = current["processing_fov"]
    source = current["source"]

    current_mask = mask_from_response(current, valid_fov)
    no_z = no_zscore_maps(source, valid_fov, simple=False)
    no_z_mask = mask_from_response(
        no_z,
        valid_fov,
        high_pct=95.5,
        low_pct=82.0,
        support_floor=0.08,
        support_pct_floor=28.0,
        min_area=10,
    )
    simple = no_zscore_maps(source, valid_fov, simple=True)
    simple_mask = mask_from_response(simple, valid_fov)

    metrics = {
        "current": binary_metrics(current_mask, gt, valid_fov),
        "no_zscore": binary_metrics(no_z_mask, gt, valid_fov),
        "simple_no_zscore": binary_metrics(simple_mask, gt, valid_fov),
    }
    tiles = [
        ("Source norm", source),
        ("Current zscore", current["zscore"]),
        ("Current support", current["support_connected"]),
        ("Current resp", current["response"]),
        ("Current mask", current_mask),
        ("Current overlay", overlay_mask(rgb, current_mask, valid_fov)),
        ("Dark source", no_z["base"]),
        ("No-z matched", no_z["matched"]),
        ("No-z Jerman", no_z["jerman"]),
        ("No-z support", no_z["support"]),
        ("No-z resp", no_z["response"]),
        ("No-z mask", no_z_mask),
        ("No-z overlay", overlay_mask(rgb, no_z_mask, valid_fov)),
        ("Simple support", simple["support"]),
        ("Simple resp", simple["response"]),
        ("Simple mask", simple_mask),
        ("Simple overlay", overlay_mask(rgb, simple_mask, valid_fov)),
    ]
    return tiles, metrics


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    all_metrics: dict[str, list[dict[str, float]]] = {"current": [], "no_zscore": [], "simple_no_zscore": []}
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = suppress_fov_border(estimate_fov_mask(rgb))
        gt = read_binary_mask(mask_path, fov.shape) if mask_path is not None else None
        tiles, metrics = workflow_tiles(rgb, fov, gt)
        for key, values in metrics.items():
            if values:
                all_metrics[key].append(values)
        display_maps: list[tuple[str, np.ndarray]] = [(f"{index}. {name}", rgb), ("GT", gt if gt is not None else np.zeros(fov.shape, dtype=bool))]
        display_maps.extend(tiles)
        row_tiles = [add_label(image, label, tile_size) for label, image in display_maps]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))

    for label, rows_metrics in all_metrics.items():
        if not rows_metrics:
            continue
        means = {
            metric: float(np.mean([row[metric] for row in rows_metrics]))
            for metric in ("precision", "recall", "dice")
        }
        print(
            f"{label}: "
            f"precision={means['precision']:.4f} "
            f"recall={means['recall']:.4f} "
            f"dice={means['dice']:.4f}"
        )

    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current vessel pipeline against no-zscore simplifications.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ABLATION_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--zhao-count", type=int, default=5)
    args = parser.parse_args()

    rows = agrawal_rows(args.retcam_count, args.neo_count)
    rows.extend(zhao_rows(ZHAO_ROOT, args.zhao_count))
    if not rows:
        raise RuntimeError("No debug images found.")
    print(build_sheet(rows, args.output, tile_size=args.tile_size, max_side=args.max_side))


if __name__ == "__main__":
    main()
