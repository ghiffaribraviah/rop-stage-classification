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
    coherence_gate,
    connected_hysteresis_mask,
    fill_outside_fov,
    local_clahe_channel,
    matched_filter_response,
    multiscale_local_dark_zscore_response,
    response_percentile_floor,
    smooth_response_preserve_edges,
    suppress_fov_border,
    zhao_rows,
)


DEFAULT_SOURCE_OUTPUT = DEFAULT_OUTPUT.with_name("debug_source_normalization_15_samples.jpg")


def clahe_green_source(
    rgb: np.ndarray,
    fov: np.ndarray,
    l_clip: float,
    green_clip: float,
    prefilter: str | None = None,
    flatten: str | None = None,
) -> np.ndarray:
    working = rgb.copy()
    if prefilter == "median3":
        working = cv2.medianBlur(working, 3)
    elif prefilter == "bilateral":
        working = cv2.bilateralFilter(working, d=5, sigmaColor=18, sigmaSpace=5)

    lab = cv2.cvtColor(working, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=float(l_clip), tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    green = enhanced[:, :, 1].astype(np.float32)

    if flatten == "gaussian_divide":
        filled = fill_outside_fov(green, fov)
        background = cv2.GaussianBlur(filled, (0, 0), sigmaX=30.0, sigmaY=30.0, borderType=cv2.BORDER_REFLECT)
        scale = float(np.median(background[fov])) if np.any(fov) else 1.0
        green = filled * scale / np.maximum(background, 1.0)
    elif flatten == "morph_close":
        filled = np.clip(fill_outside_fov(green, fov), 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        background = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel).astype(np.float32)
        center = float(np.median(background[fov])) if np.any(fov) else 0.0
        green = filled.astype(np.float32) - background + center

    green = np.clip(green, 0, 255).astype(np.uint8)
    source = local_clahe_channel(green, fov, clip_limit=float(green_clip), tile_grid_size=(16, 16))
    source[~fov] = 0
    return source


def response_maps_from_source(source: np.ndarray, fov: np.ndarray) -> dict[str, np.ndarray]:
    zscore = multiscale_local_dark_zscore_response(source, fov)
    matched = matched_filter_response(zscore, fov, size=15, sigma=2.0)
    matched_small = matched_filter_response(zscore, fov, size=9, sigma=1.2)
    jerman = jerman_vesselness_with_scales(zscore, fov, sigmas=(1.0, 2.0, 3.0, 5.0), tau=0.90)
    jerman_small = jerman_vesselness_with_scales(zscore, fov, sigmas=(0.6, 0.9, 1.2, 1.8), tau=0.90)
    frangi_map = normalize01(
        frangi(zscore, sigmas=(0.7, 1.0, 1.5, 2.5, 4.0), alpha=0.5, beta=15.0, black_ridges=False),
        fov,
    )
    raw = normalize01(
        0.36 * zscore + 0.24 * matched + 0.18 * matched_small + 0.14 * jerman + 0.08 * frangi_map,
        fov,
    )
    support = normalize01(0.36 * matched + 0.24 * matched_small + 0.28 * jerman + 0.12 * jerman_small, fov)
    support = response_percentile_floor(support, fov, floor_pct=45.0, high_pct=99.0, gamma=0.80)
    coherence = response_percentile_floor(coherence_gate(source, fov), fov, floor_pct=35.0, high_pct=97.0)
    line_gate = np.clip(0.30 + 0.70 * support * (0.35 + 0.65 * coherence), 0.0, 1.0)
    supported = normalize01(raw * line_gate, fov)
    smooth = smooth_response_preserve_edges(supported, fov)
    response = normalize01(0.70 * smooth + 0.18 * supported + 0.12 * normalize01(raw * coherence, fov), fov)
    return {"response": response, "support": support}


def balanced_hysteresis(response: np.ndarray, support: np.ndarray, fov: np.ndarray) -> np.ndarray:
    return connected_hysteresis_mask(
        response,
        fov,
        high_pct=97.0,
        low_pct=87.0,
        min_area=12,
        support=support,
        support_floor=0.12,
    )


def source_variants(rgb: np.ndarray, fov: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("C6G6", clahe_green_source(rgb, fov, l_clip=6.0, green_clip=6.0)),
        ("C4G4", clahe_green_source(rgb, fov, l_clip=4.0, green_clip=4.0)),
        ("C4G6", clahe_green_source(rgb, fov, l_clip=4.0, green_clip=6.0)),
        ("Median+C6G6", clahe_green_source(rgb, fov, l_clip=6.0, green_clip=6.0, prefilter="median3")),
        ("Bilateral+C6G6", clahe_green_source(rgb, fov, l_clip=6.0, green_clip=6.0, prefilter="bilateral")),
        ("GaussFlat+C6G6", clahe_green_source(rgb, fov, l_clip=6.0, green_clip=6.0, flatten="gaussian_divide")),
        ("MorphFlat+C6G6", clahe_green_source(rgb, fov, l_clip=6.0, green_clip=6.0, flatten="morph_close")),
    ]


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = suppress_fov_border(estimate_fov_mask(rgb))
        gt = read_binary_mask(mask_path, fov.shape)
        display_maps: list[tuple[str, np.ndarray]] = [
            (f"{index}. {name}", rgb),
            ("FOV", fov),
            ("GT", gt),
        ]
        for label, source in source_variants(rgb, fov):
            maps = response_maps_from_source(source, fov)
            hyst = balanced_hysteresis(maps["response"], maps["support"], fov)
            display_maps.extend(
                [
                    (f"{label} src", source),
                    (f"{label} resp", maps["response"]),
                    (f"{label} hyst", hyst),
                ]
            )
        row_tiles = [add_label(image, label, tile_size) for label, image in display_maps]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare source normalization variants with fixed balanced hysteresis.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
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
