from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from almeida_workflow_visualization import (
    DEFAULT_OUTPUT,
    add_label,
    agrawal_rows,
    estimate_fov_mask,
    normalize01,
    read_binary_mask,
    read_rgb,
    resize_max_side,
)
from threshold_workflow_visualization import (
    ZHAO_ROOT,
    connected_hysteresis_mask,
    fill_outside_fov,
    threshold_response_maps,
    zhao_rows,
)


DEFAULT_NOISE_OUTPUT = DEFAULT_OUTPUT.with_name("debug_noise_diagnostics_15_samples.jpg")


def local_std_map(channel: np.ndarray, fov: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    filled = fill_outside_fov(channel, fov).astype(np.float32)
    kernel_size = max(3, int(kernel_size) | 1)
    mean = cv2.blur(filled, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.blur(filled * filled, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    std[~fov] = 0
    return normalize01(std, fov)


def high_frequency_map(channel: np.ndarray, fov: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    filled = fill_outside_fov(channel, fov).astype(np.float32)
    low_pass = cv2.GaussianBlur(filled, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)
    high = np.abs(filled - low_pass)
    high[~fov] = 0
    return normalize01(high, fov)


def gaussian_background(channel: np.ndarray, fov: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    filled = fill_outside_fov(channel, fov).astype(np.float32)
    background = cv2.GaussianBlur(
        filled,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    background[~fov] = 0
    return normalize01(background, fov)


def background_flattened_source(channel: np.ndarray, fov: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    filled = fill_outside_fov(channel, fov).astype(np.float32)
    background = cv2.GaussianBlur(
        filled,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    scale = float(np.median(background[fov])) if np.any(fov) else 1.0
    corrected = filled * scale / np.maximum(background, 1.0)
    corrected[~fov] = 0
    return normalize01(corrected, fov)


def response_residual_noise(response: np.ndarray, fov: np.ndarray) -> np.ndarray:
    smooth = cv2.GaussianBlur(response.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    residual = np.maximum(response.astype(np.float32) - smooth, 0.0)
    residual[~fov] = 0
    return normalize01(residual, fov)


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


def diagnostic_tiles(maps: dict[str, np.ndarray], fov: np.ndarray) -> list[tuple[str, np.ndarray]]:
    source = maps["source"]
    response = maps["response"]
    support = maps["support"]
    return [
        ("Denoised C4G4 source", source),
        ("Local std 15", local_std_map(source, fov, kernel_size=15)),
        ("Local std 31", local_std_map(source, fov, kernel_size=31)),
        ("High freq", high_frequency_map(source, fov, sigma=1.4)),
        ("Background", gaussian_background(source, fov, sigma=30.0)),
        ("BG flattened", background_flattened_source(source, fov, sigma=30.0)),
        ("Support raw", maps["support_raw"]),
        ("Support conn", support),
        ("Support mask", maps["support_mask"]),
        ("Coherence", maps["coherence"]),
        ("Response", response),
        ("Resp residual", response_residual_noise(response, fov)),
        ("Hyst balanced", balanced_hysteresis(response, support, fov)),
    ]


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, image_path, mask_path) in enumerate(rows, start=1):
        rgb = resize_max_side(read_rgb(image_path), max_side)
        fov = estimate_fov_mask(rgb)
        gt = read_binary_mask(mask_path, fov.shape)
        maps = threshold_response_maps(rgb, fov)
        fov = maps["processing_fov"]
        display_maps: list[tuple[str, np.ndarray]] = [
            (f"{index}. {name}", rgb),
            ("FOV", fov),
            ("GT", gt),
        ]
        display_maps.extend(diagnostic_tiles(maps, fov))
        row_tiles = [add_label(image, label, tile_size) for label, image in display_maps]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize upstream noise diagnostics before thresholding.")
    parser.add_argument("--output", type=Path, default=DEFAULT_NOISE_OUTPUT)
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
