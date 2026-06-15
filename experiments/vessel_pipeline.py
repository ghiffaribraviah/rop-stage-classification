"""
Core vessel pipeline functions extracted from the ROP classification notebook.
Cleaned up, standalone, no Jupyter dependencies.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.filters import frangi, threshold_triangle, threshold_otsu
from skimage.morphology import remove_small_holes, skeletonize
from scipy import ndimage as ndi
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

# ── Types ──────────────────────────────────────────────────────────────

VesselnessMode = Literal["almeida", "ensemble", "legacy_tophat", "almeida_paper"]
ThresholdMethod = Literal["triangle", "otsu", "percentile", "hybrid"]
MorphologyMode = Literal["none", "open", "close", "open_close"]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


@dataclass(frozen=True)
class VesselPipelineConfig:
    process_max_side: int = 768
    vesselness_mode: VesselnessMode = "almeida"
    clahe_clip: float = 2.5
    background_sigma: float = 30.0
    threshold_method: ThresholdMethod = "triangle"
    target_density: float = 0.14
    fov_erode_px: int = 12
    min_component_area: int = 4
    hole_size: int = 0
    morphology: MorphologyMode = "none"
    od_suppression: bool = False
    od_soft_penalty: float = 0.35
    bilateral_d: int = 9
    bilateral_sigma_color: float = 40.0
    bilateral_sigma_space: float = 9.0
    final_skeletonize: bool = False
    skeleton_min_area: int = 20
    skeleton_dilate_iter: int = 1
    auto_binary_selection: bool = False


NORMALIZATION_RESIDUAL_TUNED_BASELINE = VesselPipelineConfig()


# ── Image I/O ──────────────────────────────────────────────────────────

def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# ── Geometry / resize ──────────────────────────────────────────────────

def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image.copy()
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def resize_soft_map(soft: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(soft.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


# ── FOV ────────────────────────────────────────────────────────────────

def estimate_fov_mask(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold = max(3, int(np.percentile(gray, 1)))
    mask = (gray > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask.astype(bool)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


# ── Normalization utils ────────────────────────────────────────────────

def normalize01(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = image[mask] if mask is not None else image.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    output = (image.astype(np.float32) - float(low)) / float(high - low)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


# ── Fill outside FOV ──────────────────────────────────────────────────

def fill_outside_fov(channel: np.ndarray, fov: np.ndarray) -> np.ndarray:
    output = channel.astype(np.float32).copy()
    if np.any(fov):
        output[~fov] = float(np.median(output[fov]))
    return output


def masked_median_value(image: np.ndarray, mask: np.ndarray) -> float:
    values = image[mask]
    if values.size == 0:
        return 0.0
    return float(np.median(values))


# ── Enhancement / channel sources ──────────────────────────────────────

def enhance_cielab_rgb(rgb: np.ndarray, fov: np.ndarray, clahe_clip: float = 2.5) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
    l_enh = clahe.apply(l_ch)
    l_blur = cv2.GaussianBlur(l_enh, (0, 0), 1.2)
    l_sharp = cv2.addWeighted(l_enh, 1.35, l_blur, -0.35, 0)
    l_sharp = np.clip(l_sharp, 0, 255).astype(np.uint8)
    enhanced_rgb = cv2.cvtColor(cv2.merge([l_sharp, a_ch, b_ch]), cv2.COLOR_LAB2RGB)
    enhanced_rgb[~fov] = 0
    return enhanced_rgb


def local_clahe_channel(channel: np.ndarray, fov: np.ndarray,
                        clip_limit: float = 2.2, tile_grid_size: int | tuple[int, int] = 16) -> np.ndarray:
    source = np.clip(channel, 0, 255).astype(np.uint8)
    if isinstance(tile_grid_size, tuple):
        tile_size = tile_grid_size
    else:
        tile_size = (tile_grid_size, tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_size)
    output = clahe.apply(source)
    output[~fov] = 0
    return output.astype(np.uint8)


def cielab_green_clahe_source(rgb: np.ndarray, fov: np.ndarray,
                               l_clip: float = 6.0, green_clip: float = 6.0) -> np.ndarray:
    """C6G6 / C4G4 style: CIELAB L* + green channel with independent CLAHE."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_clahe = cv2.createCLAHE(clipLimit=float(l_clip), tileGridSize=(8, 8)).apply(l_ch)
    l_clahe[~fov] = 0

    green = rgb[:, :, 1].copy()
    green_clahe = cv2.createCLAHE(clipLimit=float(green_clip), tileGridSize=(8, 8)).apply(green)
    green_clahe[~fov] = 0

    combined = np.clip(0.50 * l_clahe.astype(np.float32) + 0.50 * green_clahe.astype(np.float32), 0, 255).astype(np.uint8)
    # ^ source for vessel pipeline is the green channel after CIELAB processing
    # Actually the original notebook uses cielab_rgb[:,:,1] which is the A channel, not a blend.
    # Let me re-check… In the notebook:
    #   c6 = cielab_green_clahe_source(rgb, fov, config, l_clip=6.0, green_clip=6.0)
    # And in debug_flatten_darkline_filter_sources:
    #   c6_green = cielab6_green_pre_clahe(rgb, fov, config)
    # That returns green_clahe from the CIELAB pipeline.
    # Let me use the green channel from the CIELAB-enhanced image.
    return green_clahe


def gaussian_background_flatten(channel: np.ndarray, fov: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Subtract Gaussian background estimate."""
    source = np.clip(fill_outside_fov(channel, fov), 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(source.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    result = source.astype(np.float32) - bg
    result[~fov] = 0
    result = result - float(np.percentile(result[fov], 1)) if np.any(fov) else result
    result[~fov] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def divide_background_flatten(channel: np.ndarray, fov: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Divide by Gaussian background estimate (retinal flat-mount style)."""
    source = np.clip(fill_outside_fov(channel, fov), 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(source.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    bg = np.maximum(bg, 1.0)
    result = source.astype(np.float32) / bg * 128.0
    result[~fov] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def small_blackhat_boost(channel: np.ndarray, fov: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Add scaled blackhat (dark feature extraction) to the original."""
    source = np.clip(channel, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(source, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    result = source.astype(np.float32) + float(strength) * blackhat.astype(np.float32)
    result[~fov] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def blackhat_boost_then_clahe_source(channel: np.ndarray, fov: np.ndarray,
                                      strength: float = 1.25, green_clip: float = 6.0) -> np.ndarray:
    """Blackhat boost → CLAHE."""
    boosted = small_blackhat_boost(channel, fov, strength=strength)
    return local_clahe_channel(boosted, fov, clip_limit=green_clip, tile_grid_size=(8, 8))


def blackhat_isolated_dark_source(channel: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Pure blackhat: only dark ridge/line structures."""
    source = np.clip(channel, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(source, cv2.MORPH_BLACKHAT, kernel)
    # suppress weak response
    blackhat[~fov] = 0
    if np.any(fov):
        low = np.percentile(blackhat[fov], 5)
        blackhat = np.clip(blackhat.astype(np.float32) - low, 0, None).astype(np.uint8)
    blackhat = cv2.normalize(blackhat.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blackhat[~fov] = 0
    return blackhat


def bright_artifact_suppress(channel: np.ndarray, rgb: np.ndarray, fov: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    """Replace bright artifact spots with local Gaussian fill."""
    source = np.clip(fill_outside_fov(channel, fov), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    if np.any(fov):
        l_cut = np.percentile(l_channel[fov], 98.0)
        s_cut = np.percentile(source[fov], 95.0)
    else:
        l_cut, s_cut = 255, 255
    artifact = (l_channel >= l_cut) & (source >= s_cut) & fov
    artifact = cv2.dilate(artifact.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1).astype(bool)
    replacement = cv2.GaussianBlur(source.astype(np.float32), (0, 0),
                                   sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT)
    output = source.astype(np.float32)
    output[artifact] = replacement[artifact]
    output[~fov] = 0
    return np.clip(output, 0, 255).astype(np.uint8)


# ── Background normalization (green channel processing) ─────────────────

def estimate_background_mean(green: np.ndarray, fov: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(round(kernel_size)))
    filled = green.astype(np.float32).copy()
    if np.any(fov):
        filled[~fov] = float(np.median(filled[fov]))
    background = cv2.blur(filled, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT)
    background[~fov] = 0
    return np.clip(background, 0, 255).astype(np.uint8)


def bilateral_filter_green(green: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig) -> np.ndarray:
    d = int(config.bilateral_d)
    if d <= 1:
        return green.copy()
    if d % 2 == 0:
        d += 1
    filled = green.copy()
    if np.any(fov):
        filled[~fov] = int(np.median(filled[fov]))
    filtered = cv2.bilateralFilter(filled, d=d,
                                    sigmaColor=float(config.bilateral_sigma_color),
                                    sigmaSpace=float(config.bilateral_sigma_space))
    filtered[~fov] = 0
    return filtered.astype(np.uint8)


def normalize_green_with_background_residual(
    green: np.ndarray, background: np.ndarray, fov: np.ndarray,
    stats_erode_px: int = 12, vessel_contrast: float = 118.0,
    background_contrast: float = 34.0, vessel_gamma: float = 0.72,
    background_gamma: float = 1.55,
) -> np.ndarray:
    output = np.zeros(green.shape, dtype=np.float32)
    stats_fov = erode_mask(fov, int(stats_erode_px))
    if not np.any(stats_fov):
        stats_fov = fov.astype(bool)
    if not np.any(stats_fov):
        return output.astype(np.uint8)

    residual = green.astype(np.float32) - background.astype(np.float32)
    values = residual[stats_fov]
    center = float(np.median(values))
    low, high = np.percentile(values, [1.0, 99.0])
    negative_scale = max(center - float(low), 1.0)
    positive_scale = max(float(high) - center, 1.0)
    if negative_scale <= 0 and positive_scale <= 0:
        return output.astype(np.uint8)

    dark_response = np.clip((center - residual) / negative_scale, 0.0, 1.0)
    bright_response = np.clip((residual - center) / positive_scale, 0.0, 1.0)
    dark_response = np.power(dark_response, float(vessel_gamma))
    bright_response = np.power(bright_response, float(background_gamma))
    output[fov] = (142.0
                   - float(vessel_contrast) * dark_response[fov]
                   + float(background_contrast) * bright_response[fov])
    output[~fov] = 0
    return np.clip(output, 0, 255).astype(np.uint8)


def preprocess_green_channel(rgb: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig):
    """Full green-channel preprocessing pipeline."""
    cielab_rgb = enhance_cielab_rgb(rgb, fov, config.clahe_clip)
    green = cielab_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=float(config.clahe_clip), tileGridSize=(8, 8))
    green_clahe = clahe.apply(green)
    green_clahe[~fov] = 0
    green_filtered = bilateral_filter_green(green_clahe, fov, config)
    background = estimate_background_mean(green_filtered, fov, int(round(config.background_sigma)))
    normalized = normalize_green_with_background_residual(
        green_filtered, background, fov, stats_erode_px=int(config.fov_erode_px))
    return cielab_rgb, green_clahe, green_filtered, background, normalized


# ── Vessel filters ─────────────────────────────────────────────────────

def line_kernel(size: int, angle_deg: int, sigma: float) -> np.ndarray:
    center = size // 2
    yy, xx = np.mgrid[:size, :size].astype(np.float32) - center
    radians = np.deg2rad(angle_deg)
    x_rot = xx * np.cos(radians) + yy * np.sin(radians)
    y_rot = -xx * np.sin(radians) + yy * np.cos(radians)
    kernel = np.exp(-(y_rot**2) / (2.0 * sigma**2))
    kernel[np.abs(x_rot) > center] = 0
    kernel -= kernel.mean()
    norm = np.sum(np.abs(kernel))
    if norm > 0:
        kernel /= norm
    return kernel.astype(np.float32)


def matched_filter_response(inverted_float: np.ndarray, fov: np.ndarray, angle_step: int = 20) -> np.ndarray:
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    for size, sigma in ((9, 1.2), (15, 1.8), (21, 2.4), (31, 3.2)):
        for angle in range(0, 180, angle_step):
            kernel = line_kernel(size, angle, sigma)
            filtered = cv2.filter2D(inverted_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)
    response[~fov] = 0
    return normalize01(response, fov)


def matched_filter_response_small(inverted_float: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """Smaller scale matched filter for fine vessels."""
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    for size, sigma in ((5, 0.6), (7, 0.8), (9, 1.0)):
        for angle in range(0, 180, 10):
            kernel = line_kernel(size, angle, sigma)
            filtered = cv2.filter2D(inverted_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)
    response[~fov] = 0
    return normalize01(response, fov)


def modified_tophat(inverted: np.ndarray, fov: np.ndarray) -> np.ndarray:
    maps = []
    for kernel_size in (5, 7, 9, 13, 17, 23, 31):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        top_hat = cv2.subtract(inverted, opened).astype(np.float32)
        maps.append(top_hat)
    response = np.max(np.stack(maps, axis=0), axis=0)
    response = cv2.GaussianBlur(response, (0, 0), 0.8)
    return normalize01(response, fov)


def jerman_vesselness(bright_vessels: np.ndarray, fov: np.ndarray,
                       sigmas: tuple = (1.0, 1.6, 2.4, 3.2, 4.8)) -> np.ndarray:
    """Approximate Jerman-style 2D vesselness for bright tubular structures."""
    image = bright_vessels.astype(np.float32)
    output = np.zeros(image.shape, dtype=np.float32)
    for sigma in sigmas:
        hes = hessian_matrix(image, sigma=sigma, order="rc", use_gaussian_derivatives=True)
        eig_small, eig_large = hessian_matrix_eigvals(hes)
        lambda2 = -eig_large
        lambda1 = np.abs(eig_small)
        positive = lambda2 > 0
        if not np.any(positive & fov):
            continue
        lambda2_norm = np.zeros_like(lambda2, dtype=np.float32)
        scale_max = np.percentile(lambda2[positive & fov], 99)
        if scale_max <= 0:
            continue
        lambda2_norm[positive] = np.clip(lambda2[positive] / scale_max, 0.0, 1.0)
        blob_penalty = np.exp(-(lambda1**2) / (2.0 * (lambda2**2 + 1e-6)))
        vessel = lambda2_norm * blob_penalty
        vessel[~positive] = 0
        output = np.maximum(output, vessel.astype(np.float32))
    output[~fov] = 0
    return normalize01(output, fov)


def safe_filter_call(name: str, image: np.ndarray) -> np.ndarray:
    if name == "frangi":
        return frangi(image, sigmas=(1, 2, 3, 4, 5), black_ridges=False)
    if name == "sato":
        from skimage.filters import sato
        return sato(image, sigmas=(1, 2, 3, 4, 5), black_ridges=False)
    if name == "meijering":
        from skimage.filters import meijering
        return meijering(image, sigmas=(1, 2, 3, 4, 5), black_ridges=False)
    raise ValueError(f"Unknown vessel filter: {name}")


def fuse_vessel_responses(
    top_hat: np.ndarray, matched: np.ndarray, frangi_map: np.ndarray,
    jerman_map: np.ndarray, fov: np.ndarray,
    mode: VesselnessMode = "almeida",
    sato_map: np.ndarray | None = None,
    meijering_map: np.ndarray | None = None,
    local_dark: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "legacy_tophat":
        return normalize01(top_hat, fov)

    if mode == "almeida":
        fused = 0.20 * top_hat + 0.40 * matched + 0.15 * frangi_map + 0.25 * jerman_map
    elif mode == "ensemble":
        if sato_map is None or meijering_map is None or local_dark is None:
            raise ValueError("all maps required for ensemble")
        fused = (0.14 * top_hat + 0.34 * matched + 0.12 * frangi_map
                 + 0.22 * jerman_map + 0.08 * sato_map
                 + 0.06 * meijering_map + 0.04 * local_dark)
    elif mode == "almeida_paper":
        # same as almeida for our purposes
        fused = 0.20 * top_hat + 0.40 * matched + 0.15 * frangi_map + 0.25 * jerman_map
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fused = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), 0.6)
    fused[~fov] = 0
    return normalize01(fused, fov)


# ── Threshold / postprocess ────────────────────────────────────────────

def threshold_from_values(values: np.ndarray, method: ThresholdMethod, target_density: float) -> float:
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size < 16 or float(values.max() - values.min()) < 1e-6:
        return 0.5

    percentile_threshold = float(np.percentile(values, 100.0 * (1.0 - target_density)))
    if method == "percentile":
        return percentile_threshold
    try:
        if method == "triangle":
            return float(threshold_triangle(values))
        if method == "otsu":
            return float(threshold_otsu(values))
        if method == "hybrid":
            triangle = float(threshold_triangle(values))
            return min(triangle, percentile_threshold)
    except ValueError:
        return percentile_threshold
    raise ValueError(f"Unknown method: {method}")


def keep_components_at_least(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    output = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = True
    return output


def remove_border_components(mask: np.ndarray, border_px: int = 3) -> np.ndarray:
    n_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask.astype(bool)
    border = np.zeros(mask.shape, dtype=bool)
    border[:border_px, :] = True
    border[-border_px:, :] = True
    border[:, :border_px] = True
    border[:, -border_px:] = True
    output = mask.astype(bool).copy()
    for label in np.unique(labels[border]):
        if label != 0:
            output[labels == label] = False
    return output


def remove_edge_line_artifacts(mask: np.ndarray, edge_margin: int = 24, min_aspect: float = 5.0) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask.astype(bool)
    height, width = mask.shape
    output = mask.astype(bool).copy()
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        near_edge = (x <= edge_margin or y <= edge_margin
                     or x + w >= width - edge_margin or y + h >= height - edge_margin)
        if not near_edge:
            continue
        short_side = max(1, min(w, h))
        aspect = max(w, h) / short_side
        fill = area / max(w * h, 1)
        if aspect >= min_aspect and fill >= 0.12:
            output[labels == label] = False
    return output


def threshold_response_map(
    soft: np.ndarray, fov: np.ndarray, method: ThresholdMethod = "percentile",
    target_density: float = 0.10, fov_erode_px: int = 8,
) -> np.ndarray:
    """Threshold a soft vesselness map, return soft values above threshold."""
    inner_fov = erode_mask(fov, int(fov_erode_px))
    working = soft.astype(np.float32).copy()
    working[~inner_fov] = 0.0
    threshold = threshold_from_values(working[inner_fov], method, float(target_density))
    response = np.zeros(working.shape, dtype=np.float32)
    response[(working >= threshold) & inner_fov] = working[(working >= threshold) & inner_fov]
    return normalize01(response, response > 0)


def clean_triangle_response(
    triangle: np.ndarray, fov: np.ndarray,
    min_component_area: int = 16, fov_erode_px: int = 8,
) -> np.ndarray:
    inner_fov = erode_mask(fov, int(fov_erode_px))
    response = triangle.astype(np.float32).copy()
    response[~inner_fov] = 0.0
    support = response > 0
    support = keep_components_at_least(support, int(min_component_area))
    support = remove_edge_line_artifacts(support)
    response[~support] = 0.0
    return normalize01(response, response > 0)


def postprocess_vessel_map(
    soft: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig = VesselPipelineConfig(),
) -> np.ndarray:
    inner_fov = erode_mask(fov, int(config.fov_erode_px))
    working = soft.copy()
    working[~inner_fov] = 0.0
    threshold = threshold_from_values(working[inner_fov], config.threshold_method, float(config.target_density))
    binary = (working >= threshold) & inner_fov
    binary = keep_components_at_least(binary, int(config.min_component_area))
    binary = remove_border_components(binary, border_px=3)
    return binary.astype(bool)


# ── Channel source processing → vessel maps ────────────────────────────

def vessel_maps_from_enhanced_channel(
    enhanced_channel: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig,
    include_small_branch: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run vessel filter stack on a preprocessed channel → combined + P10 threshold."""
    inverted = 255 - np.clip(enhanced_channel, 0, 255).astype(np.uint8)
    inverted_float = normalize01(inverted.astype(np.float32), fov)

    top_hat = modified_tophat(inverted, fov)
    matched = matched_filter_response(inverted_float, fov)
    frangi_map = normalize01(safe_filter_call('frangi', inverted_float), fov)
    jerman_map = jerman_vesselness(inverted_float, fov)

    mode = "almeida" if config.vesselness_mode == "almeida_paper" else config.vesselness_mode
    combined = fuse_vessel_responses(top_hat, matched, frangi_map, jerman_map, fov, mode=mode)

    if include_small_branch:
        small_matched = matched_filter_response_small(inverted_float, fov)
        small_jerman = jerman_vesselness(inverted_float, fov, sigmas=(0.6, 0.8, 1.0, 1.2))
        combined = normalize01(0.72 * combined + 0.18 * small_matched + 0.10 * small_jerman, fov)

    p10 = threshold_response_map(combined, fov, method='percentile', target_density=0.10,
                                  fov_erode_px=max(0, int(config.fov_erode_px)))
    return combined, p10


def process_channel_source(
    rgb: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig,
    source_type: str,
) -> dict:
    """
    Generate vessel maps for a named channel source type.

    Returns dict with 'combined', 'p10', and optionally other thresholds.
    """
    # Generate the source channel
    if source_type == 'default':
        # Original pipeline: CIELAB green channel
        _, _, _, _, normalized = preprocess_green_channel(rgb, fov, config)
        source_channel = normalized
    elif source_type == 'C4G4':
        source_channel = cielab_green_clahe_source(rgb, fov, l_clip=4.0, green_clip=4.0)
    elif source_type == 'C6G6':
        source_channel = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
    elif source_type == 'BH isolated':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        source_channel = blackhat_isolated_dark_source(c6_green, fov)
    elif source_type == 'BHboost+G6':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        source_channel = blackhat_boost_then_clahe_source(c6_green, fov, strength=1.25, green_clip=6.0)
    elif source_type == 'FlatSub+G6':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        source_channel = gaussian_background_flatten(c6_green, fov, sigma=30.0)
        source_channel = local_clahe_channel(source_channel, fov, clip_limit=6.0, tile_grid_size=(16, 16))
    elif source_type == 'FlatDiv+G6':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        source_channel = divide_background_flatten(c6_green, fov, sigma=30.0)
        source_channel = local_clahe_channel(source_channel, fov, clip_limit=6.0, tile_grid_size=(16, 16))
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    combined, p10 = vessel_maps_from_enhanced_channel(source_channel, fov, config)

    result = {'source_channel': source_channel, 'combined': combined, 'p10': p10}

    # Also compute additional thresholds for comparison
    for density in [8, 10, 12, 14]:
        label = f'P{density:02d}'
        result[label] = threshold_response_map(
            combined, fov, method='percentile', target_density=density / 100.0,
            fov_erode_px=max(0, int(config.fov_erode_px)))

    # Triangle threshold
    result['Triangle'] = threshold_response_map(
        combined, fov, method='triangle', target_density=0.14,
        fov_erode_px=max(0, int(config.fov_erode_px)))

    result['Triangle_clean'] = clean_triangle_response(
        result['Triangle'], fov, min_component_area=max(1, int(config.min_component_area)),
        fov_erode_px=max(0, int(config.fov_erode_px)))

    return result


def process_fused_sources(
    rgb: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig,
    sources: List[Tuple[str, float]],  # [(source_type, weight), ...]
) -> dict:
    """
    Fuse multiple channel sources at the inverted-float level.

    Each source is independently processed through the matched filter,
    and results are combined with the given weights before the full
    fusion pipeline.
    """
    inverted_fused = None
    total_weight = 0.0

    for source_type, weight in sources:
        _, _, normalized = process_channel_source_raw(rgb, fov, config, source_type)
        inv = 255 - np.clip(normalized, 0, 255).astype(np.uint8)
        inv_float = normalize01(inv.astype(np.float32), fov)

        if inverted_fused is None:
            inverted_fused = inv_float * weight
        else:
            inverted_fused += inv_float * weight
        total_weight += weight

    if inverted_fused is None:
        raise ValueError("No sources to fuse")

    inverted_fused /= total_weight
    inverted_fused[~fov] = 0
    inverted = to_uint8(inverted_fused)

    # Then run the rest of the filter stack on the fused inverted
    top_hat = modified_tophat(inverted, fov)
    matched = matched_filter_response(inverted_fused, fov)
    frangi_map = normalize01(safe_filter_call('frangi', inverted_fused), fov)
    jerman_map = jerman_vesselness(inverted_fused, fov)

    mode = "almeida" if config.vesselness_mode == "almeida_paper" else config.vesselness_mode
    combined = fuse_vessel_responses(top_hat, matched, frangi_map, jerman_map, fov, mode=mode)

    result = {'source_channel': inverted, 'combined': combined}

    for density in [8, 10, 12, 14]:
        label = f'P{density:02d}'
        result[label] = threshold_response_map(
            combined, fov, method='percentile', target_density=density / 100.0,
            fov_erode_px=max(0, int(config.fov_erode_px)))

    result['Triangle'] = threshold_response_map(
        combined, fov, method='triangle', target_density=0.14,
        fov_erode_px=max(0, int(config.fov_erode_px)))

    return result


def process_channel_source_raw(
    rgb: np.ndarray, fov: np.ndarray, config: VesselPipelineConfig, source_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (source_channel, inverted_float, normalized_channel)."""
    if source_type == 'default':
        _, _, _, _, normalized = preprocess_green_channel(rgb, fov, config)
        source_channel = normalized
    elif source_type == 'C4G4':
        source_channel = cielab_green_clahe_source(rgb, fov, l_clip=4.0, green_clip=4.0)
    elif source_type == 'C6G6':
        source_channel = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
    elif source_type == 'BH isolated':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        source_channel = blackhat_isolated_dark_source(c6_green, fov)
    elif source_type == 'BHboost+G6':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        source_channel = blackhat_boost_then_clahe_source(c6_green, fov, strength=1.25, green_clip=6.0)
    elif source_type == 'FlatSub+G6':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        flat = gaussian_background_flatten(c6_green, fov, sigma=30.0)
        source_channel = local_clahe_channel(flat, fov, clip_limit=6.0, tile_grid_size=(16, 16))
    elif source_type == 'FlatDiv+G6':
        c6_green = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
        flat = divide_background_flatten(c6_green, fov, sigma=30.0)
        source_channel = local_clahe_channel(flat, fov, clip_limit=6.0, tile_grid_size=(16, 16))
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    return source_channel, None, source_channel


# ── Evaluation metrics ─────────────────────────────────────────────────

def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    eps = 1e-8
    return {
        'dice': float((2 * tp) / (2 * tp + fp + fn + eps)),
        'iou': float(tp / (tp + fp + fn + eps)),
        'precision': float(tp / (tp + fp + eps)),
        'sensitivity': float(tp / (tp + fn + eps)),
        'specificity': float(tn / (tn + fp + eps)),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn + eps)),
        'pred_density': float(pred.mean()),
        'gt_density': float(gt.mean()),
    }


def read_binary_mask(path: str | Path, target_shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Could not read mask: {path}')
    h, w = target_shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if mask.max() <= 1:
        return mask.astype(bool)
    return mask >= 127


# ── Debug / visualization helpers ──────────────────────────────────────

def gray_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        image = np.clip(image.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    return image


def resize_tile(image: np.ndarray, size: int) -> np.ndarray:
    image = gray_rgb(image)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def add_tile_label(image: np.ndarray, label: str, size: int) -> np.ndarray:
    output = resize_tile(image, size)
    cv2.rectangle(output, (0, 0), (size, 24), (0, 0, 0), -1)
    cv2.putText(output, label[:40], (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def overlay_prediction(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
    overlay[np.logical_and(pred, gt)] = (255, 255, 255)   # TP = white
    overlay[np.logical_and(pred, ~gt)] = (255, 0, 0)       # FP = red
    overlay[np.logical_and(~pred, gt)] = (0, 255, 255)     # FN = cyan
    return overlay


def empty_tile(shape: Tuple[int, int]) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.uint8)


# ── Finding Agrawal pairs ──────────────────────────────────────────────

def find_agrawal_pairs(root: Path | None) -> list[dict]:
    """Find (image_path, mask_path, subset) from Agrawal2021 HVDROPDB-BV."""
    if root is None or not root.exists():
        return []
    bv_root = root / 'HVDROPDB-BV'
    candidates = [
        ('RetCam', bv_root / 'RetCam_Vessels_images', bv_root / 'RetCam_Vessels_masks'),
        ('RetCam', bv_root / 'Retcam_Vessels_images', bv_root / 'Retcam_Vessels_masks'),
        ('Neo', bv_root / 'Neo_Vessels_images', bv_root / 'Neo_Vessels_masks'),
    ]
    rows = []
    for subset, image_dir, mask_dir in candidates:
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        for stem in sorted(set(images) & set(masks)):
            rows.append({'source': subset, 'image_path': str(images[stem]),
                         'mask_path': str(masks[stem]), 'name': images[stem].name})
    return rows
