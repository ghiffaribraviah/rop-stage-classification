from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi, threshold_otsu, threshold_triangle
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import disk, skeletonize
from skimage.transform import hough_circle, hough_circle_peaks


ROOT = Path(__file__).resolve().parents[1]
AGRAWAL_ROOT = ROOT / "data" / "Agrawal2021"
ZHAO_ROOT = ROOT / "data" / "Zhao2024"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "00_debug_baseline"
    / "normalization_residual_tuned"
    / "debug_almeida_exact_workflow_15_samples.jpg"
)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image.copy()
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def normalize01(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = image[mask] if mask is not None and np.any(mask) else image.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if hi <= lo:
        hi = float(values.max())
        lo = float(values.min())
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.float32)
    output = (image.astype(np.float32) - lo) / (hi - lo)
    output = np.clip(output, 0.0, 1.0)
    if mask is not None:
        output = output.copy()
        output[~mask] = 0
    return output.astype(np.float32)


def uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def to_rgb_tile(image: np.ndarray) -> np.ndarray:
    image = uint8_image(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    return image[:, :, :3]


def estimate_fov_mask(rgb: np.ndarray) -> np.ndarray:
    # Estimate the visible retinal field from the dark image background. Keep
    # this as the raw FOV; border artifact suppression is handled separately.
    max_channel = rgb.max(axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    visible = (max_channel > 8) | (gray > 6)

    # A small close/open smooths compression speckles and thin black seams
    # without moving the field center. Kernel sizes are image-scale dependent
    # because the debug set mixes RetCam, Neo, and cropped Zhao images.
    min_side = max(1, min(rgb.shape[:2]))
    close_radius = max(5, int(round(min_side * 0.015)))
    open_radius = max(2, int(round(min_side * 0.004)))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * close_radius + 1, 2 * close_radius + 1),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * open_radius + 1, 2 * open_radius + 1),
    )
    mask = cv2.morphologyEx(visible.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel).astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest_label
    mask = ndi.binary_fill_holes(mask)

    # Redraw the largest external contour to remove one-pixel jaggedness from
    # thresholding while preserving clipped, non-circular fields.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        smoothed = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(smoothed, [largest], -1, 1, thickness=cv2.FILLED)
        mask = ndi.binary_fill_holes(smoothed.astype(bool))
    return mask.astype(bool)


def enhance_cielab(rgb: np.ndarray, fov: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    enhanced[~fov] = 0
    return enhanced


def estimate_background_mean(green: np.ndarray, fov: np.ndarray) -> np.ndarray:
    filled = green.copy()
    if np.any(fov):
        filled[~fov] = int(np.median(green[fov]))
    background = cv2.blur(filled.astype(np.float32), (30, 30), borderType=cv2.BORDER_REFLECT)
    background[~fov] = 0
    return background


def normalize_green_with_background(green: np.ndarray, background: np.ndarray, fov: np.ndarray) -> np.ndarray:
    output = np.zeros(green.shape, dtype=np.float32)
    if not np.any(fov):
        return output.astype(np.uint8)

    green_float = green.astype(np.float32)
    background_float = np.maximum(background.astype(np.float32), 1.0)
    scale = float(np.median(background_float[fov]))
    corrected = green_float * scale / background_float
    values = corrected[fov]
    low, high = np.percentile(values, [0.5, 99.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return output.astype(np.uint8)
    output[fov] = (corrected[fov] - low) / float(high - low)
    output = np.clip(output, 0.0, 1.0)
    output = 32.0 + 223.0 * output
    output[~fov] = 0
    return np.clip(output, 0, 255).astype(np.uint8)


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


def bsgmf_response(inverted_float: np.ndarray, fov: np.ndarray) -> np.ndarray:
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    for angle in range(0, 180, 15):
        filtered = cv2.filter2D(
            inverted_float,
            cv2.CV_32F,
            line_kernel(15, angle, 2.0),
            borderType=cv2.BORDER_REFLECT,
        )
        response = np.maximum(response, filtered)
    response[~fov] = 0
    return normalize01(response, fov)


def modified_tophat(source_float: np.ndarray, fov: np.ndarray) -> np.ndarray:
    source = uint8_image(normalize01(source_float, fov))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    maps = []
    for radius in (1, 2, 3):
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        closed = cv2.morphologyEx(source, cv2.MORPH_CLOSE, close_kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel)
        maps.append(cv2.subtract(source, np.minimum(opened, source)).astype(np.float32))
    response = np.mean(np.stack(maps, axis=0), axis=0)
    response[~fov] = 0
    return normalize01(response, fov)


def jerman_vesselness(image: np.ndarray, fov: np.ndarray, tau: float = 0.90) -> np.ndarray:
    output = np.zeros(image.shape, dtype=np.float32)
    return jerman_vesselness_with_scales(image, fov, sigmas=(1.0, 3.0, 5.0, 7.0), tau=tau)


def jerman_vesselness_with_scales(
    image: np.ndarray,
    fov: np.ndarray,
    sigmas: tuple[float, ...],
    tau: float = 0.90,
) -> np.ndarray:
    output = np.zeros(image.shape, dtype=np.float32)
    for sigma in sigmas:
        hessian = hessian_matrix(image.astype(np.float32), sigma=sigma, order="rc", use_gaussian_derivatives=True)
        eig_small, eig_large = hessian_matrix_eigvals(hessian)
        vessel_strength = -eig_large.astype(np.float32)
        positive = (vessel_strength > 0) & fov
        if not np.any(positive):
            continue
        lambda_p = tau * float(np.percentile(vessel_strength[positive], 99.5))
        if lambda_p <= 0:
            continue
        rho = np.clip(vessel_strength / lambda_p, 0.0, 1.0)
        blob_penalty = np.exp(-(np.abs(eig_small) ** 2) / (2.0 * (vessel_strength**2 + 1e-6)))
        vessel = (rho**2) * (3.0 - 2.0 * rho) * blob_penalty
        vessel[~positive] = 0
        output = np.maximum(output, vessel.astype(np.float32))
    output[~fov] = 0
    return normalize01(output, fov)


def keep_components_at_least(mask: np.ndarray, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    output = np.zeros(mask.shape, dtype=bool)
    for label in range(1, n_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            output[labels == label] = True
    return output


def triangle_threshold(response: np.ndarray, fov: np.ndarray) -> np.ndarray:
    values = response[fov]
    try:
        threshold = float(threshold_triangle(values, nbins=4))
    except ValueError:
        threshold = float(np.percentile(values, 90.0)) if values.size else 1.0
    return ((response >= threshold) & fov).astype(bool)


def triangle_threshold_value(response: np.ndarray, fov: np.ndarray, nbins: int) -> float:
    values = response[fov]
    if values.size == 0:
        return 1.0
    try:
        return float(threshold_triangle(values, nbins=max(2, int(nbins))))
    except ValueError:
        return float(np.percentile(values, 90.0))


def connected_hysteresis_threshold(
    response: np.ndarray,
    fov: np.ndarray,
    strong_nbins: int = 8,
    weak_ratio: float = 0.70,
    strong_percentile: float = 90.0,
    weak_percentile: float = 75.0,
) -> np.ndarray:
    values = response[fov]
    if values.size == 0:
        return np.zeros(response.shape, dtype=bool)
    strong_threshold = max(
        triangle_threshold_value(response, fov, nbins=strong_nbins),
        float(np.percentile(values, float(strong_percentile))),
    )
    weak_threshold = max(
        float(strong_threshold) * float(weak_ratio),
        float(np.percentile(values, float(weak_percentile))),
    )
    strong = (response >= strong_threshold) & fov
    weak = (response >= weak_threshold) & fov
    if not np.any(strong) or not np.any(weak):
        return strong.astype(bool)
    labels, count = ndi.label(weak)
    if count == 0:
        return strong.astype(bool)
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels > 0]
    return (np.isin(labels, strong_labels) & fov).astype(bool)


def keep_skeleton_branches_at_least(skeleton: np.ndarray, min_length: int) -> np.ndarray:
    return keep_components_at_least(skeleton.astype(bool), int(min_length))


def almeida_small_vessel_branch(
    bsgmf: np.ndarray,
    top_hat: np.ndarray,
    fov: np.ndarray,
    rgb: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    small_sigmas = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
    small_frangi = normalize01(
        frangi(
            top_hat.astype(np.float32),
            sigmas=small_sigmas,
            alpha=0.5,
            beta=15.0,
            black_ridges=False,
        ),
        fov,
    )
    small_jerman = jerman_vesselness_with_scales(top_hat, fov, sigmas=small_sigmas, tau=0.90)
    small_response = normalize01(
        0.35 * small_frangi + 0.45 * small_jerman + 0.10 * bsgmf + 0.10 * top_hat,
        fov,
    )
    small_hyst = connected_hysteresis_threshold(
        small_response,
        fov,
        strong_nbins=8,
        weak_ratio=0.70,
        strong_percentile=90.0,
        weak_percentile=78.0,
    )
    small_clean = keep_components_at_least(small_hyst, 25)
    small_od_clean, _ = remove_optic_disc_artifact(small_clean, rgb, fov)
    small_skeleton = skeletonize(small_od_clean & fov).astype(bool)
    small_skeleton = keep_skeleton_branches_at_least(small_skeleton, 12)
    return [
        ("Small resp", small_response),
        ("Small hyst", small_hyst),
        ("Small clean", small_od_clean),
        ("Small skel", small_skeleton),
    ]


def optic_disc_centers(rgb: np.ndarray, fov: np.ndarray, max_centers: int = 8) -> list[tuple[int, int]]:
    red = rgb[:, :, 0].astype(np.float32)
    inner = cv2.erode(fov.astype(np.uint8), disk(35).astype(np.uint8), iterations=1).astype(bool)
    red[~inner] = 0
    dog = cv2.GaussianBlur(red, (0, 0), 2.0) - cv2.GaussianBlur(red, (0, 0), 8.0)
    dog = uint8_image(normalize01(dog, inner))
    edges = cv2.Sobel(dog, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(dog, cv2.CV_32F, 0, 1, ksize=3) ** 2
    edges = uint8_image(normalize01(np.sqrt(edges), inner))
    edges = cv2.Canny(edges, 40, 120)
    radii = np.arange(10, min(101, max(12, min(rgb.shape[:2]) // 3)), 4)
    if radii.size == 0 or not np.any(edges):
        return []
    hough = hough_circle(edges > 0, radii)
    _, cx, cy, _ = hough_circle_peaks(hough, radii, total_num_peaks=max_centers)
    return [(int(x), int(y)) for x, y in zip(cx, cy)]


def remove_optic_disc_artifact(segmentation: np.ndarray, rgb: np.ndarray, fov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    output = segmentation.astype(bool).copy()
    od_marker = np.zeros(segmentation.shape, dtype=bool)
    for cx, cy in optic_disc_centers(rgb, fov):
        half = 40
        y0, y1 = max(0, cy - half), min(output.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(output.shape[1], cx + half)
        window = output[y0:y1, x0:x1]
        if window.size == 0 or not np.any(window):
            continue
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(window.astype(np.uint8), 8)
        best_label = 0
        best_distance = np.inf
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not (12 <= area <= 1200):
                continue
            component = (labels == label).astype(np.uint8)
            moments = cv2.moments(component)
            if moments["m00"] == 0:
                continue
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]
            covariance = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float32)
            eigvals = np.linalg.eigvalsh(covariance)
            axis_ratio = float(np.sqrt((eigvals.max() + 1e-6) / (eigvals.min() + 1e-6)))
            if axis_ratio > 4.0:
                continue
            local_cx, local_cy = centroids[label]
            distance = float(np.hypot(local_cx + x0 - cx, local_cy + y0 - cy))
            if distance < best_distance and distance <= 30:
                best_label = label
                best_distance = distance
        if best_label:
            artifact = labels == best_label
            output[y0:y1, x0:x1][artifact] = False
            od_marker[y0:y1, x0:x1][artifact] = True
            break
    return output & fov, od_marker


def read_binary_mask(path: Path | None, target_shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(target_shape, dtype=bool)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    resized = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def almeida_workflow_maps(rgb: np.ndarray, mask_path: Path | None, max_side: int) -> list[tuple[str, np.ndarray]]:
    rgb = resize_max_side(rgb, max_side)
    fov = estimate_fov_mask(rgb)
    gt = read_binary_mask(mask_path, fov.shape)
    cielab = enhance_cielab(rgb, fov)
    green = cielab[:, :, 1].copy()
    green[~fov] = 0
    background = estimate_background_mean(green, fov)
    normalized = normalize_green_with_background(green, background, fov)
    inverted = 255 - normalized
    inverted_float = normalize01(inverted.astype(np.float32), fov)
    bsgmf = bsgmf_response(inverted_float, fov)
    top_hat = modified_tophat(bsgmf, fov)
    frangi_map = normalize01(
        frangi(top_hat.astype(np.float32), sigmas=(1, 3, 5, 7), alpha=0.5, beta=15.0, black_ridges=False),
        fov,
    )
    jerman_map = jerman_vesselness(top_hat, fov)
    combined = normalize01(0.70 * frangi_map + 0.30 * jerman_map, fov)
    triangle = triangle_threshold(combined, fov)
    area_clean = keep_components_at_least(triangle, 200)
    od_clean, _ = remove_optic_disc_artifact(area_clean, rgb, fov)
    skeleton = skeletonize(od_clean & fov).astype(bool)
    maps = [
        ("Input", rgb),
        ("GT", gt),
        ("FOV", fov),
        ("CIELAB", cielab),
        ("Green", green),
        ("Background", normalize01(background, fov)),
        ("Normalized", normalized),
        ("BSGMF", bsgmf),
        ("Mod top-hat", top_hat),
        ("Frangi", frangi_map),
        ("Jerman", jerman_map),
        ("70F 30J", combined),
        ("Triangle", triangle),
        ("Area >=200", area_clean),
        ("OD removed", od_clean),
        ("Skeleton", skeleton),
    ]
    maps.extend(almeida_small_vessel_branch(bsgmf, top_hat, fov, rgb))
    return maps


def add_label(image: np.ndarray, label: str, tile_size: int) -> np.ndarray:
    tile = cv2.resize(to_rgb_tile(image), (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    strip = np.zeros((24, tile_size, 3), dtype=np.uint8)
    cv2.putText(strip, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1, cv2.LINE_AA)
    return np.vstack([strip, tile])


def agrawal_rows(retcam_count: int, neo_count: int) -> list[tuple[str, Path, Path | None]]:
    rows: list[tuple[str, Path, Path | None]] = []
    bv_root = AGRAWAL_ROOT / "HVDROPDB-BV"
    for source, count, candidates in (
        (
            "RetCam",
            retcam_count,
            (
                ("RetCam_Vessels_images", "RetCam_Vessels_masks"),
                ("Retcam_Vessels_images", "Retcam_Vessels_masks"),
            ),
        ),
        ("Neo", neo_count, (("Neo_Vessels_images", "Neo_Vessels_masks"),)),
    ):
        for image_folder, mask_folder in candidates:
            image_dir = bv_root / image_folder
            mask_dir = bv_root / mask_folder
            if image_dir.is_dir() and mask_dir.is_dir():
                images = {path.stem: path for path in image_dir.glob("*.png")}
                masks = {path.stem: path for path in mask_dir.glob("*.png")}
                stems = sorted(set(images) & set(masks), key=lambda stem: int(stem))
                rows.extend((f"{source} {images[stem].name}", images[stem], masks[stem]) for stem in stems[:count])
                break
    return rows


def zhao_rows(root: Path, limit: int) -> list[tuple[str, Path, Path | None]]:
    if limit <= 0 or not root.is_dir():
        return []
    classes = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.lower() != "laser scars"
    ]
    class_files = {
        klass.name: sorted(path for path in klass.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        for klass in classes
    }
    rows: list[tuple[str, Path, Path | None]] = []
    round_idx = 0
    while len(rows) < limit:
        added = False
        for _, files in class_files.items():
            if round_idx < len(files):
                path = files[round_idx]
                rows.append((f"Zhao {path.name}", path, None))
                added = True
                if len(rows) >= limit:
                    break
        if not added:
            break
        round_idx += 1
    return rows


def build_sheet(rows: list[tuple[str, Path, Path | None]], output_path: Path, tile_size: int, max_side: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_rows = []
    for index, (name, path, mask_path) in enumerate(rows, start=1):
        maps = almeida_workflow_maps(read_rgb(path), mask_path=mask_path, max_side=max_side)
        row_tiles = [add_label(image, f"{index}. {name}" if col == 0 else label, tile_size) for col, (label, image) in enumerate(maps)]
        sheet_rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Almeida 2024 exact-workflow visualization grid.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=150)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--retcam-count", type=int, default=5)
    parser.add_argument("--neo-count", type=int, default=5)
    parser.add_argument("--zhao-count", type=int, default=5)
    args = parser.parse_args()

    rows = agrawal_rows(args.retcam_count, args.neo_count)
    rows.extend(zhao_rows(ZHAO_ROOT, args.zhao_count))
    if not rows:
        raise RuntimeError(f"No debug images found under {AGRAWAL_ROOT} or {ZHAO_ROOT}")
    path = build_sheet(rows, args.output, tile_size=args.tile_size, max_side=args.max_side)
    print(path)


if __name__ == "__main__":
    main()
