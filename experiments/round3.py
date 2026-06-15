"""
Round 3: Gaussian matched filter (Chaudhuri), enhanced Frangi,
region growing postprocessing, entropy thresholding.
All classical, no ML.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, cosfire_filter_response, adaptive_hysteresis, morphological_cleanup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


# ── 1. Gaussian Matched Filter (Chaudhuri 1989) ────────────────────────

def gaussian_matched_kernel(size, theta, sigma, length=5):
    """
    Chaudhuri-style Gaussian matched filter kernel.
    A 2D Gaussian elongated along one axis, rotated by theta.
    This matches the profile of a retinal vessel (dark on light background).
    """
    center = size // 2
    yy, xx = np.mgrid[-center:size-center, -center:size-center].astype(np.float32)
    # Rotate coordinates
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    # Elongated Gaussian
    sigma_y = sigma
    sigma_x = sigma * length  # longer along vessel direction
    kernel = np.exp(-0.5 * (xr**2 / sigma_x**2 + yr**2 / sigma_y**2))
    # Subtract mean to make it a matched filter (respond to dark vessels)
    kernel = -kernel  # invert because vessels are darker than background
    kernel -= kernel.mean()
    return kernel / np.sum(np.abs(kernel))


def gaussian_matched_filter_response(inverted_float, fov):
    """
    Full Gaussian matched filter bank (Chaudhuri style).
    Uses 12 orientations and 4 scales.
    """
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    sigmas = [1.0, 1.6, 2.4, 3.5]
    lengths = [5, 7, 9, 11]

    for sigma, length in zip(sigmas, lengths):
        kernel_size = int(6 * sigma * (length / 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(7, min(45, kernel_size))

        for angle_deg in range(0, 180, 12):
            theta = np.deg2rad(angle_deg)
            kernel = gaussian_matched_kernel(kernel_size, theta, sigma, length)
            # Normalize kernel to zero mean and unit energy
            kernel = kernel.astype(np.float32)
            filtered = cv2.filter2D(inverted_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)

    response[~fov] = 0
    return normalize01(response, fov)


# ── 2. Enhanced Frangi with better scale range ─────────────────────────

def enhanced_frangi(inverted_float, fov):
    """
    Multi-scale Frangi vesselness with extended scale range.
    Catches both fine and thick vessels.
    """
    from skimage.filters import frangi
    # Wider scale range for neonatal vessels (can be very fine or thick)
    result = frangi(inverted_float, sigmas=(1, 2, 3, 4, 5, 7, 9), black_ridges=False)
    result[~fov] = 0
    return normalize01(result, fov)


# ── 3. Hessian-based vesselness (direct eigenvalue analysis) ───────────

def hessian_vesselness(inverted_float, fov):
    """
    Direct Hessian eigenvalue analysis.
    For bright vessels on dark background (inverted image).
    Large eigenvalue >> small eigenvalue indicates tubular structure.
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    output = np.zeros(inverted_float.shape, dtype=np.float32)
    sigmas = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    for sigma in sigmas:
        hes = hessian_matrix(inverted_float, sigma=sigma, order='rc', use_gaussian_derivatives=True)
        eig_small, eig_large = hessian_matrix_eigvals(hes)

        # For bright vessels in inverted image: lambda2 (eig_large) > 0 and |lambda2| >> |lambda1|
        # This means the structure is tubular (one large, one small eigenvalue)
        lambda2 = eig_large  # largest eigenvalue
        lambda1 = np.abs(eig_small)  # smallest eigenvalue (absolute)

        # Vesselness: lambda2 should be large (bright tubular), lambda1 should be small
        # Ratio-based response
        lambda2_pos = np.maximum(lambda2, 0)
        ratio = np.zeros_like(lambda2_pos)
        denom = lambda2_pos + 1e-6
        # Only compute where lambda2_pos > 0 (bright structures)
        ratio[lambda2_pos > 0] = lambda1[lambda2_pos > 0] / denom[lambda2_pos > 0]

        # Vesselness: high when lambda2 > 0 (bright) and ratio is low (tubular)
        vessel = lambda2_pos * np.exp(-ratio**2 / 0.5)
        output = np.maximum(output, vessel)

    output[~fov] = 0
    return normalize01(output, fov)


# ── 4. Region growing from high-confidence seeds ───────────────────────

def region_growing_vessels(soft, fov, seed_percentile=92, grow_threshold_percentile=55, max_iterations=50):
    """
    Region growing: start from high-confidence seeds, grow into weaker regions.
    This is like hysteresis but iterative and can bridge small gaps.
    """
    inner_fov = erode_mask(fov, 8)
    vals = soft[inner_fov]
    vals = vals[vals > 0]

    if len(vals) < 50:
        return (soft > np.percentile(vals, 80)) if len(vals) > 10 else (soft > 0.5)

    seed_th = float(np.percentile(vals, seed_percentile))
    grow_th = float(np.percentile(vals, grow_threshold_percentile))

    seeds = (soft >= seed_th) & inner_fov
    candidates = (soft >= grow_th) & inner_fov

    # Iterative region growing
    result = seeds.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max_iterations):
        dilated = cv2.dilate(result.astype(np.uint8), kernel, iterations=1).astype(bool)
        new_pixels = dilated & candidates & ~result
        if not np.any(new_pixels):
            break
        result |= new_pixels

    # Clean up small components
    result = keep_components_at_least(result, 4)
    return result & fov


# ── 5. Entropy thresholding ──────────────────────────────────────────

def entropy_threshold(values):
    """
    Kapur's entropy thresholding: choose threshold that maximizes
    entropy separation between foreground and background.
    """
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size < 100:
        return float(np.percentile(values, 80)) if values.size > 10 else 0.5

    # Histogram
    hist, bins = np.histogram(values, bins=64)
    hist = hist.astype(np.float32) + 1e-10
    hist = hist / hist.sum()

    best_th = bins[0]
    best_entropy = -1e10

    for i in range(1, len(bins) - 1):
        p1 = hist[:i].sum()
        p2 = hist[i:].sum()
        if p1 < 1e-6 or p2 < 1e-6:
            continue

        h1 = -np.sum((hist[:i] / p1) * np.log(hist[:i] / p1 + 1e-10))
        h2 = -np.sum((hist[i:] / p2) * np.log(hist[i:] / p2 + 1e-10))
        total_entropy = h1 + h2

        if total_entropy > best_entropy:
            best_entropy = total_entropy
            best_th = bins[i]

    return best_th


# ── Compute all maps for an image ──────────────────────────────────────

def compute_all_maps(rgb, fov, cfg):
    c6_ch = cielab_green_clahe_source(rgb, fov, l_clip=6.0, green_clip=6.0)
    c6_inv = 255 - c6_ch
    c6_inv_f = normalize01(c6_inv.astype(np.float32), fov)

    # Standard Almeida on C6G6
    res = process_channel_source(rgb, fov, cfg, 'C6G6')
    c6_almeida = res['combined']

    # New filters
    c6_gauss = gaussian_matched_filter_response(c6_inv_f, fov)
    c6_frangi = enhanced_frangi(c6_inv_f, fov)
    c6_hessian = hessian_vesselness(c6_inv_f, fov)
    c6_gabor = gabor_filter_response(c6_inv_f, fov)
    c6_dog = cosfire_filter_response(c6_inv_f, fov)

    # Fusions
    almeida_gauss = normalize01(0.5 * c6_almeida + 0.5 * c6_gauss, fov)
    almeida_gabor_gauss = normalize01(0.35 * c6_almeida + 0.35 * c6_gabor + 0.30 * c6_gauss, fov)
    frangi_hessian = normalize01(0.5 * c6_frangi + 0.5 * c6_hessian, fov)
    mega_fusion = normalize01(
        0.25 * c6_almeida + 0.25 * c6_gabor + 0.20 * c6_gauss + 0.15 * c6_frangi + 0.15 * c6_hessian,
        fov)
    triple = normalize01(0.4 * c6_almeida + 0.3 * c6_gabor + 0.3 * c6_gauss, fov)

    return {
        'c6_almeida': c6_almeida,
        'c6_gauss': c6_gauss,
        'c6_frangi': c6_frangi,
        'c6_hessian': c6_hessian,
        'c6_gabor': c6_gabor,
        'c6_dog': c6_dog,
        'almeida_gauss': almeida_gauss,
        'almeida_gabor_gauss': almeida_gabor_gauss,
        'frangi_hessian': frangi_hessian,
        'mega_fusion': mega_fusion,
        'triple': triple,
    }


def test_thresholds(soft, fov, gt, prefix):
    results = []
    inner = erode_mask(fov, 8)
    vals = soft[inner]
    vals = vals[vals > 0]

    # Percentile thresholds
    for d in [8, 10, 12]:
        th = threshold_response_map(soft, fov, method='percentile', target_density=d/100.0)
        m = segmentation_metrics(th > 0, gt)
        m['method'] = f'{prefix}_P{d:02d}'
        results.append(m)

    # Entropy threshold
    if len(vals) > 50:
        ent_th = entropy_threshold(vals)
        ent_bin = (soft >= ent_th) & inner
        m = segmentation_metrics(ent_bin, gt)
        m['method'] = f'{prefix}_Entropy'
        results.append(m)

    # Otsu threshold
    if len(vals) > 50:
        try:
            from skimage.filters import threshold_otsu
            otsu_th = float(threshold_otsu(vals))
            otsu_bin = (soft >= otsu_th) & inner
            m = segmentation_metrics(otsu_bin, gt)
            m['method'] = f'{prefix}_Otsu'
            results.append(m)
        except:
            pass

    # Hysteresis
    if len(vals) > 50:
        hyst = adaptive_hysteresis(soft, fov)
        m = segmentation_metrics(hyst, gt)
        m['method'] = f'{prefix}_Hyst'
        results.append(m)

    # Region growing
    if len(vals) > 50:
        rg = region_growing_vessels(soft, fov)
        m = segmentation_metrics(rg, gt)
        m['method'] = f'{prefix}_RegionGrow'
        results.append(m)

    # Hysteresis + cleanup
    if len(vals) > 50:
        hc = morphological_cleanup(hyst, fov, min_vessel_area=4, close_radius=2)
        m = segmentation_metrics(hc, gt)
        m['method'] = f'{prefix}_HystClean'
        results.append(m)

    return results


def main():
    print("=" * 70)
    print("ROUND 3: Gaussian matched filter, enhanced Frangi, region growing")
    print("=" * 70)

    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    retcam = [p for p in pairs if p['source'] == 'RetCam'][:8]
    neo = [p for p in pairs if p['source'] == 'Neo'][:8]
    sample = retcam + neo
    print(f"Images: {len(sample)} (8 RetCam + 8 Neo)")

    all_metrics = []

    for idx, pair in enumerate(sample):
        rgb = read_rgb(pair['image_path'])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        maps = compute_all_maps(working, fov, CONFIG)

        for name, soft in maps.items():
            all_metrics.extend(test_thresholds(soft, fov, gt, name))

        sys.stdout.write(f"\r  [{idx+1}/{len(sample)}] {pair['name']}")
        sys.stdout.flush()

    print(f"\n\nTotal metrics: {len(all_metrics)}")
    write_csv(OUTPUT_DIR / 'round3_metrics.csv', all_metrics)

    # Summary
    methods = list(dict.fromkeys([m['method'] for m in all_metrics]))
    summary = []
    for m in methods:
        rows = [r for r in all_metrics if r['method'] == m]
        summary.append({
            'method': m,
            'dice': float(np.mean([r['dice'] for r in rows])),
            'sensitivity': float(np.mean([r['sensitivity'] for r in rows])),
            'precision': float(np.mean([r['precision'] for r in rows])),
        })

    summary.sort(key=lambda r: r['dice'], reverse=True)
    write_csv(OUTPUT_DIR / 'round3_summary.csv', summary)

    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 72)
    for r in summary[:30]:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    # Per-filter-type best
    print("\n═══ BEST PER FILTER TYPE (P10) ═══")
    for ftype in ['c6_almeida', 'c6_gauss', 'c6_frangi', 'c6_hessian', 'c6_gabor', 'c6_dog']:
        rows = [r for r in summary if r['method'].startswith(f'{ftype}_P10')]
        if rows:
            print(f"  {ftype:20s} Dice={rows[0]['dice']:.4f}")

    print("\n═══ BEST FUSIONS (P10) ═══")
    for ftype in ['almeida_gauss', 'almeida_gabor_gauss', 'frangi_hessian', 'mega_fusion', 'triple']:
        rows = [r for r in summary if r['method'].startswith(f'{ftype}_P10')]
        if rows:
            print(f"  {ftype:20s} Dice={rows[0]['dice']:.4f}")

    print("\n═══ BEST PER THRESHOLD TYPE (mega_fusion) ═══")
    for ttype in ['P08', 'P10', 'P12', 'Entropy', 'Otsu', 'Hyst', 'RegionGrow', 'HystClean']:
        rows = [r for r in summary if r['method'] == f'mega_fusion_{ttype}']
        if rows:
            print(f"  {ttype:15s} Dice={rows[0]['dice']:.4f}  Sens={rows[0]['sensitivity']:.4f}  Prec={rows[0]['precision']:.4f}")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
