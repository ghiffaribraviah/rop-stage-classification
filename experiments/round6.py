"""
Round 6: Optimize median filtering + postprocessing on gabor response.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, adaptive_hysteresis, morphological_cleanup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


def median_blur_response(response, fov, ksize=5):
    """Median filter on vesselness response."""
    r = (response * 255).astype(np.uint8)
    r = cv2.medianBlur(r, ksize)
    r[~fov] = 0
    return normalize01(r.astype(np.float32), fov)


def bilateral_blur_response(response, fov, d=7, sigma_color=50, sigma_space=7):
    """Bilateral filter preserves edges better than median."""
    r = (response * 255).astype(np.uint8)
    r[~fov] = 0
    filled = r.copy()
    if np.any(fov):
        filled[~fov] = int(np.median(r[fov]))
    filtered = cv2.bilateralFilter(filled, d, sigma_color, sigma_space)
    filtered[~fov] = 0
    return normalize01(filtered.astype(np.float32), fov)


def gaussian_blur_response(response, fov, sigma=1.0):
    """Light Gaussian blur to smooth noise."""
    r = response.astype(np.float32)
    r[~fov] = 0
    filled = r.copy()
    if np.any(fov):
        filled[~fov] = float(np.median(r[fov]))
    blurred = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma)
    blurred[~fov] = 0
    return normalize01(blurred, fov)


def stick_filter(response, fov, iterations=3):
    """
    Stick filter: enhance thin structures by suppressing non-stick noise.
    Works by checking local orientation coherence.
    Simplified: morphological opening with thin structuring elements.
    """
    result = (response * 255).astype(np.uint8)
    result[~fov] = 0
    for _ in range(iterations):
        # Opening with a thin horizontal kernel
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        opened_h = cv2.morphologyEx(result, cv2.MORPH_OPEN, kh)
        opened_v = cv2.morphologyEx(result, cv2.MORPH_OPEN, kv)
        # Keep max of both orientations
        result = np.maximum(opened_h, opened_v)
        result[~fov] = 0
    return normalize01(result.astype(np.float32), fov)


def multi_threshold_fusion(soft, fov, densities=[6, 8, 10, 12, 14]):
    """
    Fuse multiple thresholds: keep pixels that are consistently
    detected across multiple thresholds.
    """
    inner = erode_mask(fov, 8)
    votes = np.zeros(soft.shape, dtype=np.int32)
    for d in densities:
        th = threshold_response_map(soft, fov, method='percentile', target_density=d/100.0)
        votes += (th > 0).astype(np.int32)
    # Keep pixels that appear in at least N thresholds
    for n_votes in [2, 3, 4]:
        consensus = (votes >= n_votes) & inner
        yield f'consensus_{n_votes}of{len(densities)}', consensus


def main():
    print("=" * 70)
    print("ROUND 6: Median/Bilateral/Stick optimization")
    print("=" * 70)

    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    retcam = [p for p in pairs if p['source'] == 'RetCam'][:8]
    neo = [p for p in pairs if p['source'] == 'Neo'][:8]
    sample = retcam + neo
    print(f"Images: {len(sample)}")

    all_metrics = []

    for idx, pair in enumerate(sample):
        rgb = read_rgb(pair['image_path'])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair['mask_path'], fov.shape)

        # Get clean c6g6 + gabor
        c6_ch = cielab_green_clahe_source(working, fov, l_clip=6.0, green_clip=6.0)
        c6_inv_f = normalize01((255 - c6_ch).astype(np.float32), fov)
        gabor = gabor_filter_response(c6_inv_f, fov)

        # Also get Almeida
        res = process_channel_source(working, fov, CONFIG, 'C6G6')
        almeida = res['combined']

        # Fuse gabor + almeida
        fused = normalize01(0.5 * almeida + 0.5 * gabor, fov)

        soft_maps = {
            'gabor': gabor,
            'almeida': almeida,
            'fused': fused,
        }

        # Apply various smoothing filters
        for sname, soft in soft_maps.items():
            inner = erode_mask(fov, 8)
            vals = soft[inner]
            vals = vals[vals > 0]

            # No filter (baseline)
            for d in [10]:
                th = threshold_response_map(soft, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_P10'; all_metrics.append(m)

            # Median 3x3
            m3 = median_blur_response(soft, fov, 3)
            for d in [8, 10, 12]:
                th = threshold_response_map(m3, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_median3_P{d:02d}'; all_metrics.append(m)

            # Median 5x5
            m5 = median_blur_response(soft, fov, 5)
            for d in [8, 10, 12]:
                th = threshold_response_map(m5, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_median5_P{d:02d}'; all_metrics.append(m)

            # Median 7x7
            m7 = median_blur_response(soft, fov, 7)
            for d in [8, 10, 12]:
                th = threshold_response_map(m7, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_median7_P{d:02d}'; all_metrics.append(m)

            # Median 5 + Median 3 fused (multi-scale median)
            m53 = normalize01(0.6 * m5 + 0.4 * m3, fov)
            for d in [8, 10, 12]:
                th = threshold_response_map(m53, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_median53_P{d:02d}'; all_metrics.append(m)

            # Bilateral filter
            bi = bilateral_blur_response(soft, fov, d=7, sigma_color=50, sigma_space=7)
            for d in [8, 10, 12]:
                th = threshold_response_map(bi, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_bilateral_P{d:02d}'; all_metrics.append(m)

            # Light Gaussian
            gs = gaussian_blur_response(soft, fov, sigma=0.8)
            for d in [8, 10, 12]:
                th = threshold_response_map(gs, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_gauss_P{d:02d}'; all_metrics.append(m)

            # Stick filter
            st = stick_filter(soft, fov, iterations=2)
            for d in [8, 10, 12]:
                th = threshold_response_map(st, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_stick_P{d:02d}'; all_metrics.append(m)

            # Multi-threshold consensus on median5
            for label, consensus in multi_threshold_fusion(m5, fov):
                m = segmentation_metrics(consensus, gt)
                m['method'] = f'{sname}_median5_{label}'; all_metrics.append(m)

            # Hysteresis on median5
            if len(vals) > 50:
                hyst = adaptive_hysteresis(m5, fov)
                m = segmentation_metrics(hyst, gt)
                m['method'] = f'{sname}_median5_Hyst'; all_metrics.append(m)

                # Cleaned hysteresis
                cleaned = morphological_cleanup(hyst, fov, min_vessel_area=4, close_radius=2)
                m = segmentation_metrics(cleaned, gt)
                m['method'] = f'{sname}_median5_HystCln'; all_metrics.append(m)

        sys.stdout.write(f"\r  [{idx+1}/{len(sample)}] {pair['name']}")
        sys.stdout.flush()

    print(f"\n\nTotal: {len(all_metrics)} metrics")

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
    write_csv(OUTPUT_DIR / 'round6_summary.csv', summary)

    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 72)
    for r in summary:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    print("\n═══ TOP 15 ═══")
    for r in summary[:15]:
        print(f"  {r['method']:45s} Dice={r['dice']:.4f}")


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
