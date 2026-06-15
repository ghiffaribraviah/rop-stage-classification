"""
Round 5: Optic disc inpainting + simpler preprocessing + 
median filtering to suppress non-vessel structures.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, adaptive_hysteresis, morphological_cleanup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


def inpaint_optic_disc(rgb, fov):
    """
    Simple optic disc inpainting: detect the bright disc region,
    then fill it with surrounding texture using OpenCV inpainting.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Optic disc is the brightest region within the FOV
    bright = np.zeros_like(gray)
    if np.any(fov):
        p98 = np.percentile(gray[fov], 98)
        bright = ((gray >= p98) & fov).astype(np.uint8)
    
    # Clean up: largest bright connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n_labels > 1:
        sizes = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n_labels)]
        # Find the component that's most central (optic disc is usually near center)
        h, w = gray.shape
        best_label = max(sizes, key=lambda x: x[0])[1]
        disc = (labels == best_label).astype(np.uint8) * 255
        # Dilate slightly to cover the disc
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        disc = cv2.dilate(disc, k, iterations=1)
        # Inpaint
        inpainted = cv2.inpaint(rgb, disc, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
        return inpainted
    return rgb.copy()


def vessel_median_filter(response, fov, median_size=5):
    """
    Apply median filter to vesselness response to remove speckle.
    Vessels are thin and elongated, speckle noise is isolated pixels.
    Median filtering preserves edges while removing isolated noise.
    """
    filtered = cv2.medianBlur((response * 255).astype(np.uint8), median_size)
    return normalize01(filtered.astype(np.float32), fov)


def run_simple_green(rgb, fov):
    """
    Simple green channel processing: just green channel, slight CLAHE, matched filter.
    Minimal preprocessing to avoid creating false textures.
    """
    green = rgb[:,:,1].copy()
    green[~fov] = 0
    # Very mild CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    enhanced[~fov] = 0
    inverted = 255 - enhanced
    inv_f = normalize01(inverted.astype(np.float32), fov)
    gabor = gabor_filter_response(inv_f, fov)
    return gabor


def sobel_vesselness(rgb, fov):
    """Simple Sobel-based edge detection for vessels."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray[~fov] = 0
    # Use green channel instead
    green = rgb[:,:,1].copy()
    green[~fov] = 0
    # Sobel magnitude
    sobelx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=5)
    sobely = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude[~fov] = 0
    return normalize01(magnitude, fov)


def phase_congruency_vesselness(rgb, fov):
    """
    Simple phase-based approach: use local energy from quadrature filters.
    Phase is illumination-invariant, good for detecting vessels regardless
    of contrast.
    """
    green = rgb[:,:,1].astype(np.float32)
    green[~fov] = 0
    
    # Simple phase symmetry: use Log-Gabor-like approach
    # Apply difference of Gaussians at multiple scales (bandpass)
    result = np.zeros_like(green)
    for sigma in [1.0, 1.5, 2.5, 4.0]:
        k = 1.6
        inner = cv2.GaussianBlur(green, (0, 0), sigmaX=sigma)
        outer = cv2.GaussianBlur(green, (0, 0), sigmaX=sigma * k)
        dog = inner - outer
        # Take absolute value (both dark and bright edges)
        dog = np.abs(dog)
        result = np.maximum(result, dog)
    
    result[~fov] = 0
    return normalize01(result, fov)


def main():
    print("=" * 70)
    print("ROUND 5: Optic disc inpainting + simpler preprocessing + phase")
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

        # Optic disc inpainting
        inpainted = inpaint_optic_disc(working, fov)
        fov2 = estimate_fov_mask(inpainted)

        # Test maps
        maps = {}

        # Standard C6G6 with and without disc inpainting
        c6_ch = cielab_green_clahe_source(working, fov, l_clip=6.0, green_clip=6.0)
        c6_inv_f = normalize01((255 - c6_ch).astype(np.float32), fov)
        maps['c6g6_gabor'] = gabor_filter_response(c6_inv_f, fov)

        # C6G6 + disc inpainting
        c6_ch2 = cielab_green_clahe_source(inpainted, fov2, l_clip=6.0, green_clip=6.0)
        c6_inv_f2 = normalize01((255 - c6_ch2).astype(np.float32), fov2)
        maps['c6g6_inpaint_gabor'] = gabor_filter_response(c6_inv_f2, fov2)

        # Simple green (mild CLAHE)
        maps['simple_green_gabor'] = run_simple_green(working, fov)

        # Simple green + inpainting
        maps['simple_green_inpaint_gabor'] = run_simple_green(inpainted, fov2)

        # Phase congruency
        maps['phase'] = phase_congruency_vesselness(working, fov)

        # Phase guided: phase * gabor
        phase = maps['phase']
        gabor = maps['c6g6_gabor']
        maps['phase_gabor'] = normalize01(phase * gabor, fov)

        # Median filtered gabor
        maps['gabor_median'] = vessel_median_filter(gabor, fov)

        # Almeida baseline
        res = process_channel_source(working, fov, CONFIG, 'C6G6')
        maps['almeida'] = res['combined']

        # Almeida + inpainting
        res2 = process_channel_source(inpainted, fov2, CONFIG, 'C6G6')
        maps['almeida_inpaint'] = res2['combined']

        # Test all
        for sname, soft in maps.items():
            inner = erode_mask(fov, 8)
            vals = soft[inner]
            vals = vals[vals > 0]

            for d in [8, 10, 12]:
                th = threshold_response_map(soft, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_P{d:02d}'; all_metrics.append(m)

            if len(vals) > 50:
                hyst = adaptive_hysteresis(soft, fov)
                m = segmentation_metrics(hyst, gt)
                m['method'] = f'{sname}_Hyst'; all_metrics.append(m)

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
    write_csv(OUTPUT_DIR / 'round5_summary.csv', summary)

    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 72)
    for r in summary[:25]:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")


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
