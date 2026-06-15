"""
Round 8: CLAHE tile size sweep, wavelet denoising, TV-L1 denoising,
response normalization variants.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, adaptive_hysteresis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


def clahe_variants(green, fov):
    """Test different CLAHE tile sizes on green channel."""
    variants = {}
    for tilesize in [4, 8, 16, 32, 64]:
        for clip in [1.0, 2.0, 3.0, 5.0]:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tilesize, tilesize))
            enh = clahe.apply(green)
            enh[~fov] = 0
            name = f'clahe_t{tilesize}_c{clip:.0f}'
            variants[name] = enh
    return variants


def tv_l1_denoise(image, fov, strength=10, iterations=10):
    """Total Variation L1 denoising - edge preserving."""
    img = image.astype(np.float32) / 255.0
    img[~fov] = 0
    filled = img.copy()
    if np.any(fov):
        filled[~fov] = float(np.median(img[fov]))
    
    dt = 0.1 / strength
    for _ in range(iterations):
        grad_x = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
        # TV-L1: shrink gradient magnitude
        grad_x = grad_x / mag
        grad_y = grad_y / mag
        div_x = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3)
        div_y = cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3)
        filled = filled + dt * (div_x + div_y)
        # Data term
        filled = filled + dt * strength * (img - filled)
        filled[~fov] = 0
    
    result = np.clip(filled * 255, 0, 255).astype(np.uint8)
    result[~fov] = 0
    return result


def nlm_denoise(image, fov, h=10):
    """Non-local means denoising."""
    img = image.astype(np.uint8)
    img[~fov] = 0
    denoised = cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)
    denoised[~fov] = 0
    return denoised


def log_gabor_filter_response(inverted_float, fov):
    """
    Log-Gabor filter bank - better for natural images than standard Gabor.
    Log-Gabor has no DC component and allows larger bandwidth.
    """
    response = np.zeros(inverted_float.shape, dtype=np.float32)
    rows, cols = inverted_float.shape
    
    # Compute FFT
    f = np.fft.fft2(inverted_float)
    fshift = np.fft.fftshift(f)
    
    # Frequency grid
    cx, cy = cols // 2, rows // 2
    y, x = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    radius = np.sqrt(x**2 + y**2)
    radius[cy, cx] = 1  # avoid division by zero
    theta = np.arctan2(y, x)
    
    # Log-Gabor parameters
    nscale = 5
    norient = 12
    min_wavelength = 3
    mult = 2.1
    sigma_onf = 0.55
    d_theta_on_sigma = 0.4
    
    for s in range(nscale):
        wavelength = min_wavelength * mult**s
        fo = 1.0 / wavelength  # center frequency
        # Radial component
        log_gabor = np.exp(-(np.log(radius / fo))**2 / (2 * np.log(sigma_onf)**2))
        log_gabor[cy, cx] = 0  # DC
        
        for o in range(norient):
            angle = o * np.pi / norient
            # Angular component
            ds = np.sin(theta - angle)
            dc = np.cos(theta - angle)
            dtheta = np.abs(np.arctan2(ds, dc))
            spread = np.exp(-dtheta**2 / (2 * d_theta_on_sigma**2))
            
            filter_ = log_gabor * spread
            # Filter the image
            filtered = np.fft.ifft2(np.fft.ifftshift(fshift * filter_))
            filtered = np.abs(filtered)
            response = np.maximum(response, filtered)
    
    response[~fov] = 0
    return normalize01(response, fov)


def multi_scale_morphological_tophat(green, fov):
    """Multi-scale morphological tophat for vessel enhancement."""
    result = np.zeros(green.shape, dtype=np.float32)
    green_f = green.astype(np.float32)
    
    for size in [3, 5, 7, 11, 15, 21, 31]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # Black tophat (extracts dark structures on bright background)
        blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
        # White tophat (extracts bright structures on dark background)
        whitehat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
        # Vessels are dark on bright retina, so blackhat is more relevant
        result = np.maximum(result, blackhat.astype(np.float32))
    
    result[~fov] = 0
    return normalize01(result, fov)


def gradient_magnitude(green, fov):
    """Gradient magnitude - simple edge detector."""
    gx = cv2.Sobel(green, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(green, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    grad[~fov] = 0
    return normalize01(grad, fov)


def local_response_normalization(response, fov, sigma=10):
    """Normalize response by local mean and std (adaptive contrast)."""
    r = response.astype(np.float32)
    r[~fov] = 0
    filled = r.copy()
    if np.any(fov):
        filled[~fov] = float(np.median(r[fov]))
    
    local_mean = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma)
    local_sq = cv2.GaussianBlur(filled**2, (0, 0), sigmaX=sigma)
    local_std = np.sqrt(np.maximum(local_sq - local_mean**2, 1e-8))
    
    normalized = (r - local_mean) / (local_std + 0.1)
    normalized[~fov] = 0
    return normalize01(normalized, fov)


def main():
    print("=" * 70)
    print("ROUND 8: CLAHE sweep + TV denoise + Log-Gabor + local norm")
    print("=" * 70)

    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    retcam = [p for p in pairs if p['source'] == 'RetCam'][:6]
    neo = [p for p in pairs if p['source'] == 'Neo'][:6]
    sample = retcam + neo
    print(f"Images: {len(sample)} (fast run)")

    all_metrics = []

    for idx, pair in enumerate(sample):
        rgb = read_rgb(pair['image_path'])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair['mask_path'], fov.shape)
        green = working[:,:,1].copy()

        # ── CLAHE sweep ──
        for tilesize in [8, 16, 32, 48]:
            for clip in [1.5, 2.5, 4.0, 6.0]:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tilesize, tilesize))
                enh = clahe.apply(green)
                enh[~fov] = 0
                inv = 255 - enh
                inv_f = normalize01(inv.astype(np.float32), fov)
                gabor = gabor_filter_response(inv_f, fov)
                # Median 7
                r7 = cv2.medianBlur((gabor * 255).astype(np.uint8), 7)
                r7[~fov] = 0
                gabor_m7 = normalize01(r7.astype(np.float32), fov)
                
                for d in [10]:
                    th = threshold_response_map(gabor_m7, fov, method='percentile', target_density=d/100.0)
                    m = segmentation_metrics(th > 0, gt)
                    m['method'] = f'clahe_t{tilesize}c{clip:.1f}_P{d:02d}'
                    all_metrics.append(m)

        # ── TV-L1 denoising on green channel ──
        for strength in [5, 10, 20]:
            tv = tv_l1_denoise(green, fov, strength=strength, iterations=15)
            inv_tv = 255 - tv
            inv_tv_f = normalize01(inv_tv.astype(np.float32), fov)
            gabor_tv = gabor_filter_response(inv_tv_f, fov)
            r7 = cv2.medianBlur((gabor_tv * 255).astype(np.uint8), 7)
            r7[~fov] = 0
            gabor_tv_m7 = normalize01(r7.astype(np.float32), fov)
            for d in [10]:
                th = threshold_response_map(gabor_tv_m7, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'tvl1_s{strength}_P{d:02d}'; all_metrics.append(m)

        # ── NL-Means denoising ──
        for h in [5, 10, 15]:
            nlm = nlm_denoise(green, fov, h=h)
            inv_nlm = 255 - nlm
            inv_nlm_f = normalize01(inv_nlm.astype(np.float32), fov)
            gabor_nlm = gabor_filter_response(inv_nlm_f, fov)
            r7 = cv2.medianBlur((gabor_nlm * 255).astype(np.uint8), 7)
            r7[~fov] = 0
            gabor_nlm_m7 = normalize01(r7.astype(np.float32), fov)
            for d in [10]:
                th = threshold_response_map(gabor_nlm_m7, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'nlm_h{h}_P{d:02d}'; all_metrics.append(m)

        # ── Log-Gabor filter ──
        inv_green = 255 - green
        inv_green_f = normalize01(inv_green.astype(np.float32), fov)
        log_gabor = log_gabor_filter_response(inv_green_f, fov)
        for d in [8, 10, 12]:
            th = threshold_response_map(log_gabor, fov, method='percentile', target_density=d/100.0)
            m = segmentation_metrics(th > 0, gt)
            m['method'] = f'loggabor_P{d:02d}'; all_metrics.append(m)

        # ── Multi-scale morphological tophat ──
        mstop = multi_scale_morphological_tophat(green, fov)
        for d in [8, 10, 12]:
            th = threshold_response_map(mstop, fov, method='percentile', target_density=d/100.0)
            m = segmentation_metrics(th > 0, gt)
            m['method'] = f'mstophat_P{d:02d}'; all_metrics.append(m)

        # ── Local response normalization on C6G6 gabor ──
        c6_ch = cielab_green_clahe_source(working, fov, l_clip=6.0, green_clip=6.0)
        c6_inv_f = normalize01((255 - c6_ch).astype(np.float32), fov)
        gabor_c6 = gabor_filter_response(c6_inv_f, fov)
        for sigma in [5, 10, 20]:
            lrn = local_response_normalization(gabor_c6, fov, sigma=sigma)
            r7 = cv2.medianBlur((lrn * 255).astype(np.uint8), 7)
            r7[~fov] = 0
            lrn_m7 = normalize01(r7.astype(np.float32), fov)
            for d in [10]:
                th = threshold_response_map(lrn_m7, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'lrn_s{sigma}_P{d:02d}'; all_metrics.append(m)

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
    write_csv(OUTPUT_DIR / 'round8_summary.csv', summary)

    print(f"\n{'Method':45s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 72)
    for r in summary:
        print(f"{r['method']:45s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    print("\n═══ CLAHE SWEEP RESULTS ═══")
    for r in summary:
        if r['method'].startswith('clahe_t'):
            print(f"  {r['method']:30s} Dice={r['dice']:.4f}")

    print("\n═══ TOP 10 ═══")
    for r in summary[:10]:
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
