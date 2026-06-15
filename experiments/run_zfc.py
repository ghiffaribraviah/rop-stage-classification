"""
Z-Fused-Coherence implementation from friend's code.
Ported from the notebook to standalone experiment.
"""
import sys, csv, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import *
from advanced_pipeline import gabor_filter_response, adaptive_hysteresis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / 'data' / 'Agrawal2021'
CONFIG = VesselPipelineConfig()


# ── Helper: fill outside FOV ──────────────────────────────────────────

def _fill(channel, fov):
    out = channel.astype(np.float32).copy()
    if np.any(fov):
        out[~fov] = float(np.median(out[fov]))
    return out


# ── Z-score helpers ────────────────────────────────────────────────────

def local_dark_zscore_response(channel, fov, kernel_size=31, min_std=3.0, z_cap=3.0):
    """Z-score: how dark is each pixel relative to its local median."""
    source = np.clip(_fill(channel, fov), 0, 255).astype(np.uint8)
    ks = max(3, int(kernel_size) | 1)
    median = cv2.medianBlur(source, ks).astype(np.float32)
    abs_dev = np.abs(source.astype(np.float32) - median).astype(np.uint8)
    mad = cv2.medianBlur(abs_dev, ks).astype(np.float32)
    robust_sigma = np.maximum(1.4826 * mad, float(min_std))
    dark = np.maximum(median - source.astype(np.float32), 0.0)
    response = np.clip((dark / robust_sigma) / float(z_cap), 0.0, 1.0)
    response[~fov] = 0
    return response.astype(np.float32)


def multiscale_local_dark_zscore_response(channel, fov):
    responses = [local_dark_zscore_response(channel, fov, kernel_size=size, min_std=3.0, z_cap=3.0)
                 for size in (15, 31, 51)]
    response = np.max(np.stack(responses, axis=0), axis=0)
    response[~fov] = 0
    return response.astype(np.float32)


# ── Input sources ─────────────────────────────────────────────────────

def c6g6_source(rgb, fov):
    """C6G6: CLAHE tile=16, clip=6.0 on green channel from CIELAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8)).apply(l_ch)
    l_clahe[~fov] = 0
    green_clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8)).apply(rgb[:,:,1])
    green_clahe[~fov] = 0
    # C6G6 uses the green channel after CIELAB processing
    cielab_green = l_clahe  # Actually the green from the CIELAB space
    return green_clahe  # Simpler: just CLAHE on green channel at clip=6.0


def background_corrected_green_source(rgb, fov, sigma=35.0, clahe_clip=4.0):
    green = rgb[:,:,1].copy()
    # Divide background flatten
    bg = cv2.GaussianBlur(_fill(green, fov), (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = np.maximum(bg, 1.0)
    corrected = green.astype(np.float32) / bg * 128.0
    corrected[~fov] = 0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    return clahe.apply(corrected)


def lab_l_background_corrected_source(rgb, fov, sigma=35.0, clahe_clip=4.0):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch = lab[:,:,0]
    bg = cv2.GaussianBlur(_fill(l_ch, fov), (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg = np.maximum(bg, 1.0)
    corrected = l_ch.astype(np.float32) / bg * 128.0
    corrected[~fov] = 0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16))
    return clahe.apply(corrected)


# ── Z-fused mean response ─────────────────────────────────────────────

def z_fused_mean_response(rgb, fov):
    """Compute z-fused mean vesselness response."""
    c6 = c6g6_source(rgb, fov)
    bg_g = background_corrected_green_source(rgb, fov, sigma=35.0, clahe_clip=4.0)
    bg_l = lab_l_background_corrected_source(rgb, fov, sigma=35.0, clahe_clip=4.0)
    
    z_c6 = multiscale_local_dark_zscore_response(c6, fov)
    z_bg_g = multiscale_local_dark_zscore_response(bg_g, fov)
    z_bg_l = multiscale_local_dark_zscore_response(bg_l, fov)
    
    fused = normalize01(0.60 * z_c6 + 0.25 * z_bg_g + 0.15 * z_bg_l, fov)
    return c6, z_c6, fused, z_bg_g


# ── Noise floor & smoothing ──────────────────────────────────────────

def soft_noise_floor_response(response, fov, low_pct=42.0, high_pct=98.5, gamma=0.78):
    output = np.zeros(response.shape, dtype=np.float32)
    if not np.any(fov):
        return output
    values = response[fov].astype(np.float32)
    low = float(np.percentile(values, float(low_pct)))
    high = float(np.percentile(values, float(high_pct)))
    if high <= low:
        return normalize01(response, fov)
    scaled = np.clip((response.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    scaled = np.power(scaled, float(gamma))
    scaled[~fov] = 0
    return scaled.astype(np.float32)


def median_smooth_response(response, fov, median_size=3, blur_sigma=0.45):
    r = np.clip(response * 255, 0, 255).astype(np.uint8)
    ms = max(3, int(median_size) | 1)
    s = cv2.medianBlur(r, ms).astype(np.float32) / 255.0
    if blur_sigma > 0:
        s = cv2.GaussianBlur(s, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    s[~fov] = 0
    return normalize01(s, fov)


# ── Coherence gate ─────────────────────────────────────────────────────

def local_std_float(channel, fov, kernel_size=21):
    """Local standard deviation."""
    filled = _fill(channel, fov)
    ks = max(3, int(kernel_size) | 1)
    mean = cv2.blur(filled, (ks, ks))
    sq_mean = cv2.blur(filled * filled, (ks, ks))
    var = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(var)
    std[~fov] = 0
    return std


def directional_dark_ridge_kernel(length, center_width, side_offset, side_width, angle_deg):
    """Kernel for directional dark ridge detection."""
    size = length + 2 * (side_offset + side_width) + 4
    if size % 2 == 0:
        size += 1
    center = size // 2
    yy, xx = np.mgrid[:size, :size].astype(np.float32) - center
    theta = np.deg2rad(angle_deg)
    along = xx * np.cos(theta) + yy * np.sin(theta)
    across = -xx * np.sin(theta) + yy * np.cos(theta)
    
    # Center bar: vessel body (positive weight)
    center_mask = (np.abs(along) <= length / 2.0) & (np.abs(across) <= center_width / 2.0)
    # Side bars: background on each side (negative weight)
    inner = np.abs(across) - center_width / 2.0
    side_mask = (np.abs(along) <= length / 2.0) & (inner >= side_offset) & (inner <= side_offset + side_width)
    
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[center_mask] = 1.0
    kernel[side_mask] = -0.5
    kernel = kernel - kernel.mean()
    norm = np.sum(np.abs(kernel))
    if norm > 0:
        kernel /= norm
    return kernel


def directional_dark_ridge_response(channel, fov, scales=None, angle_step=15, normalize_by_local_std=True):
    if scales is None:
        scales = ((7, 1, 2, 1), (9, 1, 3, 1), (13, 1, 4, 1), (17, 2, 5, 2))
    filled = _fill(channel, fov)
    response = np.zeros(channel.shape, dtype=np.float32)
    for length, cw, soff, sw in scales:
        for angle in range(0, 180, int(angle_step)):
            kernel = directional_dark_ridge_kernel(length, cw, soff, sw, angle)
            filtered = cv2.filter2D(filled, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
            response = np.maximum(response, filtered)
    response = np.maximum(response, 0.0)
    if normalize_by_local_std:
        std = local_std_float(channel, fov, kernel_size=21)
        response = response / np.maximum(std, 3.0)
    response[~fov] = 0
    return normalize01(response, fov)


def coherence_gate(channel, fov, sigma=1.2):
    """Measure local structure coherence."""
    filled = _fill(channel, fov)
    # Structure tensor components
    ix = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    ix2 = cv2.GaussianBlur(ix * ix, (0, 0), sigmaX=sigma, sigmaY=sigma)
    iy2 = cv2.GaussianBlur(iy * iy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    ixy = cv2.GaussianBlur(ix * iy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Coherence = (λ1 - λ2) / (λ1 + λ2) where λ are eigenvalues
    trace = ix2 + iy2
    det = ix2 * iy2 - ixy * ixy
    sqrt_disc = np.sqrt(np.maximum(trace * trace / 4.0 - det, 0.0))
    lambda1 = trace / 2.0 + sqrt_disc
    lambda2 = np.maximum(trace / 2.0 - sqrt_disc, 0.0)
    coherence = np.zeros_like(lambda1)
    denom = lambda1 + lambda2
    mask = denom > 1e-8
    coherence[mask] = ((lambda1[mask] - lambda2[mask]) / denom[mask])
    coherence[~fov] = 0
    return coherence.astype(np.float32)


def coherence_weight_response(response, source, fov, floor=0.55):
    """Weight response by coherence gate."""
    coh = coherence_gate(source, fov, sigma=1.2)
    weighted = response.astype(np.float32) * (float(floor) + (1.0 - float(floor)) * coh)
    weighted[~fov] = 0
    return normalize01(weighted, fov)


# ── Main Z-Fused-Coherence pipeline ───────────────────────────────────

def z_fused_coherence_pipeline(rgb, fov):
    """Full Z-Fused-Coherence pipeline."""
    c6, z_c6, z_fused, _ = z_fused_mean_response(rgb, fov)
    # Darken: noise floor removal
    z_fused_dark = soft_noise_floor_response(z_fused, fov, low_pct=42.0, high_pct=98.5, gamma=0.78)
    # Coherence weight
    z_fused_coherence = coherence_weight_response(z_fused_dark, c6, fov, floor=0.62)
    return {
        'z_c6': z_c6,
        'z_fused': z_fused,
        'z_fused_dark': z_fused_dark,
        'z_fused_coherence': z_fused_coherence,
        'c6': c6,
    }


def main():
    print("=" * 70)
    print("Z-FUSED-COHERENCE FROM FRIEND'S CODE")
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

        # Z-Fused-Coherence
        maps = z_fused_coherence_pipeline(working, fov)

        # Also our best for comparison
        green = working[:,:,1].copy()
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16))
        enh = clahe.apply(green); enh[~fov] = 0
        inv = 255 - enh; inv_f = normalize01(inv.astype(np.float32), fov)
        gab = gabor_filter_response(inv_f, fov)
        r7 = cv2.medianBlur((gab*255).astype(np.uint8), 7); r7[~fov] = 0
        maps['our_best'] = normalize01(r7.astype(np.float32), fov)

        # Test all maps
        for sname, soft in maps.items():
            for d in [8, 10, 12]:
                th = threshold_response_map(soft, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_P{d:02d}'; all_metrics.append(m)

            # Try with median filtering (might help)
            r7 = cv2.medianBlur((soft*255).astype(np.uint8), 7)
            r7[~fov] = 0
            soft_m7 = normalize01(r7.astype(np.float32), fov)
            for d in [10]:
                th = threshold_response_map(soft_m7, fov, method='percentile', target_density=d/100.0)
                m = segmentation_metrics(th > 0, gt)
                m['method'] = f'{sname}_M7_P{d:02d}'; all_metrics.append(m)

            # Hysteresis
            inner = erode_mask(fov, 8)
            vals = soft[inner]
            vals = vals[vals > 0]
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
    write_csv(OUTPUT_DIR / 'zfc_summary.csv', summary)

    print(f"\n{'Method':40s} {'Dice':>8s} {'Sens':>8s} {'Prec':>8s}")
    print("-" * 68)
    for r in summary[:25]:
        print(f"{r['method']:40s} {r['dice']:>8.4f} {r['sensitivity']:>8.4f} {r['precision']:>8.4f}")

    print("\n═══ Z-Fused-Coherence vs Our Best ═══")
    for target in ['z_fused_coherence', 'z_fused', 'z_c6', 'our_best']:
        rows = [r for r in summary if r['method'].startswith(f'{target}_P10')]
        if rows:
            print(f"  {target:30s} Dice={rows[0]['dice']:.4f}  Sens={rows[0]['sensitivity']:.4f}  Prec={rows[0]['precision']:.4f}")


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
