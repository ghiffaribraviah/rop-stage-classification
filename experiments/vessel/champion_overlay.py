"""
Render the vessel-segmentation CHAMPION overlay (g40_m60_fine).

This is the exact vessel channel the masked-CNN classification champion consumes
(see CHAMPION_RESULTS.md): soft = norm01(0.40*gabor_tophat + 0.60*meijering_fine),
then P0.16 + top-3 CC + close 3x3.

Unlike experiments/output/overlay_results.jpg (pure-Gabor figure), this overlays
the *fused* champion recipe so the figure matches what the classifier sees.

Output: experiments/output/champion_overlay.jpg
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from vessel_pipeline import (  # noqa: E402
    VesselPipelineConfig,
    find_agrawal_pairs,
    read_rgb,
    resize_max_side,
    estimate_fov_mask,
    read_binary_mask,
    normalize01,
    segmentation_metrics,
)
# Reuse the EXACT champion building blocks (no reimplementation).
from vessel_round7 import gabor_ch, mei_ch, predict  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AGRAWAL_ROOT = PROJECT_ROOT / "data" / "Agrawal2021"
CONFIG = VesselPipelineConfig()

# Champion recipe: g40_m60_fine
W_GABOR, W_MEI, SIGMA_SET = 0.40, 0.60, "fine"


def champion_soft(rgb, fov):
    """Fused champion soft map, identical to vessel_round7.score()."""
    d = {"rgb": rgb, "fov": fov}
    g = gabor_ch(d)
    m = mei_ch(d, SIGMA_SET)
    return normalize01(W_GABOR * g + W_MEI * m, fov)


def red_overlay(rgb, pred_mask, alpha=0.55):
    """Blend the binary prediction onto the image in red."""
    out = rgb.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255.0
    m = pred_mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * red[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    pairs = find_agrawal_pairs(AGRAWAL_ROOT)
    selected = (
        [p for p in pairs if p["source"] == "RetCam"][:4]
        + [p for p in pairs if p["source"] == "Neo"][:4]
    )

    n = len(selected)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    dices = []
    for i, pair in enumerate(selected):
        rgb = read_rgb(pair["image_path"])
        working = resize_max_side(rgb, CONFIG.process_max_side)
        fov = estimate_fov_mask(working)
        gt = read_binary_mask(pair["mask_path"], fov.shape)

        soft = champion_soft(working, fov)
        pred = predict(soft, fov)
        m = segmentation_metrics(pred, gt)
        dices.append(m["dice"])

        ov = red_overlay(working, pred)

        axes[i, 0].imshow(working)
        axes[i, 0].set_title(f"{pair['source']} {pair['name']}  (original)", fontsize=10)
        axes[i, 1].imshow(soft, cmap="inferno")
        axes[i, 1].set_title("champion soft map  0.40g+0.60m(fine)", fontsize=10)
        axes[i, 2].imshow(ov)
        axes[i, 2].set_title(f"prediction overlay  Dice={m['dice']:.4f}", fontsize=10)
        # GT contour over original
        gt_vis = working.copy()
        cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_vis, cnts, -1, (0, 255, 0), 2)
        axes[i, 3].imshow(gt_vis)
        axes[i, 3].set_title("ground-truth vessels", fontsize=10)
        for j in range(4):
            axes[i, j].axis("off")

    mean_dice = float(np.mean(dices))
    fig.suptitle(
        f"Vessel champion overlay  (g40_m60_fine, Dice {mean_dice:.4f} over {n} imgs)",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()
    out_path = OUTPUT_DIR / "champion_overlay.jpg"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"mean Dice = {mean_dice:.4f} over {n} images")
    for pair, d in zip(selected, dices):
        print(f"  {pair['source']:8s} {pair['name']:12s} dice={d:.4f}")
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
