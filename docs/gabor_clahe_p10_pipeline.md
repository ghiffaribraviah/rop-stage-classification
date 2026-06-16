# Gabor + CLAHE P10 Vessel Segmentation Pipeline

This document describes a portable vessel-segmentation pipeline for retinal images. It is written as an implementation spec, not tied to any specific repository.

The pipeline uses:

- green-channel retinal image processing
- illumination normalization
- dark-vessel enhancement
- CLAHE
- multi-orientation Gabor filtering
- two-stage hysteresis thresholding
- connected-component filtering
- FOV-border subtraction

The final output is a binary vessel mask.

## Required Inputs

For each image:

- `rgb`: RGB retinal image as an array with shape `(H, W, 3)`.
- `fov`: boolean field-of-view mask with shape `(H, W)`.

The FOV mask should represent the visible retinal image area. The pipeline does **not** shrink or reshape this FOV. It only subtracts a thin FOV outline later to avoid selecting the image border as a vessel.

## Best Tuned Configuration

Use this configuration as the default starting point:

```python
config = {
    "target_density": 0.115,
    "main_low_mult": 1.45,
    "main_high_mult": 0.60,
    "residual_enabled": True,
    "residual_low_mult": 1.10,
    "recovery_axis_ratio": 2.6,
    "recovery_skeleton_length": 18,
    "recovery_branch_density": 0.10,
}
```

Validation result on 100 GT masks from the Agrawal vessel dataset:

| Metric | Score |
|---|---:|
| clDice | 0.4856 |
| Dice / F1 | 0.4779 |
| Precision | 0.4952 |
| Recall | 0.4617 |
| Accuracy | 0.8641 |

If higher recall is preferred over precision, use:

```python
config = {
    "target_density": 0.115,
    "main_low_mult": 1.45,
    "main_high_mult": 0.60,
    "residual_enabled": True,
    "residual_low_mult": 1.10,
    "recovery_axis_ratio": 2.0,
    "recovery_skeleton_length": 12,
    "recovery_branch_density": 0.20,
}
```

That recall-focused config reached recall `0.4638`, Dice `0.4757`, clDice `0.4801`, and precision `0.4882`.

## High-Level Algorithm

1. Extract the green channel.
2. Fill pixels outside the FOV before filtering.
3. Normalize illumination using a large Gaussian background.
4. Enhance small dark vessel structures with multi-scale dark response.
5. Apply CLAHE to the boosted green channel.
6. Invert the result so vessels become bright ridges.
7. Apply a multi-scale, multi-orientation Gabor filter bank.
8. Smooth the Gabor response with median filtering.
9. Apply a second CLAHE.
10. Build a soft vessel response.
11. Subtract the FOV outline from the soft response.
12. Run threshold pass 1 to get main vessel candidates.
13. Run threshold pass 2 on residual soft response to recover separated vessels.
14. Final mask is:
    - top 2 components from threshold pass 1
    - plus top 2 components from threshold pass 2

The final mask is **not** selected as the 2 largest components after merging. The union rule is important because threshold pass 2 should add missing vessels without removing useful threshold-pass-1 vessels.

## Step 1: FOV Outline Subtraction Mask

Do not shrink the FOV. Instead, create a thin outline mask from the FOV contour and subtract it later.

Recommended thickness:

```python
min_side = min(height, width)
thickness = clip(round(min_side * 0.034), 12, 26)
```

Implementation idea:

```python
outline = zeros_like(fov)
contours = find_external_contours(fov)
draw_contours(outline, contours, thickness=thickness)
fov_outline_subtraction = outline & fov
```

Use this subtraction mask on:

- `soft_response`
- threshold masks
- final mask

This prevents bright FOV edges from becoming false vessel paths.

## Step 2: Green Channel and Outside-FOV Filling

Use the green channel:

```python
green = rgb[:, :, 1]
```

Before filtering, fill pixels outside the FOV using nearby valid FOV pixels. A nearest-neighbor fill is sufficient.

Conceptually:

```python
green_filled = fill_outside_fov_nearest(green, fov)
```

This prevents black-background pixels from influencing blur, CLAHE, and Gabor filtering.

## Step 3: Illumination Flattening

Estimate a smooth background:

```python
median_value = median(green_filled[fov])
background_sigma = clip(round(min_side * 0.035), 12, 32)
background = gaussian_blur(green_filled, sigma=background_sigma)
background = maximum(background, 1.0)
flattened = green_filled * median_value / background
```

The goal is to reduce uneven illumination while keeping vessel contrast.

## Step 4: Multi-Scale Dark-Vessel Boost

Compute a local dark response:

```python
local_background = gaussian_blur(flattened, sigma=2.0)
local_dark = max(local_background - flattened, 0)
local_dark = gaussian_blur(local_dark, sigma=0.35)
local_dark_norm = normalize01(local_dark inside fov)
```

Compute multi-scale dark responses:

```python
scale_sigmas = (1.4, 2.2, 3.6, 5.5, 8.0)
```

For each `sigma`:

```python
scale_background = gaussian_blur(flattened, sigma=sigma)
scale_dark = max(scale_background - flattened, 0)
scale_dark = gaussian_blur(scale_dark, sigma=0.35)
scale_dark_norm = normalize01(scale_dark inside fov)
```

Then:

```python
multiscale_dark = max(all_scale_dark_norm_maps)
multiscale_dark = median_blur(multiscale_dark, kernel_size=3)
```

Build the vessel boost:

```python
vessel_boost = normalize01(0.35 * local_dark_norm + 0.65 * multiscale_dark)
boosted_green = flattened - 42.0 * vessel_boost
boosted_green = gaussian_blur(boosted_green, sigma=0.25)
```

Finally stretch intensities inside the FOV:

```python
boosted_green = percentile_stretch(boosted_green, fov, low_pct=0.7, high_pct=99.3)
boosted_green[~fov] = 0
```

## Step 5: First CLAHE and Inversion

Apply CLAHE:

```python
clahe1 = CLAHE(boosted_green, clip_limit=6.0, tile_grid_size=(16, 16))
```

Invert:

```python
inverted = 255 - clahe1
```

For pixels outside the FOV, fill with the median inside-FOV value before Gabor filtering:

```python
inverted[~fov] = median(inverted[fov])
```

## Step 6: Gabor Filter Bank

Use a multi-scale, multi-orientation Gabor bank.

Recommended parameters:

```python
wavelengths = (8.0, 12.0, 16.0)
sigma = 4.0
gamma = 0.50
angles = range(0, 180, 15)
```

For each wavelength and angle:

```python
ksize = odd(max(17, round(wavelength * 2.6)))
kernel = gabor_kernel(
    ksize=ksize,
    sigma=4.0,
    theta=angle,
    lambd=wavelength,
    gamma=0.50,
    psi=0.0,
)
kernel -= mean(kernel)
kernel /= sum(abs(kernel))
filtered = filter2d(inverted / 255.0, kernel)
response = max(response, filtered)
```

Then:

```python
response = max(response, 0)
gabor_norm = normalize01(response inside fov)
gabor_norm[~fov] = 0
```

## Step 7: Median, Second CLAHE, and Soft Response

Apply median filtering:

```python
median7 = median_blur(gabor_norm, kernel_size=7)
```

Apply second CLAHE:

```python
clahe2 = CLAHE(median7, clip_limit=12.0, tile_grid_size=(12, 12))
```

Build the soft response:

```python
median_norm = normalize01(median7 inside fov)
clahe2_norm = normalize01(clahe2 inside fov)
soft_response = normalize01(0.65 * median_norm + 0.35 * clahe2_norm)
soft_response[~fov] = 0
soft_response[fov_outline_subtraction] = 0
```

## Step 8: Hysteresis Density Threshold

Both threshold stages use the same hysteresis-density method.

Inputs:

```python
response
fov
target_density
low_mult
high_mult
```

Compute low and high densities:

```python
low_density = clip(target_density * low_mult, 0.015, 0.20)
high_density = clip(target_density * high_mult, 0.008, low_density * 0.85)
```

Convert density to percentile thresholds:

```python
values = response[fov]
low_threshold = percentile(values, 100 * (1 - low_density))
high_threshold = percentile(values, 100 * (1 - high_density))
```

Build low and high masks:

```python
low_mask = response >= low_threshold
high_mask = response >= high_threshold
low_mask &= fov
high_mask &= fov
```

Keep only low-mask components that contain a high-mask seed:

```python
labels, stats = connected_components(low_mask)
seed_labels = unique(labels[high_mask])
```

Reject tiny components:

```python
min_area = clip(round(min_side * min_side * 0.000035), 8, 28)
```

Final hysteresis mask:

```python
mask = components where label in seed_labels and area >= min_area
```

## Step 9: Threshold Pass 1

Use:

```python
target_density = 0.115
low_mult = 1.45
high_mult = 0.60
```

Run hysteresis threshold on `soft_response`.

Then:

```python
raw_mask &= ~fov_outline_subtraction
threshold1_top2 = keep_largest_components(raw_mask, count=2)
threshold1_final = morphology_close(threshold1_top2, ellipse_3x3, iterations=1)
threshold1_final &= fov
threshold1_final &= ~fov_outline_subtraction
```

## Step 10: Threshold Pass 2

Threshold pass 2 searches for separated vessels not already captured by threshold pass 1.

Remove the area around threshold-1 vessels:

```python
residual_block = dilate(threshold1_final, ellipse_5x5, iterations=1)
residual_fov = fov & ~residual_block & ~fov_outline_subtraction
```

Build residual soft response:

```python
residual_soft = soft_response.copy()
residual_soft[~residual_fov] = 0
residual_soft = normalize01(residual_soft inside residual_fov)
```

Run hysteresis threshold again:

```python
target_density_2 = target_density * residual_low_mult
target_density_2 = 0.115 * 1.10
```

Use the same `low_mult` and `high_mult` as threshold pass 1:

```python
low_mult = 1.45
high_mult = 0.60
```

This gives `residual_candidates`.

## Step 11: Vessel-Like Component Filtering

Filter threshold-2 candidates component by component.

For each connected component:

1. Reject if area is too small.
2. Compute axis ratio using PCA/eigenvalues.
3. Skeletonize the component.
4. Reject if skeleton is too short.
5. Reject if branch-point density is too high.
6. Reject if response intensity is too weak.

Recommended parameters:

```python
min_area = clip(round(min_side * min_side * 0.000025), 8, 24)
axis_ratio_min = 2.6
skeleton_length_min = 18
branch_density_max = 0.10
```

Axis ratio:

```python
coords = component_pixel_coordinates
covariance = covariance_matrix(coords)
eigenvalues = eigvalsh(covariance)
axis_ratio = sqrt(largest_eigenvalue / smallest_eigenvalue)
```

Skeleton length:

```python
skeleton = skeletonize(component)
skeleton_length = count_nonzero(skeleton)
```

Branch density:

```python
neighbor_count = count_3x3_neighbors(skeleton)
branch_points = count(skeleton pixels with neighbor_count >= 5)
branch_density = branch_points / skeleton_length
```

Intensity rule:

```python
p45 = percentile(residual_soft[residual_fov], 45)
p82 = percentile(residual_soft[residual_fov], 82)
p93 = percentile(residual_soft[residual_fov], 93)

strong_enough = (
    mean_response >= p45 and max_response >= p82
) or (
    max_response >= p93
)
```

Keep only components that pass all filters.

Then:

```python
threshold2_largest2 = keep_largest_components(recovered_vessels, count=2)
threshold2_largest2 &= fov
threshold2_largest2 &= ~fov_outline_subtraction
```

## Step 12: Final Mask

The final mask must preserve threshold-pass-1 vessels and add threshold-pass-2 vessels.

Use:

```python
final_candidates = threshold1_final | threshold2_largest2
mask_final = morphology_close(final_candidates, ellipse_3x3, iterations=1)
mask_final &= fov
mask_final &= ~fov_outline_subtraction
```

Do **not** keep only the largest 2 components after this union. That can remove useful threshold-pass-1 vessels when threshold-pass-2 components change the ranking.

## Output

Return at least:

```python
{
    "soft_response": soft_response,
    "threshold1_final": threshold1_final,
    "residual_soft": residual_soft,
    "residual_candidates": residual_candidates,
    "threshold2_largest2": threshold2_largest2,
    "mask_final": mask_final,
}
```

For debugging, also return:

```python
{
    "green": green,
    "green_filled": green_filled,
    "boosted_green": boosted_green,
    "clahe1": clahe1,
    "inverted": inverted,
    "gabor_norm": gabor_norm,
    "median7": median7,
    "clahe2": clahe2,
    "raw_mask": raw_mask,
    "fov_outline_subtraction": fov_outline_subtraction,
}
```

## Implementation Notes

- Use boolean masks for all mask operations.
- Normalize only inside the relevant FOV or residual FOV.
- Always zero out values outside the FOV after each major response image.
- Use nearest-FOV fill before blur/CLAHE/Gabor to avoid black-border artifacts.
- Keep FOV shape unchanged.
- Subtract only the FOV outline from response and masks.
- Threshold pass 2 should add candidates, not replace threshold pass 1.
- Frangi filtering can be useful for visualization, but it is not part of this final mask.
- Vessel connection/bridging is not used in this final version.
