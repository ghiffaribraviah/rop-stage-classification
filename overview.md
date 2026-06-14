# ROP Stage Classification Workflow

This project is a notebook-first workflow for Retinopathy of Prematurity (ROP)
stage classification. The main entry point is `rop-stage-classification.ipynb`.

## Data

The expected local data layout is:

```text
data/
  Zhao2024/
    Normal/
    Stage1/
    Stage2/
    Stage3/
    laser scars/
  Agrawal2021/
    HVDROPDB-BV/
```

`Zhao2024` is used for stage classification. The `laser scars` class is kept in
the dataset folder but excluded from the four-class classification workflow.

`Agrawal2021/HVDROPDB-BV` is used for vessel segmentation debugging and metric
checks because it includes vessel masks.

## Main Workflow

The notebook builds four deterministic image scenarios:

1. `S1_raw`: resized RGB fundus image.
2. `S2_enhanced`: conservative CIELAB/CLAHE-enhanced RGB image.
3. `S3_vessel_mask`: vesselness map repeated into three channels.
4. `S4_vessel_guided`: enhanced RGB image guided by the vesselness map.

The classification model is a small custom residual CNN. Training is disabled
by default so the notebook can run lightweight preprocessing and debugging
without starting a full experiment.

## Fixed Vessel Baseline

The classical vessel pipeline is inlined directly inside
`rop-stage-classification.ipynb`. The fixed baseline is:

```text
vesselness_mode      = almeida
threshold_method     = triangle
target_density       = 0.14
fov_erode_px         = 12
min_component_area   = 4
background_sigma     = 30.0
bilateral_d          = 9
final_skeletonize    = False
auto_binary_selection = False
```

This corresponds to the `normalization_residual_tuned` pipeline. The debug
visualization shows the single intended processing path, including residual
normalization, rather than comparing multiple normalization methods.

## Debug Mode

`cfg.debug_mode = True` regenerates a 15-image per-process visualization:

- 5 RetCam images from Agrawal2021
- 5 Neo images from Agrawal2021
- 5 Zhao2024 images

The debug output is saved to:

```text
output/00_debug_baseline/normalization_residual_tuned/
```

Generated files:

```text
debug_baseline_15_samples.jpg
debug_agrawal_sample_metrics.csv
debug_agrawal_sample_summary.csv
debug_baseline_config.csv
```

## Output Convention

The repository uses numbered output folders:

```text
output/
  00_debug_baseline/
  01_preprocessing_cache/
  02_training_runs/
  03_evaluation/
  04_figures/
```

Large binary files, datasets, images, model weights, and arrays are configured
for Git LFS through `.gitattributes`.
