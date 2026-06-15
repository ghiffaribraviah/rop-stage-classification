# Experiments Index

All scripts live flat in this directory by design — 29 of 36 use sibling imports
(`from vessel_pipeline import ...`) or `sys.path.insert`, so moving them into
subfolders would break those imports. This index is the map instead.

Research logs: see [`VESSEL_FINDINGS.md`](./VESSEL_FINDINGS.md) (vessel Dice log,
locked eval protocol) and the project-root `RESULTS.md` (ROP staging results).

## ROP stage classification

| Script | Purpose |
| --- | --- |
| `rop_classical.py` | Handcrafted features + classical ML. The 0.5147 macro-F1 baseline. |
| `rop_ridge_classical.py` | Classical ridge / demarcation-line enhancement (no learned model). |
| `rop_ridge_unet.py` | From-scratch ridge/demarcation-line U-Net. |
| `masked_cnn_cv.py` | Masked-TinyResNet under the same 5-fold CV protocol as the classical baseline. |
| `rop_experiment.py` | ResNet50 (ImageNet) + S1_raw + WeightedRandomSampler. |
| `rop_training.py` | v2: wider model, MixUp, focal loss, from scratch. |
| `rop_ssl.py` | SimCLR self-supervised pretraining on all 1099 Zhao2024 images. |
| `train_local.py` | Local ROP classification training entrypoint. |

## Vessel segmentation pipeline

| Script | Purpose |
| --- | --- |
| `vessel_pipeline.py` | Core vessel pipeline functions (shared import target). |
| `vessel_eval.py` | Full-dataset eval harness for Agrawal2021 (100 pairs). |
| `vessel_modal.py` | Modal-hosted vessel segmentation runner. |
| `advanced_pipeline.py` | Advanced multi-stage vessel segmentation pipeline. |
| `novel_approaches.py` | Exploratory novel vessel segmentation approaches. |

## Vessel tuning rounds (chronological)

| Script | Purpose |
| --- | --- |
| `round2.py` | Color channel fusion + tuned hysteresis + preprocessing. |
| `round3.py` | Gaussian matched filter (Chaudhuri), enhanced Frangi. |
| `round4.py` | Entropy seeds → region growing (precision-focused). |
| `round5.py` | Optic disc inpainting + simpler preprocessing. |
| `round6.py` | Optimize median filtering + postprocessing on Gabor response. |
| `round7.py` | Z-Fused-Coherence: anisotropic diffusion preprocessing. |
| `tv_denoise_enhencement.py` | Round 8: CLAHE tile sweep, wavelet + TV-L1 denoising. |
| `vessel_gabor_tune.py` | Round 2 soft-map: tune Gabor filter bank + preprocessing. |
| `vessel_round3.py` | Round 3: tune around `tophat_pre` (TEST 0.4600). |
| `vessel_round4.py` | Round 4: change the soft-map builder itself. |
| `vessel_round5.py` | Round 5: refine `blend_sato_50` (TEST 0.4655). |
| `vessel_round6.py` | Round 6: 3-way detector fusion vs round-5 ceiling (TEST 0.4667). |
| `vessel_round7.py` | Round 7: fine-tune Gabor+Meijering fusion. |
| `vessel_round8.py` | Round 8: re-optimize post-processing on round-7 winners. |
| `vessel_variants_v1.py` | Variant search, tune on TRAIN report TEST. |
| `vessel_variants_fast.py` | Fast variant search: precompute soft maps once per (builder, image). |

## Overlay / U-Net / utilities

| Script | Purpose |
| --- | --- |
| `overlay_config_metrics.py` | Reproduce the exact pipeline behind `output/overlay_results.jpg`. |
| `overlay_tune.py` | Sweep overlay pipeline variations to beat mean Dice 0.466. |
| `train_unet.py` | UNet for retinal vessel segmentation on Agrawal2021. |
| `run_experiments.py` | Experiment runner: compare channel sources + fusion strategies. |
| `run_zfc.py` | Z-Fused-Coherence reference implementation. |
| `show_thresholds.py` | Visual threshold viewer for vessel detection. |
| `test_enhancements.py` | Precompute vesselness maps, then vary thresholding/postprocessing. |
