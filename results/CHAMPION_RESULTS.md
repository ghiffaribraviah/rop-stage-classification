# Masked-CNN Champion — Results

5-fold cross-validated head-to-head between a small from-scratch CNN and the
classical ML baseline, using the **same vessel/ridge evidence** the classical
model saw. Script: [`trashbin/experiments/masked_cnn_cv_champion.py`](../trashbin/experiments/masked_cnn_cv_champion.py).

## Idea

The 0.5147 classical baseline (`rop_classical.py`) hand-engineers vessel and ridge
features, then feeds a classical ML head. Question: **given the identical
hand-engineered evidence, does a small CNN learn a better decision boundary than
the classical head?** Same folds, same images, same features → a fair comparison
that isolates the model, not the features.

## Approach

- **Input (not raw RGB):** a byte-identical 3-channel masked construction —
  `[vessel_softmap, ridge_softmap, masked_CLAHE_green]`, all FOV-masked, 224×224.
- **Vessel channel = the locked vessel-segmentation champion** `g40_m60_fine`
  (Dice 0.4739, see [`experiments/vessel/VESSEL_FINDINGS.md`](../experiments/vessel/VESSEL_FINDINGS.md)):
  `0.40 · gabor_tophat + 0.60 · meijering_fine`, FOV-masked and renormalized.
  The binarization stage (P0.16 + top-3 CC + close 3×3) is intentionally dropped —
  the CNN consumes the continuous soft map, not a binary mask.
- **Model:** TinyResNetV2 (3 stages, widths 48/96/192), focal loss (γ=1.0) with
  inverse-frequency class weights, AdamW lr 1e-3, cosine schedule, 80 epochs,
  flip+rotate augmentation. No ImageNet weights (channels are softmaps, not RGB).
- **Protocol:** `StratifiedKFold(5, shuffle=True, random_state=42)` over all 756
  ROP images; out-of-fold predictions pooled; macro-F1 reported.
- **Compute:** Modal, A100-80GB. Inputs generated once and cached on a volume.

## Key decisions

- **Recipe-faithfulness fix (pre-run).** The ported vessel channel was audited
  line-by-line against the source of truth (`vessel_round7.py`,
  `vessel_round3.build_soft`, `vessel_champion_config.csv`). The Gabor term was
  byte-identical (corr 1.0). The **meijering term was not**: the port fed meijering
  `norm01(255-enh)` (percentile 1–99 stretch) instead of the source's
  `(255-enh)/255.0` (plain scaling). Meijering's Hessian eigen-analysis is
  intensity-scale sensitive, so the stretch diverged on 50–78% of FOV pixels
  (corr 0.33–0.58, mean Δ≈0.15) — corrupting the **0.60-weight dominant channel**.
  Fixed to match the source; re-verified byte-identical (meanΔ=0.000000, corr 1.0).
- **Cache invalidation.** The Modal volume held 756 masked inputs cached from the
  pre-fix recipe. Purged `/masked3ch` so all inputs regenerated with the corrected
  vessel channel before training.

## Measured results

**Masked-TinyResNet, 5-fold OOF (StratifiedKFold seed=42), 756 images:**

| Metric | Value |
| --- | --- |
| OOF macro-F1 | **0.7332** |
| OOF accuracy | 0.7606 |
| OOF precision (macro) | 0.7576 |
| OOF recall (macro) | 0.7383 |
| Classical baseline macro-F1 | 0.5147 |
| Δ vs baseline | **+0.2185** |

Per-fold macro-F1: 0.7267 / 0.7515 / 0.7952 / 0.6771 / 0.7083.

Per-class (OOF):

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal | 0.96 | 0.88 | 0.92 | 236 |
| Stage1 | 0.62 | 0.56 | 0.59 | 94 |
| Stage2 | 0.52 | 0.83 | 0.64 | 165 |
| Stage3 | 0.94 | 0.68 | 0.79 | 261 |

Modal run: `fc-01KV78S57P9SDCD6EC68NCA0H6`
(`https://modal.com/apps/chairulridjaal/main/ap-xB3Y6YhrpS9NK9jS4TupaE`).

## Important caveat: data leakage in the CV protocol

The macro-F1 of 0.7332 is **inflated by data leakage and is not a clean estimate
of generalization to unseen patients.** This was identified before finalizing and
is documented here for honesty.

- **Confirmed (byte-level, md5):** 6 byte-identical duplicate image groups (12
  files) exist among the 756 ROP images. **3 are cross-class label conflicts** —
  the exact same image filed under two different stages:
  - `Stage_2_ROP_154.jpg` ≡ `Stage_3_ROP_210.jpg`
  - `Stage_2_ROP_52.jpg`  ≡ `Stage_3_ROP_254.jpg`
  - `Stage_1_ROP_42.jpg`  ≡ `Stage_2_ROP_4.jpg`
- The split is over **individual files**, so duplicates and (more importantly)
  multiple images of the same eye/exam can land on both sides of a fold boundary,
  letting the CNN recognize the eye rather than the disease stage. Near-duplicate
  analysis (NCC ≥ 0.99) flags ~63 images, supporting same-eye spread across folds.
- 12 byte-identical files (~1.6% of data) cannot by themselves explain the
  0.51 → 0.73 jump; the larger driver is the per-image (non-grouped) split.

**Why GroupKFold-by-patient was not used:** the dataset does not expose reliable
patient IDs, and the per-stage patient counts are too small for grouped splitting
(e.g. Stage 3 maps to only ~3 distinct patients). Grouping would collapse minority
classes and make stratified CV impossible. Per-image stratified CV — matching the
classical baseline's protocol — was therefore the pragmatic choice.

**Fair-comparison note:** the 0.5147 classical baseline uses the *same* per-image
StratifiedKFold split, so the comparison is apples-to-apples in leakiness. Both
numbers are upper bounds, not patient-level generalization estimates. The CNN's
**relative** improvement under the identical protocol and identical features is the
meaningful signal here; the absolute 0.7332 should not be quoted as clean
held-out performance.
