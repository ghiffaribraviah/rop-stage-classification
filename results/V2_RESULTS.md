# Masked-CNN v2 — 5-class ROP staging toward 0.80 macro-F1 (from scratch)

Script: [`experiments/cnn/masked_cnn_cv_v2.py`](../experiments/cnn/masked_cnn_cv_v2.py).
Constraint held throughout: **no pretraining** (no ImageNet, no external weights).
All channels are FOV-masked softmaps, not RGB, so ImageNet weights would not transfer anyway.

## Goal

Move ROP stage classification from the lineage's **0.73–0.74 macro-F1** plateau to the
published **0.80+ benchmark** (Zhao et al., *Sci Data* 2024: ResNet50/ImageNet, 5-class,
F1 0.8281) **without** any pretraining.

## What changed between the lineage and v2

| Run | Classes | Images | Vessel channel | OOF macro-F1 |
| --- | --- | --- | --- | --- |
| `masked_cnn_cv.py` | 4 | 756 | plain-Gabor | 0.7425 |
| `masked_cnn_cv_champion.py` | 4 | 756 | Dice-0.4739 fusion | 0.7332 ← **regression** |
| **`masked_cnn_cv_v2.py`** | **5** | **1099** | plain-Gabor (reverted) | **0.7802 / 0.7853** |

### Root-cause finding carried into v2 (the champion regression)

The "champion" vessel softmap maximized **vessel-segmentation Dice** (0.4739) by fusing
`0.40·gabor_tophat + 0.60·meijering_fine`. That is the **wrong objective for staging**:
maximizing vessel overlap homogenizes the demarcation-line / ridge texture that separates
Stage2 from Stage3, and it cost **−0.06 F1 on Stage3** (0.85→0.79) and dropped 4-class macro
0.7425 → 0.7332. **Vessel-Dice and stage-classification are different tasks.** v2 reverts to
the plain-Gabor vessel map. The Dice-champion fusion is retained only as a documented dead end.

### Problem redefinition (4-class → 5-class)

The 0.80 target is the **published 5-class benchmark**: the full Zhao2024 set is **1099 images**
including **343 laser-scar images**, not the 4-class / 756-image subset the lineage scored.
v2 adopts the 5-class task so the number is comparable to the benchmark. Laser scars are an
easy, visually distinct class that lifts the macro floor before any model gains.

Class distribution (1099): `Laser 343, Stage3 261, Normal 236, Stage2 165, Stage1 94`.

### From-scratch regularization stack

Techniques proven to give from-scratch gains on small medical datasets (ResNet-RS,
"ResNet strikes back", CutMix):
- **MixUp + CutMix** (per-batch, p=0.5, MixUp α=0.2 / CutMix α=1.0)
- **Label smoothing** 0.1
- **Stochastic depth** (DropPath), linear schedule 0 → 0.2 across 6 residual blocks
- **Weight EMA** (decay 0.999, warmup so the average is not dominated by random init)
- **Flip test-time augmentation** at OOF inference
- AdamW lr 1e-3 / wd 5e-4, cosine schedule, 160 epochs, inverse-frequency class weights

## Measured results — v2, 5-class, 1099 images, no pretraining

Two CV protocols were run in parallel on A100-80GB. Both pool out-of-fold predictions.

### Protocol A — StratifiedKFold (comparable to lineage + benchmark)

| Metric | Value |
| --- | --- |
| OOF macro-F1 | **0.7802** |
| OOF accuracy | 0.8271 |
| OOF precision (macro) | 0.7875 |
| OOF recall (macro) | 0.7989 |

Per-fold macro-F1: 0.8283 / 0.7805 / 0.7584 / 0.7501 / 0.7833.

Per-class OOF:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal | 0.97 | 0.88 | 0.92 | 236 |
| Stage1 | 0.41 | 0.78 | 0.54 | 94 |
| Stage2 | 0.71 | 0.59 | 0.65 | 165 |
| Stage3 | 0.85 | 0.83 | 0.84 | 261 |
| Laser | 1.00 | 0.92 | 0.95 | 343 |

Modal run: `ap-uYELPWpJlXHWuNA4nOAKAX`.

### Protocol B — StratifiedGroupKFold (leakage-corrected, honest estimate)

Near-duplicate / same-eye groups (`clean_manifest.json`) are kept on one side of every fold
boundary so the model cannot recognize the eye instead of the disease stage. Laser-scar rows
without a group mapping become singleton groups.

| Metric | Value |
| --- | --- |
| OOF macro-F1 | **0.7853** |
| OOF accuracy | 0.8317 |
| OOF precision (macro) | 0.7867 |
| OOF recall (macro) | 0.8028 |

Per-fold macro-F1: 0.7747 / 0.7881 / 0.7597 / 0.7834 / 0.8203.

Per-class OOF:

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Normal | 0.95 | 0.90 | 0.93 | 236 |
| Stage1 | 0.45 | 0.77 | 0.57 | 94 |
| Stage2 | 0.70 | 0.61 | 0.65 | 165 |
| Stage3 | 0.83 | 0.82 | 0.83 | 261 |
| Laser | 1.00 | 0.92 | 0.96 | 343 |

Modal run: `ap-E1aQngVFNxOurMW3ot7C8j`.

> Note: the grouped (honest) number (0.7853) is **slightly higher** than the stratified one
> (0.7802) here — leakage correction did not inflate the result, which strengthens confidence
> that ~0.78 is a real, generalizing estimate rather than a leakage artifact.

## Where we stand vs the 0.80 target

- **Honest 5-class macro-F1 = 0.7853**, ~0.015 short of 0.80, achieved **from scratch**.
- Easy classes are essentially solved: **Laser 0.95–0.96, Normal 0.92–0.93, Stage3 0.83–0.84**.
- The macro average is dragged down by the **Stage1 / Stage2 pair**:
  - **Stage1 F1 ≈ 0.54–0.57** with precision ≈ 0.41–0.45 but recall ≈ 0.77–0.78 — the model
    over-predicts Stage1, i.e. it confuses **Stage2 → Stage1** (adjacent severity).
  - **Stage2 F1 ≈ 0.65** with recall ≈ 0.59–0.61 — Stage2 examples leak into Stage1 (low) and
    Stage3 (high).
- This is the **classic adjacency confusion of an ordinal target** treated as nominal: the
  cross-entropy head has no notion that Stage1 < Stage2 < Stage3 and pays no extra penalty for
  predicting Stage3 when the truth is Stage1.

## Recommended next lever (highest ROI): ordinal-aware head

Treat severity as **ordinal** so adjacent-class errors are penalized less than far errors, and
the decision boundary respects Stage1 < Stage2 < Stage3.

- **CORN / CORAL ordinal head** (rank-consistent ordinal regression) on the 3 ROP stages only;
  Normal and Laser stay nominal (they are not on the severity axis). Practically: a shared
  trunk, an ordinal sub-head over {Stage1, Stage2, Stage3}, and a nominal head that gates
  Normal vs Laser vs "is-ROP".
- Expected effect: tighten the Stage1 precision leak (0.41–0.45) without sacrificing its recall,
  and recover Stage2 recall — the two terms that are holding the macro average under 0.80.
- This is the **single highest-ROI change** because every other class is already ≥ 0.83.

Secondary levers (only if ordinal head is not enough):
- Stage-boundary-focused softmap tuning (sharpen the demarcation-line channel for Stage1).
- Class-balanced sampling / minority oversampling on Stage1 (n=94).

## Reproduce

```bash
cd /home/ubuntu/rop-stage-classification
# stratified (benchmark-comparable)
python experiments/cnn/masked_cnn_cv_v2.py --mode full --protocol stratified --epochs 160
# grouped (leakage-corrected, honest)
python experiments/cnn/masked_cnn_cv_v2.py --mode full --protocol group --epochs 160
```

Logs: `experiments/v2_stratified.log`, `experiments/v2_group.log`.
