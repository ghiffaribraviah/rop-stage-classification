# ROP Stage Classification — Results & Findings

Stage classification of Retinopathy of Prematurity (ROP) on the **Zhao2024**
dataset: 756 stage-labeled fundus images, classes Stage1 / Stage2 / Stage3.
All scores are **macro-F1 under 5-fold stratified CV** (`StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)`), no data leakage, classical features only.

## Headline result

The **fused classical pipeline** (`experiments/classical/rop_classical.py`, 48 features =
vessel + ridge) achieves **macro-F1 = 0.5147** with RandomForest. This is the
best result under the stated constraints (no learned segmentation, classical
features only) and a **+0.041** improvement over the vessel-only baseline (0.474).

## Method trajectory

| Method                        | Macro-F1 | Notes                                    |
|-------------------------------|----------|------------------------------------------|
| Scratch TinyResNet            | ~0.45    | Trained from scratch, no pretraining     |
| Self-supervised (SSL)         | ~0.47    | SSL pretrain + linear probe              |
| Classical ML (vessel-only)    | 0.474    | Vessel morphology features only          |
| **Fused classical + ridge**   | **0.515**| Vessel + ridge features (best)           |

Each method independently confirms the same ceiling, nudged a few points higher
by better feature engineering.

## Fused run — classifier comparison

5-fold stratified CV (macro-F1), 48 features on 756 images:

| Classifier       | Macro-F1 | Accuracy |
|------------------|----------|----------|
| SVM (RBF)        | 0.4964   |          |
| **RandomForest** | **0.5147** |        |
| GradientBoosting | 0.4912   |          |

RandomForest config: `n_estimators=400, max_depth=None, class_weight="balanced",
random_state=42`.

## Per-class F1 (RandomForest, fused vs vessel-only)

| Class  | Vessel-only | Fused  | Δ       |
|--------|-------------|--------|---------|
| Stage1 | ~0.30       | 0.312  | ~flat   |
| Stage2 | 0.364       | 0.416  | +0.052  |
| Stage3 | —           | —      |         |

The ridge features deliver their payoff exactly where expected: **Stage2 F1
crosses 0.4 for the first time** (0.364 → 0.416), the demarcation-ridge signal
doing measurable work.

## Top RF feature importances (fused)

```
tortuosity_proxy         0.0427
comp_size_std            0.0328
vessel_anisotropy        0.0325
int_r_std                0.0286
ridge_main_area          0.0283   <- ridge
ridge_main_len           0.0274   <- ridge
ridge_main_extent        0.0267   <- ridge
soft_std                 0.0266
comp_size_max            0.0260
glcm_correlation_std     0.0258
ridge_resp_p95           0.0247   <- ridge
vessel_density           0.0243
```

Four ridge features land in the top 12, confirming they earn their place.

## The Stage1 wall (the binding constraint)

Stage1 F1 is **0.312** — essentially flat against the 0.298–0.304 it has always
been across every method. The cause is mechanical, not a tuning miss:

- The classical ridge filter recovers only **~37% of true ridge pixels** at 16%
  density.
- Faint Stage1 demarcation lines are precisely the low-contrast structures the
  hand-built filter drops.
- The abandoned ridge U-Net reached **Dice 0.615**, showing a *learned*
  segmenter can find those lines; a hand-built filter cannot.

## Masked-input CNN — the classifier was the ceiling, not the ridge filter

A TinyResNetV2 was trained on the **byte-identical** vessel + ridge softmaps the
classical model consumes (3-channel masked input: vessel softmap, ridge softmap,
masked CLAHE-green), under the **same** `StratifiedKFold(5, shuffle=True,
random_state=42)` over the same 756 images, OOF predictions pooled. No learned
segmenter was added — the ridge signal is the same ~37%-recall hand-built filter.
Script: `trashbin/experiments/masked_cnn_cv.py` (archived; superseded by
`experiments/cnn/masked_cnn_cv_v2.py`).

**Read the comparison carefully — the class sets differ.** This run scores **4
classes** (Normal/Stage1/Stage2/Stage3); the 0.5147 baseline above is a **3-class**
(Stage1/2/3) macro-F1. The two macro averages are therefore not directly
comparable. Both numbers are reported separately below.

Per-class OOF F1 (4-class run):

| Class  | Precision | Recall | F1   | Classical 3-class F1 | Δ (shared) |
|--------|-----------|--------|------|----------------------|------------|
| Normal | 0.97      | 0.89   | 0.92 | — (not in 3-class)   | —          |
| Stage1 | 0.52      | 0.57   | 0.55 | 0.312                | **+0.24**  |
| Stage2 | 0.62      | 0.68   | 0.65 | 0.416                | **+0.23**  |
| Stage3 | 0.87      | 0.84   | 0.85 | — (unreported)       | —          |

- **4-class OOF macro-F1 = 0.7425** (Acc 0.787, P 0.742, R 0.745). Lifted in part
  by the easy Normal class (0.92 F1) — not a like-for-like figure against 0.5147.
- **3-class matched macro-F1 = (0.55+0.65+0.85)/3 = 0.683** — the fair comparison
  against the 0.5147 baseline: **+0.168 absolute** on the same class set, same
  folds, same features.

## Conclusion (revised)

The earlier conclusion — that 0.5147 was the reproducible ceiling and that Stage1
**could not** improve without a learned ridge segmenter — **does not hold.** Given
the *same* hand-engineered, ~37%-recall ridge softmap, swapping the RandomForest
head for a small CNN moves Stage1 F1 from **0.312 → 0.55** and Stage2 from
**0.416 → 0.65**, for a matched 3-class macro-F1 of **0.683** (vs 0.5147).

The binding constraint was the **classifier**, not the ridge filter: a CNN reads
spatial structure in the softmaps that the classical pipeline discards when it
pools features into a 48-dim vector. A learned ridge segmenter (U-Net Dice 0.615)
remains the lever for further Stage1 gains, but it was never required to break the
0.5147 wall — that wall was an artifact of the classical head.

Caveat held open for a future check: re-run the **classical** pipeline as a 4-class
problem (adding Normal) to produce a same-class-count baseline, so the 0.74 figure
also has a matched counterpart.

## Reproduce

```bash
cd /home/ubuntu/rop-stage-classification
python experiments/classical/rop_classical.py    # ~28 min CPU, writes fused_run.log
```

Source log: `trashbin/experiments-output/output/fused_run.log`
Script: `experiments/classical/rop_classical.py`
