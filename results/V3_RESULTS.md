# Masked-CNN v3 — ordinal-aware staging (ONSCE + Stage1 bias sweep)

Script: [`experiments/cnn/masked_cnn_cv_v3.py`](../experiments/cnn/masked_cnn_cv_v3.py).
Constraint held: **no pretraining** (no ImageNet, no external weights).
Modal run: `ap-ENQn00oBfzxEO3l5h9r20X` (A100-80GB, 5-fold, 160 epochs).

## Goal

v2 plateaued at **0.7853** macro-F1, dragged down by the **Stage1 / Stage2 pair**:
Stage1 F1 ≈ 0.57 with **precision 0.45 / recall 0.77** — the model over-predicts Stage1,
confusing adjacent severity (Stage2 → Stage1). The cross-entropy head has no notion that
Stage1 < Stage2 < Stage3. v3 adds ordinal structure to tighten the Stage1 precision leak
**without** sacrificing its recall.

## What v3 changed vs v2

| Component | v2 | v3 |
| --- | --- | --- |
| Soft targets | hard scatter | **ONSCE** ordinal-neighbor smoothing (eps_ord=0.15) |
| Class weights | inv-freq | **capped sqrt(inv-freq)**, cap=4.0, renorm mean 1 |
| Rank-aux loss | — | optional cumulative-rank head (disabled this run) |
| Stage1 calibration | — | post-hoc **Stage1 logit-bias sweep** on pooled OOF |
| Reporting | raw | raw **+ swept** |

## Results — 5-fold OOF, group-aware, 160 epochs

Per-fold raw macro-F1: **0.7780 / 0.7731 / 0.7645 / 0.7964 / 0.8089** (mean 0.7842, std ≈ 0.016).

| Metric | v3 raw | v3 swept (bias=−0.30) | v2 champion |
| --- | --- | --- | --- |
| **Macro-F1** | 0.7851 | **0.7890** | **0.7853** |
| Accuracy | 0.8417 | 0.8499 | — |
| Precision (macro) | 0.7860 | 0.7906 | 0.7867 |
| Recall (macro) | 0.7925 | 0.7910 | — |

**Aggregate Δ vs v2: +0.0037.** This is inside the fold-noise band (±0.02–0.04). **Statistical tie, not a win.** Target 0.80 not reached.

## Per-class — and why v3 failed on its own terms

The Stage1 sweep lifted aggregate by only +0.0039 (base 0.7851 → swept 0.7890), far below
the +0.031 the single smoke fold suggested — expected regression to reality on pooled OOF.

| Stage1 | v2 | v3 swept | Δ |
| --- | --- | --- | --- |
| Precision | 0.45 | 0.49 | +0.04 |
| Recall | 0.77 | **0.52** | **−0.25** |
| **F1** | **0.57** | **0.51** | **−0.06 (worse)** |

v3 **traded Stage1 recall for a little precision and ended with a lower Stage1 F1.** The ordinal
machinery moved the error around rather than fixing it. Aggregate held even only because the
easy classes (Laser 0.99, Normal 0.94, Stage3 0.86) carry the mean.

v3 swept per-class: Normal P0.99/R0.89/F0.94 · Stage1 P0.49/R0.52/F0.51 · Stage2 P0.60/R0.72/F0.66 · Stage3 P0.88/R0.84/F0.86 · Laser P0.99/R0.98/F0.99.

## Verdict

**Keep v2 (0.7853) as champion.** v3's ordinal hypothesis — that neighbor-smoothing + capped
weights + a Stage1 bias would tighten the Stage1 leak — **failed**: the aggregate is a
noise-level tie and the targeted Stage1 F1 regressed. ONSCE + rank-aux + sweep added real
complexity for no payoff. Retained as a documented dead end, same as the Dice-champion fusion.

## Why the Stage1 problem persists

Stage1↔Stage2 is an **adjacent-severity** confusion on the smallest class (n=94). Global ordinal
smoothing is too blunt: it softens *all* boundaries, including ones that were already fine
(Laser, Normal), while the actual error is concentrated in one class pair. The next lever has to
be **targeted at Stage1/Stage2**, not applied globally.

## Follow-up #1 — Stage1 batch oversampling (WeightedRandomSampler ×3): DEAD END

Hypothesis: feed more Stage1 per batch (sampler, not loss) to tighten its boundary without
the loss-distortion that hurt v3. Run `ap-33fo5NFCEJfiWsgfmEJm7k`, group, 160ep, cap 4.0, ×3.

| Fold | v3 baseline | +oversample ×3 | Δ |
| --- | --- | --- | --- |
| 0 | 0.7780 | 0.7165 | −0.062 |
| 1 | 0.7731 | 0.6842 | −0.089 |

Killed after 2 folds (both collapsing, trend worsening — on track for ~0.70). Oversampling the
same 94 Stage1 images per batch **amplified** v2's existing over-prediction pathology (Stage1
recall was already 0.77; the problem was precision 0.45). Stacking the sampler on top of capped
class weights pushed the same direction — too far. **Confirms the broader lesson: every lever
that "shouts Stage1 louder" trades precision for recall and tanks the macro average.** The fix
must reduce Stage1 false positives, not increase its exposure.

## Definitive 2×2 ablation — what actually helps? (the fair comparison)

v3's swept 0.7890 was compared against v2's *raw* 0.7853 — unfair, because the Stage1 sweep is a
**post-hoc, model-agnostic** step (add a scalar bias to the Stage1 logit on pooled OOF) that v2
never received. To settle it, a copy of v2 (`masked_cnn_cv_v2_ablation.py`, app
`ap-G4dCncVPlEAVY8lJJU8tur`) was run as a clean 2×2: ordinal {off, on} × sweep {off, on}, same
5-fold group-aware OOF protocol, 160ep. The ablation `train_one_fold` returns logits so the
sweep can be applied to any config.

| Config | Raw macro-F1 | Swept macro-F1 |
| --- | --- | --- |
| **v2 base** | **0.7927** | **0.8018** |
| v2 + ordinal (ONSCE eps=0.15) | 0.7372 | 0.7767 |
| v3 (ordinal, separate run) | 0.7851 | 0.7890 |

v2 base per-fold raw: 0.7848 / 0.8068 / 0.7662 / 0.7665 / 0.8388 (this run landed slightly above
the documented 0.7853 — normal run-to-run variance).

### Conclusions

1. **v3's apparent edge was entirely the sweep, not the ordinal training.** v2 base swept
   (0.8018) beats v3 swept (0.7890) by +0.013, and v2 base beats v3 on *raw* too (0.7927 vs
   0.7851). Give plain v2 the same post-hoc sweep and it is unambiguously better.

2. **Ordinal smoothing (ONSCE) actively hurts.** Added to the v2 base it dropped raw by −0.055
   (0.7927→0.7372) and swept by −0.025 (0.8018→0.7767). Now confirmed on two independent bases
   (standalone v3, and v2+ordinal here). Documented dead end.

3. **Best honest result: plain v2 + post-hoc Stage1 sweep.**
   - **Clean CV headline: raw 0.7927.**
   - **With post-hoc Stage1 calibration: 0.8018** — crosses the 0.80 target.

### Caveat on the swept number (report honestly)

The swept 0.8018 selects the Stage1 bias (−1.40) on the *same* pooled OOF it is scored on, so it
carries a mild in-sample optimism. It is a legitimate, commonly-reported calibration technique,
but it must be labeled **"with post-hoc Stage1 calibration"** — not presented as a clean
cross-validated number. The defensible headline remains the raw **0.7927**.


