# Vessel Segmentation Findings — ROP Project

Persistent research log. Goal: maximize mean Dice for retinal vessel segmentation
on the Agrawal2021 dataset (100 image/mask pairs: 50 RetCam + 50 Neo).

**Evaluation protocol (locked, use this always):**
- Dataset: ALL 100 pairs via `experiments/vessel_eval.py::load_dataset()`
- Split: deterministic stratified-by-source 60/40 train/test, seed=42
- Metric: mean per-image Dice. Tune on TRAIN, report TEST as the honest number.
- NEVER tune+report on the same small subset (the old 8-image scores overfit badly).

---

## Baselines (mean Dice)

| Config | 8-img (old, optimistic) | ALL 100 | TRAIN 60 | TEST 40 |
|---|---|---|---|---|
| `overlay_results` (Gabor + 2×CLAHE + P0.16 + top2) | 0.4662 | **0.4501** | 0.4522 | 0.4469 |
| `overlay_best` (8-img tuned: clahe2=9, top3, close5) | 0.4749 | 0.4423 | 0.4453 | 0.4377 |

**KEY LESSON:** `overlay_best` *beat* `overlay_results` on the 8 cherry-picked images
(+0.009) but is *worse* on the full set (−0.008). 8-image tuning overfit. The real
target to beat is **TEST Dice = 0.4469** (overlay_results), or ALL = 0.4501.

### `overlay_results` reference config (current champion)
- channel: green
- CLAHE #1: clip 6.0, tile 16×16
- invert → normalize → Gabor filter response (`advanced_pipeline.gabor_filter_response`)
- median blur 7 → renormalize
- CLAHE #2 (sharpen): clip 12.0, tile 12×12
- threshold: percentile, target_density 0.16
- cleanup: keep 2 largest connected components

---

## Research notes (techniques to try)

From arXiv search (retinal vessel segmentation, classical/unsupervised):
1. **LS-CF (local-sensitive connectivity filter)** — post-processing that fills
   Frangi/Hessian discontinuities via pixel-level continuity + local tolerance.
   Beats morphological closing. (Rodrigues et al.) → try as post-proc on our soft map.
2. **Multiscale 2D Gabor wavelets** (Kumar et al.) — top-hat + CLAHE preproc, then
   Gabor at multiple scales/orientations for thick AND thin vessels. ~94% acc on DRIVE.
   → our Gabor is single-config; try multi-scale bank.
3. **Gabor + entropic threshold** (Waly et al.) — entropic thresholding gives fewer FP
   than percentile. → try entropy threshold vs percentile.
4. **Kernel matched filter + Otsu** (Saroj et al.) — well-matched MF kernel + Otsu.
5. **Frangi-Net** — trainable Frangi; +17% F1. (deep, skip for classical track)

---

## Experiments log

(append results below as variants are tested)

### Round 1: soft-builder × threshold × cleanup (vessel_variants_fast.py)
Protocol: precompute soft maps once, sweep threshold+cleanup, tune TRAIN(60)/report TEST(40).

**Best validated config — BEATS BASELINE:**
- `gabor_c12` soft map (= overlay_results soft: Gabor + median7 + CLAHE2 clip12 tile12)
- threshold: percentile P0.16
- cleanup: keep top-3 components + morphological close 3×3
- **TEST Dice = 0.4505** (TRAIN 0.4508) vs baseline TEST 0.4469 → **+0.0036**, no overfit (train≈test)

What worked / didn't:
- top-3 components > top-2 (baseline used top-2). Small consistent gain.
- morphological close 3×3: tiny positive.
- minarea filter: neutral (no effect with top-k already applied).
- **Fusion (Gabor+Frangi+Jerman+matched): WORSE** than Gabor-only (~0.41 train). Frangi/Jerman dilute the Gabor response.
- **LS-CF connectivity filling: no improvement**, slow. Gabor map is already fairly connected.
- **Otsu/Triangle thresholds: worse** than percentile P0.16 (percentile controls density directly).
- clahe2 clip 12 > clip 9 on full set (the clip-9 "overlay_best" was an 8-img overfit artifact).

Gain is small. Threshold/cleanup are plateaued; the lever is the SOFT MAP itself. Next: tune the Gabor bank + preprocessing.

### Round 2: Gabor bank + preprocessing (vessel_gabor_tune.py)
Fixed downstream: P0.16 + top-3 + close 3×3. Tune TRAIN(60)/report TEST(40).

**NEW BEST — BIG WIN:**
- `tophat_pre`: blend modified top-hat with inverted-green BEFORE the Gabor bank
  (inv_f = 0.5*normalize(inv) + 0.5*tophat), then standard Gabor bank + median7 + CLAHE2.
- **TEST Dice = 0.4600** (TRAIN 0.4666) vs baseline 0.4469 → **+0.0131**, train≈test (generalizes).

What worked / didn't (TRAIN dice):
- **top-hat preprocessing: 0.4666** (best). Enhances dark line structures before Gabor. KEY.
- thick_scales (bigger sigmas): 0.4513, ~neutral.
- more_scales (6 scales): 0.4511, ~neutral (more compute, no gain).
- finer_angles (10°): 0.4507, neutral. coarse (30°): 0.4382, worse.
- **thin_scales (small sigmas): 0.389, much WORSE** — over-detects noise/capillaries.
- median5: 0.439 < median7. no_median: 0.420 (median7 helps denoise Gabor).
- tophat_thin: 0.409 — top-hat helps only with the standard scale set, not thin.

Lever ranking: preprocessing (top-hat) >> scales/angles/threshold/cleanup. Next: tune top-hat blend ratio.
