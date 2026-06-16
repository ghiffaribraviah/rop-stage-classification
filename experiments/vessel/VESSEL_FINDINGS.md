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

### Round 3: top-hat blend ratio + kernel tuning (vessel_round3.py)
Fixed downstream: P0.16 + top-3 + close 3×3. Swept the top-hat blend weight and structuring-element size.

**Result: PLATEAU — no improvement over Round 2.**
- best TEST Dice = 0.4599 (≈ Round 2's 0.4600, within noise).
- blend weight 0.5 is robust: 0.4/0.5/0.6 all land within ±0.001. The blend is insensitive — good for stability, but no lever here.
- top-hat kernel size: neutral across tested radii.

Conclusion: preprocessing is now saturated. Threshold, cleanup, AND preprocessing are all plateaued at TEST≈0.460. The remaining lever is the SOFT-MAP BUILDER itself (the detector), not its inputs/outputs.

### Round 4: change the soft-map builder — Hessian ridge fusion (vessel_round4.py)
New lever: multiscale Hessian ridge filters (Frangi / Sato / Meijering) on CLAHE-enhanced
inverted green. Tubularness via local Hessian eigenstructure — fundamentally different from
oriented Gabor energy. Tested (a) pure ridge maps and (b) ridge BLENDED with the round-2 Gabor+tophat soft.

**NEW BEST — BIG WIN:**
- `blend_sato_50`: 0.5 × Sato ridge + 0.5 × gabor_tophat soft, then P0.16 + top-3 + close 3×3.
- **TEST Dice = 0.4655** (vs Round 2's 0.4600) → **+0.0055**, generalizes (train≈test).

What worked / didn't:
- **Sato + Gabor blend beats Frangi + Gabor.** Sato's neuriteness normalization complements Gabor better than Frangi's vesselness.
- Pure ridge filters alone (no Gabor): worse than the blend — Gabor still carries the thick-vessel signal.
- 50/50 blend is the sweet spot among coarse sweeps.

### Round 5: refine the Sato+Gabor blend (vessel_round5.py)
Levers at locked downstream: (a) Sato sigma-range (finer/denser/extended), (b) beta fine grid 0.40–0.60, (c) FOV percentile clip-normalize on the heavy-tailed Sato map before blending.

**Small win:**
- `fine_050`: 0.5 × Sato(fine sigmas) + 0.5 × gabor_tophat → **TEST Dice = 0.4667** (+0.0012 over Round 4).
- Finer Sato sigmas marginally help; beta stays at 0.50; percentile clip ~neutral.

### Round 6: 3-way detector fusion (vessel_round6.py)
Add Meijering (neuriteness, different Hessian eigenvalue normalization than Sato) as a 3rd channel:
blend = wg·gabor_tophat + ws·sato_fine + wm·meijering_fine (weights sum to 1).

**Finding: Meijering REPLACES Sato — it does not add to it.**
- Best 3-way configs collapse toward Gabor+Meijering (ws→0). Sato and Meijering are redundant; Meijering is the stronger partner for Gabor.
- This motivated Round 7: drop Sato, tune a clean Gabor + Meijering 2-way fusion.

### Round 7: Gabor + Meijering 2-way fusion — CHAMPION (vessel_round7.py)
blend = wg·gabor_tophat + wm·meijering(sigma_set). Swept weights and Meijering sigma sets
(fine / thin / thick). Locked downstream: P0.16 (FOV-eroded) + top-3 CC + close 3×3.

**NEW CHAMPION — overall best:**
- `g40_m60_fine`: **0.40 × Gabor + 0.60 × Meijering(fine)** → **TEST Dice = 0.4739**
  (+0.0072 over Round 5, **+6.0% over the 0.4469 baseline**). TRAIN≈TEST, clean generalization.
- `g45_m55_thin` ties at **0.4739** (0.45 Gabor + 0.55 Meijering, thin sigmas) — also generalizes cleanly.
- Meijering-dominant blends (wm≥0.55) win: neuriteness recovers thin vessels Gabor misses, Gabor anchors the thick trunks.

**Champion recipe (locked):**
1. Green channel → CLAHE clip 6.0, tile 16×16
2. invert → normalize → Gabor filter bank (sigmas=[1.5,2.5,3.5,5.0], lambdas=[3,5,7,10], 15° angles), with top-hat blend preproc
3. median blur 7 → renormalize → CLAHE clip 12.0, tile 12×12  → gabor_tophat soft
4. Meijering neuriteness, fine sigmas=(0.8,1.4,2.0,2.8,3.6,4.5)
5. fuse: 0.40·gabor_tophat + 0.60·meijering
6. threshold: percentile P0.16 with FOV erosion
7. cleanup: keep top-3 CC + morphological close 3×3

### Round 8: push past the champion (vessel_round8.py) — DEAD END
Attempted further gains beyond 0.4739 (extra detector channels / weight & sigma micro-tuning).

**Result: REGRESSION. No config beat Round 7.** Round 8 reverted (uncommitted edit discarded).
Research has converged — **Round 7 `g40_m60_fine` (TEST 0.4739) is the validated peak.**

---

## Final summary

| Round | Lever | Best TEST Dice | Δ vs baseline |
|---|---|---|---|
| baseline | overlay_results (Gabor + 2×CLAHE + P0.16 + top2) | 0.4469 | — |
| 1 | threshold / cleanup | 0.4505 | +0.0036 |
| 2 | top-hat preprocessing before Gabor | 0.4600 | +0.0131 |
| 3 | top-hat blend ratio | 0.4599 | (plateau) |
| 4 | Sato + Gabor fusion | 0.4655 | +0.0186 |
| 5 | fine Sato sigmas, beta | 0.4667 | +0.0198 |
| 6 | 3-way (Meijering replaces Sato) | — | (redundancy found) |
| **7** | **Gabor + Meijering fusion** | **0.4739** | **+0.0270 (+6.0%)** |
| 8 | push beyond | (regression) | dead end |

**CHAMPION: `vessel_round7.py` → `g40_m60_fine` (also `g45_m55_thin`), TEST Dice = 0.4739.**
Lever ranking overall: detector fusion (Gabor+Meijering) > top-hat preprocessing > Sato blend > threshold/cleanup.
