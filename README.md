# ROP Stage Classification

Retinopathy of Prematurity (ROP) stage classification on retinal fundus images,
combining a classical-ML baseline with a from-scratch deep-learning champion. The
project also studies whether vessel- and ridge-focused preprocessing improves
staging performance.

## Quick Start

**Data:** Zhao2024 (3 ROP stages) + Agrawal2021 (5 ROP stages, with vessel masks).

**Champion result:** **0.7853 macro-F1** — 5-fold group-aware cross-validation.
- Model: Masked CNN (TinyResNetV2) trained from scratch.
- Preprocessing: vessel + ridge softmaps (Gabor + Meijering fusion).
- Classical baseline: **0.5147 macro-F1** (48 handcrafted features + classical ML).

The full source of truth for results is [`results/FINAL_RESULT.md`](results/FINAL_RESULT.md).

## Classes

- `Normal`
- `Stage1`
- `Stage2`
- `Stage3`

The main question is not only whether a model can classify the stages, but also how different image-processing choices affect the result. The project compares several ways of presenting the same retinal image to a classifier:

- the original image,
- an enhanced image,
- a vessel mask,
- and an enhanced image guided by vessel information.

This makes the project a complete image-analysis pipeline rather than only a model-training exercise.

## Project Process

The work is organized into these phases:

1. **Problem definition**

   Define the target task as ROP stage classification and limit the main classes to Normal, Stage 1, Stage 2, and Stage 3. The `laser scars` class is excluded because it describes post-treatment appearance, not a primary ROP stage.

2. **Dataset preparation**

   Use the Zhao2024 dataset as the main classification dataset. Images are grouped by class, checked locally, and split into train, validation, and test sets with a fixed random seed.

3. **Image preprocessing**

   Resize images to a consistent input size and prepare the same train/validation/test split for every experiment. This keeps comparisons fair across all scenarios.

4. **Image enhancement**

   Apply conservative retinal image enhancement inspired by prior ROP image-restoration work. The purpose is to test whether improving contrast and illumination helps reveal stage-related structures.

5. **Vessel-focused processing**

   Use a classical vessel segmentation pipeline to produce vessel masks and vessel-guided images. This tests whether vascular structure alone, or vascular emphasis, contributes useful information for ROP staging.

6. **Classification experiments**

   Train the same CNN architecture on each input scenario so that performance differences are mainly caused by the input representation, not by a different model.

7. **Evaluation and analysis**

   Compare the scenarios using classification metrics, confusion matrices, and visual inspection. The analysis focuses on which preprocessing choices help, which do not, and what limitations remain.

8. **Reporting**

   Summarize the project workflow, assumptions, results, and limitations for the final report or presentation.

## Experimental Scenarios

The project compares five scenarios:

| Scenario | Input | Purpose |
|---|---|---|
| `S1_raw` | Original resized RGB image | Baseline stage classification |
| `S2_enhanced` | Enhanced RGB image | Tests whether contrast and illumination correction improve classification |
| `S3_vessel_mask` | Vessel mask only | Tests whether vessel structure alone is useful for staging |
| `S4_vessel_guided` | Enhanced image guided by vessel information | Tests whether emphasizing vessels helps while preserving retinal context |
| `S5_masked_cnn_input` | Three-channel masked-CNN input: P10 soft response, P10 final mask, masked enhanced-green channel | Optional copied masked TinyResNetV2 experiment while preserving this repo's preprocessing |

All scenarios use the same dataset split.

## Data Used

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
    HVDROPDB-RIDGE/
```

`Zhao2024` is used for the main ROP stage classification task.

`Agrawal2021` is used as supporting data for vessel segmentation checks because it includes vessel-related masks.

The current split is image-level because patient or eye identifiers are not available in the local filenames. This is an important limitation: reported performance should be interpreted as image-level performance, not fully patient-independent generalization.

## Repository Structure

```text
rop-stage-classification/
├── README.md                       # This file
├── overview.md                     # High-level technical workflow note
├── rop-stage-classification.ipynb  # Original notebook-first exploration
├── pyproject.toml / uv.lock        # uv-managed dependencies
│
├── data/                           # Local datasets (Git LFS)
│   ├── Zhao2024/                   # 3 ROP stages
│   └── Agrawal2021/                # 5 ROP stages + vessel masks
│
├── experiments/                    # Keeper scripts (see experiments/README.md)
│   ├── cnn/                        # Masked-CNN classification (Modal)
│   ├── vessel/                     # Vessel-segmentation pipeline + champion recipe
│   └── classical/                  # Classical ML baselines
│
├── results/                        # Result write-ups (source of truth)
│   ├── FINAL_RESULT.md             # End-to-end narrative, raw RGB → champion
│   ├── RESULTS.md                  # ROP staging results & findings
│   ├── CHAMPION_RESULTS.md         # Masked-CNN champion head-to-head
│   ├── V2_RESULTS.md               # Masked-CNN v2 (5-class, toward 0.80)
│   └── V3_RESULTS.md               # Masked-CNN v3 (ordinal-aware)
│
├── scripts/                        # Standalone visualization / diagnostic scripts
├── docs/                           # Proposal, references, assignment material, LaTeX template
└── trashbin/                       # Quarantined dead-end work (git-ignored, kept on disk)
```

Large image files, model checkpoints, arrays, and archives are tracked with Git
LFS through [`.gitattributes`](.gitattributes).

## Datasets

```text
data/
  Zhao2024/        # main 3-stage classification dataset
  Agrawal2021/     # 5-stage staging + vessel masks (HVDROPDB-BV, HVDROPDB-RIDGE)
```

- **Zhao2024** — primary stage-classification dataset.
- **Agrawal2021** — 5-stage staging plus vessel/ridge masks, used both for the
  champion CNN and for vessel-segmentation evaluation.

The cross-validation is **group-aware** to avoid leaking related images across
folds. Reported performance should still be read as research-grade, not as a
validated clinical generalization estimate.

## Method Overview

1. **Classical baseline** — 48 handcrafted features (color, texture, vessel,
   ridge descriptors) fed to classical ML. Establishes the 0.5147 macro-F1 floor.
2. **Vessel segmentation** — a classical pipeline producing vesselness softmaps,
   tuned over multiple rounds to a champion recipe (Gabor + Meijering fusion).
   See [`experiments/vessel/VESSEL_FINDINGS.md`](experiments/vessel/VESSEL_FINDINGS.md).
3. **Masked CNN** — a small from-scratch residual CNN (TinyResNetV2) that consumes
   the RGB image plus vessel/ridge softmap channels, cross-validated 5-fold under
   the classical baseline protocol. Reaches the 0.7853 macro-F1 champion result.

## How To Run

Install dependencies with `uv`:

```bash
uv sync
```

### CNN classification (Modal)

The CNN scripts use Modal with **relative** paths, so they must be invoked **from
the repo root**:

```bash
modal run experiments/cnn/masked_cnn_cv_v2.py
```

### Vessel & classical scripts

Vessel and classical scripts are self-contained (they resolve their own paths)
and run from any working directory:

```bash
uv run python experiments/vessel/vessel_round7.py
uv run python experiments/classical/rop_classical.py
```

See [`experiments/README.md`](experiments/README.md) for the per-script index.

### Notebook

The original exploration lives in
[`rop-stage-classification.ipynb`](rop-stage-classification.ipynb):

```bash
uv run jupyter notebook
```

## References

Papers are stored under [`docs/references`](docs/references), including:

- **Zhao2024** — main ROP classification dataset.
- **Agrawal2021** — vessel/ridge segmentation reference data and 5-stage staging.
- **Almeida2024** — vessel enhancement and segmentation methods.
- **Rahim2024** — ROP image enhancement ideas.
- **Vahidmoghadam2026** — combined image + vessel-mask scenario inspiration.

## Scope

This project is for research and coursework. It is not a clinical diagnostic
system, and its results must not be used for medical decision-making.
