# ROP Stage Classification

This project studies Retinopathy of Prematurity (ROP) stage classification from retinal fundus images. The goal is to build and evaluate an image-analysis workflow that can classify images into ROP stages while also examining whether image enhancement and vessel-focused processing help the classification task.

The project is developed as a notebook-first research workflow in [`rop-stage-classification.ipynb`](rop-stage-classification.ipynb). Supporting proposal material and references are stored under [`docs/`](docs).

## What We Are Doing

ROP is a retinal disease affecting premature infants. Clinical staging depends on visual signs such as vascular changes, demarcation lines, ridges, and other retinal structures. This project focuses on a four-class image classification task:

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
.
|-- rop-stage-classification.ipynb        # Main project notebook
|-- overview.md                           # Short technical workflow note
|-- pyproject.toml                        # Python dependency metadata
|-- uv.lock                               # Locked dependency versions
|-- data/                                 # Local datasets
|-- docs/                                 # Proposal, references, and assignment material
`-- output/                               # Generated debug, preprocessing, training, and evaluation outputs
```

Large image files, model checkpoints, arrays, and archives are configured for Git LFS through [`.gitattributes`](.gitattributes).

## How To Run

Install dependencies with `uv`:

```bash
uv sync
```

Start Jupyter from the project root:

```bash
uv run jupyter notebook
```

Open:

```text
rop-stage-classification.ipynb
```

The notebook is designed to run locally and can also adapt to Kaggle-style paths.

## Outputs

The project writes generated artifacts under `output/`:

```text
output/
  00_debug_baseline/        # Vessel-processing debug visualizations
  01_preprocessing_cache/   # Cached scenario images and dataset split
  02_training_runs/         # Model checkpoints
  03_evaluation/            # Metrics, reports, and workflow summaries
  04_figures/               # Plots for reports and presentations
```

These outputs support reproducibility and reporting, but the main source of the project workflow is the notebook.

## References

The project uses papers stored in [`docs/references`](docs/references), including:

- Zhao2024 for the main ROP classification dataset.
- Rahim2024 for ROP image enhancement ideas.
- Almeida2024 for vessel enhancement and segmentation methods.
- Agrawal2021 for vessel segmentation reference data.
- Vahidmoghadam2026 for combined image and vessel-mask scenario inspiration.

## Scope

This project is for research and coursework. It is not a clinical diagnostic system, and its results should not be used for medical decision-making.
