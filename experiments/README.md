# Experiments Index

Scripts are grouped by function into three subfolders. Only the keeper scripts
are listed here; dead-end exploration was quarantined under the repo-root
`trashbin/` (git-ignored, kept on disk for reference).

```
experiments/
├── cnn/        Modal-hosted CNN classification (run from repo root)
├── vessel/     Vessel-segmentation pipeline + champion recipe + eval
└── classical/  Classical ML baselines (handcrafted features)
```

Research logs: [`vessel/VESSEL_FINDINGS.md`](./vessel/VESSEL_FINDINGS.md) (vessel
Dice log, locked eval protocol) and the project-root `results/RESULTS.md` (ROP
staging results).

## cnn/ — CNN classification (Modal)

| Script | Purpose |
| --- | --- |
| `masked_cnn_cv_v2.py` | Masked-TinyResNet, 5-fold CV under the classical baseline protocol. |
| `masked_cnn_cv_v2_ablation.py` | v2 ablation sweep (mask channel / augmentation variants). |
| `masked_cnn_cv_v3.py` | v3: wider model + revised augmentation/caching. |
| `clean_manifest.json` | Curated image manifest consumed by the CNN scripts. |
| `v2_group.log`, `v2_group_rgb.log`, `v2_stratified.log` | v2 run logs (group / group-RGB / stratified CV). |

> **Run location:** these use Modal with **relative** paths
> (`.add_local_dir("data/Zhao2024", ...)`, `.add_local_file("experiments/cnn/clean_manifest.json", ...)`),
> so they must be invoked **from the repo root**:
> `modal run experiments/cnn/masked_cnn_cv_v2.py`.

## vessel/ — vessel segmentation

| Script | Purpose |
| --- | --- |
| `vessel_pipeline.py` | Core vessel pipeline functions (shared import target). |
| `advanced_pipeline.py` | Advanced multi-stage segmentation (Gabor kernels, fusion helpers). |
| `vessel_eval.py` | Locked full-dataset eval harness for Agrawal2021 (100 pairs). |
| `vessel_round3.py` | Tuning round 3: Gabor soft-map (`tophat_pre`). |
| `vessel_round7.py` | Champion recipe: Gabor + Meijering fusion. |
| `champion_overlay.py` | Render the champion overlay from the round-7 recipe. |
| `ncc_groups.json` | Precomputed NCC grouping used by the eval split. |

> Self-contained: every script does `sys.path.insert(0, Path(__file__).parent)`
> for sibling imports and resolves output via `parents[2]/experiments/output`,
> so they run from any working directory.

## classical/ — classical ML baselines

| Script | Purpose |
| --- | --- |
| `rop_classical.py` | Handcrafted features + classical ML. The 0.5147 macro-F1 baseline. |
| `rop_ridge_classical.py` | Classical ridge / demarcation-line enhancement (no learned model). |

> Data resolved via `parents[2]/data/...`, so they run from any working directory.
