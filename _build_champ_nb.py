#!/usr/bin/env python3
"""Build an 11-code-cell champion notebook from masked_cnn_cv_v2.py.

Splits the source at blank-line boundaries so concatenating the code cells
reproduces the source byte-for-byte. Emits unexecuted nbformat-4 JSON.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "experiments/cnn/masked_cnn_cv_v2.py"
OUT = ROOT / "rop-stage-classification-champ.ipynb"

src = SRC.read_text()
lines = src.split("\n")  # source has trailing newline -> last element is ""

# (start, end) 1-indexed inclusive line ranges for each code segment.
# Boundaries chosen on blank lines so the segments tile the file with no gaps.
RANGES = [
    (1, 61),     # docstring + imports + modal app/image + class constants
    (63, 77),    # shared primitives: norm01, est_fov
    (79, 103),   # vessel softmap: gabor_resp, vessel_softmap
    (105, 131),  # ridge softmap: source channel, hessian, tophat, ridge_softmap
    (133, 171),  # build_channels + build_channels_rgb
    (173, 207),  # model: drop_path, BasicBlock, TinyResNetV2
    (209, 254),  # ModelEMA, soft_ce, mix_batch
    (256, 272),  # CacheDataset
    (274, 299),  # load_manifest, ensure_cache
    (301, 338),  # train_one_fold, _report
    (340, 437),  # run_full, quick_test, main entrypoint
]

# Markdown headers interleaved before selected code cells.
MARKDOWN = {
    0: "# ROP Stage Classification — Champion (Masked-TinyResNet v2)\n\n"
       "From-scratch 5-class ROP staging. Reverts the vessel channel to the "
       "plain-Gabor map (the Dice-champion fusion regressed staging F1) and adds "
       "MixUp/CutMix, label smoothing, stochastic depth, weight-EMA, and flip-TTA.\n\n"
       "Source: `experiments/cnn/masked_cnn_cv_v2.py`. Write-only — nothing runs "
       "until `main()` is given an explicit `--mode`.",
    1: "## Image primitives & preprocessing\n\n"
       "FOV estimation, percentile normalization, vessel/ridge soft-maps, and the "
       "3-channel input builders (softmap + RGB-ablation control).",
    5: "## Model, training & CV run\n\n"
       "TinyResNet v2 (from scratch), EMA + soft-CE + MixUp/CutMix, the per-fold "
       "trainer, and the Modal entrypoints (`run_full`, `quick_test`, `main`).",
}


def seg_text(start, end):
    # join inclusive 1-indexed range; no trailing newline inside a cell source
    return "\n".join(lines[start - 1:end])


def code_cell(text):
    # nbformat stores source as a list of lines, each (except last) ending in \n
    parts = text.split("\n")
    source = [p + "\n" for p in parts[:-1]] + [parts[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(text):
    parts = text.split("\n")
    source = [p + "\n" for p in parts[:-1]] + [parts[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


cells = []
for i, (s, e) in enumerate(RANGES):
    if i in MARKDOWN:
        cells.append(md_cell(MARKDOWN[i]))
    cells.append(code_cell(seg_text(s, e)))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))

n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f"WROTE {OUT}")
print(f"total_cells={len(cells)} code_cells={n_code} markdown_cells={n_md}")

# ---- self-verification: code cells must concatenate to the source ----
code_sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
rebuilt = "\n".join(code_sources)
# the source file has a trailing newline; our rebuilt has none -> add it back
expected = src[:-1] if src.endswith("\n") else src
assert rebuilt == expected, "FIDELITY FAIL: code cells do not reproduce source"
print("FIDELITY OK: code cells concatenate to exact source")
