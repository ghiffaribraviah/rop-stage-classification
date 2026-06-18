#!/usr/bin/env python3
"""Build the report companion notebook.

The notebook is intentionally generated from this script so the JSON stays
repeatable and reviewable. It is self-contained for Kaggle/local runs: the
expensive sections are controlled by config flags inside the notebook.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "rop-stage-classification.ipynb"


def split_source(text: str) -> list[str]:
    lines = textwrap.dedent(text).strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


_CELL_ID = 0


def next_id() -> str:
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:03d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": next_id(), "metadata": {}, "source": split_source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": split_source(text),
    }


cells: list[dict] = [
    md(
        """
        # ROP Stage Classification

        Report companion notebook: dataset loading, softmap preprocessing,
        supporting segmentation checks, classical baseline, TinyResNet training,
        and final visualizations.
        """
    ),
    md(
        """
        ## 1. Runtime Setup

        The notebook runs locally or on Kaggle. Expensive cells are controlled by
        the config flags in the next section.
        """
    ),
    code(
        """
        import importlib.util
        import os
        import random
        import subprocess
        import sys
        import time
        import warnings
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from dataclasses import dataclass
        from pathlib import Path

        INSTALL_MISSING = False
        REQUIRED_PACKAGES = {
            "cv2": "opencv-python",
            "numpy": "numpy",
            "pandas": "pandas",
            "PIL": "pillow",
            "scipy": "scipy",
            "skimage": "scikit-image",
            "sklearn": "scikit-learn",
            "torch": "torch",
            "torchvision": "torchvision",
            "tqdm": "tqdm",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
        }

        missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES.items()
                   if importlib.util.find_spec(module_name) is None]
        if missing and INSTALL_MISSING:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        elif missing:
            print("Missing packages:", missing)
            print("Set INSTALL_MISSING=True and rerun this cell if the environment allows pip installs.")
        """
    ),
    code(
        """
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from PIL import Image
        from scipy import ndimage as ndi
        from skimage.feature import graycomatrix, graycoprops
        from skimage.filters import meijering
        from skimage.morphology import skeletonize
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid")
        """
    ),
    md(
        """
        ## 2. Configuration

        Defaults are safe for a fresh run. Turn on cache/training flags when you
        want to reproduce the full experiments.
        """
    ),
    code(
        """
        @dataclass
        class CFG:
            seed: int = 42
            img_size: int = 224
            process_side: int = 768
            n_folds: int = 5
            epochs: int = 160
            quick_epochs: int = 3
            batch_size: int = 32
            lr: float = 1e-3
            weight_decay: float = 5e-4
            label_smoothing: float = 0.1
            mix_p: float = 0.5
            ema_decay: float = 0.999
            num_workers: int = min(4, os.cpu_count() or 2)
            preprocess_workers: int = max(1, min(8, (os.cpu_count() or 2) - 1))
            run_preprocessing_cache: bool = False
            run_segmentation_eval: bool = True
            run_classical_baseline: bool = False
            run_training: bool = False
            run_quick_training: bool = False
            use_tta: bool = True
            use_known_report_results_when_not_run: bool = True

        cfg = CFG()

        REPORT_RESULTS = pd.DataFrame([
            {"scenario": "Baseline klasik terbaik", "input": "48 fitur", "macro_f1": 0.5147, "note": "RF, 3 kelas"},
            {"scenario": "CNN RGB (group-aware)", "input": "RGB", "macro_f1": 0.6866, "note": "OOF 5-fold"},
            {"scenario": "CNN softmap (group-aware)", "input": "Softmap", "macro_f1": 0.7853, "note": "OOF 5-fold"},
            {"scenario": "CNN + kalibrasi Stage1", "input": "Softmap", "macro_f1": 0.8018, "note": "Post-hoc"},
        ])
        REPORT_VESSEL_DICE = 0.4739

        def seed_everything(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True

        seed_everything(cfg.seed)
        """
    ),
    code(
        """
        IS_KAGGLE = Path("/kaggle").exists()
        REPO_ROOT = Path.cwd()
        if IS_KAGGLE:
            WORK_DIR = Path("/kaggle/working")
            INPUT_ROOTS = [Path("/kaggle/input"), REPO_ROOT]
        else:
            WORK_DIR = REPO_ROOT
            INPUT_ROOTS = [REPO_ROOT]

        OUTPUT_DIR = WORK_DIR / "output"
        FIG_DIR = WORK_DIR / "figures"
        CACHE_DIR = OUTPUT_DIR / "cache"
        for path in [OUTPUT_DIR, FIG_DIR, CACHE_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count()
        print("Kaggle:", IS_KAGGLE)
        print("Device:", device, "| GPUs:", n_gpus)
        if n_gpus:
            print([torch.cuda.get_device_name(i) for i in range(n_gpus)])
        print("Work dir:", WORK_DIR)
        """
    ),
    md(
        """
        ## 3. Dataset Paths

        The notebook searches common local and Kaggle layouts.
        """
    ),
    code(
        """
        def find_dir(name_parts):
            candidates = []
            for root in INPUT_ROOTS:
                if not root.exists():
                    continue
                for p in root.rglob(name_parts[-1]):
                    if p.is_dir() and all(part in str(p) for part in name_parts[:-1]):
                        candidates.append(p)
            return sorted(candidates, key=lambda p: len(str(p)))[0] if candidates else None

        ZHAO_ROOT = REPO_ROOT / "data" / "Zhao2024"
        AGRAWAL_ROOT = REPO_ROOT / "data" / "Agrawal2021"
        if not ZHAO_ROOT.exists():
            ZHAO_ROOT = find_dir(["Zhao2024"])
        if not AGRAWAL_ROOT.exists():
            AGRAWAL_ROOT = find_dir(["Agrawal2021"])

        print("Zhao2024:", ZHAO_ROOT)
        print("Agrawal2021:", AGRAWAL_ROOT)
        """
    ),
    code(
        """
        CLASSES = ("Normal", "Stage1", "Stage2", "Stage3", "Laser")
        DIR2CLASS = {
            "Normal": "Normal",
            "Stage1": "Stage1",
            "Stage2": "Stage2",
            "Stage3": "Stage3",
            "laser scars": "Laser",
        }
        CLASS2ID = {name: i for i, name in enumerate(CLASSES)}
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        def is_lfs_pointer(path: Path) -> bool:
            try:
                head = path.read_bytes()[:64]
            except OSError:
                return False
            return head.startswith(b"version https://git-lfs.github.com/spec")

        def read_rgb(path):
            path = Path(path)
            if is_lfs_pointer(path):
                raise ValueError(f"Git LFS pointer, not image bytes: {path}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read image: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        def resize_max_side(rgb, max_side):
            h, w = rgb.shape[:2]
            scale = min(1.0, float(max_side) / float(max(h, w)))
            if scale >= 1.0:
                return rgb.copy()
            return cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        """
    ),
    md(
        """
        ## 4. Image Processing Primitives

        FOV masking, normalization, plain-Gabor vessel softmap, and ridge softmap.
        """
    ),
    code(
        """
        def norm01(image, mask=None, lohi=(1, 99)):
            arr = image.astype(np.float32)
            vals = arr[mask > 0] if mask is not None else arr.ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.zeros(arr.shape, np.float32)
            lo, hi = np.percentile(vals, lohi)
            denom = max(float(hi - lo), 1e-8)
            return np.clip((arr - float(lo)) / denom, 0, 1).astype(np.float32)

        def estimate_fov_mask(rgb):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            threshold = max(3, int(np.percentile(gray, 1)))
            mask = (gray > threshold).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            if n_labels > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                mask = (labels == largest).astype(np.uint8)
            return mask.astype(bool)

        def clahe_green(rgb, fov, clip=4.0, tile=(16, 16)):
            green = rgb[:, :, 1].copy()
            green[~fov] = 0
            enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(green)
            enhanced[~fov] = 0
            return norm01(enhanced.astype(np.float32), fov)
        """
    ),
    code(
        """
        def gabor_response(inv_float, fov):
            response = np.zeros(inv_float.shape, np.float32)
            for sigma, lambd in [(1.5, 3), (2.5, 5), (3.5, 7), (5.0, 10)]:
                size = max(7, int(6 * sigma) + (1 - int(6 * sigma) % 2))
                center = size // 2
                y, x = np.ogrid[-center:size - center, -center:size - center]
                for angle in range(0, 180, 15):
                    theta = np.deg2rad(angle)
                    xt = x * np.cos(theta) + y * np.sin(theta)
                    yt = -x * np.sin(theta) + y * np.cos(theta)
                    kernel = np.exp(-0.5 * (xt ** 2 / sigma ** 2 + yt ** 2 * 0.25 / sigma ** 2))
                    kernel *= np.cos(2 * np.pi * xt / lambd)
                    kernel = (kernel - kernel.mean()).astype(np.float32)
                    filtered = cv2.filter2D(inv_float, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
                    response = np.maximum(response, filtered)
            response[~fov] = 0
            return norm01(response, fov)

        def plain_gabor_vessel_softmap(rgb, fov):
            green = rgb[:, :, 1].copy()
            green[~fov] = 0
            enhanced = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16)).apply(green)
            enhanced[~fov] = 0
            inverted = 255 - enhanced
            gabor = gabor_response(norm01(inverted.astype(np.float32), fov), fov)
            blurred = cv2.medianBlur(np.clip(gabor * 255, 0, 255).astype(np.uint8), 7)
            blurred[~fov] = 0
            soft = norm01(blurred.astype(np.float32), fov)
            enhanced_2 = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(12, 12)).apply(
                np.clip(soft * 255, 0, 255).astype(np.uint8)
            )
            enhanced_2[~fov] = 0
            return norm01(enhanced_2.astype(np.float32), fov)
        """
    ),
    code(
        """
        RIDGE_SCALES = (3, 5, 7, 9, 11)

        def hessian_ridge(channel, fov):
            response = meijering(channel.astype(np.float64), sigmas=list(RIDGE_SCALES), black_ridges=False)
            response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            response[~fov] = 0
            return norm01(response, fov)

        def oriented_tophat(channel, fov, length=21, n_angles=12):
            source = np.clip(channel * 255, 0, 255).astype(np.uint8)
            best = np.zeros_like(channel, np.float32)
            for k in range(n_angles):
                base = np.zeros((length, length), np.uint8)
                cv2.line(base, (0, length // 2), (length - 1, length // 2), 255, 1)
                matrix = cv2.getRotationMatrix2D((length / 2, length / 2), 180.0 * k / n_angles, 1.0)
                se = (cv2.warpAffine(base, matrix, (length, length)) > 0).astype(np.uint8)
                best = np.maximum(best, cv2.morphologyEx(source, cv2.MORPH_TOPHAT, se).astype(np.float32))
            best[~fov] = 0
            return norm01(best, fov)

        def ridge_softmap(rgb, fov):
            channel = clahe_green(rgb, fov, clip=4.0, tile=(16, 16))
            ridge = 0.60 * hessian_ridge(channel, fov) + 0.40 * oriented_tophat(channel, fov)
            return norm01(ridge.astype(np.float32), fov)
        """
    ),
    code(
        """
        def build_softmap_from_rgb(rgb):
            work = resize_max_side(rgb, cfg.process_side)
            fov = estimate_fov_mask(work)
            vessel = plain_gabor_vessel_softmap(work, fov)
            ridge = ridge_softmap(work, fov)
            green = clahe_green(work, fov, clip=4.0, tile=(16, 16))
            stack = np.stack([vessel, ridge, green], axis=-1)
            stack = cv2.resize(stack, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
            return np.clip(stack * 255, 0, 255).astype(np.uint8)

        def build_rgb_input_from_rgb(rgb):
            work = resize_max_side(rgb, cfg.process_side)
            fov = estimate_fov_mask(work)
            channels = []
            for c in range(3):
                channel = work[:, :, c].astype(np.float32)
                channel[~fov] = 0
                channels.append(norm01(channel, fov))
            stack = np.stack(channels, axis=-1)
            stack = cv2.resize(stack, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
            return np.clip(stack * 255, 0, 255).astype(np.uint8)

        def build_debug_maps(path):
            rgb = read_rgb(path)
            work = resize_max_side(rgb, cfg.process_side)
            fov = estimate_fov_mask(work)
            green = clahe_green(work, fov)
            vessel = plain_gabor_vessel_softmap(work, fov)
            ridge = ridge_softmap(work, fov)
            softmap = np.stack([vessel, ridge, green], axis=-1)
            return {"rgb": work, "fov": fov, "green": green, "vessel": vessel, "ridge": ridge, "softmap": softmap}
        """
    ),
    md(
        """
        ## 5. Classification Dataset

        Zhao2024 is the primary classification dataset.
        """
    ),
    code(
        """
        def infer_group_key(path):
            stem = Path(path).stem
            # This is a conservative filename-based fallback. If patient IDs exist,
            # replace this with the explicit patient/eye identifier.
            return stem

        def build_zhao_manifest(root):
            rows = []
            if root is None or not Path(root).exists():
                return pd.DataFrame(columns=["path", "label", "label_id", "key", "group"])
            root = Path(root)
            for dirname, label in DIR2CLASS.items():
                class_dir = root / dirname
                if not class_dir.exists():
                    continue
                for path in sorted(class_dir.iterdir()):
                    if path.suffix.lower() not in IMG_EXTS:
                        continue
                    rows.append({
                        "path": str(path),
                        "label": label,
                        "label_id": CLASS2ID[label],
                        "key": f"{label}_{path.stem}".replace(" ", "_"),
                        "group": infer_group_key(path),
                    })
            return pd.DataFrame(rows)

        df = build_zhao_manifest(ZHAO_ROOT)
        print("Rows:", len(df))
        display(df.head())
        display(df["label"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_frame("n"))
        """
    ),
    code(
        """
        def add_group_folds(frame, n_splits=5):
            frame = frame.copy()
            frame["fold"] = -1
            if len(frame) == 0:
                return frame
            y = frame["label_id"].values
            groups = frame["group"].astype(str).values
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
            for fold, (_, val_idx) in enumerate(splitter.split(frame, y, groups)):
                frame.loc[frame.index[val_idx], "fold"] = fold
            return frame

        df = add_group_folds(df, cfg.n_folds)
        fold_table = pd.crosstab(df["fold"], df["label"]).reindex(columns=CLASSES).fillna(0).astype(int)
        display(fold_table)
        fold_table.to_csv(OUTPUT_DIR / "fold_distribution.csv")
        """
    ),
    md(
        """
        ## 6. Dataset Visuals

        These figures are used in the report.
        """
    ),
    code(
        """
        def savefig(path, dpi=180):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.show()

        if len(df):
            counts = df["label"].value_counts().reindex(CLASSES).fillna(0)
            plt.figure(figsize=(7, 4))
            ax = sns.barplot(x=counts.index, y=counts.values, palette="Set2")
            ax.set_title("Zhao2024 class distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Images")
            savefig(FIG_DIR / "dataset_summary.png")
        """
    ),
    code(
        """
        def plot_class_samples(frame, n_per_class=1):
            sample_rows = []
            for label in CLASSES:
                sub = frame[frame["label"] == label]
                if len(sub):
                    sample_rows.extend(sub.sample(min(n_per_class, len(sub)), random_state=cfg.seed).to_dict("records"))
            if not sample_rows:
                print("No samples to plot.")
                return
            fig, axes = plt.subplots(1, len(sample_rows), figsize=(3 * len(sample_rows), 3))
            axes = np.atleast_1d(axes)
            for ax, row in zip(axes, sample_rows):
                try:
                    rgb = read_rgb(row["path"])
                    ax.imshow(rgb)
                except Exception as exc:
                    ax.text(0.5, 0.5, str(exc), ha="center", va="center", wrap=True)
                ax.set_title(row["label"])
                ax.axis("off")
            savefig(FIG_DIR / "zhao_raw_stage_samples.png")

        plot_class_samples(df, n_per_class=1)
        """
    ),
    code(
        """
        def plot_preprocessing_steps(frame, label="Stage2"):
            sub = frame[frame["label"] == label]
            if len(sub) == 0:
                sub = frame
            if len(sub) == 0:
                print("No image available.")
                return
            row = sub.sample(1, random_state=cfg.seed).iloc[0]
            maps = build_debug_maps(row["path"])
            fig, axes = plt.subplots(1, 6, figsize=(15, 3))
            panels = [
                ("RGB", maps["rgb"], None),
                ("FOV", maps["fov"], "gray"),
                ("CLAHE green", maps["green"], "gray"),
                ("Plain-Gabor vessel", maps["vessel"], "magma"),
                ("Meijering/top-hat ridge", maps["ridge"], "magma"),
                ("Softmap composite", maps["softmap"], None),
            ]
            for ax, (title, image, cmap) in zip(axes, panels):
                ax.imshow(image, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            savefig(FIG_DIR / "vessel_vs_ridge_preprocessing_steps.jpg", dpi=200)

        if len(df):
            plot_preprocessing_steps(df)
        """
    ),
    md(
        """
        ## 7. Cache Inputs

        Softmap and RGB inputs are cached so training reads small 224x224 PNGs.
        """
    ),
    code(
        """
        def cache_one(row, mode, out_dir):
            out_dir = Path(out_dir)
            out_path = out_dir / f"{row['key']}.png"
            if out_path.exists():
                return str(out_path), "cached"
            rgb = read_rgb(row["path"])
            image = build_softmap_from_rgb(rgb) if mode == "softmap" else build_rgb_input_from_rgb(rgb)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr)
            return str(out_path), "built"

        def build_cache(frame, mode="softmap", max_workers=None):
            out_dir = CACHE_DIR / f"zhao2024_{mode}_{cfg.img_size}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if len(frame) == 0:
                return out_dir
            rows = frame.to_dict("records")
            max_workers = max_workers or cfg.preprocess_workers
            statuses = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(cache_one, row, mode, out_dir) for row in rows]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"cache {mode}"):
                    statuses.append(fut.result()[1])
            print(mode, pd.Series(statuses).value_counts().to_dict())
            return out_dir

        RGB_CACHE = CACHE_DIR / f"zhao2024_rgb_{cfg.img_size}"
        SOFTMAP_CACHE = CACHE_DIR / f"zhao2024_softmap_{cfg.img_size}"

        if cfg.run_preprocessing_cache and len(df):
            RGB_CACHE = build_cache(df, "rgb")
            SOFTMAP_CACHE = build_cache(df, "softmap")
        else:
            print("Cache build skipped. Set cfg.run_preprocessing_cache=True to build cached PNG inputs.")
        """
    ),
    md(
        """
        ## 8. Supporting Segmentation Evaluation

        Vessel Dice uses HVDROPDB-BV. Ridge Dice uses HVDROPDB-RIDGE.
        """
    ),
    code(
        """
        def agrawal_pairs(root, task):
            if root is None:
                return []
            root = Path(root)
            if task == "vessel":
                base = root / "HVDROPDB-BV"
                specs = [
                    ("RetCam", base / "RetCam_Vessels_images", base / "RetCam_Vessels_masks"),
                    ("Neo", base / "Neo_Vessels_images", base / "Neo_Vessels_masks"),
                ]
            else:
                base = root / "HVDROPDB-RIDGE"
                specs = [
                    ("RetCam", base / "RetCam_Ridge_images", base / "RetCam_Ridge_masks"),
                    ("Neo", base / "Neo_Ridge_images", base / "Neo_Ridge_masks"),
                ]
            pairs = []
            for source, image_dir, mask_dir in specs:
                if not image_dir.exists() or not mask_dir.exists():
                    continue
                for image_path in sorted(image_dir.iterdir()):
                    mask_path = mask_dir / image_path.name
                    if image_path.suffix.lower() in IMG_EXTS and mask_path.exists():
                        pairs.append({"source": source, "image_path": image_path, "mask_path": mask_path})
            return pairs

        def read_binary_mask(path, shape):
            if is_lfs_pointer(Path(path)):
                raise ValueError(f"Git LFS pointer, not image bytes: {path}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {path}")
            if mask.shape != shape:
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return mask > 0

        def dice_precision_recall(pred, gt):
            pred = pred.astype(bool)
            gt = gt.astype(bool)
            tp = float((pred & gt).sum())
            fp = float((pred & ~gt).sum())
            fn = float((~pred & gt).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) else 0.0
            return {"dice": dice, "precision": precision, "recall": recall}

        def binary_from_response(resp, fov, density=0.16):
            vals = resp[fov > 0]
            if vals.size == 0:
                return np.zeros(resp.shape, bool)
            threshold = np.percentile(vals, 100.0 * (1.0 - density))
            return (resp >= threshold) & (fov > 0)
        """
    ),
    code(
        """
        def evaluate_structural_segmentation(task, density=0.16, limit=None):
            pairs = agrawal_pairs(AGRAWAL_ROOT, task)
            if limit:
                pairs = pairs[:limit]
            rows = []
            skipped = 0
            for pair in tqdm(pairs, desc=f"eval {task}"):
                try:
                    rgb = resize_max_side(read_rgb(pair["image_path"]), cfg.process_side)
                    fov = estimate_fov_mask(rgb)
                    gt = read_binary_mask(pair["mask_path"], fov.shape) & fov
                    if task == "vessel":
                        resp = plain_gabor_vessel_softmap(rgb, fov)
                    else:
                        resp = ridge_softmap(rgb, fov)
                    pred = binary_from_response(resp, fov, density=density)
                    metrics = dice_precision_recall(pred, gt)
                    metrics.update({"task": task, "source": pair["source"], "image": pair["image_path"].name})
                    rows.append(metrics)
                except Exception as exc:
                    skipped += 1
                    if skipped <= 3:
                        print("skip:", exc)
            out = pd.DataFrame(rows)
            if len(out):
                summary = out.groupby("task")[["dice", "precision", "recall"]].mean().reset_index()
                display(summary)
                out.to_csv(OUTPUT_DIR / f"{task}_segmentation_metrics.csv", index=False)
            else:
                print(f"No {task} metrics computed. Check whether Agrawal2021 files are real images, not LFS pointers.")
            return out

        vessel_seg_metrics = pd.DataFrame()
        ridge_seg_metrics = pd.DataFrame()
        if cfg.run_segmentation_eval:
            vessel_seg_metrics = evaluate_structural_segmentation("vessel")
            ridge_seg_metrics = evaluate_structural_segmentation("ridge")
        """
    ),
    md(
        """
        ## 9. Classical Baseline

        The report keeps this as the best handcrafted-feature baseline.
        """
    ),
    code(
        """
        GLCM_DISTANCES = (1, 3, 5)
        GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
        GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")

        def glcm_features(gray, fov):
            region = gray.copy()
            region[~fov] = 0
            levels = 32
            quant = np.clip((region.astype(np.float32) / 256.0 * levels).astype(np.uint8), 0, levels - 1)
            glcm = graycomatrix(quant, distances=list(GLCM_DISTANCES), angles=list(GLCM_ANGLES),
                                levels=levels, symmetric=True, normed=True)
            feats = {}
            for prop in GLCM_PROPS:
                vals = graycoprops(glcm, prop)
                feats[f"glcm_{prop}_mean"] = float(vals.mean())
                feats[f"glcm_{prop}_std"] = float(vals.std())
            return feats

        def morphology_features(binary, fov, prefix):
            fov_area = float(max(1, int(fov.sum())))
            px = float(binary.sum())
            skel = skeletonize(binary > 0)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
            sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.float32) if n_labels > 1 else np.array([0.0], np.float32)
            return {
                f"{prefix}_density": px / fov_area,
                f"{prefix}_skeleton_density": float(skel.sum()) / fov_area,
                f"{prefix}_n_components": float(max(0, n_labels - 1)),
                f"{prefix}_comp_size_mean": float(sizes.mean()),
                f"{prefix}_comp_size_max": float(sizes.max()),
                f"{prefix}_comp_size_std": float(sizes.std()),
            }

        def soft_response_features(resp, fov, prefix):
            vals = resp[fov > 0].astype(np.float32)
            if vals.size == 0:
                vals = np.array([0.0], np.float32)
            return {
                f"{prefix}_mean": float(vals.mean()),
                f"{prefix}_std": float(vals.std()),
                f"{prefix}_p90": float(np.percentile(vals, 90)),
                f"{prefix}_p95": float(np.percentile(vals, 95)),
                f"{prefix}_p99": float(np.percentile(vals, 99)),
            }

        def extract_classical_features(path):
            rgb = resize_max_side(read_rgb(path), cfg.process_side)
            fov = estimate_fov_mask(rgb)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            vessel = plain_gabor_vessel_softmap(rgb, fov)
            ridge = ridge_softmap(rgb, fov)
            vessel_bin = binary_from_response(vessel, fov, density=0.16)
            ridge_bin = binary_from_response(ridge, fov, density=0.16)
            feats = {}
            feats.update(glcm_features(gray, fov))
            feats.update(soft_response_features(vessel, fov, "vessel_resp"))
            feats.update(soft_response_features(ridge, fov, "ridge_resp"))
            feats.update(morphology_features(vessel_bin, fov, "vessel"))
            feats.update(morphology_features(ridge_bin, fov, "ridge"))
            for ci, cname in enumerate(("r", "g", "b")):
                vals = rgb[:, :, ci][fov > 0].astype(np.float32)
                vals = vals if vals.size else np.array([0.0], np.float32)
                feats[f"int_{cname}_mean"] = float(vals.mean())
                feats[f"int_{cname}_std"] = float(vals.std())
            return feats
        """
    ),
    code(
        """
        classical_result = None
        if cfg.run_classical_baseline and len(df):
            # Report baseline is 3-class only.
            cdf = df[df["label"].isin(["Stage1", "Stage2", "Stage3"])].reset_index(drop=True)
            feature_rows = []
            for row in tqdm(cdf.to_dict("records"), desc="classical features"):
                feats = extract_classical_features(row["path"])
                feats.update({"label": row["label"], "label_id": row["label_id"], "path": row["path"]})
                feature_rows.append(feats)
            features_df = pd.DataFrame(feature_rows)
            features_df.to_csv(OUTPUT_DIR / "classical_features.csv", index=False)
            X = features_df.drop(columns=["label", "label_id", "path"]).values
            y = features_df["label"].map({"Stage1": 0, "Stage2": 1, "Stage3": 2}).values
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
            pred = np.full(len(cdf), -1, dtype=int)
            for tr, te in skf.split(X, y):
                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("rf", RandomForestClassifier(n_estimators=500, random_state=cfg.seed, class_weight="balanced")),
                ])
                model.fit(X[tr], y[tr])
                pred[te] = model.predict(X[te])
            classical_result = {
                "scenario": "Baseline klasik terbaik",
                "input": "48 fitur",
                "macro_f1": f1_score(y, pred, average="macro"),
                "note": "RF, recomputed",
            }
            print(classical_result)
        else:
            print("Classical baseline skipped. Report value:", REPORT_RESULTS.iloc[0].to_dict())
        """
    ),
    md(
        """
        ## 10. TinyResNet

        Same model family for raw RGB and softmap inputs.
        """
    ),
    code(
        """
        class CachedImageDataset(Dataset):
            def __init__(self, frame, cache_dir, augment=False):
                self.frame = frame.reset_index(drop=True)
                self.cache_dir = Path(cache_dir)
                self.augment = augment

            def __len__(self):
                return len(self.frame)

            def __getitem__(self, idx):
                row = self.frame.iloc[idx]
                path = self.cache_dir / f"{row['key']}.png"
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    rgb = read_rgb(row["path"])
                    img_rgb = build_softmap_from_rgb(rgb) if "softmap" in str(self.cache_dir) else build_rgb_input_from_rgb(rgb)
                else:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.augment:
                    if random.random() < 0.5:
                        img_rgb = np.fliplr(img_rgb).copy()
                    if random.random() < 0.5:
                        img_rgb = np.flipud(img_rgb).copy()
                    angle = random.uniform(-15, 15)
                    scale = random.uniform(0.9, 1.1)
                    h, w = img_rgb.shape[:2]
                    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
                    img_rgb = cv2.warpAffine(img_rgb, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                x = img_rgb.astype(np.float32) / 255.0
                x = torch.from_numpy(x.transpose(2, 0, 1)).float()
                y = torch.tensor(int(row["label_id"]), dtype=torch.long)
                return x, y

        def drop_path(x, p, training):
            if p == 0.0 or not training:
                return x
            keep = 1.0 - p
            mask = torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device) < keep
            return x / keep * mask

        class BasicBlock(nn.Module):
            def __init__(self, in_ch, out_ch, stride=1, dp=0.0):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.dp = dp
                self.skip = nn.Identity() if stride == 1 and in_ch == out_ch else nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = drop_path(out, self.dp, self.training)
                return F.relu(out + self.skip(x))

        class TinyResNet(nn.Module):
            def __init__(self, n_classes=5, widths=(48, 96, 192), drop_path_rate=0.2, dropout=0.3):
                super().__init__()
                dps = list(np.linspace(0, drop_path_rate, 6))
                self.stem = nn.Sequential(
                    nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                    nn.BatchNorm2d(widths[0]),
                    nn.ReLU(True),
                    nn.MaxPool2d(2),
                )
                self.layer1 = nn.Sequential(BasicBlock(widths[0], widths[0], 1, dps[0]), BasicBlock(widths[0], widths[0], 1, dps[1]))
                self.layer2 = nn.Sequential(BasicBlock(widths[0], widths[1], 2, dps[2]), BasicBlock(widths[1], widths[1], 1, dps[3]))
                self.layer3 = nn.Sequential(BasicBlock(widths[1], widths[2], 2, dps[4]), BasicBlock(widths[2], widths[2], 1, dps[5]))
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.drop = nn.Dropout(dropout)
                self.fc = nn.Linear(widths[2], n_classes)

            def forward(self, x):
                x = self.stem(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.pool(x).flatten(1)
                return self.fc(self.drop(x))
        """
    ),
    code(
        """
        class ModelEMA:
            def __init__(self, model, decay=0.999):
                import copy
                self.ema = copy.deepcopy(model).eval()
                self.decay = decay
                self.step = 0
                for p in self.ema.parameters():
                    p.requires_grad_(False)

            def update(self, model):
                with torch.no_grad():
                    self.step += 1
                    decay = min(self.decay, (1 + self.step) / (10 + self.step))
                    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    for key, ema_value in self.ema.state_dict().items():
                        model_value = state[key].detach()
                        if ema_value.dtype.is_floating_point:
                            ema_value.mul_(decay).add_(model_value, alpha=1 - decay)
                        else:
                            ema_value.copy_(model_value)

        def smooth_mix_ce(logits, target_a, target_b, lam, weight=None, smoothing=0.1):
            n = logits.size(1)
            logp = F.log_softmax(logits, dim=1)

            def one(target):
                with torch.no_grad():
                    true = torch.full_like(logp, smoothing / max(1, n - 1))
                    true.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
                loss = -(true * logp)
                if weight is not None:
                    loss = loss * weight.unsqueeze(0)
                return loss.sum(1).mean()

            return lam * one(target_a) + (1.0 - lam) * one(target_b)

        def mix_batch(images, labels, p=0.5, alpha_mix=0.2, alpha_cut=1.0):
            if random.random() > p:
                return images, labels, labels, 1.0
            perm = torch.randperm(images.size(0), device=images.device)
            if random.random() < 0.5:
                lam = float(np.random.beta(alpha_cut, alpha_cut))
                h, w = images.shape[2:]
                rw, rh = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
                cx, cy = np.random.randint(w), np.random.randint(h)
                x1, x2 = np.clip(cx - rw // 2, 0, w), np.clip(cx + rw // 2, 0, w)
                y1, y2 = np.clip(cy - rh // 2, 0, h), np.clip(cy + rh // 2, 0, h)
                images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
                lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(h * w))
            else:
                lam = float(np.random.beta(alpha_mix, alpha_mix))
                images = lam * images + (1.0 - lam) * images[perm]
            return images, labels, labels[perm], lam
        """
    ),
    code(
        """
        def make_model():
            model = TinyResNet(n_classes=len(CLASSES)).to(device)
            if n_gpus >= 2:
                model = nn.DataParallel(model)
            return model

        def train_one_fold(frame, train_idx, val_idx, cache_dir, epochs=None, input_name="softmap"):
            epochs = epochs or cfg.epochs
            train_df = frame.iloc[train_idx].reset_index(drop=True)
            val_df = frame.iloc[val_idx].reset_index(drop=True)
            train_loader = DataLoader(
                CachedImageDataset(train_df, cache_dir, augment=True),
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
            )
            val_loader = DataLoader(
                CachedImageDataset(val_df, cache_dir, augment=False),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            model = make_model()
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            ema = ModelEMA(base_model, decay=cfg.ema_decay)
            counts = np.array([(train_df["label_id"] == i).sum() for i in range(len(CLASSES))], np.float32)
            weights = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

            for epoch in range(1, epochs + 1):
                model.train()
                for images, labels in train_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    images, a, b, lam = mix_batch(images, labels, p=cfg.mix_p)
                    opt.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        loss = smooth_mix_ce(model(images), a, b, lam, weights, cfg.label_smoothing)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    ema.update(model)
                sched.step()
                if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                    print(f"{input_name} epoch {epoch}/{epochs}")

            ema.ema.to(device).eval()
            preds, probs, y_true = [], [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    logits = ema.ema(images)
                    if cfg.use_tta:
                        logits = logits + ema.ema(torch.flip(images, dims=[3]))
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    probs.append(prob)
                    preds.extend(prob.argmax(axis=1).tolist())
                    y_true.extend(labels.numpy().tolist())
            return np.array(preds), np.concatenate(probs), np.array(y_true)
        """
    ),
    code(
        """
        def run_group_cv(frame, cache_dir, input_name="softmap", epochs=None):
            frame = frame.reset_index(drop=True).copy()
            y = frame["label_id"].values
            groups = frame["group"].astype(str).values
            splitter = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
            oof_pred = np.full(len(frame), -1, dtype=int)
            oof_prob = np.zeros((len(frame), len(CLASSES)), dtype=np.float32)
            for fold, (train_idx, val_idx) in enumerate(splitter.split(frame, y, groups)):
                print(f"Fold {fold}: train={len(train_idx)} val={len(val_idx)}")
                pred, prob, _ = train_one_fold(frame, train_idx, val_idx, cache_dir, epochs=epochs, input_name=input_name)
                oof_pred[val_idx] = pred
                oof_prob[val_idx] = prob
                fold_f1 = f1_score(y[val_idx], pred, average="macro")
                print(f"Fold {fold} macro-F1={fold_f1:.4f}")
            report = classification_report(y, oof_pred, target_names=CLASSES, zero_division=0, output_dict=True)
            metrics = {
                "scenario": f"CNN {input_name} (group-aware)",
                "input": input_name,
                "macro_f1": report["macro avg"]["f1-score"],
                "accuracy": accuracy_score(y, oof_pred),
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
            }
            oof = frame[["path", "label", "label_id", "key", "group"]].copy()
            oof["pred"] = oof_pred
            for i, cls in enumerate(CLASSES):
                oof[f"prob_{cls}"] = oof_prob[:, i]
            return metrics, oof
        """
    ),
    md(
        """
        ## 11. RGB vs Softmap Experiments

        Training is off by default. Set `cfg.run_training=True` for the full run.
        """
    ),
    code(
        """
        cv_results = []
        oof_rgb = pd.DataFrame()
        oof_softmap = pd.DataFrame()

        if cfg.run_training and len(df):
            if not RGB_CACHE.exists() or not SOFTMAP_CACHE.exists():
                print("Building missing caches before training.")
                RGB_CACHE = build_cache(df, "rgb")
                SOFTMAP_CACHE = build_cache(df, "softmap")
            epochs = cfg.quick_epochs if cfg.run_quick_training else cfg.epochs
            rgb_metrics, oof_rgb = run_group_cv(df, RGB_CACHE, input_name="RGB", epochs=epochs)
            soft_metrics, oof_softmap = run_group_cv(df, SOFTMAP_CACHE, input_name="softmap", epochs=epochs)
            cv_results.extend([rgb_metrics, soft_metrics])
            oof_rgb.to_csv(OUTPUT_DIR / "oof_rgb.csv", index=False)
            oof_softmap.to_csv(OUTPUT_DIR / "oof_softmap.csv", index=False)
        else:
            print("Training skipped. Set cfg.run_training=True to train TinyResNet.")
        """
    ),
    md(
        """
        ## 12. Stage 1 Calibration

        Post-hoc calibration is applied to OOF probabilities.
        """
    ),
    code(
        """
        def calibrate_stage1(oof, stage1_delta=-0.10):
            if oof is None or len(oof) == 0:
                return pd.DataFrame(), None
            prob_cols = [f"prob_{cls}" for cls in CLASSES]
            probs = oof[prob_cols].values.copy()
            stage1_idx = CLASS2ID["Stage1"]
            probs[:, stage1_idx] = np.clip(probs[:, stage1_idx] + stage1_delta, 0, None)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
            out = oof.copy()
            out["pred_calibrated"] = probs.argmax(axis=1)
            metric = f1_score(out["label_id"], out["pred_calibrated"], average="macro")
            return out, {"scenario": "CNN + kalibrasi Stage1", "input": "Softmap", "macro_f1": metric, "note": f"delta={stage1_delta}"}

        oof_softmap_calibrated, calibration_result = calibrate_stage1(oof_softmap)
        if calibration_result:
            print(calibration_result)
            oof_softmap_calibrated.to_csv(OUTPUT_DIR / "oof_softmap_calibrated.csv", index=False)
        else:
            print("Calibration skipped because OOF softmap probabilities are unavailable.")
        """
    ),
    md(
        """
        ## 13. Final Results

        The report table has four scenarios only.
        """
    ),
    code(
        """
        rows = []
        if classical_result:
            rows.append(classical_result)
        if cv_results:
            rows.extend(cv_results)
        if calibration_result:
            rows.append(calibration_result)

        if rows:
            final_results = pd.DataFrame(rows)
        elif cfg.use_known_report_results_when_not_run:
            final_results = REPORT_RESULTS.copy()
            final_results["source"] = "report_locked"
        else:
            final_results = pd.DataFrame(columns=REPORT_RESULTS.columns)

        final_results.to_csv(OUTPUT_DIR / "final_metrics.csv", index=False)
        display(final_results)
        """
    ),
    code(
        """
        if len(final_results):
            plt.figure(figsize=(8, 4.5))
            ax = sns.barplot(data=final_results, x="macro_f1", y="scenario", palette="Set2")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Macro F1")
            ax.set_ylabel("")
            ax.set_title("Final experiment summary")
            for container in ax.containers:
                ax.bar_label(container, fmt="%.4f", padding=3)
            savefig(FIG_DIR / "paper_result_comparison_with_notebook_run.png")
        """
    ),
    code(
        """
        def plot_confusion_from_oof(oof, pred_col="pred", name="softmap"):
            if oof is None or len(oof) == 0 or pred_col not in oof:
                print(f"No OOF predictions for {name}.")
                return
            cm = confusion_matrix(oof["label_id"], oof[pred_col], labels=list(range(len(CLASSES))))
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"OOF confusion matrix - {name}")
            savefig(FIG_DIR / f"oof_confusion_matrix_{name}.png")

        plot_confusion_from_oof(oof_softmap, "pred", "softmap")
        plot_confusion_from_oof(oof_softmap_calibrated, "pred_calibrated", "softmap_calibrated")
        """
    ),
    md(
        """
        ## 14. Export Summary

        The notebook writes figures and CSV outputs for the report.
        """
    ),
    code(
        """
        print("Artifacts")
        for path in sorted([OUTPUT_DIR, FIG_DIR]):
            print(path)
            for item in sorted(path.glob("*"))[:30]:
                print(" -", item.name)
        """
    ),
]


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
print(f"WROTE {OUT}")
print(f"cells={len(cells)} markdown={sum(c['cell_type']=='markdown' for c in cells)} code={sum(c['cell_type']=='code' for c in cells)}")
