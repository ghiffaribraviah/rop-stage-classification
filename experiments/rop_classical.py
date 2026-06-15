"""ROP stage classification with handcrafted features + classical ML.

No neural network of any kind: vessel masks come from the existing classical
pipeline (Gabor / matched-filter / CLAHE), features are engineered (GLCM
texture, vessel morphology, tortuosity, intensity/region statistics), and the
classifier is SVM / RandomForest / GradientBoosting trained from scratch.

Runs fully on CPU.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support, classification_report,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vessel_pipeline import (  # noqa: E402
    read_rgb, estimate_fov_mask, resize_max_side, process_channel_source,
    VesselPipelineConfig, normalize01, threshold_response_map,
)
from advanced_pipeline import gabor_filter_response  # noqa: E402
from rop_ridge_classical import ridge_response_map  # noqa: E402

CLASSES = ("Normal", "Stage1", "Stage2", "Stage3")
CLS2ID = {n: i for i, n in enumerate(CLASSES)}
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "Zhao2024"
GLCM_DISTANCES = (1, 3, 5)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


def glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict:
    region = gray.copy()
    region[mask == 0] = 0
    levels = 32
    quant = (region.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    quant = np.clip(quant, 0, levels - 1)
    glcm = graycomatrix(
        quant, distances=list(GLCM_DISTANCES), angles=list(GLCM_ANGLES),
        levels=levels, symmetric=True, normed=True,
    )
    feats = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)
        feats[f"glcm_{prop}_mean"] = float(vals.mean())
        feats[f"glcm_{prop}_std"] = float(vals.std())
    return feats


def vessel_morphology_features(binary: np.ndarray, fov: np.ndarray) -> dict:
    fov_area = float(max(1, int((fov > 0).sum())))
    vessel_px = float((binary > 0).sum())
    skel = skeletonize(binary > 0)
    skel_len = float(skel.sum())

    num, labels = cv2.connectedComponents((binary > 0).astype(np.uint8))
    comp_sizes = np.array(
        [int((labels == i).sum()) for i in range(1, num)], dtype=np.float32
    ) if num > 1 else np.array([0.0], dtype=np.float32)

    ys, xs = np.where(binary > 0)
    if xs.size > 4:
        cx, cy = xs.mean(), ys.mean()
        spread = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).mean()))
        cov = np.cov(np.stack([xs, ys]))
        eigs = np.linalg.eigvalsh(cov) if cov.shape == (2, 2) else np.array([0.0, 0.0])
        eigs = np.sort(np.abs(eigs))
        anisotropy = float(eigs[0] / (eigs[1] + 1e-6))
    else:
        spread, anisotropy = 0.0, 0.0

    return {
        "vessel_density": vessel_px / fov_area,
        "skeleton_density": skel_len / fov_area,
        "tortuosity_proxy": vessel_px / (skel_len + 1e-6),
        "n_components": float(num - 1),
        "comp_size_mean": float(comp_sizes.mean()),
        "comp_size_max": float(comp_sizes.max()),
        "comp_size_std": float(comp_sizes.std()),
        "vessel_spread": spread,
        "vessel_anisotropy": anisotropy,
    }


def softmap_features(soft: np.ndarray, fov: np.ndarray) -> dict:
    vals = soft[fov > 0].astype(np.float32)
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float32)
    grad = ndimage.sobel(soft.astype(np.float32))
    gvals = np.abs(grad[fov > 0])
    return {
        "soft_mean": float(vals.mean()),
        "soft_std": float(vals.std()),
        "soft_p90": float(np.percentile(vals, 90)),
        "soft_p99": float(np.percentile(vals, 99)),
        "soft_skew": float(((vals - vals.mean()) ** 3).mean() / (vals.std() ** 3 + 1e-6)),
        "soft_grad_mean": float(gvals.mean()),
        "soft_grad_p95": float(np.percentile(gvals, 95)) if gvals.size else 0.0,
    }


def ridge_features(rgb: np.ndarray, fov: np.ndarray) -> dict:
    resp = ridge_response_map(rgb, fov, method="meijering", use_tophat=True, tophat_weight=0.4)
    vals = resp[fov > 0].astype(np.float32)
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float32)

    thr = np.percentile(vals, 84.0)
    binary = ((resp >= thr) & (fov > 0)).astype(np.uint8)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(binary, 8)
    feats = {
        "ridge_resp_mean": float(vals.mean()),
        "ridge_resp_p95": float(np.percentile(vals, 95)),
        "ridge_resp_p99": float(np.percentile(vals, 99)),
        "ridge_resp_std": float(vals.std()),
        "ridge_density": float((binary > 0).sum()) / float(max(1, (fov > 0).sum())),
        "ridge_n_components": float(max(0, num - 1)),
    }

    if num <= 1:
        feats.update({
            "ridge_main_area": 0.0, "ridge_main_elong": 0.0, "ridge_main_extent": 0.0,
            "ridge_main_len": 0.0, "ridge_main_curv": 0.0, "ridge_main_contrast": 0.0,
            "ridge_center_dist": 0.0, "ridge_main_solidity": 0.0,
        })
        return feats

    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    comp = (labels == main)
    ys, xs = np.where(comp)
    area = float(comp.sum())

    cov = np.cov(np.stack([xs, ys]).astype(np.float32)) if xs.size > 2 else np.eye(2)
    eig = np.sort(np.abs(np.linalg.eigvalsh(cov)))
    elong = float(eig[1] / (eig[0] + 1e-6))
    main_len = float(4.0 * np.sqrt(eig[1])) if eig[1] > 0 else 0.0

    skel = skeletonize(comp)
    skel_len = float(skel.sum())
    curvature = float(skel_len / (main_len + 1e-6)) if main_len > 0 else 0.0

    region_vals = resp[comp]
    surround = cv2.dilate(comp.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = surround & ~comp & (fov > 0)
    contrast = float(region_vals.mean() - (resp[ring].mean() if ring.sum() else 0.0))

    h, w = fov.shape
    cx, cy = xs.mean(), ys.mean()
    center_dist = float(np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2) / (np.sqrt(w * w + h * h) / 2))

    bx, by, bw, bh = (stats[main, cv2.CC_STAT_LEFT], stats[main, cv2.CC_STAT_TOP],
                      stats[main, cv2.CC_STAT_WIDTH], stats[main, cv2.CC_STAT_HEIGHT])
    extent = area / float(max(1, bw * bh))
    hull_area = area
    try:
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
            hull_area = max(area, float(cv2.contourArea(hull)))
    except Exception:  # noqa: BLE001
        pass
    solidity = area / (hull_area + 1e-6)

    feats.update({
        "ridge_main_area": area / float(max(1, (fov > 0).sum())),
        "ridge_main_elong": elong,
        "ridge_main_extent": extent,
        "ridge_main_len": main_len / float(max(h, w)),
        "ridge_main_curv": curvature,
        "ridge_main_contrast": contrast,
        "ridge_center_dist": center_dist,
        "ridge_main_solidity": solidity,
    })
    return feats


def intensity_features(rgb: np.ndarray, fov: np.ndarray) -> dict:
    feats = {}
    for ci, cname in enumerate(("r", "g", "b")):
        ch = rgb[:, :, ci][fov > 0].astype(np.float32)
        if ch.size == 0:
            ch = np.array([0.0], dtype=np.float32)
        feats[f"int_{cname}_mean"] = float(ch.mean())
        feats[f"int_{cname}_std"] = float(ch.std())
    return feats


def vessel_overlay_config(rgb: np.ndarray, fov: np.ndarray, density: float = 0.16, top: int = 2):
    green = rgb[:, :, 1].copy()
    green[~fov] = 0
    enh = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(16, 16)).apply(green)
    enh[~fov] = 0
    inv = 255 - enh
    inv_f = normalize01(inv.astype(np.float32), fov)
    gab = gabor_filter_response(inv_f, fov)
    r7 = cv2.medianBlur((gab * 255).astype(np.uint8), 7)
    r7[~fov] = 0
    soft = normalize01(r7.astype(np.float32), fov)
    u8 = np.clip(soft * 255, 0, 255).astype(np.uint8)
    u8[~fov] = 0
    enh_s = cv2.createCLAHE(clipLimit=12, tileGridSize=(12, 12)).apply(u8)
    enh_s[~fov] = 0
    sharp = normalize01(enh_s.astype(np.float32), fov)
    th = threshold_response_map(sharp, fov, method="percentile", target_density=density) > 0
    nl, labels, stats, _ = cv2.connectedComponentsWithStats(th.astype(np.uint8), 8)
    if nl > 1:
        areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, nl)), reverse=True)
        keep = {idx for _, idx in areas[: min(top, len(areas))]}
        th = np.isin(labels, list(keep))
    return sharp, th.astype(np.uint8)


def extract_features(path: str) -> dict:
    rgb = read_rgb(path)
    rgb = resize_max_side(rgb, 768)
    fov = estimate_fov_mask(rgb)
    soft, binary = vessel_overlay_config(rgb, fov, density=0.16, top=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    feats: dict = {}
    feats.update(glcm_features(gray, fov))
    feats.update(vessel_morphology_features(binary, fov))
    feats.update(softmap_features(soft, fov))
    feats.update(intensity_features(rgb, fov))
    feats.update(ridge_features(rgb, fov))
    return feats


def build_dataset() -> pd.DataFrame:
    rows = []
    exts = {".jpg", ".jpeg", ".png"}
    for c in CLASSES:
        d = DATA_ROOT / c
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                rows.append({"path": str(p), "label": c, "label_id": CLS2ID[c]})
    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    print(f"Dataset: {len(df)} labeled images")
    print(df["label"].value_counts().to_string())

    t0 = time.time()
    feat_rows = []
    for i, row in df.iterrows():
        try:
            feat_rows.append(extract_features(row["path"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  feature error on {row['path']}: {exc}")
            feat_rows.append({})
        if (i + 1) % 50 == 0:
            print(f"  features {i + 1}/{len(df)} ({time.time() - t0:.0f}s)")

    X = pd.DataFrame(feat_rows).fillna(0.0)
    y = df["label_id"].to_numpy()
    feature_names = list(X.columns)
    print(f"\nExtracted {X.shape[1]} features in {time.time() - t0:.0f}s")
    X.to_csv(Path(__file__).resolve().parent / "output" / "classical_features.csv", index=False)

    models = {
        "svm_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=42,
        ),
        "grad_boost": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Xv = X.to_numpy()
    print(f"\n{'=' * 56}\n5-fold stratified CV (macro-F1)\n{'=' * 56}")
    for name, model in models.items():
        yp = cross_val_predict(model, Xv, y, cv=cv, n_jobs=-1)
        f1 = f1_score(y, yp, average="macro")
        acc = accuracy_score(y, yp)
        print(f"\n[{name}] macro-F1={f1:.4f}  acc={acc:.4f}")
        p, r, f, _ = precision_recall_fscore_support(y, yp, average=None, zero_division=0)
        for ci, cname in enumerate(CLASSES):
            print(f"   {cname:8s} P={p[ci]:.3f} R={r[ci]:.3f} F1={f[ci]:.3f}")

    rf = models["random_forest"].fit(Xv, y)
    imp = sorted(zip(feature_names, rf.feature_importances_), key=lambda kv: -kv[1])
    print(f"\nTop 12 RF feature importances:")
    for fn, iv in imp[:12]:
        print(f"   {fn:24s} {iv:.4f}")


if __name__ == "__main__":
    main()
