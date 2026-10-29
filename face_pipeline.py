#!/usr/bin/env python3
"""CS228 Checkpoint 1: auditable masked-face preprocessing and verification.

Sections follow the checkpoint: data engineering (II), EDA (III), baseline (IV).
The CLI also provides a structure-only inventory and an optional pixel/PCA sanity
baseline. The requested MaskInv baseline always requires real pretrained weights;
there is no random-model fallback. See README.md for assumptions and commands.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from typing import Any
import warnings

import matplotlib
matplotlib.use("Agg")  # Works on headless servers and Windows without a GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import laplace
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve

LOGGER = logging.getLogger("faces")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
FEATURES = ["width", "height", "aspect_ratio", "brightness", "contrast", "entropy", "sharpness", "dark_fraction", "bright_fraction"]
CONDITIONS = {"NM": ("unmasked", "masked"), "MM": ("masked", "masked"), "NN": ("unmasked", "unmasked")}
CAPTIONS: dict[str, str] = {}


# ============================================================================
# 0. Reproducibility, serialization, and figure conventions
# ============================================================================
def write_json(path: Path, value: Any) -> None:
    """Write strict JSON: NaN/Infinity are forbidden to prevent ambiguous metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy; torch is configured separately only if requested."""
    random.seed(seed)
    np.random.seed(seed)


def configure_torch(seed: int, threads: int) -> None:
    """Apply deterministic torch settings before either alignment or inference."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("This mode needs PyTorch; see requirements-model.txt and README") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(threads)


def sha256_file(path: Path) -> str:
    """Hash large model files in chunks rather than loading them a second time."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(image: Image.Image) -> str:
    """Detect exact decoded duplicates even when filenames/encodings differ."""
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def package_versions() -> dict:
    """Record actual installed versions, not just dependency constraints."""
    import importlib.metadata
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "pandas", "Pillow", "scipy", "scikit-learn", "matplotlib", "torch", "facenet-pytorch"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def git_state(repo: Path) -> dict:
    """Record commit and dirty state; return unknown outside a Git repository."""
    def run(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def save_figure(fig, out: Path, name: str, caption: str) -> None:
    """Every figure gets a title/axes, an embedded caption, and a caption index."""
    out.mkdir(parents=True, exist_ok=True)
    import textwrap
    wrapped = textwrap.fill(caption, width=110)
    fig.text(0.02, 0.015, wrapped, va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.06 + 0.022 * (len(wrapped.splitlines()) - 1), 1, 1))
    fig.savefig(out / f"{name}.png", dpi=160)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    CAPTIONS[name] = caption
    write_json(out / "captions.json", CAPTIONS)


# ============================================================================
# II-A. Input schema, identity parsing, and structure-only inventory
# ============================================================================
def canonical_identity(origin: str, folder: str, aliases: dict) -> str:
    """Apply explicit aliases first, then the documented verification-name rule.

    '00_0001_tankai_07' -> 'tankai'. Prefix/suffix meanings are NOT inferred.
    This provisional equivalence is emitted in the audit for manual review.
    """
    if (origin, folder) in aliases:
        return aliases[(origin, folder)]
    match = re.fullmatch(r"\d+_\d+_(.+)_\d+", folder) if origin == "verification" else None
    return (match.group(1) if match else folder).strip().casefold()


def read_aliases(path: Path | None) -> dict:
    """Read origin,folder,identity CSV; reject ambiguous mappings."""
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"origin", "folder", "identity"}.issubset(frame):
        raise ValueError("Alias CSV needs origin,folder,identity columns")
    result = {}
    for row in frame.itertuples():
        key = (row.origin, row.folder)
        identity = row.identity.strip().casefold()
        if row.origin not in {"train", "verification"} or not identity:
            raise ValueError(f"Invalid identity alias: {key}")
        if key in result and result[key] != identity:
            raise ValueError(f"Conflicting alias: {key}")
        result[key] = identity
    return result


def structure_inventory(path: Path, out: Path, aliases: dict) -> dict:
    """Parse Windows TREE /F text without assuming its filenames are decodable.

    Counts are listing-derived, NOT counts of valid images. Empty directories are
    included. The live image audit remains authoritative for all real results.
    """
    stack: list[str] = []
    records = []
    folders = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = re.match(r"^(.*?)[+\\]---(.*)$", line)
        if match:
            depth = len(match.group(1)) // 4
            stack = stack[:depth] + [match.group(2)]
            if depth == 2 and stack[0] in {"train", "verification"}:
                folders.append({"origin": stack[0], "branch": stack[1], "folder": stack[2],
                                "identity": canonical_identity(stack[0], stack[2], aliases)})
        else:
            name = line.strip(" |\t")
            if len(stack) == 3 and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"origin": stack[0], "branch": stack[1], "folder": stack[2],
                                "condition": "unmasked" if (stack[1] == "unmasked" or Path(name).stem.casefold() == "unmasked") else "masked"})
    f = pd.DataFrame(folders)
    r = pd.DataFrame(records)
    if f.empty or r.empty:
        raise ValueError("No recognized TREE /F dataset structure found")
    train = set(f.loc[f.origin == "train", "identity"])
    verify = set(f.loc[f.origin == "verification", "identity"])
    result = {"source": str(path), "evidence": "directory listing only; image content not available",
              "image_entries": r.groupby(["origin", "branch"]).size().to_dict(),
              "folder_counts": f.groupby(["origin", "branch"]).size().to_dict(),
              "train_identity_names": len(train), "verification_identity_names": len(verify),
              "provisional_overlapping_identity_names": sorted(train & verify),
              "decoding_warning": "Replacement characters may occur in legacy Windows filenames."}
    # JSON object keys must be strings, whereas groupby creates tuple keys.
    for key in ["image_entries", "folder_counts"]:
        result[key] = {"/".join(k): int(v) for k, v in result[key].items()}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "structure_inventory.json", result)
    f.to_csv(out / "structure_identity_mapping.csv", index=False)
    return result


def scan_dataset(root: Path, aliases: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load exactly train/{masked,unmasked}/ID and verification/masked_whn/ID.

    Numbered verification files are assumed masked; 'unmasked.*' is unmasked.
    Unknown files and empty folders are logged, never silently counted as images.
    Unicode paths are preserved; basenames are never used as unique identifiers.
    """
    rows, issues = [], []
    branches = [("train", "masked", root / "train" / "masked"),
                ("train", "unmasked", root / "train" / "unmasked"),
                ("verification", None, root / "verification" / "masked_whn")]
    for origin, branch_condition, branch in branches:
        if not branch.is_dir():
            raise ValueError(f"Required directory missing: {branch}")
        for folder in sorted(branch.iterdir()):
            if not folder.is_dir():
                issues.append({"path": folder.relative_to(root).as_posix(), "issue": "unexpected_branch_file"})
                continue
            files = sorted(folder.iterdir())
            found = 0
            for path in files:
                rel = path.relative_to(root).as_posix()
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    issues.append({"path": rel, "issue": "unsupported_file_or_nested_directory"})
                    continue
                condition = branch_condition or ("unmasked" if path.stem.casefold() == "unmasked" else "masked")
                rows.append({"sample_id": hashlib.sha256(rel.encode("utf-8")).hexdigest()[:24],
                             "relative_path": rel, "origin": origin, "folder": folder.name,
                             "identity": canonical_identity(origin, folder.name, aliases),
                             "condition": condition, "session": "unknown", "extension": path.suffix.lower()})
                found += 1
            if not found:
                issues.append({"path": folder.relative_to(root).as_posix(), "issue": "empty_identity_folder"})
    if not rows:
        raise ValueError("Dataset contains no supported image files")
    return pd.DataFrame(rows), pd.DataFrame(issues, columns=["path", "issue"])


# ============================================================================
# II-B. Image integrity, quality measurements, cropping/alignment, and caching
# ============================================================================
def image_statistics(image: Image.Image) -> dict:
    """Measure original oriented RGB image; sharpness is a proxy, not face quality.

    Gray intensity is in [0,1]. Laplacian variance is computed after resizing the
    gray image to 112x112, reducing resolution confounding across conditions.
    """
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    histogram = np.bincount(np.asarray(gray_image).ravel(), minlength=256).astype(float)
    probability = histogram[histogram > 0] / histogram.sum()
    small = np.asarray(gray_image.resize((112, 112), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    w, h = image.size
    return {"width": w, "height": h, "aspect_ratio": w / h,
            "brightness": float(gray.mean()), "contrast": float(gray.std()),
            "entropy": float(-(probability * np.log2(probability)).sum()),
            "sharpness": float(laplace(small).var()),
            "dark_fraction": float((gray < 0.05).mean()),
            "bright_fraction": float((gray > 0.95).mean())}


def make_aligner(mode: str, device: str):
    """Optional five-point MTCNN similarity alignment; no detection fallback.

    MTCNN detection alone is not alignment: eyes/nose/mouth landmarks are fitted
    to a standard 112x112 face template with a rotation+scale+translation.
    """
    if mode != "mtcnn":
        return None
    try:
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError("MTCNN mode needs torch, torchvision and facenet-pytorch; see README") from exc
    return MTCNN(keep_all=True, device=device)


def similarity_align(image: Image.Image, landmarks: np.ndarray) -> Image.Image:
    """Fit an orientation-preserving similarity transform via SVD (Umeyama).

    Pillow needs the inverse output-to-input affine map. Reject degenerate
    landmark geometry instead of manufacturing an aligned crop.
    """
    source = np.asarray(landmarks, dtype=np.float64)
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    variance = float((x * x).sum() / len(x))
    if variance < 1e-10 or not np.isfinite(source).all():
        raise ValueError("degenerate_face_landmarks")
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float((singular * np.diag(correction)).sum() / variance)
    matrix = np.eye(3)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = target_mean - matrix[:2, :2] @ source_mean
    inverse = np.linalg.inv(matrix)
    return image.transform((112, 112), Image.Transform.AFFINE, tuple(inverse[:2].ravel()), Image.Resampling.BICUBIC)


def preprocess_image(image: Image.Image, mode: str, aligner, multiple_faces: str) -> tuple[Image.Image, dict]:
    """Return RGB 112x112 and auditable transformation metadata.

    pad: aspect-preserving letterbox, no alignment claim.
    prealigned: require 112x112 input and trust documented prior alignment.
    mtcnn: detect + five-point alignment; reject failures/multiple faces by default.
    """
    meta = {"face_count": None, "face_confidence": None}
    if mode == "prealigned":
        if image.size != (112, 112):
            raise ValueError("prealigned_input_must_be_112x112")
        return image, meta
    if mode == "pad":
        return ImageOps.pad(image, (112, 112), method=Image.Resampling.LANCZOS, color=(0, 0, 0)), meta
    boxes, probs, landmarks = aligner.detect(image, landmarks=True)
    if boxes is None or len(boxes) == 0:
        raise ValueError("face_not_detected")
    if len(boxes) > 1 and multiple_faces == "reject":
        raise ValueError("multiple_faces_detected")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(area))
    if not np.isfinite(probs[index]) or probs[index] < 0.9:
        raise ValueError("face_confidence_below_0.9")
    meta = {"face_count": int(len(boxes)), "face_confidence": float(probs[index])}
    return similarity_align(image, landmarks[index]), meta


def audit_images(frame: pd.DataFrame, root: Path, out: Path, args) -> pd.DataFrame:
    """Decode every image, quarantine bad inputs, and save lossless processed PNGs.

    Originals are never edited. Missingness is an audit statistic; no fabricated
    identity/condition/image values are imputed. Very dark or blurred images are
    flagged for review, retained by default to avoid selective evaluation bias.
    """
    aligner = make_aligner(args.alignment, args.device)
    cache = out / "processed"
    cache.mkdir()
    records = []
    for index, row in enumerate(frame.to_dict("records")):
        record = {**row, "status": "ok", "reason": "", "processed_path": ""}
        try:
            path = root / row["relative_path"]
            record["file_sha256"] = sha256_file(path)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as handle:
                    if getattr(handle, "n_frames", 1) != 1:
                        raise ValueError("animated_or_multiframe_image")
                    handle.load()  # Strict decoder; do not enable LOAD_TRUNCATED_IMAGES.
                    record["source_mode"] = handle.mode
                    image = ImageOps.exif_transpose(handle).convert("RGB")
            record.update(image_statistics(image))
            record["pixel_sha256"] = pixel_hash(image)
            if min(image.size) < args.min_side:
                raise ValueError("image_below_minimum_side")
            transformed, meta = preprocess_image(image, args.alignment, aligner, args.multiple_faces)
            record.update(meta)
            record["processed_sha256"] = pixel_hash(transformed)
            target = cache / f"{row['sample_id']}.png"
            transformed.save(target)
            record["processed_path"] = target.relative_to(out).as_posix()
        except (OSError, ValueError, RuntimeError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            record.update(status="rejected", reason=f"{type(exc).__name__}: {exc}")
        records.append(record)
        if (index + 1) % 500 == 0:
            LOGGER.info("Audited %d/%d images", index + 1, len(frame))
    return pd.DataFrame(records)


def quarantine_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find raw/decoded/processed exact duplicates; quarantine conflicting labels.

    Build duplicate components across all three hashes. A component may contain
    different raw files that become identical after preprocessing. If identity OR
    mask labels conflict, all component members are quarantined for manual review.
    Otherwise prefer verification over training and retain one stable representative.
    Near duplicates require human review; this is not a perceptual deduplicator.
    """
    frame = frame.copy()
    active = list(frame.index[frame.status == "ok"])
    parent = {i: i for i in active}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for column in ["file_sha256", "pixel_sha256", "processed_sha256"]:
        if column not in frame:
            continue
        first = {}
        for i in active:
            value = frame.at[i, column]
            if pd.isna(value):
                continue
            if value in first:
                union(first[value], i)
            else:
                first[value] = i
    groups = defaultdict(list)
    for i in active:
        groups[find(i)].append(i)
    details = []
    for members in groups.values():
        if len(members) < 2:
            continue
        group = frame.loc[members]
        conflict = group.identity.nunique() > 1 or group.condition.nunique() > 1
        winner = None if conflict else sorted(members, key=lambda i: (frame.at[i, "origin"] != "verification", frame.at[i, "relative_path"]))[0]
        component_id = min(group.sample_id)
        for i in members:
            details.append({"duplicate_group": component_id, "sample_id": frame.at[i, "sample_id"],
                            "relative_path": frame.at[i, "relative_path"], "label_conflict": conflict,
                            "retained": i == winner})
            if i != winner:
                frame.loc[i, ["status", "reason"]] = ["quarantined", "duplicate_label_conflict" if conflict else "exact_duplicate"]
    return frame, pd.DataFrame(details, columns=["duplicate_group", "sample_id", "relative_path", "label_conflict", "retained"])


# ============================================================================
# II-C. Identity-disjoint train/calibration/test split and quality diagnostics
# ============================================================================
def assign_splits(frame: pd.DataFrame, seed: int, calibration_fraction: float) -> pd.DataFrame:
    """Purge train identities seen in verification; split verification by identity.

    Hold out all verification identities, even ones with rejected images. This
    prevents future restoration of a failed image from introducing hidden leakage.
    At least four usable verification identities are required for pair metrics.
    """
    frame = frame.copy()
    frame["split"] = "excluded"
    verification_ids = set(frame.loc[frame.origin == "verification", "identity"])
    train_ok = (frame.origin == "train") & (frame.status == "ok")
    overlap = train_ok & frame.identity.isin(verification_ids)
    frame.loc[overlap, ["status", "reason"]] = ["excluded", "train_identity_in_verification"]
    frame.loc[(frame.origin == "train") & (frame.status == "ok"), "split"] = "train"
    usable = sorted(frame.loc[(frame.origin == "verification") & (frame.status == "ok"), "identity"].unique())
    np.random.default_rng(seed).shuffle(usable)
    if len(usable) < 4:
        raise ValueError("Need >=4 usable verification identities for disjoint calibration/test pair sets")
    n_calibration = min(len(usable) - 2, max(2, round(len(usable) * calibration_fraction)))
    calibration = set(usable[:n_calibration])
    for identity in usable:
        select = (frame.origin == "verification") & (frame.status == "ok") & (frame.identity == identity)
        frame.loc[select, "split"] = "calibration" if identity in calibration else "test"
    assert_disjoint(frame)
    return frame


def assert_disjoint(frame: pd.DataFrame) -> None:
    """Fail immediately if any identity/hash occurs in two model partitions."""
    used = frame[frame.split != "excluded"]
    for column in ["identity", "file_sha256", "pixel_sha256", "processed_sha256"]:
        if column not in used:
            continue
        if (used.groupby(column).split.nunique() > 1).any():
            raise ValueError(f"Cross-split leakage detected by {column}")


def quality_tables(frame: pd.DataFrame, out: Path) -> dict:
    """Write missingness, IQR outliers and per-identity counts for the report."""
    frame.isna().sum().rename("missing_count").to_csv(out / "missing_values.csv")
    valid = frame[frame.status == "ok"]
    outliers = []
    for column in FEATURES:
        if column not in valid or valid[column].dropna().empty:
            continue
        q1, q3 = valid[column].quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        for row in valid[(valid[column] < low) | (valid[column] > high)].itertuples():
            outliers.append({"sample_id": row.sample_id, "feature": column, "value": getattr(row, column), "lower": low, "upper": high})
    pd.DataFrame(outliers, columns=["sample_id", "feature", "value", "lower", "upper"]).to_csv(out / "iqr_outliers.csv", index=False)
    counts = frame.groupby(["origin", "identity", "condition", "status", "split"]).size().rename("samples").reset_index()
    counts.to_csv(out / "identity_counts.csv", index=False)
    frame[["origin", "folder", "identity"]].drop_duplicates().sort_values(["identity", "origin"]).to_csv(out / "identity_mapping.csv", index=False)
    result = {"scanned_images": len(frame), "status_counts": {str(k): int(v) for k, v in frame.status.value_counts().items()},
              "retained_split_counts": {}, "sessions": "unknown", "outlier_rows": len(outliers),
              "identity_names_are_provisional": True,
              "limitations": ["No demographic, session, spoof/live, or provenance labels inferred from faces or filenames.",
                              "Exact duplicate checks do not establish absence of near duplicates or mislabeled identities.",
                              "Pretrained-model training identities may overlap this dataset; their overlap is unknown."]}
    for split, group in frame[frame.split != "excluded"].groupby("split"):
        result["retained_split_counts"][split] = {"images": len(group), "identities": int(group.identity.nunique()),
                                                  "conditions": {str(k): int(v) for k, v in group.condition.value_counts().items()}}
    write_json(out / "data_summary.json", result)
    return result


# ============================================================================
# III. EDA: quality distributions, feature relationships, balance and examples
# ============================================================================
def run_eda(frame: pd.DataFrame, out: Path, seed: int) -> None:
    """Produce report figures without fitting or tuning anything on test labels.

    Descriptive test EDA is allowed; its inspection must not lead to retuning the
    chosen threshold. Mask state is categorical; identity codes are NOT correlated
    as arbitrary numbers. Color-grouped scatter plots show label relationships.
    """
    figures = out / "figures"
    # Raw readable images, including later exclusions, reveal selection effects.
    raw = frame[frame.brightness.notna()].copy() if "brightness" in frame else pd.DataFrame()
    if raw.empty:
        raise ValueError("No readable images for EDA")
    palette = {"masked": "#d97706", "unmasked": "#2563eb"}
    numeric = ["brightness", "contrast", "entropy", "sharpness", "width", "height"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feature in zip(axes.ravel(), numeric):
        for condition, color in palette.items():
            values = raw.loc[raw.condition == condition, feature]
            if len(values):
                ax.hist(values, bins=35, density=True, alpha=0.5, color=color, label=condition)
        ax.set(title=f"Original {feature} distribution", xlabel=feature, ylabel="Density")
        ax.legend(fontsize=8)
    save_figure(fig, figures, "01_univariate_histograms", "Compare masked and unmasked image statistics to identify quality and resolution shifts before exclusions.")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feature in zip(axes.ravel(), numeric[:4]):
        groups = [raw.loc[raw.condition == c, feature].dropna() for c in palette]
        ax.boxplot(groups, tick_labels=list(palette), showfliers=False)
        ax.set(title=f"{feature} by mask state", xlabel="Mask state", ylabel=feature)
    save_figure(fig, figures, "02_quality_boxplots", "Boxes show medians and spread by mask state; extreme points are listed separately in the IQR outlier audit.")
    corr = raw[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    artist = ax.imshow(corr.fillna(0), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(xticks=range(len(FEATURES)), yticks=range(len(FEATURES)), xticklabels=FEATURES, yticklabels=FEATURES,
           title="Image-statistic Pearson correlations", xlabel="Image statistic", ylabel="Image statistic")
    plt.setp(ax.get_xticklabels(), rotation=50, ha="right")
    fig.colorbar(artist, ax=ax, label="Pearson r (constant-feature cells shown as 0)")
    corr.to_csv(out / "feature_correlations.csv")
    save_figure(fig, figures, "03_correlations", "Correlations expose redundant image statistics; constant-feature correlations are undefined in the CSV and rendered as zero.")
    sample = raw.sample(min(5000, len(raw)), random_state=seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition, color in palette.items():
        group = sample[sample.condition == condition]
        axes[0].scatter(group.brightness, group.sharpness, s=9, alpha=0.4, c=color, label=condition)
        axes[1].scatter(group.width, group.height, s=9, alpha=0.4, c=color, label=condition)
    axes[0].set(title="Brightness versus sharpness", xlabel="Mean gray intensity [0,1]", ylabel="112x112 Laplacian variance")
    axes[1].set(title="Original image geometry", xlabel="Width (pixels)", ylabel="Height (pixels)")
    for ax in axes:
        ax.legend()
    save_figure(fig, figures, "04_bivariate_scatter", "A fixed-seed sample of at most 5,000 images relates quality and geometry to mask labels without treating identity IDs as numeric features.")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    frame.groupby(["origin", "condition"]).size().unstack(fill_value=0).plot.bar(ax=axes[0], color=palette)
    axes[0].set(title="Input mask balance", xlabel="Original partition", ylabel="Image count")
    per_id = frame.groupby(["origin", "identity"]).size()
    for origin in ["train", "verification"]:
        if origin in per_id.index.get_level_values(0):
            axes[1].hist(per_id.loc[origin], bins=30, alpha=0.5, label=origin)
    axes[1].set(title="Samples per identity", xlabel="Images per identity", ylabel="Identity count")
    axes[1].legend()
    frame.groupby(["split", "condition"]).size().unstack(fill_value=0).plot.bar(ax=axes[2], color=palette)
    axes[2].set(title="Retained and excluded images", xlabel="Assigned partition", ylabel="Image count")
    save_figure(fig, figures, "05_class_balance", "Imbalance motivates capped genuine pairs per identity, balanced impostor sampling, and macro identity error rates rather than image oversampling.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    status = frame.status.value_counts()
    axes[0].bar(status.index, status.values, color="#64748b")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set(title="Preprocessing disposition", xlabel="Audit status", ylabel="Image count")
    overlap_sets = [set(frame.loc[frame.origin == o, "identity"]) for o in ["train", "verification"]]
    values = [len(overlap_sets[0] - overlap_sets[1]), len(overlap_sets[0] & overlap_sets[1]), len(overlap_sets[1] - overlap_sets[0])]
    axes[1].bar(["Train only", "Overlap", "Verification only"], values, color=["#2563eb", "#dc2626", "#16a34a"])
    axes[1].set(title="Provisional identity overlap", xlabel="Identity membership before purge", ylabel="Identity count")
    save_figure(fig, figures, "06_cleaning_and_overlap", "Training identities that also occur in verification are excluded before fitting any reduction or selecting thresholds.")
    examples = frame[(frame.status == "ok") & (frame.processed_path != "")]
    if len(examples):
        examples = examples.sample(min(8, len(examples)), random_state=seed)
        fig, axes = plt.subplots(len(examples), 2, figsize=(7, max(4, 2.4 * len(examples))), squeeze=False)
        root = Path(json.loads((out / "run_config.json").read_text())["dataset_root"])
        for axes_row, row in zip(axes, examples.itertuples()):
            with Image.open(root / row.relative_path) as original:
                axes_row[0].imshow(ImageOps.exif_transpose(original).convert("RGB"))
            with Image.open(out / row.processed_path) as transformed:
                axes_row[1].imshow(transformed)
            for ax, title in zip(axes_row, [f"Original: {row.condition}", "Processed 112x112"]):
                ax.set(title=title, xlabel="Pixel x", ylabel="Pixel y")
        save_figure(fig, figures, "07_preprocessing_examples", "Original and processed samples make cropping, padding, and alignment artifacts inspectable before baseline evaluation.")


# ============================================================================
# IV-A. Requested baseline: pretrained MaskInv IResNet-100 embedding extraction
# ============================================================================
def load_maskinv(args) -> tuple[Any, dict]:
    """Import the requested repository's iresnet100 and strictly load its weights.

    No upstream training scripts are executed; the network is frozen. Evaluation
    on this dataset is a transfer baseline, not a reproduction of paper results.
    The model's own embedding size and checkpoint compatibility are validated.
    """
    if args.repo is None or args.checkpoint is None:
        raise ValueError("maskinv needs --repo and --checkpoint; see README for model downloads")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to run the requested MaskInv baseline") from exc
    repo = args.repo.resolve()
    backbone_file = repo / "backbones" / "iresnet.py"
    if not backbone_file.is_file():
        raise ValueError(f"Expected upstream backbones/iresnet.py at {repo}")
    # Load only the backbone file: importing backbones/__init__ can pull unrelated deps.
    specification = importlib.util.spec_from_file_location("checkpoint_maskinv_iresnet", backbone_file)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    configure_torch(args.seed, args.torch_threads)
    model = module.iresnet100(num_features=512)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "backbone", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Expected a tensor state dictionary; do not supply a serialized full module")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    return model, {"method": "MaskInv frozen IResNet-100 + cosine similarity", "upstream": git_state(repo),
                   "backbone_sha256": sha256_file(backbone_file), "checkpoint_sha256": sha256_file(args.checkpoint),
                   "checkpoint_name": args.checkpoint.name, "alignment": args.alignment,
                   "normalization": args.normalization, "channel_order": args.channel_order,
                   "flip_test": args.flip_test, "device": args.device,
                   "training": "none: pretrained model evaluation; threshold fitted on calibration identities"}


def normalize_embeddings(array: np.ndarray) -> np.ndarray:
    """Normalize robustly; zero/nonfinite embeddings are errors, never valid faces."""
    if not np.isfinite(array).all():
        raise ValueError("Embedding contains NaN or Infinity")
    lengths = np.linalg.norm(array, axis=1, keepdims=True)
    if (lengths < 1e-12).any():
        raise ValueError("Zero embedding encountered")
    return array / lengths


def torch_input(images: np.ndarray, normalization: str, channel_order: str) -> np.ndarray:
    """Explicit color/normalization contract; compare with your checkpoint evaluator.

    Standard ArcFace-style transform is (RGB/255 - .5)/.5. Legacy evaluators may
    use (RGB - 127.5)/128; both are selectable and recorded, never mixed silently.
    """
    array = images.astype(np.float32)
    if channel_order == "BGR":
        array = array[..., ::-1]
    array = array / 127.5 - 1.0 if normalization == "half" else (array - 127.5) / 128.0
    return np.ascontiguousarray(array.transpose(0, 3, 1, 2))


def maskinv_embeddings(frame: pd.DataFrame, out: Path, args) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Batch only calibration/test images to avoid needless 90k-image CNN inference."""
    model, provenance = load_maskinv(args)
    import torch
    selected = frame[frame.split.isin(["calibration", "test"])].reset_index(drop=True)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            batch = selected.iloc[start:start + args.batch_size]
            images = []
            for row in batch.itertuples():
                with Image.open(args.cache_root / row.processed_path) as image:
                    images.append(np.asarray(image.convert("RGB")))
            tensor = torch.from_numpy(torch_input(np.stack(images), args.normalization, args.channel_order)).to(args.device)
            embeddings = model(tensor)
            if args.flip_test:
                embeddings = embeddings + model(torch.flip(tensor, dims=[3]))
            vectors.append(embeddings.float().cpu().numpy())
            LOGGER.info("Embedded %d/%d verification images", min(start + args.batch_size, len(selected)), len(selected))
    return selected, normalize_embeddings(np.concatenate(vectors)), provenance


def pixel_pca_embeddings(frame: pd.DataFrame, out: Path, args) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Optional small sanity baseline; never presented as the requested KD model.

    Fit PCA only on retained training identities, cap memory through fixed-seed
    sampling, and transform held-out images. Gray 32x32 features need no external
    model weights, enabling an end-to-end pipeline check without a GPU.
    """
    train = frame[frame.split == "train"]
    if len(train) < 3:
        raise ValueError("Pixel/PCA sanity baseline needs >=3 retained training images")
    train = train.sample(min(len(train), args.pca_fit_cap), random_state=args.seed)
    selected = frame[frame.split.isin(["calibration", "test"])].reset_index(drop=True)
    def pixels(rows):
        result = []
        for row in rows.itertuples():
            with Image.open(args.cache_root / row.processed_path) as image:
                result.append(np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32).ravel() / 255.0)
        return np.stack(result)
    x_train = pixels(train)
    components = min(args.pca_components, len(train) - 1, x_train.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=args.seed)
    pca.fit(x_train)
    embeddings = normalize_embeddings(pca.transform(pixels(selected)))
    np.savez_compressed(out / "pca_fit.npz", components=pca.components_, mean=pca.mean_, explained_variance_ratio=pca.explained_variance_ratio_)
    train[["sample_id", "identity"]].to_csv(out / "pca_training_sample.csv", index=False)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, components + 1), cumulative, marker="o", markersize=3)
    ax.set(title="Train-only pixel PCA reduction", xlabel="Principal components", ylabel="Cumulative explained variance")
    save_figure(fig, out / "figures", "08_pca_variance", "PCA is fitted only on retained training identities; the variance curve documents the optional pixel sanity baseline's reduction.")
    return selected, embeddings, {"method": "pixel-PCA sanity baseline (NOT MaskInv)", "pca_components": components,
                                  "train_fit_images": len(train), "train_identity_count": int(train.identity.nunique()),
                                  "training": "train-only unsupervised PCA; calibration-only threshold fitting"}


# ============================================================================
# IV-B. Deterministic, bounded, identity-balanced verification-pair sampling
# ============================================================================
def sample_pairs(frame: pd.DataFrame, split: str, scenario: str, seed: int,
                 per_identity: int, maximum: int) -> tuple[pd.DataFrame, dict]:
    """Create capped genuine and balanced impostor pairs without quadratic lists.

    Identity is the verification label: same=1, different=0. Mask state defines
    NM/MM/NN conditions, not the identity target. Genuine candidates are sampled
    per identity; impostor identities are drawn uniformly from eligible pools.
    Pairs may share endpoints, so these are not independent statistical trials.
    """
    rng = np.random.default_rng(seed)
    a_condition, b_condition = CONDITIONS[scenario]
    rows = frame[frame.split == split]
    pools = {}
    for identity, group in rows.groupby("identity", sort=True):
        pools[identity] = {c: sorted(group.loc[group.condition == c, "sample_id"]) for c in ["masked", "unmasked"]}
    genuine = []
    for identity, pool in sorted(pools.items()):
        left, right = pool[a_condition], pool[b_condition]
        same_condition = a_condition == b_condition
        possible = len(left) * (len(left) - 1) // 2 if same_condition else len(left) * len(right)
        target = min(possible, per_identity)
        selected = set()
        if possible <= per_identity:
            selected = set(itertools.combinations(left, 2)) if same_condition else set(itertools.product(left, right))
        else:
            # Only O(per_identity) storage, even for identities with many photos.
            for _ in range(max(100, target * 50)):
                if len(selected) >= target:
                    break
                a, b = left[int(rng.integers(len(left)))], right[int(rng.integers(len(right)))]
                if a == b:
                    continue
                selected.add(tuple(sorted((a, b))) if same_condition else (a, b))
        genuine.extend((a, b, 1) for a, b in sorted(selected))
    if len(genuine) > maximum:
        indices = rng.choice(len(genuine), size=maximum, replace=False)
        genuine = [genuine[i] for i in sorted(indices)]
    a_ids = [i for i in sorted(pools) if pools[i][a_condition]]
    b_ids = [i for i in sorted(pools) if pools[i][b_condition]]
    impostors = set()
    target = len(genuine)
    for _ in range(max(100, target * 100)):
        if len(impostors) >= target or not a_ids or not b_ids:
            break
        ai, bi = a_ids[int(rng.integers(len(a_ids)))], b_ids[int(rng.integers(len(b_ids)))]
        if ai == bi:
            continue
        a = pools[ai][a_condition][int(rng.integers(len(pools[ai][a_condition])))]
        b = pools[bi][b_condition][int(rng.integers(len(pools[bi][b_condition])))]
        impostors.add(tuple(sorted((a, b))) if a_condition == b_condition else (a, b))
    # Downsample genuine if the available impostor space is smaller. Report it.
    positives_before_balance = len(genuine)
    if len(impostors) < len(genuine):
        indices = rng.choice(len(genuine), size=len(impostors), replace=False)
        genuine = [genuine[i] for i in sorted(indices)]
    pairs = genuine + [(a, b, 0) for a, b in sorted(impostors)]
    rng.shuffle(pairs)
    result = pd.DataFrame(pairs, columns=["sample_a", "sample_b", "genuine"])
    lookup = frame.set_index("sample_id")
    for endpoint in ["a", "b"]:
        result[f"identity_{endpoint}"] = result[f"sample_{endpoint}"].map(lookup.identity)
    result["split"], result["scenario"] = split, scenario
    diagnostics = {"split": split, "scenario": scenario, "genuine": len(genuine), "impostor": len(impostors),
                   "genuine_before_final_balance": positives_before_balance,
                   "genuine_identity_count": int(result.loc[result.genuine == 1, "identity_a"].nunique()),
                   "total_split_identities": len(pools), "pairs_per_identity_cap": per_identity,
                   "requested_per_class_cap": maximum,
                   "protocol": "custom fixed-seed pairs; NOT the supplied masked_pairs.txt benchmark protocol"}
    return result, diagnostics


def attach_scores(pairs: pd.DataFrame, samples: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """Cosine similarity of unit embeddings, with vectorized O(number of pairs) work."""
    index = {sid: i for i, sid in enumerate(samples.sample_id)}
    a = [index[sid] for sid in pairs.sample_a]
    b = [index[sid] for sid in pairs.sample_b]
    result = pairs.copy()
    result["score"] = np.clip(np.einsum("ij,ij->i", embeddings[a], embeddings[b]).astype(np.float64), -1.0, 1.0)
    return result


# ============================================================================
# IV-C. Count-backed metrics, calibration-only operating points, and ROC/EER
# ============================================================================
def threshold_table(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """All ROC thresholds with their integer counts and corresponding FAR/FRR."""
    scores = np.asarray(scores, dtype=np.float64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold analysis requires both genuine and impostor pairs")
    far, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    thresholds = np.where(np.isfinite(thresholds), thresholds, np.nextafter(float(scores.max()), math.inf))
    n_genuine, n_impostor = int((labels == 1).sum()), int((labels == 0).sum())
    false_accept = np.rint(far * n_impostor).astype(int)
    true_accept = np.rint(tpr * n_genuine).astype(int)
    return pd.DataFrame({"threshold": thresholds, "false_accept": false_accept,
                         "impostor_count": n_impostor, "false_reject": n_genuine - true_accept,
                         "genuine_count": n_genuine, "FAR": far, "FRR": 1.0 - tpr, "TAR": tpr})


def eer_summary(table: pd.DataFrame) -> dict:
    """Linearly interpolate ROC crossing; also expose closest realizable point.

    Interpolated EER is a descriptive curve statistic, not a deployable threshold.
    A deterministic threshold can only achieve one of the empirical table rows.
    """
    difference = table.FAR.to_numpy() - table.FRR.to_numpy()
    closest = int(np.argmin(np.abs(difference)))
    crossings = np.flatnonzero(difference >= 0)
    right = int(crossings[0])
    if right == 0 or difference[right] == 0:
        value = float(table.FAR.iloc[right])
    else:
        left = right - 1
        weight = -difference[left] / (difference[right] - difference[left])
        value = float(table.FAR.iloc[left] + weight * (table.FAR.iloc[right] - table.FAR.iloc[left]))
    return {"interpolated_EER": value, "closest_empirical_threshold": float(table.threshold.iloc[closest]),
            "closest_empirical_FAR": float(table.FAR.iloc[closest]), "closest_empirical_FRR": float(table.FRR.iloc[closest]),
            "genuine_count": int(table.genuine_count.iloc[0]), "impostor_count": int(table.impostor_count.iloc[0]),
            "interpretation": "Interpolated curve statistic; see threshold CSV for integer-count realizable points."}


def select_threshold(calibration: pd.DataFrame, target_far: float) -> float:
    """Choose best calibration TAR subject to empirical FAR <= the requested rate.

    ROC rows are ordered by decreasing threshold. argmax resolves ties toward the
    most conservative threshold. Reject-all is allowed when evidence is limited.
    """
    eligible = calibration[calibration.FAR <= target_far + 1e-15]
    best = int(np.argmax(eligible.TAR.to_numpy()))
    return float(eligible.threshold.iloc[best])


def metrics_at_threshold(pairs: pd.DataFrame, threshold: float) -> dict:
    """Return numerator/denominator for EVERY reported empirical rate."""
    labels = pairs.genuine.to_numpy(dtype=int)
    accept = pairs.score.to_numpy(dtype=np.float64) >= threshold
    tp = int(((labels == 1) & accept).sum())
    fn = int(((labels == 1) & ~accept).sum())
    fp = int(((labels == 0) & accept).sum())
    tn = int(((labels == 0) & ~accept).sum())
    def rate(numerator, denominator):
        return {"value": numerator / denominator if denominator else None,
                "numerator": numerator, "denominator": denominator}
    return {"threshold": threshold, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "FAR": rate(fp, fp + tn), "FRR": rate(fn, fn + tp), "TAR": rate(tp, fn + tp),
            "accuracy": rate(tp + tn, len(labels)), "precision": rate(tp, tp + fp),
            "F1": rate(2 * tp, 2 * tp + fp + fn), "specificity": rate(tn, tn + fp),
            "balanced_accuracy": {"value": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                                  "components": [rate(tp, tp + fn), rate(tn, tn + fp)]}}


def error_analysis(pairs: pd.DataFrame, frame: pd.DataFrame, threshold: float, out: Path,
                   cache_root: Path, scenario: str) -> dict:
    """Export failures, macro identity FRR, quality bins, and inspectable examples.

    Quality hypotheses are descriptive, not established causes. Pair endpoints are
    dependent; no false precision from an independent-pair confidence interval.
    """
    lookup = frame.set_index("sample_id")
    detail = pairs.copy()
    detail["accepted"] = detail.score.to_numpy(dtype=np.float64) >= threshold
    detail["error"] = detail.accepted != detail.genuine.astype(bool)
    detail["error_type"] = np.where(~detail.error, "correct", np.where(detail.genuine == 1, "false_reject", "false_accept"))
    for endpoint in ["a", "b"]:
        for column in ["relative_path", "condition", "brightness", "sharpness", "contrast"]:
            detail[f"{column}_{endpoint}"] = detail[f"sample_{endpoint}"].map(lookup[column])
    detail.to_csv(out / f"{scenario}_test_predictions.csv", index=False)
    failures = detail[detail.error]
    failures.to_csv(out / f"{scenario}_errors.csv", index=False)
    genuine = detail[detail.genuine == 1]
    by_identity = genuine.groupby("identity_a").agg(false_rejects=("error", "sum"), genuine_pairs=("error", "size"))
    by_identity["FRR"] = by_identity.false_rejects / by_identity.genuine_pairs
    by_identity.to_csv(out / f"{scenario}_per_identity_FRR.csv")
    # Probe endpoint b defines bins. Fixed bins avoid post-hoc tuning from errors.
    detail["brightness_bin"] = pd.cut(detail.brightness_b, bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=["dark", "lower-mid", "upper-mid", "bright"])
    by_quality = detail.groupby(["genuine", "brightness_bin"], observed=True).agg(errors=("error", "sum"), pairs=("error", "size"))
    by_quality["error_rate"] = by_quality.errors / by_quality.pairs
    by_quality.to_csv(out / f"{scenario}_errors_by_brightness.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name in [(0, "FAR (impostor)"), (1, "FRR (genuine)")]:
        if label in by_quality.index.get_level_values(0):
            part = by_quality.loc[label]
            ax.plot(part.index.astype(str), part.error_rate, marker="o", label=name)
    ax.set(title=f"{scenario}: errors by probe brightness", xlabel="Probe mean intensity bin", ylabel="Error rate (counts in CSV)", ylim=(-0.02, 1.02))
    ax.legend()
    save_figure(fig, out / "figures", f"{scenario}_error_quality", "Brightness-stratified errors suggest whether exposure may contribute to failures; this association is not a causal explanation.")
    if len(failures):
        # Show strongest mistaken matches and weakest mistaken genuine pairs.
        examples = pd.concat([failures[failures.genuine == 0].nlargest(3, "score"), failures[failures.genuine == 1].nsmallest(3, "score")])
        fig, axes = plt.subplots(len(examples), 2, figsize=(7, 2.7 * len(examples)), squeeze=False)
        for axes_row, row in zip(axes, examples.itertuples()):
            for ax, sid, endpoint in zip(axes_row, [row.sample_a, row.sample_b], ["A", "B"]):
                with Image.open(cache_root / lookup.loc[sid, "processed_path"]) as image:
                    ax.imshow(image)
                ax.set(title=f"{row.error_type}: {endpoint}; score={row.score:.3f}", xlabel="Pixel x", ylabel="Pixel y")
        save_figure(fig, out / "figures", f"{scenario}_error_examples", "High-score impostor acceptances and low-score genuine rejections provide concrete cases for manual label, pose, occlusion, and alignment review.")
    return {"false_accept_count": int(((detail.genuine == 0) & detail.error).sum()),
            "false_reject_count": int(((detail.genuine == 1) & detail.error).sum()),
            "macro_identity_FRR": {"value": float(by_identity.FRR.mean()),
                                   "sum_identity_FRR": float(by_identity.FRR.sum()), "identity_count": len(by_identity)},
            "hypotheses_to_check": ["Occlusion may remove identity cues.", "Pose, exposure, and alignment errors may degrade similarity.",
                                    "Large genuine errors or strong impostor matches may reflect incorrect identity labels."],
            "causal_status": "Hypotheses only; inspect exported cases before drawing conclusions."}


def baseline_figures(cal: pd.DataFrame, test: pd.DataFrame, ctable: pd.DataFrame, ttable: pd.DataFrame,
                     threshold: float, metrics: dict, scenario: str, out: Path) -> None:
    """Score separation, ROC, FAR/FRR thresholds, confusion matrix, pair balance."""
    figures = out / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pairs, name in zip(axes, [cal, test], ["Calibration", "Test"]):
        for label, color, title in [(0, "#dc2626", "Impostor"), (1, "#16a34a", "Genuine")]:
            ax.hist(pairs.loc[pairs.genuine == label, "score"], bins=30, alpha=0.5, density=True, color=color, label=title)
        ax.axvline(threshold, color="black", linestyle="--", label="Frozen threshold")
        ax.set(title=f"{scenario}: {name} cosine scores", xlabel="Cosine similarity", ylabel="Density")
        ax.legend(fontsize=8)
    save_figure(fig, figures, f"{scenario}_score_distributions", "Greater genuine/impostor separation makes verification easier; the marked operating threshold is selected only on calibration identities.")
    fig, ax = plt.subplots(figsize=(7, 5))
    for table, name in [(ctable, "Calibration"), (ttable, "Test")]:
        ax.plot(table.FAR, table.TAR, label=f"{name}, AUC={auc(table.FAR, table.TAR):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.scatter(metrics["FAR"]["value"], metrics["TAR"]["value"], c="black", label="Test at frozen threshold", zorder=4)
    ax.set(title=f"{scenario}: verification ROC", xlabel="False acceptance rate", ylabel="True acceptance rate")
    ax.legend()
    save_figure(fig, figures, f"{scenario}_roc", "The ROC displays discrimination while the marked test point evaluates the calibration-selected security operating point.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, table, name in zip(axes, [ctable, ttable], ["Calibration", "Test diagnostic"]):
        ax.plot(table.threshold, table.FAR, label="FAR")
        ax.plot(table.threshold, table.FRR, label="FRR")
        ax.axvline(threshold, color="black", linestyle="--", label="Frozen threshold")
        ax.set(title=f"{scenario}: {name} threshold tradeoff", xlabel="Cosine acceptance threshold", ylabel="Error rate")
        ax.legend()
    save_figure(fig, figures, f"{scenario}_far_frr", "Raising the acceptance threshold trades fewer false acceptances for more false rejections; test curves are diagnostic and never used for threshold choice.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    matrix = np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])
    axes[0].imshow(matrix, cmap="Blues")
    for (r, c), count in np.ndenumerate(matrix):
        axes[0].text(c, r, str(count), ha="center", va="center", color="black", bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
    axes[0].set(title=f"{scenario}: test confusion counts", xticks=[0, 1], yticks=[0, 1], xticklabels=["Reject", "Accept"],
                yticklabels=["Impostor", "Genuine"], xlabel="Decision", ylabel="Ground truth")
    counts = pd.DataFrame({"Calibration": cal.genuine.value_counts(), "Test": test.genuine.value_counts()}).fillna(0)
    counts.index = ["Impostor" if i == 0 else "Genuine" for i in counts.index]
    counts.plot.bar(ax=axes[1])
    axes[1].set(title=f"{scenario}: verification class balance", xlabel="Pair ground truth", ylabel="Pair count")
    save_figure(fig, figures, f"{scenario}_confusion_and_balance", "Integer confusion counts and pair class counts support every rate; balanced sampled pairs do not imply real-world genuine/impostor prevalence.")


def evaluate_baseline(frame: pd.DataFrame, out: Path, args) -> dict:
    """Extract requested features, freeze thresholds, then evaluate held-out pairs."""
    if args.baseline == "maskinv":
        samples, embeddings, provenance = maskinv_embeddings(frame, out, args)
    elif args.baseline == "pixel-pca":
        samples, embeddings, provenance = pixel_pca_embeddings(frame, out, args)
    else:
        return {}
    np.savez_compressed(out / "embeddings.npz", sample_ids=samples.sample_id.to_numpy(dtype=str), embeddings=embeddings)
    result = {"provenance": provenance, "seed": args.seed, "target_calibration_FAR": args.target_far,
              "threshold_rationale": "Maximize empirical calibration TAR while FAR is at or below the requested budget; freeze before testing.",
              "pair_dependence": "Pairs share identities/images; rate denominators are pair counts, not independent people.",
              "scope": "Within-dataset transfer evaluation, not reproduction of the paper benchmark or KD training.",
              "scenarios": {}}
    for scenario_number, scenario in enumerate(args.scenarios):
        pair_sets, tables, diagnostics = {}, {}, []
        for number, split in enumerate(["calibration", "test"]):
            pairs, diagnostic = sample_pairs(frame, split, scenario, args.seed + 100 * scenario_number + number,
                                            args.pairs_per_identity, args.max_pairs_per_class)
            diagnostics.append(diagnostic)
            pairs.to_csv(out / f"{scenario}_{split}_pairs.csv", index=False)
            if pairs.empty or pairs.genuine.nunique() != 2:
                break
            scored = attach_scores(pairs, samples, embeddings)
            scored.to_csv(out / f"{scenario}_{split}_scores.csv", index=False)
            pair_sets[split] = scored
            tables[split] = threshold_table(scored.genuine.to_numpy(), scored.score.to_numpy())
            tables[split].to_csv(out / f"{scenario}_{split}_thresholds.csv", index=False)
        if len(pair_sets) != 2:
            result["scenarios"][scenario] = {"status": "unavailable", "reason": "Insufficient distinct genuine/impostor pairs in calibration or test.", "pair_diagnostics": diagnostics}
            LOGGER.warning("Skipping %s: insufficient pairs; NN usually has one reference per identity", scenario)
            continue
        threshold = select_threshold(tables["calibration"], args.target_far)
        metrics = {split: metrics_at_threshold(pairs, threshold) for split, pairs in pair_sets.items()}
        scenario_result = {"status": "evaluated", "pair_diagnostics": diagnostics,
                           "calibration_operating_point": metrics["calibration"], "test_at_frozen_threshold": metrics["test"],
                           "calibration_EER": eer_summary(tables["calibration"]), "test_EER_diagnostic": eer_summary(tables["test"]),
                           "ROC_AUC": {split: {"value": float(auc(table.FAR, table.TAR)),
                                               "genuine_count": int(table.genuine_count.iloc[0]),
                                               "impostor_count": int(table.impostor_count.iloc[0])}
                                       for split, table in tables.items()},
                           "FAR_resolution": {split: 1.0 / int(table.impostor_count.iloc[0]) for split, table in tables.items()}}
        scenario_result["error_analysis"] = error_analysis(pair_sets["test"], frame, threshold, out, args.cache_root, scenario)
        baseline_figures(pair_sets["calibration"], pair_sets["test"], tables["calibration"], tables["test"], threshold, metrics["test"], scenario, out)
        result["scenarios"][scenario] = scenario_result
    write_json(out / "baseline_results.json", result)
    if not any(item["status"] == "evaluated" for item in result["scenarios"].values()):
        raise ValueError("No scenario could be evaluated; inspect baseline_results.json and pair coverage")
    return result


# ============================================================================
# I-IV. Reviewable report scaffold and end-user CLI
# ============================================================================
def build_report(out: Path, summary: dict, result: dict, metadata: dict) -> None:
    """Generate a standalone HTML report scaffold with measured outputs and figures.
    """
    import html
    escape = lambda value: html.escape(str(value))
    blocks = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Masked-face checkpoint outputs</title>",
              "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172033}img{max-width:100%}pre{white-space:pre-wrap;background:#f1f5f9;padding:18px}figure{margin:30px 0}h2{margin-top:45px}</style>",
              "<h1>Masked-face verification: checkpoint evidence</h1>",
              "<p>Review scaffold. Add team, proposal changes, verified provenance and interpretation; prepare the final 3-4 page IEEE/ACM PDF separately.</p>",
              "<h2>I. Problem statement and motivation</h2>",
              "<p>Test the feasibility of verifying a claimed identity when a probe face is masked, using an unmasked reference. False acceptance can permit unauthorized access; false rejection can deny a legitimate user. Compare NM with MM and NN where sufficient images exist.</p>",
              "<p>Team action: state project nature, security setting and any scope changes since the proposal in a different text color.</p>",
              "<h2>II. Data engineering</h2>", "<h3>Supplied provenance metadata (verify before submission)</h3>",
              "<pre>" + escape(json.dumps(metadata, indent=2)) + "</pre>",
              "<h3>Measured dataset summary</h3><pre>" + escape(json.dumps(summary, indent=2)) + "</pre>",
              "<p>EXIF orientation, strict RGB decoding, minimum-size validation, configured 112x112 preprocessing, exact-duplicate quarantine, and identity-disjoint splits are recorded in manifest.csv. Sessions, demographics and live/spoof labels are unknown.</p>",
              "<h2>III. Exploratory data analysis</h2>"]
    for path in sorted((out / "figures").glob("0*.png")):
        blocks.append(f"<figure><img src='figures/{escape(path.name)}' alt='{escape(path.stem)}'><figcaption>{escape(CAPTIONS.get(path.stem, path.stem))}</figcaption></figure>")
    blocks.append("<h2>IV. Baseline implementation and preliminary results</h2>")
    if result:
        blocks.append("<pre>" + escape(json.dumps(result, indent=2)) + "</pre>")
        blocks.append("<p>Every operating rate includes counts. EER uses interpolation; inspect threshold CSVs for realizable rates. A zero observed FAR does not establish zero population risk, particularly when the impostor count is small. These custom sampled-pair results are not directly comparable to the paper's published benchmark numbers.</p>")
        for path in sorted((out / "figures").glob("*.png")):
            if not path.name.startswith("0"):
                blocks.append(f"<figure><img src='figures/{escape(path.name)}' alt='{escape(path.stem)}'><figcaption>{escape(CAPTIONS.get(path.stem, path.stem))}</figcaption></figure>")
    else:
        blocks.append("<p>Baseline has not been run. Supply the upstream pretrained MaskInv checkpoint and execute the baseline command in README.md. No performance numbers are fabricated.</p>")
    blocks.extend(["<h2>Reproducibility and references</h2>", "<p>See run_config.json, environment.json, manifest.csv, pair CSVs, model hashes and source_snapshot.py. Commit the reviewed code and record its Git commit.</p>",
                   "<p>Huber et al., Mask-invariant Face Recognition through Template-level Knowledge Distillation, FG 2021, DOI: 10.1109/FG52635.2021.9667081. <a href='https://github.com/fdbtrs/Masked-Face-Recognition-KD'>Official repository</a>.</p></html>"])
    (out / "report.html").write_text("\n".join(blocks), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    """Define all options in one documented CLI; a fresh output directory is required."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, help="Dataset containing train/ and verification/")
    p.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    p.add_argument("--structure", type=Path, help="Optional TREE /F listing; alone produces listing-only inventory")
    p.add_argument("--evaluate-run", type=Path, help="Reuse an existing prepared run in a new output directory")
    p.add_argument("--metadata", type=Path, help="Verified dataset provenance JSON")
    p.add_argument("--aliases", type=Path, help="Audited origin,folder,identity CSV")
    p.add_argument("--alignment", choices=["pad", "mtcnn", "prealigned"], default="pad")
    p.add_argument("--multiple-faces", choices=["reject", "largest"], default="reject")
    p.add_argument("--min-side", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-fraction", type=float, default=0.3)
    p.add_argument("--baseline", choices=["none", "maskinv", "pixel-pca"], default="none")
    p.add_argument("--repo", type=Path, help="Local clone of official Masked-Face-Recognition-KD repository")
    p.add_argument("--checkpoint", type=Path, help="Official pretrained MaskInv backbone state dictionary")
    p.add_argument("--device", default="cpu", help="Torch device: cpu, cuda, cuda:0, ...")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--normalization", choices=["half", "arcface128"], default="half")
    p.add_argument("--channel-order", choices=["RGB", "BGR"], default="RGB")
    p.add_argument("--flip-test", action="store_true", help="Sum original and flipped embeddings before unit normalization")
    p.add_argument("--target-far", type=float, default=0.01)
    p.add_argument("--scenarios", nargs="+", choices=list(CONDITIONS), default=["NM", "MM", "NN"])
    p.add_argument("--pairs-per-identity", type=int, default=20)
    p.add_argument("--max-pairs-per-class", type=int, default=10000)
    p.add_argument("--pca-components", type=int, default=64)
    p.add_argument("--pca-fit-cap", type=int, default=5000)
    return p


def main(argv=None) -> int:
    """Run listing inventory, preprocessing/EDA, or evaluation of a prepared run."""
    import shutil
    args = parser().parse_args(argv)
    CAPTIONS.clear()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not 0 < args.calibration_fraction < 1 or not 0 <= args.target_far < 1:
        raise ValueError("Calibration fraction must be in (0,1) and target FAR in [0,1)")
    for value in [args.min_side, args.batch_size, args.pairs_per_identity, args.max_pairs_per_class,
                  args.pca_components, args.pca_fit_cap, args.torch_threads]:
        if value < 1:
            raise ValueError("Sizes, counts and thread limits must be positive")
    if args.pca_fit_cap < 3:
        raise ValueError("--pca-fit-cap must be at least 3")
    if len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("Each scenario may be specified only once")
    if args.evaluate_run and (args.data_root or args.structure or args.aliases):
        raise ValueError("--evaluate-run reuses the source manifest; do not also supply input data/aliases/structure")
    if args.evaluate_run and args.baseline == "none":
        raise ValueError("--evaluate-run requires a baseline")
    if not args.data_root and not args.structure and not args.evaluate_run:
        raise ValueError("Supply --data-root, --structure or --evaluate-run")
    # Required before the first CUDA call, including optional face detection.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    aliases = read_aliases(args.aliases)
    if args.structure:
        result = structure_inventory(args.structure, out, aliases)
        LOGGER.info("Listing inventory: %s", out / "structure_inventory.json")
        if not args.data_root:
            return 0
    write_json(out / "environment.json", package_versions())
    shutil.copy2(Path(__file__), out / "source_snapshot.py")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else {
        "name": "unknown", "origin_url": "unknown", "citation": "unknown", "license": "unknown",
        "access_terms": "unknown", "sessions": "unknown", "identity_mapping_verified": False}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["source_git"] = git_state(Path(__file__).parent)
    config["source_sha256"] = sha256_file(Path(__file__))
    config["dataset_metadata"] = metadata
    if args.evaluate_run:
        source = args.evaluate_run.resolve()
        old_config = json.loads((source / "run_config.json").read_text())
        if args.seed != old_config["seed"]:
            raise ValueError("Reuse the prepared run seed via --seed to preserve its documented protocol")
        args.cache_root = Path(old_config.get("cache_root", str(source)))
        args.alignment = old_config["alignment"]
        config.update(dataset_root=old_config["dataset_root"], alignment=args.alignment, cache_root=str(args.cache_root))
        if not args.metadata:
            metadata = old_config["dataset_metadata"]
            config["dataset_metadata"] = metadata
        frame = pd.read_csv(source / "manifest.csv", keep_default_na=False)
        for column in FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        assert_disjoint(frame)
        summary = json.loads((source / "data_summary.json").read_text())
        shutil.copy2(source / "run_config.json", out / "preparation_config.json")
        shutil.copy2(source / "environment.json", out / "preparation_environment.json")
        shutil.copy2(source / "manifest.csv", out / "manifest.csv")
        shutil.copy2(source / "data_summary.json", out / "data_summary.json")
        if (source / "figures").exists():
            # Only copy preparation figures, not an earlier baseline's evaluation.
            (out / "figures").mkdir()
            for path in (source / "figures").glob("0*"):
                if not path.name.startswith("08_"):
                    shutil.copy2(path, out / "figures" / path.name)
            caption_path = source / "figures" / "captions.json"
            if caption_path.exists():
                CAPTIONS.update({k: v for k, v in json.loads(caption_path.read_text()).items() if k.startswith("0") and not k.startswith("08_")})
    else:
        root = args.data_root.resolve()
        args.cache_root = out
        config.update(dataset_root=str(root), cache_root=str(out))
        write_json(out / "run_config.json", config)
        if args.alignment == "mtcnn":
            configure_torch(args.seed, args.torch_threads)
        frame, issues = scan_dataset(root, aliases)
        issues.to_csv(out / "schema_issues.csv", index=False)
        pair_file = root / "verification" / "masked_pairs.txt"
        config["supplied_pairs_file"] = {"present": pair_file.exists(), "sha256": sha256_file(pair_file) if pair_file.exists() else None,
                                         "used": False, "reason": "Unknown format; custom disjoint protocol generated instead. See README."}
        frame = audit_images(frame, root, out, args)
        frame, duplicates = quarantine_duplicates(frame)
        duplicates.to_csv(out / "duplicate_audit.csv", index=False)
        # Save audit even if too few usable identities prevent a valid split.
        frame.to_csv(out / "image_audit.csv", index=False)
        frame = assign_splits(frame, args.seed, args.calibration_fraction)
        frame.to_csv(out / "manifest.csv", index=False)
        summary = quality_tables(frame, out)
        run_eda(frame, out, args.seed)
    write_json(out / "run_config.json", config)
    if args.baseline == "maskinv" and args.alignment == "pad":
        LOGGER.warning("Padding does not align faces. Treat pad-mode model results as a preprocessing ablation, not an aligned MaskInv reference.")
    # Both embedding extractors resolve cached images through the prepared run.
    result = evaluate_baseline(frame, out, args)
    build_report(out, summary, result, metadata)
    LOGGER.info("Complete: %s", out / "report.html")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2)
