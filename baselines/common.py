"""Shared Sen1Floods11 hand-set IO, metrics, and viz helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import rasterio
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install rasterio: pip install rasterio") from e


IGNORE_INDEX = -1
WATER = 1
NOT_WATER = 0

METRIC_PRINT_KEYS = (
    "water_iou",
    "miou",
    "precision_water",
    "recall_water",
    "f1_water",
    "pixel_acc",
)

# ImageNet stats used by SegFormer / mit-* checkpoints
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_split_stems(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, header=None)
    col = df.iloc[:, 0].astype(str).str.strip()
    stems = []
    for raw in col:
        if not raw or raw.lower() in {"nan", "none"}:
            continue
        stem = Path(raw).stem
        stem = re.sub(r"_(S1Hand|LabelHand|S2Hand)$", "", stem)
        stems.append(stem)
    return stems


def resolve_pair(
    data_root: Path, s1_dir: str, label_dir: str, stem: str
) -> tuple[Path, Path]:
    s1 = data_root / s1_dir / f"{stem}_S1Hand.tif"
    lab = data_root / label_dir / f"{stem}_LabelHand.tif"
    if not s1.exists():
        matches = sorted((data_root / s1_dir).glob(f"{stem}*.tif"))
        if not matches:
            raise FileNotFoundError(f"Missing S1 chip for stem={stem}: {s1}")
        s1 = matches[0]
    if not lab.exists():
        matches = sorted((data_root / label_dir).glob(f"{stem}*.tif"))
        if not matches:
            raise FileNotFoundError(f"Missing label for stem={stem}: {lab}")
        lab = matches[0]
    return s1, lab


def load_s1_vv_vh(s1_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return VV, VH float32 arrays (band1=VV, band2=VH)."""
    with rasterio.open(s1_path) as src:
        if src.count < 2:
            raise ValueError(f"{s1_path} has {src.count} bands; need VV+VH")
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
    return vv, vh


def load_label(label_path: Path) -> np.ndarray:
    with rasterio.open(label_path) as src:
        return src.read(1).astype(np.int16)


def load_s1_and_label(
    s1_path: Path, label_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vv, vh, label); raises if shapes mismatch."""
    vv, vh = load_s1_vv_vh(s1_path)
    label = load_label(label_path)
    if vv.shape != label.shape or vh.shape != label.shape:
        raise ValueError(
            f"Shape mismatch {s1_path.name}: vv={vv.shape} vh={vh.shape} vs {label.shape}"
        )
    return vv, vh, label


def load_vh_and_label(s1_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vh, label) for VH-threshold baseline."""
    _, vh, label = load_s1_and_label(s1_path, label_path)
    return vh, label


def empty_confusion() -> dict[str, int]:
    return {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


def add_confusion(totals: dict[str, int], counts: dict[str, int]) -> None:
    for k in totals:
        totals[k] += counts[k]


def confusion_counts(pred: np.ndarray, label: np.ndarray) -> dict[str, int]:
    valid = label != IGNORE_INDEX
    p = pred[valid].astype(np.int32)
    y = label[valid].astype(np.int32)
    return {
        "tp": int(((p == WATER) & (y == WATER)).sum()),
        "tn": int(((p == NOT_WATER) & (y == NOT_WATER)).sum()),
        "fp": int(((p == WATER) & (y == NOT_WATER)).sum()),
        "fn": int(((p == NOT_WATER) & (y == WATER)).sum()),
    }


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    water_iou = tp / (tp + fp + fn + eps)
    not_water_iou = tn / (tn + fp + fn + eps)
    miou = 0.5 * (water_iou + not_water_iou)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "water_iou": float(water_iou),
        "not_water_iou": float(not_water_iou),
        "miou": float(miou),
        "precision_water": float(precision),
        "recall_water": float(recall),
        "f1_water": float(f1),
        "pixel_acc": float(pixel_acc),
        **{k: int(v) for k, v in c.items()},
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for k in METRIC_PRINT_KEYS:
        print(f"  {k:18s} {metrics[k]:.4f}")


def save_metrics_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def percentile_stretch(x: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[valid], [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def make_error_map(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """RGB: green=TP, red=FP, blue=FN, gray=TN, black=ignore."""
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    valid = label != IGNORE_INDEX
    tp = valid & (pred == WATER) & (label == WATER)
    tn = valid & (pred == NOT_WATER) & (label == NOT_WATER)
    fp = valid & (pred == WATER) & (label == NOT_WATER)
    fn = valid & (pred == NOT_WATER) & (label == WATER)
    rgb[tp] = (0.2, 0.8, 0.2)
    rgb[tn] = (0.55, 0.55, 0.55)
    rgb[fp] = (0.9, 0.2, 0.2)
    rgb[fn] = (0.2, 0.4, 0.95)
    return rgb


def make_gt_rgb(label: np.ndarray) -> np.ndarray:
    """RGB: white=not-water, blue=water, black=no-data (-1)."""
    h, w = label.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    rgb[label == WATER] = (0.12, 0.35, 0.85)
    rgb[label == IGNORE_INDEX] = (0.0, 0.0, 0.0)
    return rgb


def choose_stems_for_viz(
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    stems: list[str],
    num_viz: int,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng() if seed == -1 else np.random.default_rng(seed)
    available = []
    for stem in stems:
        try:
            resolve_pair(data_root, s1_dir, label_dir, stem)
            available.append(stem)
        except FileNotFoundError:
            continue
    if not available:
        return []
    n = min(num_viz, len(available))
    return list(rng.choice(available, size=n, replace=False))


def visualize_prediction(
    stem: str,
    sar_panel: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    *,
    sar_title: str,
    pred_title: str,
    model_input: np.ndarray | None = None,
    model_input_title: str = "Model input",
) -> None:
    """Panels: SAR [/ model input] / GT / pred / error map.

    ``model_input`` should be H×W×3 in [0, 1] (e.g. VV/VH/VH before ImageNet norm).
    """
    chip_metrics = metrics_from_counts(confusion_counts(pred, label))
    n_panels = 5 if model_input is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.6))
    axes[0].imshow(percentile_stretch(sar_panel), cmap="gray")
    axes[0].set_title(f"{sar_title}\n{stem}")
    col = 1
    if model_input is not None:
        axes[col].imshow(np.clip(model_input, 0.0, 1.0))
        axes[col].set_title(model_input_title)
        col += 1
    axes[col].imshow(make_gt_rgb(label))
    axes[col].set_title("Ground truth\n(blue=water, black=no-data)")
    axes[col + 1].imshow(pred, cmap="Blues", vmin=0, vmax=1)
    axes[col + 1].set_title(pred_title)
    axes[col + 2].imshow(make_error_map(pred, label))
    axes[col + 2].set_title(
        f"Error (G=TP R=FP B=FN black=no-data)\nwater IoU={chip_metrics['water_iou']:.3f}"
    )
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def s1_to_pseudo_rgb(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    db_min: float = -30.0,
    db_max: float = 0.0,
) -> np.ndarray:
    """Stack VV/VH/VH, clip dB → [0,1]. Returns (H, W, 3) float32."""
    # stack = np.stack([vv, vh, vh], axis=-1).astype(np.float32)
    stack = np.stack([vv, vh, 0.5*(vv+vh)], axis=-1).astype(np.float32)
    stack = np.clip(stack, db_min, db_max)
    stack = (stack - db_min) / max(db_max - db_min, 1e-6)
    return stack


def normalize_imagenet_chw(rgb01: np.ndarray) -> np.ndarray:
    """(H, W, 3) in [0,1] → (3, H, W) ImageNet-normalized float32."""
    x = (rgb01 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x.astype(np.float32), (2, 0, 1))
