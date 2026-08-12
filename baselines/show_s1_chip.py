#!/usr/bin/env python3
"""Show VV, VH, and ground-truth (if present) for a Sen1Floods11 chip.

Examples:
  python baselines/show_s1_chip.py \\
    --chip data/sen1floods11_hand/S1/Bolivia_23014_S1Hand.tif

  python baselines/show_s1_chip.py \\
    --data-root data/sen1floods11_hand --stem Bolivia_23014 --save
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install rasterio: pip install rasterio") from e


IGNORE_INDEX = -1
WATER = 1
NOT_WATER = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize VV, VH, and ground truth from a Sen1Floods11 chip"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--chip", type=Path, help="Path to *_S1Hand.tif")
    g.add_argument("--stem", type=str, help="Chip stem, e.g. Bolivia_23014")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/sen1floods11_hand"),
        help="Hand-set root with S1/ and Labels/ (used with --stem, or to find GT)",
    )
    p.add_argument("--s1-dir", type=str, default="S1")
    p.add_argument("--label-dir", type=str, default="Labels")
    p.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=Path("outputs/s1_chip_preview.png"),
        default=None,
        help="Save figure. Optional path; default outputs/s1_chip_preview.png",
    )
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    p.add_argument("--vmin", type=float, default=-30.0, help="dB colormap min")
    p.add_argument("--vmax", type=float, default=0.0, help="dB colormap max")
    return p.parse_args()


def stem_from_path(path: Path) -> str:
    return re.sub(r"_(S1Hand|LabelHand|S2Hand)$", "", path.stem)


def resolve_chip(args: argparse.Namespace) -> Path:
    if args.chip is not None:
        path = args.chip
    else:
        path = args.data_root / args.s1_dir / f"{args.stem}_S1Hand.tif"
        if not path.exists():
            matches = sorted((args.data_root / args.s1_dir).glob(f"{args.stem}*.tif"))
            if not matches:
                raise SystemExit(f"Missing S1 chip: {path}")
            path = matches[0]
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return path


def resolve_label(args: argparse.Namespace, stem: str, s1_path: Path) -> Path | None:
    candidates = [
        args.data_root / args.label_dir / f"{stem}_LabelHand.tif",
        s1_path.parent.parent / args.label_dir / f"{stem}_LabelHand.tif",
        s1_path.parent / f"{stem}_LabelHand.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    label_dirs = [
        args.data_root / args.label_dir,
        s1_path.parent.parent / args.label_dir,
    ]
    for d in label_dirs:
        if d.is_dir():
            matches = sorted(d.glob(f"{stem}*.tif"))
            if matches:
                return matches[0]
    return None


def load_vv_vh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        if src.count < 2:
            raise SystemExit(f"{path} has {src.count} bands; need VV+VH")
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)
    return vv, vh


def load_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def make_gt_rgb(label: np.ndarray) -> np.ndarray:
    """RGB: white=not-water, blue=water, black=no-data (-1)."""
    rgb = np.ones((*label.shape, 3), dtype=np.float32)
    rgb[label == WATER] = (0.12, 0.35, 0.85)
    rgb[label == IGNORE_INDEX] = (0.0, 0.0, 0.0)
    return rgb


def summarize(name: str, arr: np.ndarray) -> None:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"  {name}: all non-finite")
        return
    print(
        f"  {name}: shape={arr.shape}  "
        f"min={finite.min():.2f}  max={finite.max():.2f}  "
        f"mean={finite.mean():.2f}  std={finite.std():.2f}"
    )


def summarize_label(label: np.ndarray) -> None:
    vals, counts = np.unique(label, return_counts=True)
    parts = []
    names = {IGNORE_INDEX: "nodata", NOT_WATER: "not-water", WATER: "water"}
    for v, c in zip(vals, counts):
        parts.append(f"{names.get(int(v), v)}={int(c)}")
    print(f"  GT: shape={label.shape}  " + "  ".join(parts))


def main() -> None:
    args = parse_args()

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1_path = resolve_chip(args)
    stem = stem_from_path(s1_path) if args.stem is None else args.stem
    vv, vh = load_vv_vh(s1_path)
    label_path = resolve_label(args, stem, s1_path)
    label = load_label(label_path) if label_path is not None else None

    print(f"chip: {s1_path}")
    summarize("VV", vv)
    summarize("VH", vh)
    if label is not None:
        print(f"label: {label_path}")
        summarize_label(label)
    else:
        print("label: not found (showing VV/VH only)")

    n = 3 if label is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 2:
        axes = list(axes)

    for ax, band, title in zip(axes[:2], (vv, vh), ("VV (dB)", "VH (dB)")):
        im = ax.imshow(band, cmap="gray", vmin=args.vmin, vmax=args.vmax)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if label is not None:
        axes[2].imshow(make_gt_rgb(label))
        axes[2].set_title("Ground truth\n(blue=water, black=no-data)")
        axes[2].axis("off")

    fig.suptitle(stem, fontsize=12)
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=140, bbox_inches="tight")
        print(f"saved: {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
