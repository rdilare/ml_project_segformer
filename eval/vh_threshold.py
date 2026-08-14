#!/usr/bin/env python3
"""VH backscatter threshold baseline for Sen1Floods11 hand-labeled chips.

Tunes a VH (dB) threshold on the validation split, evaluates on test,
writes metrics JSON/CSV and side-by-side visualizations.

Does NOT download data. Point --data-root at your existing hand set:

  DATA_ROOT/
    S1/                 *_S1Hand.tif   (bands: VV, VH)
    Labels/             *_LabelHand.tif  (-1 nodata, 0 not-water, 1 water)
    splits/
      flood_valid_data.csv
      flood_test_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `python eval/vh_threshold.py` from repo root or eval/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import (  # noqa: E402
    add_confusion,
    choose_stems_for_viz,
    confusion_counts,
    empty_confusion,
    load_vh_and_label,
    metrics_from_counts,
    print_metrics,
    read_split_stems,
    resolve_pair,
    save_metrics_json,
    visualize_prediction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sen1Floods11 VH threshold baseline")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder with S1/, Labels/, splits/ (hand set you already downloaded)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/vh_baseline"),
        help="Where to write metrics and figures",
    )
    p.add_argument("--s1-dir", type=str, default="S1")
    p.add_argument("--label-dir", type=str, default="Labels")
    p.add_argument("--splits-dir", type=str, default="splits")
    p.add_argument("--thresh-min", type=float, default=-30.0)
    p.add_argument("--thresh-max", type=float, default=-10.0)
    p.add_argument("--thresh-step", type=float, default=0.5)
    p.add_argument("--num-viz", type=int, default=8)
    p.add_argument("--seed", type=int, default=-1)
    return p.parse_args()


def predict_water(vh: np.ndarray, threshold_db: float) -> np.ndarray:
    """Water where VH <= threshold (darker = more likely water)."""
    return (vh <= threshold_db).astype(np.uint8)


def accumulate_split(
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    stems: list[str],
    threshold_db: float,
) -> dict[str, float]:
    totals = empty_confusion()
    missing = 0
    for stem in stems:
        try:
            s1_path, lab_path = resolve_pair(data_root, s1_dir, label_dir, stem)
            vh, label = load_vh_and_label(s1_path, lab_path)
        except FileNotFoundError:
            missing += 1
            continue
        pred = predict_water(vh, threshold_db)
        add_confusion(totals, confusion_counts(pred, label))
    if missing:
        print(f"  warning: skipped {missing}/{len(stems)} missing chips")
    return metrics_from_counts(totals)


def tune_threshold(
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    val_stems: list[str],
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
) -> tuple[float, pd.DataFrame]:
    grid = np.arange(thresh_min, thresh_max + 1e-9, thresh_step)
    rows = []
    best_t, best_iou = None, -1.0
    for t in grid:
        m = accumulate_split(data_root, s1_dir, label_dir, val_stems, float(t))
        rows.append({"threshold_db": float(t), **m})
        if m["water_iou"] > best_iou:
            best_iou = m["water_iou"]
            best_t = float(t)
        print(
            f"  val  VH<= {t:6.2f} dB  water_IoU={m['water_iou']:.4f}  mIoU={m['miou']:.4f}"
        )
    assert best_t is not None
    return best_t, pd.DataFrame(rows)


def visualize_split(
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    stems: list[str],
    threshold_db: float,
    out_dir: Path,
    num_viz: int,
    seed: int,
) -> list[str]:
    chosen = choose_stems_for_viz(data_root, s1_dir, label_dir, stems, num_viz, seed)
    if not chosen:
        print("  no chips available for visualization")
        return []
    viz_dir = out_dir / "figures"
    for stem in chosen:
        s1_path, lab_path = resolve_pair(data_root, s1_dir, label_dir, stem)
        vh, label = load_vh_and_label(s1_path, lab_path)
        pred = predict_water(vh, threshold_db)
        visualize_prediction(
            stem,
            vh,
            label,
            pred,
            viz_dir / f"{stem}.png",
            sar_title="VH (dB)",
            pred_title=f"Pred VH≤{threshold_db:.1f} dB",
        )
    return chosen


def plot_tuning_curve(df: pd.DataFrame, best_t: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["threshold_db"], df["water_iou"], label="water IoU", marker="o", ms=3)
    ax.plot(df["threshold_db"], df["miou"], label="mIoU", marker="o", ms=3)
    ax.axvline(best_t, color="crimson", ls="--", label=f"best={best_t:.1f} dB")
    ax.set_xlabel("VH threshold (dB)")
    ax.set_ylabel("Score")
    ax.set_title("VH threshold tuning on validation split")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.data_root
    splits = root / args.splits_dir
    for name in ("flood_valid_data.csv", "flood_test_data.csv"):
        if not (splits / name).exists():
            raise SystemExit(f"Missing split file: {splits / name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_stems = read_split_stems(splits / "flood_valid_data.csv")
    test_stems = read_split_stems(splits / "flood_test_data.csv")
    print(f"val chips: {len(val_stems)}  test chips: {len(test_stems)}")
    print("Tuning VH threshold on validation (hand set only)...")

    best_t, tune_df = tune_threshold(
        root,
        args.s1_dir,
        args.label_dir,
        val_stems,
        args.thresh_min,
        args.thresh_max,
        args.thresh_step,
    )
    tune_df.to_csv(args.out_dir / "val_threshold_sweep.csv", index=False)
    plot_tuning_curve(tune_df, best_t, args.out_dir / "figures" / "val_threshold_curve.png")

    print(f"\nBest threshold (max val water IoU): {best_t:.2f} dB")
    print("Evaluating on test...")
    test_metrics = accumulate_split(root, args.s1_dir, args.label_dir, test_stems, best_t)
    val_metrics = accumulate_split(root, args.s1_dir, args.label_dir, val_stems, best_t)

    summary = {
        "method": "VH_threshold",
        "threshold_db": best_t,
        "tuned_on": "flood_valid_data.csv",
        "evaluated_on": "flood_test_data.csv",
        "val": val_metrics,
        "test": test_metrics,
    }
    save_metrics_json(args.out_dir / "metrics.json", summary)

    print_metrics("Validation @ best threshold", val_metrics)
    print_metrics("Test @ best threshold", test_metrics)

    print("\nSaving visualizations...")
    chosen = visualize_split(
        root,
        args.s1_dir,
        args.label_dir,
        test_stems,
        best_t,
        args.out_dir,
        args.num_viz,
        args.seed,
    )
    print(f"Wrote {len(chosen)} figures under {args.out_dir / 'figures'}")
    print(f"Metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
