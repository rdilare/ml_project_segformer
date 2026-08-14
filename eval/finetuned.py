#!/usr/bin/env python3
"""Load fine-tuned SegFormer weights (models/best_hf/) and evaluate a chip or split.

Same SAR → VV/VH/mean → ImageNet-norm pipeline as training. Writes metrics
JSON and SAR / GT / pred / error-map figures.

Examples::

  python eval/finetuned.py --stem Ghana_313799

  python eval/finetuned.py \\
      --chip data/sen1floods11_hand/S1/Ghana_313799_S1Hand.tif

  python eval/finetuned.py --split test --num-viz 8
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import (  # noqa: E402
    add_confusion,
    choose_stems_for_viz,
    confusion_counts,
    empty_confusion,
    load_s1_and_label,
    load_s1_vv_vh,
    metrics_from_counts,
    normalize_imagenet_chw,
    print_metrics,
    read_split_stems,
    resolve_pair,
    s1_to_pseudo_rgb,
    save_metrics_json,
    visualize_prediction,
)

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install torch: pip install torch") from e

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install transformers: pip install transformers") from e


SPLIT_CSV = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate fine-tuned SegFormer (models/best_hf/) on a chip or split"
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--stem", type=str, help="Chip stem, e.g. Ghana_313799")
    target.add_argument("--chip", type=Path, help="Path to *_S1Hand.tif")
    target.add_argument(
        "--split",
        type=str,
        choices=sorted(set(SPLIT_CSV)),
        help="Evaluate every chip in an official split",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=_REPO_ROOT / "models" / "best_hf",
        help="Hugging Face folder with config.json + model.safetensors",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=_REPO_ROOT / "data" / "sen1floods11_hand",
        help="Folder with S1/, Labels/, splits/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "finetuned_eval",
        help="Metrics JSON and figures",
    )
    p.add_argument("--s1-dir", type=str, default="S1")
    p.add_argument("--label-dir", type=str, default="Labels")
    p.add_argument("--splits-dir", type=str, default="splits")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--db-min", type=float, default=-30.0)
    p.add_argument("--db-max", type=float, default=0.0)
    p.add_argument("--device", type=str, default="auto", help="cuda | mps | cpu | auto")
    p.add_argument("--fp16", action="store_true", help="CUDA autocast fp16")
    p.add_argument("--num-viz", type=int, default=8, help="Figures when using --split")
    p.add_argument("--seed", type=int, default=24)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def stem_from_path(path: Path) -> str:
    return re.sub(r"_(S1Hand|LabelHand|S2Hand)$", "", path.stem)


def resolve_label(data_root: Path, label_dir: str, stem: str, s1_path: Path) -> Path | None:
    candidates = [
        data_root / label_dir / f"{stem}_LabelHand.tif",
        s1_path.parent.parent / label_dir / f"{stem}_LabelHand.tif",
        s1_path.parent / f"{stem}_LabelHand.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    for d in (data_root / label_dir, s1_path.parent.parent / label_dir):
        if d.is_dir():
            matches = sorted(d.glob(f"{stem}*Label*.tif"))
            if matches:
                return matches[0]
    return None


def resolve_one_chip(
    *,
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    stem: str | None,
    chip: Path | None,
) -> tuple[str, Path, Path | None]:
    if chip is not None:
        s1_path = chip
        if not s1_path.exists():
            raise SystemExit(f"File not found: {s1_path}")
        stem = stem_from_path(s1_path)
        return stem, s1_path, resolve_label(data_root, label_dir, stem, s1_path)
    assert stem is not None
    stem = stem_from_path(Path(stem))
    try:
        s1_path, lab_path = resolve_pair(data_root, s1_dir, label_dir, stem)
        return stem, s1_path, lab_path
    except FileNotFoundError:
        s1_path = data_root / s1_dir / f"{stem}_S1Hand.tif"
        if not s1_path.exists():
            matches = sorted((data_root / s1_dir).glob(f"{stem}*.tif"))
            if not matches:
                raise SystemExit(
                    f"Missing S1 chip for stem={stem} under {data_root / s1_dir}"
                )
            s1_path = matches[0]
        return stem, s1_path, resolve_label(data_root, label_dir, stem, s1_path)


def load_model(model_dir: Path, device: torch.device) -> SegformerForSemanticSegmentation:
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"Missing config.json in {model_dir}")
    weights = model_dir / "model.safetensors"
    if not weights.is_file() and not (model_dir / "pytorch_model.bin").is_file():
        raise SystemExit(f"Missing model.safetensors in {model_dir}")
    print(f"Loading {model_dir} ...")
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_chip(
    model: SegformerForSemanticSegmentation,
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    image_size: int,
    db_min: float,
    db_max: float,
    device: torch.device,
    use_fp16: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred uint8 HxW at original size, pseudo-RGB HxWx3 in [0,1])."""
    rgb01 = s1_to_pseudo_rgb(vv, vh, db_min=db_min, db_max=db_max)
    chw = normalize_imagenet_chw(rgb01)
    t = torch.from_numpy(chw).unsqueeze(0)
    if t.shape[-2] != image_size or t.shape[-1] != image_size:
        t = F.interpolate(
            t, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    t = t.to(device)
    if use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values=t).logits.float()
    else:
        logits = model(pixel_values=t).logits.float()
    out_hw = (int(vv.shape[0]), int(vv.shape[1]))
    if logits.shape[-2:] != out_hw:
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred, rgb01


def save_chip_figure(
    *,
    stem: str,
    vh: np.ndarray,
    pred: np.ndarray,
    rgb01: np.ndarray,
    label: np.ndarray | None,
    out_path: Path,
) -> None:
    if label is not None:
        visualize_prediction(
            stem,
            vh,
            label,
            pred,
            out_path,
            sar_title="VH (dB)",
            pred_title="SegFormer fine-tuned\nargmax",
            model_input=rgb01,
            model_input_title="SegFormer input\n(VV/VH/mean → [0,1])",
        )
        return

    import matplotlib.pyplot as plt
    from eval.common import percentile_stretch

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    axes[0].imshow(percentile_stretch(vh), cmap="gray")
    axes[0].set_title(f"VH (dB)\n{stem}")
    axes[1].imshow(np.clip(rgb01, 0.0, 1.0))
    axes[1].set_title("SegFormer input\n(VV/VH/mean → [0,1])")
    axes[2].imshow(pred, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("SegFormer fine-tuned\nargmax (no GT)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_one(args: argparse.Namespace, model, device: torch.device, use_fp16: bool) -> None:
    stem, s1_path, lab_path = resolve_one_chip(
        data_root=args.data_root,
        s1_dir=args.s1_dir,
        label_dir=args.label_dir,
        stem=args.stem,
        chip=args.chip,
    )
    print(f"chip: {s1_path}")
    if lab_path is not None:
        print(f"label: {lab_path}")
    else:
        print("label: not found (metrics skipped)")

    if lab_path is not None:
        vv, vh, label = load_s1_and_label(s1_path, lab_path)
    else:
        vv, vh = load_s1_vv_vh(s1_path)
        label = None
    pred, rgb01 = predict_chip(
        model,
        vv,
        vh,
        image_size=args.image_size,
        db_min=args.db_min,
        db_max=args.db_max,
        device=device,
        use_fp16=use_fp16,
    )
    metrics = None if label is None else metrics_from_counts(confusion_counts(pred, label))
    if metrics is not None:
        print_metrics(stem, metrics)

    fig_path = args.out_dir / "figures" / f"{stem}.png"
    save_chip_figure(
        stem=stem, vh=vh, pred=pred, rgb01=rgb01, label=label, out_path=fig_path
    )
    summary = {
        "method": "SegFormer_finetune",
        "model_dir": str(args.model_dir),
        "stem": stem,
        "s1": str(s1_path),
        "label": str(lab_path) if lab_path is not None else None,
        "decision": "argmax",
        "device": str(device),
        "metrics": metrics,
        "figure": str(fig_path),
    }
    save_metrics_json(args.out_dir / "metrics.json", summary)
    print(f"Figure: {fig_path}")
    print(f"Metrics: {args.out_dir / 'metrics.json'}")


def run_split(args: argparse.Namespace, model, device: torch.device, use_fp16: bool) -> None:
    csv_name = SPLIT_CSV[args.split]
    csv_path = args.data_root / args.splits_dir / csv_name
    if not csv_path.exists():
        raise SystemExit(f"Missing split file: {csv_path}")
    stems = read_split_stems(csv_path)
    print(f"split={args.split}  chips listed={len(stems)}")

    totals = empty_confusion()
    per_chip: list[dict] = []
    pred_by_stem: dict[str, np.ndarray] = {}
    ok = 0
    missing = 0
    for stem in stems:
        try:
            s1_path, lab_path = resolve_pair(
                args.data_root, args.s1_dir, args.label_dir, stem
            )
        except FileNotFoundError:
            missing += 1
            continue
        vv, vh, label = load_s1_and_label(s1_path, lab_path)
        pred, _rgb = predict_chip(
            model,
            vv,
            vh,
            image_size=args.image_size,
            db_min=args.db_min,
            db_max=args.db_max,
            device=device,
            use_fp16=use_fp16,
        )
        counts = confusion_counts(pred, label)
        add_confusion(totals, counts)
        chip_metrics = metrics_from_counts(counts)
        per_chip.append({"stem": stem, **chip_metrics})
        pred_by_stem[stem] = pred
        ok += 1
        print(
            f"  {stem:40s}  water_iou={chip_metrics['water_iou']:.4f}  "
            f"miou={chip_metrics['miou']:.4f}"
        )

    if missing:
        print(f"warning: skipped {missing}/{len(stems)} missing chips")
    if ok == 0:
        raise SystemExit(f"No chips found under {args.data_root}")

    metrics = metrics_from_counts(totals)
    print_metrics(f"{args.split} (fine-tuned, n={ok})", metrics)

    chosen: list[str] = []
    if args.num_viz > 0:
        chosen = choose_stems_for_viz(
            args.data_root,
            args.s1_dir,
            args.label_dir,
            list(pred_by_stem.keys()),
            args.num_viz,
            args.seed,
        )
        viz_dir = args.out_dir / "figures"
        for stem in chosen:
            s1_path, lab_path = resolve_pair(
                args.data_root, args.s1_dir, args.label_dir, stem
            )
            vv, vh, label = load_s1_and_label(s1_path, lab_path)
            rgb01 = s1_to_pseudo_rgb(vv, vh, db_min=args.db_min, db_max=args.db_max)
            save_chip_figure(
                stem=stem,
                vh=vh,
                pred=pred_by_stem[stem],
                rgb01=rgb01,
                label=label,
                out_path=viz_dir / f"{stem}.png",
            )
        print(f"Wrote {len(chosen)} figures under {args.out_dir / 'figures'}")

    summary = {
        "method": "SegFormer_finetune",
        "model_dir": str(args.model_dir),
        "split": args.split,
        "decision": "argmax",
        "device": str(device),
        "n": ok,
        "n_missing": missing,
        "metrics": metrics,
        "per_chip": per_chip,
        "figures": [str(args.out_dir / "figures" / f"{s}.png") for s in chosen],
    }
    save_metrics_json(args.out_dir / "metrics.json", summary)
    print(f"Metrics: {args.out_dir / 'metrics.json'}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  fp16={use_fp16}")
    print(f"data-root={args.data_root}")
    print(f"out-dir={args.out_dir}")

    model = load_model(args.model_dir, device)
    if args.split is not None:
        run_split(args, model, device, use_fp16)
    else:
        run_one(args, model, device, use_fp16)


if __name__ == "__main__":
    main()
