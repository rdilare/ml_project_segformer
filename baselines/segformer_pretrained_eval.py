#!/usr/bin/env python3
"""Protocol A: pretrained SegFormer (mit-b0) before fine-tuning on Sen1Floods11.

Loads ImageNet-pretrained encoder + randomly initialized 2-class decode head
(no training). S1 is adapted as VV/VH/VH → ImageNet-normalized 3ch.

Same splits / ignore rules / metrics as the VH threshold baseline.

Does NOT download data. Point --data-root at your existing hand set:

  DATA_ROOT/
    S1/                 *_S1Hand.tif   (bands: VV, VH)
    Labels/             *_LabelHand.tif  (-1 nodata, 0 not-water, 1 water)
    splits/
      flood_valid_data.csv
      flood_test_data.csv

Example:
  python baselines/segformer_pretrained_eval.py \\
    --data-root data/sen1floods11_hand \\
    --out-dir outputs/segformer_pretrained
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python baselines/segformer_pretrained_eval.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines.common import (  # noqa: E402
    WATER,
    add_confusion,
    choose_stems_for_viz,
    confusion_counts,
    empty_confusion,
    load_s1_and_label,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sen1Floods11 SegFormer pretrained (no fine-tune) eval — protocol A"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder with S1/, Labels/, splits/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/segformer_pretrained"),
        help="Where to write metrics and figures",
    )
    p.add_argument("--s1-dir", type=str, default="S1")
    p.add_argument("--label-dir", type=str, default="Labels")
    p.add_argument("--splits-dir", type=str, default="splits")
    p.add_argument(
        "--model-id",
        type=str,
        default="nvidia/mit-b0",
        help="HF checkpoint with ImageNet-pretrained encoder",
    )
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--db-min", type=float, default=-30.0)
    p.add_argument("--db-max", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cuda | mps | cpu | auto",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 autocast on CUDA (Colab T4)",
    )
    p.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Sweep water-prob threshold on val (else argmax)",
    )
    p.add_argument("--thresh-min", type=float, default=0.05)
    p.add_argument("--thresh-max", type=float, default=0.95)
    p.add_argument("--thresh-step", type=float, default=0.05)
    p.add_argument("--num-viz", type=int, default=8)
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


def set_seed(seed: int) -> None:
    """Seed RNGs so the randomly initialized 2-class head is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_id: str, device: torch.device) -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "not_water", 1: "water"},
        label2id={"not_water": 0, "water": 1},
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()
    return model


def preprocess_chip(
    vv: np.ndarray,
    vh: np.ndarray,
    *,
    image_size: int,
    db_min: float,
    db_max: float,
) -> torch.Tensor:
    """Return (3, image_size, image_size) float32 tensor."""
    rgb01 = s1_to_pseudo_rgb(vv, vh, db_min=db_min, db_max=db_max)
    chw = normalize_imagenet_chw(rgb01)
    t = torch.from_numpy(chw).unsqueeze(0)  # 1,3,H,W
    if t.shape[-2] != image_size or t.shape[-1] != image_size:
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return t.squeeze(0)


@torch.inference_mode()
def predict_logits(
    model: SegformerForSemanticSegmentation,
    pixel_values: torch.Tensor,
    *,
    out_hw: tuple[int, int],
    use_fp16: bool,
) -> torch.Tensor:
    """pixel_values: (B,3,H,W) on device → logits (B,2,out_h,out_w) float32."""
    device = pixel_values.device
    if use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(pixel_values=pixel_values)
            logits = out.logits.float()
    else:
        out = model(pixel_values=pixel_values)
        logits = out.logits.float()
    if logits.shape[-2:] != out_hw:
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
    return logits


def logits_to_pred(logits: torch.Tensor, water_thresh: float | None) -> np.ndarray:
    """logits (2,H,W) → uint8 pred (H,W)."""
    if water_thresh is None:
        pred = logits.argmax(dim=0)
    else:
        prob = torch.softmax(logits, dim=0)[WATER]
        pred = (prob >= water_thresh).to(torch.uint8)
    return pred.cpu().numpy().astype(np.uint8)


def iter_batches(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def collect_chip_tensors(
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    stems: list[str],
    *,
    image_size: int,
    db_min: float,
    db_max: float,
) -> tuple[list[str], list[torch.Tensor], list[np.ndarray], list[tuple[int, int]]]:
    """Load available chips; return stems, tensors (CPU), labels, original HWs."""
    ok_stems: list[str] = []
    tensors: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    shapes: list[tuple[int, int]] = []
    missing = 0
    for stem in stems:
        try:
            s1_path, lab_path = resolve_pair(data_root, s1_dir, label_dir, stem)
            vv, vh, label = load_s1_and_label(s1_path, lab_path)
        except FileNotFoundError:
            missing += 1
            continue
        t = preprocess_chip(vv, vh, image_size=image_size, db_min=db_min, db_max=db_max)
        ok_stems.append(stem)
        tensors.append(t)
        labels.append(label)
        shapes.append(label.shape)
    if missing:
        print(f"  warning: skipped {missing}/{len(stems)} missing chips")
    return ok_stems, tensors, labels, shapes


@torch.inference_mode()
def run_split_logits(
    model: SegformerForSemanticSegmentation,
    tensors: list[torch.Tensor],
    shapes: list[tuple[int, int]],
    device: torch.device,
    batch_size: int,
    use_fp16: bool,
) -> list[torch.Tensor]:
    """Return list of CPU logits (2,H,W) at original label resolution."""
    all_logits: list[torch.Tensor] = []
    for batch_idx in iter_batches(list(range(len(tensors))), batch_size):
        # Chips may have different original sizes; process one-by-one if mixed,
        # but batch when sizes match (hand chips are typically 512x512).
        groups: dict[tuple[int, int], list[int]] = {}
        for i in batch_idx:
            groups.setdefault(shapes[i], []).append(i)
        for hw, idxs in groups.items():
            pv = torch.stack([tensors[i] for i in idxs], dim=0).to(device)
            logits = predict_logits(model, pv, out_hw=hw, use_fp16=use_fp16)
            for j, i in enumerate(idxs):
                all_logits.append((i, logits[j].cpu()))
    all_logits.sort(key=lambda x: x[0])
    return [logit for _, logit in all_logits]


def metrics_from_logits(
    logits_list: list[torch.Tensor],
    labels: list[np.ndarray],
    water_thresh: float | None,
) -> dict[str, float]:
    totals = empty_confusion()
    for logits, label in zip(logits_list, labels):
        pred = logits_to_pred(logits, water_thresh)
        add_confusion(totals, confusion_counts(pred, label))
    return metrics_from_counts(totals)


def tune_water_threshold(
    logits_list: list[torch.Tensor],
    labels: list[np.ndarray],
    thresh_min: float,
    thresh_max: float,
    thresh_step: float,
) -> tuple[float, list[dict]]:
    grid = np.arange(thresh_min, thresh_max + 1e-9, thresh_step)
    rows = []
    best_t, best_iou = 0.5, -1.0
    for t in grid:
        m = metrics_from_logits(logits_list, labels, float(t))
        rows.append({"water_prob_thresh": float(t), **m})
        if m["water_iou"] > best_iou:
            best_iou = m["water_iou"]
            best_t = float(t)
        print(
            f"  val  P(water)>= {t:4.2f}  water_IoU={m['water_iou']:.4f}  mIoU={m['miou']:.4f}"
        )
    return best_t, rows


def visualize_split(
    data_root: Path,
    s1_dir: str,
    label_dir: str,
    stems: list[str],
    pred_by_stem: dict[str, np.ndarray],
    out_dir: Path,
    num_viz: int,
    seed: int,
    pred_title: str,
    *,
    db_min: float,
    db_max: float,
) -> list[str]:
    chosen = choose_stems_for_viz(data_root, s1_dir, label_dir, stems, num_viz, seed)
    if not chosen:
        print("  no chips available for visualization")
        return []
    viz_dir = out_dir / "figures"
    for stem in chosen:
        if stem not in pred_by_stem:
            continue
        s1_path, lab_path = resolve_pair(data_root, s1_dir, label_dir, stem)
        vv, vh, label = load_s1_and_label(s1_path, lab_path)
        # Same VV/VH/VH [0,1] stack fed to SegFormer (before ImageNet norm).
        model_rgb = s1_to_pseudo_rgb(vv, vh, db_min=db_min, db_max=db_max)
        visualize_prediction(
            stem,
            vh,
            label,
            pred_by_stem[stem],
            viz_dir / f"{stem}.png",
            sar_title="VH (dB)",
            pred_title=pred_title,
            model_input=model_rgb,
            model_input_title="SegFormer input\n(VV/VH/VH → [0,1])",
        )
    return chosen


def main() -> None:
    args = parse_args()
    root = args.data_root
    splits = root / args.splits_dir
    for name in ("flood_valid_data.csv", "flood_test_data.csv"):
        if not (splits / name).exists():
            raise SystemExit(f"Missing split file: {splits / name}")

    device = pick_device(args.device)
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    print(f"device={device}  fp16={use_fp16}  model={args.model_id}  seed={args.seed}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_stems = read_split_stems(splits / "flood_valid_data.csv")
    test_stems = read_split_stems(splits / "flood_test_data.csv")
    print(f"val chips: {len(val_stems)}  test chips: {len(test_stems)}")

    set_seed(args.seed)
    print("Loading model (pretrained encoder, new 2-class head)...")
    model = load_model(args.model_id, device)

    prep_kw = dict(
        image_size=args.image_size,
        db_min=args.db_min,
        db_max=args.db_max,
    )

    print("Loading validation chips...")
    val_ok, val_tensors, val_labels, val_shapes = collect_chip_tensors(
        root, args.s1_dir, args.label_dir, val_stems, **prep_kw
    )
    print("Loading test chips...")
    test_ok, test_tensors, test_labels, test_shapes = collect_chip_tensors(
        root, args.s1_dir, args.label_dir, test_stems, **prep_kw
    )

    print("Running inference on val...")
    val_logits = run_split_logits(
        model, val_tensors, val_shapes, device, args.batch_size, use_fp16
    )
    print("Running inference on test...")
    test_logits = run_split_logits(
        model, test_tensors, test_shapes, device, args.batch_size, use_fp16
    )

    water_thresh: float | None = None
    tune_rows: list[dict] | None = None
    if args.tune_threshold:
        print("Tuning water-prob threshold on validation...")
        water_thresh, tune_rows = tune_water_threshold(
            val_logits,
            val_labels,
            args.thresh_min,
            args.thresh_max,
            args.thresh_step,
        )
        print(f"Best val water-prob threshold: {water_thresh:.2f}")
        pd.DataFrame(tune_rows).to_csv(
            args.out_dir / "val_prob_threshold_sweep.csv", index=False
        )
    else:
        print("Using argmax (no threshold tuning)")

    val_metrics = metrics_from_logits(val_logits, val_labels, water_thresh)
    test_metrics = metrics_from_logits(test_logits, test_labels, water_thresh)

    summary = {
        "method": "SegFormer_pretrained_protocol_A",
        "model_id": args.model_id,
        "num_labels": 2,
        "input_adapt": "VV_VH_VH_clip_db_imagenet_norm",
        "db_min": args.db_min,
        "db_max": args.db_max,
        "image_size": args.image_size,
        "decision": (
            f"P(water)>={water_thresh:.2f}" if water_thresh is not None else "argmax"
        ),
        "water_prob_thresh": water_thresh,
        "tuned_on": "flood_valid_data.csv" if water_thresh is not None else None,
        "evaluated_on": "flood_test_data.csv",
        "device": str(device),
        "fp16": use_fp16,
        "val": val_metrics,
        "test": test_metrics,
        "n_val": len(val_ok),
        "n_test": len(test_ok),
    }
    save_metrics_json(args.out_dir / "metrics.json", summary)

    print_metrics("Validation (pretrained, no FT)", val_metrics)
    print_metrics("Test (pretrained, no FT)", test_metrics)

    pred_title = (
        f"SegFormer pretrained\nP(w)≥{water_thresh:.2f}"
        if water_thresh is not None
        else "SegFormer pretrained\nargmax"
    )
    pred_by_stem = {
        stem: logits_to_pred(logits, water_thresh)
        for stem, logits in zip(test_ok, test_logits)
    }

    print("\nSaving visualizations...")
    chosen = visualize_split(
        root,
        args.s1_dir,
        args.label_dir,
        test_ok,
        pred_by_stem,
        args.out_dir,
        args.num_viz,
        args.seed,
        pred_title,
        db_min=args.db_min,
        db_max=args.db_max,
    )
    print(f"Wrote {len(chosen)} figures under {args.out_dir / 'figures'}")
    print(f"Metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
