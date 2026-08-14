#!/usr/bin/env python3
"""Fine-tune SegFormer (mit-b0) on Sen1Floods11 hand-labeled chips.

Designed for Colab T4: fp16, batch 4–8, grad accumulation, early stop on
val water IoU. Checkpoints + metrics go under --out-dir (put that on Drive).

Loss is weighted CE (water up-weighted) + soft Dice so thin water is not
drowned by land pixels. Train sampling over-represents sparse-water chips;
optional water-centered crops zoom streams without changing 512 output size.
Dry chips are kept as negatives (no copy-paste).

Every epoch writes ``last.pt`` (full train state), ``history.json``, and
``figures/training_curves.png``. On improvement, ``best.pt`` + ``best_hf/``.
By default the script auto-resumes from ``out-dir/last.pt`` (else ``best.pt``)
when present; otherwise starts from scratch. Use ``--no-resume`` to force a
fresh run.

Does NOT download data. Point --data-root at your hand set::

  DATA_ROOT/
    S1/       *_S1Hand.tif
    Labels/   *_LabelHand.tif
    splits/   flood_{train,valid,test}_data.csv

Colab example::

  python train/finetune_segformer.py \\
    --data-root /content/drive/MyDrive/sen1floods11_hand \\
    --out-dir /content/drive/MyDrive/sen1floods11_hand/runs/segformer_ft \\
    --fp16

  # After interrupt / disconnect, re-run the same command (auto-resumes).
  # Force a fresh run: add --no-resume
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import (  # noqa: E402
    IGNORE_INDEX,
    WATER,
    add_confusion,
    choose_stems_for_viz,
    confusion_counts,
    empty_confusion,
    load_s1_and_label,
    metrics_from_counts,
    print_metrics,
    resolve_pair,
    s1_to_pseudo_rgb,
    save_metrics_json,
    visualize_prediction,
)
from train.dataset import Sen1Floods11SegDataset, collate_fn  # noqa: E402

try:
    from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install transformers: pip install transformers") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SegFormer on Sen1Floods11")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Folder with S1/, Labels/, splits/",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/segformer_ft"),
        help="Checkpoints, metrics, figures (use a Drive path on Colab)",
    )
    p.add_argument("--s1-dir", type=str, default="S1")
    p.add_argument("--label-dir", type=str, default="Labels")
    p.add_argument("--splits-dir", type=str, default="splits")
    p.add_argument("--model-id", type=str, default="nvidia/mit-b0")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--db-min", type=float, default=-30.0)
    p.add_argument("--db-max", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4, help="Per-step batch (T4: 4–8)")
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)",
    )
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=8, help="Early stop patience (epochs)")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--device", type=str, default="auto", help="cuda | mps | cpu | auto")
    p.add_argument("--fp16", action="store_true", help="CUDA autocast fp16 (Colab T4)")
    p.add_argument("--num-viz", type=int, default=8)
    p.add_argument(
        "--ce-water-weight",
        type=float,
        default=4.0,
        help="CE class weight for water (not-water stays 1.0)",
    )
    p.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight of soft Dice (water class) added to CE; 0 disables Dice",
    )
    p.add_argument(
        "--no-oversample-sparse",
        action="store_true",
        help="Uniform chip sampling instead of 1/sqrt(water pixels)",
    )
    p.add_argument(
        "--water-crop-p",
        type=float,
        default=0.5,
        help="Train-only P(crop around a water pixel). 0 disables. Dry chips unchanged",
    )
    p.add_argument(
        "--water-crop-size",
        type=int,
        default=256,
        help="Side length of water-centered crop before resize to --image-size",
    )
    p.add_argument(
        "--small-water-max",
        type=int,
        default=5000,
        help="GT water-pixel cutoff for val small-water IoU",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One train step + one val batch, print peak CUDA memory, exit",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default="auto",
        help=(
            "Resume if a checkpoint exists under --out-dir (default: auto = "
            "last.pt, else best.pt). Pass a .pt path to force that file, or "
            "use --no-resume to start from scratch"
        ),
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and train from scratch",
    )
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
    # Sen1Floods11 nodata is -1 (HF default ignore is 255)
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    model.to(device)
    return model


def _align_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
    return logits


def water_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1.0,
) -> torch.Tensor:
    """Soft Dice on water; ignore pixels are masked out of both pred and target."""
    prob = torch.softmax(logits, dim=1)[:, WATER]
    valid = (labels != ignore_index).to(dtype=prob.dtype)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    pred = prob * valid
    target = (labels == WATER).to(dtype=prob.dtype) * valid
    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (denom + eps)


def segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ce_water_weight: float = 4.0,
    dice_weight: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """fp32 weighted CE + Dice; 0 if a batch has no valid (non-ignore) pixels.

    HF CE under autocast / all-ignore batches can yield NaN and poison training.
    """
    logits = _align_logits(logits.float(), labels)
    if not (labels != ignore_index).any():
        return logits.sum() * 0.0
    weight = torch.tensor(
        [1.0, float(ce_water_weight)], device=logits.device, dtype=logits.dtype
    )
    ce = F.cross_entropy(logits, labels, weight=weight, ignore_index=ignore_index)
    if dice_weight == 0.0:
        return ce
    return ce + float(dice_weight) * water_dice_loss(
        logits, labels, ignore_index=ignore_index
    )


def extra_split_metrics(
    chip_counts: list[dict[str, int]],
    *,
    small_water_max: int,
) -> dict[str, float | int | None]:
    """Pooled IoU on sparse-water chips + dry-chip false-positive rate."""
    small = empty_confusion()
    n_small = 0
    n_dry = 0
    n_dry_fp = 0
    for c in chip_counts:
        gt_water = int(c["tp"]) + int(c["fn"])
        if 0 < gt_water <= int(small_water_max):
            add_confusion(small, c)
            n_small += 1
        if gt_water == 0:
            n_dry += 1
            if int(c["fp"]) > 0:
                n_dry_fp += 1
    small_iou = None
    if n_small > 0:
        small_iou = float(metrics_from_counts(small)["water_iou"])
    return {
        "small_water_iou": small_iou,
        "n_small_water": n_small,
        "dry_fp_rate": (n_dry_fp / n_dry) if n_dry else 0.0,
        "n_dry": n_dry,
        "n_dry_fp": n_dry_fp,
    }


def sparse_sample_weights(water_counts: np.ndarray) -> torch.Tensor:
    """Higher weight for dry / thin-water chips; mean weight is 1."""
    w = 1.0 / np.sqrt(water_counts.astype(np.float64) + 1.0)
    w = w / max(float(w.mean()), 1e-9)
    return torch.as_tensor(w, dtype=torch.double)


def _finite_xy(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in history:
        val = row.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            continue
        xs.append(int(row["epoch"]))
        ys.append(f)
    return xs, ys


def plot_training_curves(
    history: list[dict],
    out_path: Path,
    *,
    best_epoch: int | None = None,
) -> None:
    """Write a 2×2 dashboard: loss, IoU, precision/recall, dry FP rate."""
    if not history:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    ax = axes[0, 0]
    for key, label in (("train_loss", "train"), ("val_loss", "val")):
        xs, ys = _finite_xy(history, key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=label)
    ax.set_title("Loss (weighted CE + Dice)")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    ax = axes[0, 1]
    for key, label in (
        ("val_water_iou", "val water IoU (pooled)"),
        ("val_small_water_iou", "val small-water IoU"),
        ("val_miou", "val mIoU"),
    ):
        xs, ys = _finite_xy(history, key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=label)
    ax.set_title("IoU")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    ax = axes[1, 0]
    for key, label in (
        ("val_precision_water", "val precision"),
        ("val_recall_water", "val recall"),
        ("val_f1_water", "val F1"),
    ):
        xs, ys = _finite_xy(history, key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=label)
    ax.set_title("Water precision / recall")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    ax = axes[1, 1]
    xs, ys = _finite_xy(history, "val_dry_fp_rate")
    if xs:
        ax.plot(xs, ys, marker="o", markersize=3, color="C3", label="dry-chip FP rate")
    ax.set_title("Dry chips with any false water")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend()

    if best_epoch is not None and best_epoch >= 0:
        for ax in axes.ravel():
            ax.axvline(best_epoch, color="0.4", linestyle="--", linewidth=1, alpha=0.8)
        fig.suptitle(f"Training curves  (best val water IoU @ epoch {best_epoch})", fontsize=12)
    else:
        fig.suptitle("Training curves", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def forward_logits(
    model: SegformerForSemanticSegmentation,
    pixel_values: torch.Tensor,
    *,
    use_fp16: bool,
    device: torch.device,
) -> torch.Tensor:
    if use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(pixel_values=pixel_values).logits
    return model(pixel_values=pixel_values).logits


@torch.no_grad()
def evaluate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    *,
    ce_water_weight: float,
    dice_weight: float,
    small_water_max: int,
) -> dict[str, float]:
    model.eval()
    totals = empty_confusion()
    chip_counts: list[dict[str, int]] = []
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = forward_logits(model, pixel_values, use_fp16=use_fp16, device=device)
        loss = segmentation_loss(
            logits,
            labels,
            ce_water_weight=ce_water_weight,
            dice_weight=dice_weight,
        )
        if torch.isfinite(loss):
            loss_sum += float(loss)
            n_batches += 1
        logits_f = _align_logits(logits.float(), labels)
        preds = logits_f.argmax(dim=1)
        for i in range(preds.shape[0]):
            counts = confusion_counts(
                preds[i].cpu().numpy().astype(np.uint8),
                labels[i].cpu().numpy(),
            )
            add_confusion(totals, counts)
            chip_counts.append(counts)
    metrics = metrics_from_counts(totals)
    metrics["loss"] = loss_sum / max(n_batches, 1)
    metrics.update(extra_split_metrics(chip_counts, small_water_max=small_water_max))
    return metrics


def train_one_epoch(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    *,
    use_fp16: bool,
    grad_accum: int,
    scaler: torch.amp.GradScaler | None,
    ce_water_weight: float,
    dice_weight: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, n_steps = 0.0, 0
    skipped = 0
    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if not torch.isfinite(pixel_values).all():
            pixel_values = torch.nan_to_num(pixel_values, nan=0.0, posinf=0.0, neginf=0.0)

        logits = forward_logits(model, pixel_values, use_fp16=use_fp16, device=device)
        loss = segmentation_loss(
            logits,
            labels,
            ce_water_weight=ce_water_weight,
            dice_weight=dice_weight,
        ) / grad_accum

        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None and use_fp16 and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_sum += float(loss.detach()) * grad_accum
        n_steps += 1

        if step % grad_accum == 0 or step == len(loader):
            # optimizer.step() before lr_scheduler.step(); with GradScaler,
            # skip the LR step if the optimizer step was skipped (inf/nan grads).
            stepped = True
            if scaler is not None and use_fp16 and device.type == "cuda":
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            else:
                optimizer.step()
            if stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    if skipped:
        print(f"  warning: skipped {skipped} non-finite loss batches this epoch")
    if n_steps == 0:
        return float("nan")
    return loss_sum / n_steps


@torch.inference_mode()
def predict_split(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> dict[str, np.ndarray]:
    """stem → uint8 pred at label resolution."""
    model.eval()
    pred_by_stem: dict[str, np.ndarray] = {}
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"]
        stems = batch["stem"]
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(pixel_values=pixel_values)
        else:
            out = model(pixel_values=pixel_values)
        logits = out.logits.float()
        hw = (labels.shape[-2], labels.shape[-1])
        if logits.shape[-2:] != hw:
            logits = F.interpolate(logits, size=hw, mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for i, stem in enumerate(stems):
            pred_by_stem[stem] = preds[i]
    return pred_by_stem


def save_checkpoint(
    path: Path,
    model: SegformerForSemanticSegmentation,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    *,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    bad_epochs: int,
    history: list[dict],
    args: argparse.Namespace,
    save_hf: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_water_iou": best_metric,
        "best_epoch": best_epoch,
        "bad_epochs": bad_epochs,
        "history": history,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    torch.save(payload, path)
    if save_hf:
        hf_dir = path.parent / "best_hf"
        hf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(hf_dir)


def load_train_state(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_resume_path(resume: str | None, out_dir: Path, *, no_resume: bool) -> Path | None:
    """Pick a checkpoint to resume, or None to start from scratch."""
    if no_resume or resume is None:
        return None
    if resume == "auto":
        for name in ("last.pt", "best.pt"):
            cand = out_dir / name
            if cand.is_file():
                return cand
        return None
    path = Path(resume)
    return path if path.is_file() else None


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler | None:
    if not enabled:
        return None
    # torch.amp.GradScaler('cuda') — torch.cuda.amp.GradScaler is deprecated
    return torch.amp.GradScaler("cuda", enabled=True)


def smoke_test(
    model: SegformerForSemanticSegmentation,
    train_loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    *,
    ce_water_weight: float = 4.0,
    dice_weight: float = 1.0,
) -> None:
    model.train()
    batch = next(iter(train_loader))
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    if not torch.isfinite(pixel_values).all():
        n_bad = int((~torch.isfinite(pixel_values)).sum())
        print(f"smoke warning: {n_bad} non-finite input values (will nan_to_num)")
        pixel_values = torch.nan_to_num(pixel_values, nan=0.0, posinf=0.0, neginf=0.0)
    uniq = torch.unique(labels).tolist()
    print(f"smoke labels unique={uniq}  ignore_index={IGNORE_INDEX}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    logits = forward_logits(model, pixel_values, use_fp16=use_fp16, device=device)
    loss = segmentation_loss(
        logits,
        labels,
        ce_water_weight=ce_water_weight,
        dice_weight=dice_weight,
    )
    if not torch.isfinite(loss):
        raise SystemExit(
            f"smoke FAILED: non-finite loss={loss}. Check SAR NaNs / label values."
        )
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"smoke OK  loss={float(loss):.4f}  time={time.time()-t0:.2f}s  peak_vram={peak_mb:.0f} MiB")
    else:
        print(f"smoke OK  loss={float(loss):.4f}  time={time.time()-t0:.2f}s  device={device}")


def main() -> None:
    args = parse_args()
    root = args.data_root
    splits = root / args.splits_dir
    for name in (
        "flood_train_data.csv",
        "flood_valid_data.csv",
        "flood_test_data.csv",
    ):
        if not (splits / name).exists():
            raise SystemExit(f"Missing split file: {splits / name}")

    set_seed(args.seed)
    device = pick_device(args.device)
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device}  fp16={use_fp16}  model={args.model_id}  "
        f"batch={args.batch_size}x{args.grad_accum}  seed={args.seed}"
    )
    print(
        f"loss: CE water weight={args.ce_water_weight:g}  dice weight={args.dice_weight:g}  "
        f"oversample_sparse={not args.no_oversample_sparse}  "
        f"water_crop_p={args.water_crop_p:g}"
    )
    print(f"data-root={root}")
    print(f"out-dir={args.out_dir}")

    ds_kw = dict(
        s1_dir=args.s1_dir,
        label_dir=args.label_dir,
        image_size=args.image_size,
        db_min=args.db_min,
        db_max=args.db_max,
    )
    print("Building datasets...")
    train_ds = Sen1Floods11SegDataset(
        root,
        splits / "flood_train_data.csv",
        augment=True,
        water_crop_p=args.water_crop_p,
        water_crop_size=args.water_crop_size,
        **ds_kw,
    )
    val_ds = Sen1Floods11SegDataset(
        root, splits / "flood_valid_data.csv", augment=False, **ds_kw
    )
    test_ds = Sen1Floods11SegDataset(
        root, splits / "flood_test_data.csv", augment=False, **ds_kw
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_counts = train_ds.water_pixel_counts()
    n_dry = int((train_counts == 0).sum())
    n_small = int(((train_counts > 0) & (train_counts <= args.small_water_max)).sum())
    print(
        f"  train water pixels: dry={n_dry}  "
        f"small(1..{args.small_water_max})={n_small}  "
        f"large={len(train_counts) - n_dry - n_small}"
    )

    train_kw: dict = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )
    if args.no_oversample_sparse:
        train_loader = DataLoader(train_ds, shuffle=True, **train_kw)
        print("  train sampler: uniform")
    else:
        sampler = WeightedRandomSampler(
            sparse_sample_weights(train_counts),
            num_samples=len(train_ds),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **train_kw)
        print("  train sampler: WeightedRandomSampler 1/sqrt(water+1)")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    print("Loading model...")
    model = load_model(args.model_id, device)

    eval_kw = dict(
        ce_water_weight=args.ce_water_weight,
        dice_weight=args.dice_weight,
        small_water_max=args.small_water_max,
    )
    curves_path = args.out_dir / "figures" / "training_curves.png"

    if args.smoke:
        smoke_test(
            model,
            train_loader,
            device,
            use_fp16,
            ce_water_weight=args.ce_water_weight,
            dice_weight=args.dice_weight,
        )
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = make_grad_scaler(use_fp16) if device.type == "cuda" else None

    history: list[dict] = []
    best_iou = -1.0
    best_epoch = -1
    bad_epochs = 0
    start_epoch = 1
    last_path = args.out_dir / "last.pt"
    best_path = args.out_dir / "best.pt"

    resume_path = resolve_resume_path(args.resume, args.out_dir, no_resume=args.no_resume)
    if args.no_resume:
        print("--no-resume: starting from scratch")
    elif resume_path is not None:
        state = load_train_state(resume_path, device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        history = list(state.get("history") or [])
        best_iou = float(state.get("best_val_water_iou", -1.0))
        best_epoch = int(state.get("best_epoch", state.get("epoch", -1)))
        bad_epochs = int(state.get("bad_epochs", 0))
        start_epoch = int(state["epoch"]) + 1
        if any(
            (not math.isfinite(float(r.get("train_loss", 0))))
            or (not math.isfinite(float(r.get("val_loss", 0))))
            for r in history
        ):
            raise SystemExit(
                f"Checkpoint {resume_path} has non-finite losses (poisoned run). "
                f"Delete last.pt/best.pt under {args.out_dir} and re-run with --no-resume."
            )
        print(
            f"Resumed from {resume_path}  "
            f"(next epoch={start_epoch}, best_val_water_iou={best_iou:.4f} @ epoch {best_epoch})"
        )
        if start_epoch > args.epochs:
            print(f"Checkpoint already finished {args.epochs} epochs; skipping train loop.")
    else:
        if args.resume not in (None, "auto"):
            print(f"Checkpoint not found: {args.resume}; starting from scratch")
        else:
            print(
                f"No checkpoint found under {args.out_dir} "
                f"(looked for last.pt / best.pt); starting from scratch"
            )

    print(
        f"Training epochs {start_epoch}..{args.epochs}  "
        f"(early stop patience={args.patience} on val water IoU)..."
    )
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_fp16=use_fp16,
            grad_accum=args.grad_accum,
            scaler=scaler,
            ce_water_weight=args.ce_water_weight,
            dice_weight=args.dice_weight,
        )
        val_metrics = evaluate(model, val_loader, device, use_fp16, **eval_kw)
        dt = time.time() - t0
        small_iou = val_metrics.get("small_water_iou")
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_water_iou": val_metrics["water_iou"],
            "val_miou": val_metrics["miou"],
            "val_precision_water": val_metrics["precision_water"],
            "val_recall_water": val_metrics["recall_water"],
            "val_f1_water": val_metrics["f1_water"],
            "val_small_water_iou": small_iou,
            "val_dry_fp_rate": val_metrics["dry_fp_rate"],
            "sec": dt,
        }
        history.append(row)
        small_s = "n/a" if small_iou is None else f"{small_iou:.4f}"
        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_water_iou={val_metrics['water_iou']:.4f}  "
            f"val_small_water_iou={small_s}  "
            f"val_recall={val_metrics['recall_water']:.4f}  "
            f"val_dry_fp_rate={val_metrics['dry_fp_rate']:.3f}  ({dt:.1f}s)"
        )

        if not math.isfinite(train_loss) or not math.isfinite(val_metrics["loss"]):
            raise SystemExit(
                "Non-finite loss — aborting so we don't write a poisoned checkpoint. "
                "Re-copy the latest train/ + eval/common.py, delete bad "
                f"{args.out_dir}/last.pt and best.pt if present, then re-run with --no-resume."
            )

        is_best = val_metrics["water_iou"] > best_iou
        if is_best:
            best_iou = val_metrics["water_iou"]
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Always write last.pt so Colab disconnects can resume.
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            best_metric=best_iou,
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            history=history,
            args=args,
            save_hf=False,
        )
        with open(args.out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, curves_path, best_epoch=best_epoch)

        if is_best:
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                best_metric=best_iou,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                history=history,
                args=args,
                save_hf=True,
            )
            print(f"  ↑ new best val water IoU={best_iou:.4f}  saved {best_path}")

        if bad_epochs >= args.patience:
            print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    if history:
        plot_training_curves(history, curves_path, best_epoch=best_epoch)

    # Reload best weights
    ckpt_path = best_path
    if ckpt_path.exists():
        state = load_train_state(ckpt_path, device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Reloaded best checkpoint from epoch {state['epoch']}")
    elif last_path.exists():
        state = load_train_state(last_path, device)
        model.load_state_dict(state["model_state_dict"])
        print(f"No best.pt; using last checkpoint from epoch {state['epoch']}")
        ckpt_path = last_path
    else:
        print("WARNING: no checkpoint found; evaluating current in-memory weights")

    print("Evaluating best checkpoint...")
    val_metrics = evaluate(model, val_loader, device, use_fp16, **eval_kw)
    test_metrics = evaluate(model, test_loader, device, use_fp16, **eval_kw)
    print_metrics("Validation (best)", val_metrics)
    small_v = val_metrics.get("small_water_iou")
    print(
        f"  {'small_water_iou':18s} "
        f"{'n/a' if small_v is None else f'{small_v:.4f}'}  "
        f"dry_fp_rate={val_metrics['dry_fp_rate']:.3f}  "
        f"(n_small={val_metrics['n_small_water']} n_dry={val_metrics['n_dry']})"
    )
    print_metrics("Test (best)", test_metrics)
    small_t = test_metrics.get("small_water_iou")
    print(
        f"  {'small_water_iou':18s} "
        f"{'n/a' if small_t is None else f'{small_t:.4f}'}  "
        f"dry_fp_rate={test_metrics['dry_fp_rate']:.3f}  "
        f"(n_small={test_metrics['n_small_water']} n_dry={test_metrics['n_dry']})"
    )

    summary = {
        "method": "SegFormer_finetune",
        "model_id": args.model_id,
        "num_labels": 2,
        "input_adapt": "VV_VH_mean_clip_db_imagenet_norm",
        "db_min": args.db_min,
        "db_max": args.db_max,
        "image_size": args.image_size,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "ce_water_weight": args.ce_water_weight,
        "dice_weight": args.dice_weight,
        "oversample_sparse": not args.no_oversample_sparse,
        "water_crop_p": args.water_crop_p,
        "water_crop_size": args.water_crop_size,
        "small_water_max": args.small_water_max,
        "fp16": use_fp16,
        "device": str(device),
        "seed": args.seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "checkpoint": str(ckpt_path),
        "hf_dir": str(args.out_dir / "best_hf"),
        "training_curves": str(curves_path),
        "val": val_metrics,
        "test": test_metrics,
        "vh_baseline_test_water_iou_ref": 0.53,
    }
    save_metrics_json(args.out_dir / "metrics.json", summary)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if history:
        plot_training_curves(history, curves_path, best_epoch=best_epoch)

    # Qualitative panels on test
    if args.num_viz > 0:
        print("Saving visualizations...")
        pred_by_stem = predict_split(model, test_loader, device, use_fp16)
        chosen = choose_stems_for_viz(
            root,
            args.s1_dir,
            args.label_dir,
            list(pred_by_stem.keys()),
            args.num_viz,
            args.seed,
        )
        viz_dir = args.out_dir / "figures"
        for stem in chosen:
            s1_path, lab_path = resolve_pair(root, args.s1_dir, args.label_dir, stem)
            vv, vh, label = load_s1_and_label(s1_path, lab_path)
            model_rgb = s1_to_pseudo_rgb(vv, vh, db_min=args.db_min, db_max=args.db_max)
            visualize_prediction(
                stem,
                vh,
                label,
                pred_by_stem[stem],
                viz_dir / f"{stem}.png",
                sar_title="VH (dB)",
                pred_title="SegFormer fine-tuned\nargmax",
                model_input=model_rgb,
                model_input_title="SegFormer input\n(VV/VH/mean → [0,1])",
            )
        print(f"Wrote {len(chosen)} figures under {viz_dir}")

    print(f"Metrics: {args.out_dir / 'metrics.json'}")
    print(f"Curves: {curves_path}")
    print(f"Best HF weights: {args.out_dir / 'best_hf'}")
    print(
        f"Compare test water IoU={test_metrics['water_iou']:.4f} "
        f"vs VH baseline ~0.53"
    )


if __name__ == "__main__":
    main()
