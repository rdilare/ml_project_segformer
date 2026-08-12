#!/usr/bin/env python3
"""Fine-tune SegFormer (mit-b0) on Sen1Floods11 hand-labeled chips.

Designed for Colab T4: fp16, batch 4–8, grad accumulation, early stop on
val water IoU. Checkpoints + metrics go under --out-dir (put that on Drive).

Every epoch writes ``last.pt`` (full train state) and, on improvement,
``best.pt`` + ``best_hf/``. Resume after a disconnect with ``--resume``.

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

  # After interrupt / disconnect:
  python train/finetune_segformer.py ... --fp16 --resume
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines.common import (  # noqa: E402
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
        "--smoke",
        action="store_true",
        help="One train step + one val batch, print peak CUDA memory, exit",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Resume from a checkpoint. Use --resume alone for out-dir/last.pt, "
            "or --resume PATH to a .pt file"
        ),
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
    # Sen1Floods11 nodata; HF default ignore is 255
    model.config.semantic_loss_ignore_index = -1
    model.to(device)
    return model


@torch.no_grad()
def evaluate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> dict[str, float]:
    model.eval()
    totals = empty_confusion()
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(pixel_values=pixel_values, labels=labels)
        else:
            out = model(pixel_values=pixel_values, labels=labels)
        loss_sum += float(out.loss.detach())
        n_batches += 1
        logits = out.logits.float()
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        preds = logits.argmax(dim=1)
        for i in range(preds.shape[0]):
            add_confusion(
                totals,
                confusion_counts(
                    preds[i].cpu().numpy().astype(np.uint8),
                    labels[i].cpu().numpy(),
                ),
            )
    metrics = metrics_from_counts(totals)
    metrics["loss"] = loss_sum / max(n_batches, 1)
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
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, n_steps = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        if use_fp16 and device.type == "cuda" and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
        else:
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()

        loss_sum += float(out.loss.detach())
        n_steps += 1

        if step % grad_accum == 0 or step == len(loader):
            if scaler is not None and use_fp16 and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return loss_sum / max(n_steps, 1)


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
    scaler: torch.cuda.amp.GradScaler | None,
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


def resolve_resume_path(resume: str | None, out_dir: Path) -> Path | None:
    if resume is None:
        return None
    if resume == "auto":
        return out_dir / "last.pt"
    return Path(resume)


def smoke_test(
    model: SegformerForSemanticSegmentation,
    train_loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> None:
    model.train()
    batch = next(iter(train_loader))
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    if use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(pixel_values=pixel_values, labels=labels)
        out.loss.backward()
    else:
        out = model(pixel_values=pixel_values, labels=labels)
        out.loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"smoke OK  loss={float(out.loss):.4f}  time={time.time()-t0:.2f}s  peak_vram={peak_mb:.0f} MiB")
    else:
        print(f"smoke OK  loss={float(out.loss):.4f}  time={time.time()-t0:.2f}s  device={device}")


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
        root, splits / "flood_train_data.csv", augment=True, **ds_kw
    )
    val_ds = Sen1Floods11SegDataset(
        root, splits / "flood_valid_data.csv", augment=False, **ds_kw
    )
    test_ds = Sen1Floods11SegDataset(
        root, splits / "flood_test_data.csv", augment=False, **ds_kw
    )
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )
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

    if args.smoke:
        smoke_test(model, train_loader, device, use_fp16)
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if device.type == "cuda" else None

    history: list[dict] = []
    best_iou = -1.0
    best_epoch = -1
    bad_epochs = 0
    start_epoch = 1
    last_path = args.out_dir / "last.pt"
    best_path = args.out_dir / "best.pt"

    resume_path = resolve_resume_path(args.resume, args.out_dir)
    if resume_path is not None:
        if not resume_path.exists():
            raise SystemExit(f"--resume requested but checkpoint not found: {resume_path}")
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
        print(
            f"Resumed from {resume_path}  "
            f"(next epoch={start_epoch}, best_val_water_iou={best_iou:.4f} @ epoch {best_epoch})"
        )
        if start_epoch > args.epochs:
            print(f"Checkpoint already finished {args.epochs} epochs; skipping train loop.")

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
        )
        val_metrics = evaluate(model, val_loader, device, use_fp16)
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_water_iou": val_metrics["water_iou"],
            "val_miou": val_metrics["miou"],
            "sec": dt,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_water_iou={val_metrics['water_iou']:.4f}  "
            f"val_miou={val_metrics['miou']:.4f}  ({dt:.1f}s)"
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
    val_metrics = evaluate(model, val_loader, device, use_fp16)
    test_metrics = evaluate(model, test_loader, device, use_fp16)
    print_metrics("Validation (best)", val_metrics)
    print_metrics("Test (best)", test_metrics)

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
        "fp16": use_fp16,
        "device": str(device),
        "seed": args.seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "checkpoint": str(ckpt_path),
        "hf_dir": str(args.out_dir / "best_hf"),
        "val": val_metrics,
        "test": test_metrics,
        "vh_baseline_test_water_iou_ref": 0.53,
    }
    save_metrics_json(args.out_dir / "metrics.json", summary)
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

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
    print(f"Best HF weights: {args.out_dir / 'best_hf'}")
    print(
        f"Compare test water IoU={test_metrics['water_iou']:.4f} "
        f"vs VH baseline ~0.53"
    )


if __name__ == "__main__":
    main()
