#!/usr/bin/env python3
"""Streamlit workspace for Sen1Floods11 SegFormer.

Launch from the repo root::

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.common import (  # noqa: E402
    METRIC_PRINT_KEYS,
    choose_stems_for_viz,
    confusion_counts,
    load_s1_and_label,
    load_s1_vv_vh,
    make_error_map,
    make_gt_rgb,
    metrics_from_counts,
    percentile_stretch,
    read_split_stems,
    resolve_pair,
    s1_to_pseudo_rgb,
    save_metrics_json,
)

SPLIT_CSV = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
DEFAULT_DATA = REPO / "data" / "sen1floods11_hand"
DEFAULT_MODEL = REPO / "models" / "best_hf"
DEFAULT_VH_DB = -23.1
METRIC_COLS = list(METRIC_PRINT_KEYS)


def stem_from_path(path: Path) -> str:
    return re.sub(r"_(S1Hand|LabelHand|S2Hand)$", "", path.stem)


def country_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def torch_ok() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def device_choices() -> list[str]:
    opts = ["auto", "cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            opts.insert(1, "cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            opts.insert(1, "mps")
    except ImportError:
        pass
    return opts


def eval_mod():
    from eval import finetuned as ev

    return ev


@st.cache_data(show_spinner=False)
def list_stems(data_root: str, split: str) -> list[str]:
    root = Path(data_root)
    s1_dir = root / "S1"
    if split == "all chips":
        if not s1_dir.is_dir():
            return []
        return sorted(stem_from_path(p) for p in s1_dir.glob("*_S1Hand.tif"))
    csv_path = root / "splits" / SPLIT_CSV[split]
    if not csv_path.is_file():
        return []
    stems = read_split_stems(csv_path)
    out: list[str] = []
    for stem in stems:
        direct = s1_dir / f"{stem}_S1Hand.tif"
        if direct.exists() or (s1_dir.is_dir() and any(s1_dir.glob(f"{stem}*.tif"))):
            out.append(stem)
    return out


@st.cache_data(show_spinner=False)
def scan_workspace(data_root: str, model_dir: str) -> dict:
    root = Path(data_root)
    s1 = root / "S1"
    labels = root / "Labels"
    splits = root / "splits"
    n_s1 = len(list(s1.glob("*.tif"))) if s1.is_dir() else 0
    n_lab = len(list(labels.glob("*.tif"))) if labels.is_dir() else 0
    present = [name for name in SPLIT_CSV.values() if (splits / name).is_file()]
    model = Path(model_dir)
    has_model = (model / "config.json").is_file() and (
        (model / "model.safetensors").is_file() or (model / "pytorch_model.bin").is_file()
    )
    return {
        "n_s1": n_s1,
        "n_lab": n_lab,
        "splits": present,
        "has_model": has_model,
        "s1_ok": s1.is_dir(),
        "root_ok": root.is_dir(),
    }


@st.cache_data(show_spinner=False)
def load_chip_arrays(s1_path: str, lab_path: str | None):
    s1 = Path(s1_path)
    if lab_path:
        vv, vh, label = load_s1_and_label(s1, Path(lab_path))
        return vv, vh, label
    vv, vh = load_s1_vv_vh(s1)
    return vv, vh, None


def resolve_chip_paths(data_root: Path, stem: str) -> tuple[Path, Path | None]:
    try:
        return resolve_pair(data_root, "S1", "Labels", stem)
    except FileNotFoundError:
        s1 = data_root / "S1" / f"{stem}_S1Hand.tif"
        if not s1.exists():
            matches = sorted((data_root / "S1").glob(f"{stem}*.tif")) if (data_root / "S1").is_dir() else []
            if not matches:
                raise
            s1 = matches[0]
        lab = data_root / "Labels" / f"{stem}_LabelHand.tif"
        return s1, lab if lab.exists() else None


def gray01(arr: np.ndarray) -> np.ndarray:
    return percentile_stretch(arr)


def pred_rgb(pred: np.ndarray) -> np.ndarray:
    rgb = np.full((*pred.shape, 3), 0.9, dtype=np.float32)
    rgb[pred == 1] = (0.12, 0.40, 0.90)
    return rgb


def flood_overlay(vh: np.ndarray, pred: np.ndarray, alpha: float) -> np.ndarray:
    base = gray01(vh)
    rgb = np.stack([base, base, base], axis=-1)
    water = pred == 1
    color = np.array([0.12, 0.45, 0.95], dtype=np.float32)
    rgb[water] = (1.0 - alpha) * rgb[water] + alpha * color
    return np.clip(rgb, 0.0, 1.0)


def label_counts(label: np.ndarray) -> str:
    names = {-1: "nodata", 0: "not-water", 1: "water"}
    vals, counts = np.unique(label, return_counts=True)
    parts = [f"{names.get(int(v), v)}={int(c):,}" for v, c in zip(vals, counts)]
    return "  ".join(parts)


def show_metrics(metrics: dict) -> None:
    cols = st.columns(5)
    labels = [
        ("Water IoU", "water_iou"),
        ("mIoU", "miou"),
        ("Precision", "precision_water"),
        ("Recall", "recall_water"),
        ("F1", "f1_water"),
    ]
    for col, (title, key) in zip(cols, labels):
        col.metric(title, f"{metrics[key]:.3f}")


def metrics_or_none(pred: np.ndarray, label: np.ndarray | None) -> dict | None:
    if label is None:
        return None
    return metrics_from_counts(confusion_counts(pred, label))


@st.cache_resource(show_spinner="Loading SegFormer weights…")
def load_finetuned(model_dir: str, device_name: str):
    ev = eval_mod()
    device = ev.pick_device(device_name)
    model = ev.load_model(Path(model_dir), device)
    return model, device


@st.cache_data(show_spinner="Running SegFormer…", max_entries=16)
def infer_chip_cached(
    model_dir: str,
    device_name: str,
    s1_path: str,
    image_size: int,
    db_min: float,
    db_max: float,
    use_fp16: bool,
):
    """Cache logits-side outputs per chip so the threshold slider is instant."""
    import torch
    import torch.nn.functional as F
    from eval.common import normalize_imagenet_chw

    model, device = load_finetuned(model_dir, device_name)
    vv, vh = load_s1_vv_vh(Path(s1_path))
    rgb01 = s1_to_pseudo_rgb(vv, vh, db_min=db_min, db_max=db_max)
    chw = normalize_imagenet_chw(rgb01)
    t = torch.from_numpy(chw).unsqueeze(0)
    if t.shape[-2] != image_size or t.shape[-1] != image_size:
        t = F.interpolate(
            t, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    t = t.to(device)
    with torch.inference_mode():
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=t).logits.float()
        else:
            logits = model(pixel_values=t).logits.float()
        out_hw = (int(vv.shape[0]), int(vv.shape[1]))
        if logits.shape[-2:] != out_hw:
            logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
    return vv, vh, rgb01, prob


def stream_command(cmd: list[str]) -> tuple[int, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    box = st.empty()
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        box.code("\n".join(lines[-80:]), language="text")
    code = proc.wait()
    return code, "\n".join(lines)


def extract_score_rows(path: Path, payload: dict) -> list[dict]:
    method = str(payload.get("method", path.parent.name))
    run = str(path.parent.relative_to(REPO)) if path.is_relative_to(REPO) else str(path.parent)
    rows: list[dict] = []

    def row(split: str, metrics: dict | None, extra: dict | None = None) -> None:
        if not isinstance(metrics, dict) or "water_iou" not in metrics:
            return
        item = {
            "run": run,
            "method": method,
            "split": split,
            **{k: metrics.get(k) for k in METRIC_COLS},
        }
        if extra:
            item.update(extra)
        rows.append(item)

    for split_key in ("val", "test"):
        if split_key in payload:
            row(split_key, payload.get(split_key))
    if "metrics" in payload:
        split = str(payload.get("split") or payload.get("stem") or "chip")
        row(split, payload.get("metrics"))
    return rows


@st.cache_data(show_spinner=False)
def load_run_summaries(outputs_dir: str) -> tuple[pd.DataFrame, list[str]]:
    root = Path(outputs_dir)
    rows: list[dict] = []
    runs: list[str] = []
    if not root.is_dir():
        return pd.DataFrame(), []
    for metrics_path in sorted(root.glob("**/metrics.json")):
        runs.append(str(metrics_path.parent))
        try:
            payload = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(extract_score_rows(metrics_path, payload))
    df = pd.DataFrame(rows)
    return df, runs


def panel_row(images: list[tuple[np.ndarray, str]]) -> None:
    cols = st.columns(len(images))
    for col, (img, caption) in zip(cols, images):
        col.image(img, caption=caption, use_container_width=True, clamp=True)


def save_inference_outputs(
    *,
    stem: str,
    vh: np.ndarray,
    pred: np.ndarray,
    rgb01: np.ndarray,
    label: np.ndarray | None,
    metrics: dict | None,
    out_dir: Path,
    model_dir: Path,
    device: str,
) -> Path:
    ev = eval_mod()
    fig_path = out_dir / "figures" / f"{stem}.png"
    ev.save_chip_figure(
        stem=stem, vh=vh, pred=pred, rgb01=rgb01, label=label, out_path=fig_path
    )
    summary = {
        "method": "SegFormer_finetune",
        "model_dir": str(model_dir),
        "stem": stem,
        "decision": "P(water) threshold / argmax",
        "device": device,
        "metrics": metrics,
        "figure": str(fig_path),
    }
    save_metrics_json(out_dir / "metrics.json", summary)
    return fig_path


def sidebar_config() -> dict:
    st.sidebar.header("Workspace")
    data_root = Path(
        st.sidebar.text_input("Data root", value=str(DEFAULT_DATA))
    ).expanduser()
    model_dir = Path(
        st.sidebar.text_input("Model dir (models/best_hf)", value=str(DEFAULT_MODEL))
    ).expanduser()
    out_dir = Path(
        st.sidebar.text_input("Output dir", value=str(REPO / "outputs" / "finetuned_eval"))
    ).expanduser()

    status = scan_workspace(str(data_root), str(model_dir))
    st.sidebar.caption(
        f"S1 chips: {status['n_s1']} · labels: {status['n_lab']} · "
        f"model: {'yes' if status['has_model'] else 'missing'}"
    )
    if not status["root_ok"]:
        st.sidebar.error("Data root not found")
    elif not status["s1_ok"]:
        st.sidebar.warning("No S1/ folder under data root")

    if st.sidebar.button("Refresh file lists"):
        list_stems.clear()
        scan_workspace.clear()
        load_chip_arrays.clear()
        load_run_summaries.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Chip")
    split = st.sidebar.selectbox(
        "Split",
        ["test", "valid", "train", "bolivia", "all chips"],
    )
    stems = list_stems(str(data_root), split)
    countries = sorted({country_of(s) for s in stems})
    country = st.sidebar.selectbox("Country", ["All"] + countries)
    if country != "All":
        stems = [s for s in stems if country_of(s) == country]
    stem = st.sidebar.selectbox(
        f"Chip ({len(stems)})",
        stems if stems else ["(none)"],
        disabled=not stems,
    )

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    device_name = st.sidebar.selectbox("Device", device_choices())
    vh_db = st.sidebar.slider(
        "VH threshold (dB)",
        min_value=-30.0,
        max_value=-10.0,
        value=float(DEFAULT_VH_DB),
        step=0.1,
    )
    overlay_alpha = st.sidebar.slider("Flood overlay opacity", 0.1, 0.9, 0.55, 0.05)
    with st.sidebar.expander("Advanced"):
        image_size = st.number_input("Image size", min_value=128, max_value=1024, value=512, step=32)
        db_min = st.number_input("dB min", value=-30.0, step=1.0)
        db_max = st.number_input("dB max", value=0.0, step=1.0)
        use_fp16 = st.checkbox("CUDA fp16", value=False)
    return {
        "data_root": data_root,
        "model_dir": model_dir,
        "out_dir": out_dir,
        "status": status,
        "split": split,
        "stems": stems,
        "stem": None if stem == "(none)" else stem,
        "device_name": device_name,
        "image_size": int(image_size),
        "db_min": float(db_min),
        "db_max": float(db_max),
        "use_fp16": bool(use_fp16),
        "vh_db": float(vh_db),
        "overlay_alpha": float(overlay_alpha),
    }


def page_browse(cfg: dict) -> None:
    stem = cfg["stem"]
    if stem is None:
        st.info("Point the sidebar at a data root that contains S1/ chips.")
        return
    try:
        s1_path, lab_path = resolve_chip_paths(cfg["data_root"], stem)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    vv, vh, label = load_chip_arrays(str(s1_path), str(lab_path) if lab_path else None)
    st.subheader(stem)
    st.caption(str(s1_path))

    images = [
        (gray01(vv), "VV (dB, percentile stretch)"),
        (gray01(vh), "VH (dB, percentile stretch)"),
    ]
    if label is not None:
        images.append((make_gt_rgb(label), "Ground truth (blue=water, black=nodata)"))
    panel_row(images)

    c1, c2, c3 = st.columns(3)
    finite_vv = vv[np.isfinite(vv)]
    finite_vh = vh[np.isfinite(vh)]
    c1.write(f"VV  min {finite_vv.min():.1f}  max {finite_vv.max():.1f}  mean {finite_vv.mean():.1f}")
    c2.write(f"VH  min {finite_vh.min():.1f}  max {finite_vh.max():.1f}  mean {finite_vh.mean():.1f}")
    if label is not None:
        c3.write(label_counts(label))
    else:
        c3.write("No label found")

    st.divider()
    st.markdown("**VH threshold baseline on this chip**")
    pred_vh = (vh <= cfg["vh_db"]).astype(np.uint8)
    metrics = metrics_or_none(pred_vh, label)
    if metrics:
        show_metrics(metrics)
    vh_panels = [
        (pred_rgb(pred_vh), f"Pred  VH ≤ {cfg['vh_db']:.1f} dB"),
        (flood_overlay(vh, pred_vh, cfg["overlay_alpha"]), "Overlay on VH"),
    ]
    if label is not None:
        vh_panels.append((make_error_map(pred_vh, label), "Error  G=TP  R=FP  B=FN"))
    panel_row(vh_panels)


def page_infer(cfg: dict) -> None:
    st.subheader("Fine-tuned SegFormer inference")
    if not torch_ok():
        st.error("PyTorch is not installed in this environment. `pip install -r requirements.txt`")
        return
    if not cfg["status"]["has_model"]:
        st.warning(f"No Hugging Face weights at `{cfg['model_dir']}` (need config.json + model.safetensors).")

    source = st.radio("Input", ["Dataset chip", "Upload GeoTIFF"], horizontal=True)
    s1_path: Path | None = None
    lab_path: Path | None = None
    stem = cfg["stem"] or "upload"

    if source == "Dataset chip":
        if stem is None:
            st.info("Select a chip in the sidebar.")
            return
        try:
            s1_path, lab_path = resolve_chip_paths(cfg["data_root"], stem)
        except FileNotFoundError as e:
            st.error(str(e))
            return
    else:
        uploaded = st.file_uploader("Sentinel-1 GeoTIFF (VV + VH bands)", type=["tif", "tiff"])
        if uploaded is None:
            st.caption("Upload a two-band S1 chip, or switch back to a dataset stem.")
            return
        tmp_dir = Path(st.session_state.setdefault("_upload_dir", str(REPO / "outputs" / "_uploads")))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        s1_path = tmp_dir / uploaded.name
        s1_path.write_bytes(uploaded.getbuffer())
        stem = stem_from_path(s1_path)
        cand = cfg["data_root"] / "Labels" / f"{stem}_LabelHand.tif"
        lab_path = cand if cand.exists() else None

    vv, vh, label = load_chip_arrays(str(s1_path), str(lab_path) if lab_path else None)
    st.caption(f"{stem}  ·  {s1_path}")

    thresh = st.slider(
        "Water probability threshold (0.5 ≈ argmax)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
    )
    if st.button("Run inference", type="primary"):
        st.session_state["infer_s1"] = str(s1_path)

    if st.session_state.get("infer_s1") != str(s1_path):
        st.caption("Click **Run inference** to load weights and predict this chip.")
        return

    try:
        _vv, _vh, rgb01, prob = infer_chip_cached(
            str(cfg["model_dir"]),
            cfg["device_name"],
            str(s1_path),
            cfg["image_size"],
            cfg["db_min"],
            cfg["db_max"],
            cfg["use_fp16"],
        )
    except SystemExit as e:
        st.error(str(e))
        return
    except Exception as e:
        st.exception(e)
        return

    pred = (prob >= thresh).astype(np.uint8)
    metrics = metrics_or_none(pred, label)
    if metrics:
        show_metrics(metrics)
    else:
        st.caption("No ground-truth label — showing prediction only.")

    top = [
        (gray01(vh), "VH"),
        (np.clip(rgb01, 0, 1), "Model input (VV / VH / mean)"),
    ]
    if label is not None:
        top.append((make_gt_rgb(label), "Ground truth"))
    top.append((pred_rgb(pred), f"Pred  P(water) ≥ {thresh:.2f}"))
    panel_row(top)
    bottom = [
        (flood_overlay(vh, pred, cfg["overlay_alpha"]), "Flood overlay"),
        (prob, "P(water)"),
    ]
    if label is not None:
        bottom.append((make_error_map(pred, label), "Error  G=TP  R=FP  B=FN"))
    panel_row(bottom)

    pred_vh = (vh <= cfg["vh_db"]).astype(np.uint8)
    with st.expander("Compare with VH threshold", expanded=False):
        cmp_panels = [
            (pred_rgb(pred_vh), f"VH ≤ {cfg['vh_db']:.1f} dB"),
            (pred_rgb(pred), "SegFormer"),
        ]
        if label is not None:
            cmp_panels.append((make_gt_rgb(label), "Ground truth"))
            m_vh = metrics_or_none(pred_vh, label)
            if m_vh and metrics:
                c1, c2 = st.columns(2)
                c1.metric("VH water IoU", f"{m_vh['water_iou']:.3f}")
                c2.metric("SegFormer water IoU", f"{metrics['water_iou']:.3f}")
        panel_row(cmp_panels)

    if st.button("Save figure + metrics.json"):
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        fig_path = save_inference_outputs(
            stem=stem,
            vh=vh,
            pred=pred,
            rgb01=rgb01,
            label=label,
            metrics=metrics,
            out_dir=cfg["out_dir"],
            model_dir=cfg["model_dir"],
            device=cfg["device_name"],
        )
        load_run_summaries.clear()
        st.success(f"Wrote {fig_path} and {cfg['out_dir'] / 'metrics.json'}")


def page_evaluate(cfg: dict) -> None:
    st.subheader("Evaluate a split")
    st.caption("Runs the fine-tuned model on every chip in an official split and writes metrics + figures.")
    if not torch_ok():
        st.error("PyTorch is not installed in this environment.")
        return

    c1, c2, c3 = st.columns(3)
    split = c1.selectbox("Split to score", ["test", "valid", "train", "bolivia"], index=0)
    num_viz = c2.number_input("Figures to save", min_value=0, max_value=32, value=8)
    seed = c3.number_input("Viz seed", min_value=0, max_value=10_000, value=24)

    if st.button("Evaluate split", type="primary"):
        ev = eval_mod()
        try:
            model, device = load_finetuned(str(cfg["model_dir"]), cfg["device_name"])
        except SystemExit as e:
            st.error(str(e))
            return

        csv_path = cfg["data_root"] / "splits" / SPLIT_CSV[split]
        if not csv_path.exists():
            st.error(f"Missing split file: {csv_path}")
            return
        stems = read_split_stems(csv_path)
        totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        per_chip: list[dict] = []
        pred_by_stem: dict[str, np.ndarray] = {}
        missing = 0
        bar = st.progress(0.0, text="Starting…")
        n = max(len(stems), 1)
        for i, stem in enumerate(stems):
            try:
                s1_path, lab_path = resolve_pair(cfg["data_root"], "S1", "Labels", stem)
            except FileNotFoundError:
                missing += 1
                bar.progress((i + 1) / n, text=f"skip {stem} ({i + 1}/{len(stems)})")
                continue
            vv, vh, label = load_s1_and_label(s1_path, lab_path)
            pred, rgb01 = ev.predict_chip(
                model,
                vv,
                vh,
                image_size=cfg["image_size"],
                db_min=cfg["db_min"],
                db_max=cfg["db_max"],
                device=device,
                use_fp16=cfg["use_fp16"],
            )
            counts = confusion_counts(pred, label)
            for k in totals:
                totals[k] += counts[k]
            chip_metrics = metrics_from_counts(counts)
            per_chip.append({"stem": stem, "country": country_of(stem), **chip_metrics})
            pred_by_stem[stem] = pred
            bar.progress(
                (i + 1) / n,
                text=f"{stem}  water IoU={chip_metrics['water_iou']:.3f}  ({i + 1}/{len(stems)})",
            )

        if not per_chip:
            st.error("No chips found for that split.")
            return

        metrics = metrics_from_counts(totals)
        st.success(f"Scored {len(per_chip)} chips" + (f" · skipped {missing} missing" if missing else ""))
        show_metrics(metrics)

        df = pd.DataFrame(per_chip).sort_values("water_iou")
        st.markdown("Per-chip water IoU (worst first)")
        st.dataframe(
            df[["stem", "country"] + METRIC_COLS],
            hide_index=True,
            use_container_width=True,
            height=360,
        )
        edges = np.linspace(0.0, 1.0, 11)
        counts, _ = np.histogram(df["water_iou"].to_numpy(), bins=edges)
        hist_df = pd.DataFrame(
            {
                "water IoU": [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(len(counts))],
                "chips": counts,
            }
        ).set_index("water IoU")
        st.caption("Water IoU distribution")
        st.bar_chart(hist_df)

        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        chosen: list[str] = []
        if num_viz > 0 and pred_by_stem:
            chosen = choose_stems_for_viz(
                cfg["data_root"], "S1", "Labels", list(pred_by_stem.keys()), int(num_viz), int(seed)
            )
            viz_dir = cfg["out_dir"] / "figures"
            for stem in chosen:
                s1_path, lab_path = resolve_pair(cfg["data_root"], "S1", "Labels", stem)
                vv, vh, label = load_s1_and_label(s1_path, lab_path)
                rgb01 = s1_to_pseudo_rgb(vv, vh, db_min=cfg["db_min"], db_max=cfg["db_max"])
                ev.save_chip_figure(
                    stem=stem,
                    vh=vh,
                    pred=pred_by_stem[stem],
                    rgb01=rgb01,
                    label=label,
                    out_path=viz_dir / f"{stem}.png",
                )
            cols = st.columns(min(4, len(chosen)) or 1)
            for i, stem in enumerate(chosen):
                cols[i % len(cols)].image(
                    str(viz_dir / f"{stem}.png"),
                    caption=stem,
                    use_container_width=True,
                )

        summary = {
            "method": "SegFormer_finetune",
            "model_dir": str(cfg["model_dir"]),
            "split": split,
            "decision": "argmax",
            "device": str(device),
            "n": len(per_chip),
            "n_missing": missing,
            "metrics": metrics,
            "per_chip": per_chip,
            "figures": [str(cfg["out_dir"] / "figures" / f"{s}.png") for s in chosen],
        }
        save_metrics_json(cfg["out_dir"] / "metrics.json", summary)
        load_run_summaries.clear()
        st.caption(f"Saved {cfg['out_dir'] / 'metrics.json'}")

    st.divider()
    st.markdown("**Training smoke test**")
    st.caption("One train step on the current data root. Useful before a Colab run; not a full training job.")
    if st.button("Run smoke test"):
        cmd = [
            sys.executable,
            "-u",
            str(REPO / "train" / "finetune_segformer.py"),
            "--data-root",
            str(cfg["data_root"]),
            "--out-dir",
            str(REPO / "outputs" / "segformer_ft_smoke"),
            "--device",
            cfg["device_name"],
            "--smoke",
        ]
        if cfg["use_fp16"]:
            cmd.append("--fp16")
        code, _log = stream_command(cmd)
        if code == 0:
            st.success("Smoke test finished")
        else:
            st.error(f"Smoke test exited with code {code}")


def page_baselines(cfg: dict) -> None:
    st.subheader("Baselines")
    st.caption("Same CLIs as the README, with live logs. Results land under outputs/ and show up in the Results tab.")
    tab_vh, tab_pt = st.tabs(["VH threshold", "Pretrained SegFormer"])

    with tab_vh:
        odir = st.text_input("VH output dir", value=str(REPO / "outputs" / "vh_baseline"))
        c1, c2, c3, c4 = st.columns(4)
        tmin = c1.number_input("Thresh min (dB)", value=-30.0)
        tmax = c2.number_input("Thresh max (dB)", value=-10.0)
        tstep = c3.number_input("Step (dB)", value=0.5)
        nviz = c4.number_input("Num viz", min_value=0, max_value=32, value=8, key="vh_nviz")
        if st.button("Tune + evaluate VH threshold", type="primary"):
            cmd = [
                sys.executable,
                "-u",
                str(REPO / "eval" / "vh_threshold.py"),
                "--data-root",
                str(cfg["data_root"]),
                "--out-dir",
                odir,
                "--thresh-min",
                str(tmin),
                "--thresh-max",
                str(tmax),
                "--thresh-step",
                str(tstep),
                "--num-viz",
                str(int(nviz)),
            ]
            code, _log = stream_command(cmd)
            load_run_summaries.clear()
            if code == 0:
                st.success(f"Finished → {odir}")
                metrics_path = Path(odir) / "metrics.json"
                if metrics_path.exists():
                    payload = json.loads(metrics_path.read_text())
                    if "test" in payload:
                        show_metrics(payload["test"])
                curve = Path(odir) / "figures" / "val_threshold_curve.png"
                if curve.exists():
                    st.image(str(curve), caption="Validation threshold sweep", use_container_width=True)
            else:
                st.error(f"Exited with code {code}")

    with tab_pt:
        if not torch_ok():
            st.error("PyTorch is not installed in this environment.")
        odir = st.text_input("Pretrained output dir", value=str(REPO / "outputs" / "segformer_pretrained"))
        model_id = st.text_input("HF model id", value="nvidia/mit-b0")
        tune = st.checkbox("Tune P(water) threshold on val", value=False)
        nviz_pt = st.number_input("Num viz", min_value=0, max_value=32, value=8, key="pt_nviz")
        if st.button("Evaluate pretrained SegFormer", type="primary", disabled=not torch_ok()):
            cmd = [
                sys.executable,
                "-u",
                str(REPO / "eval" / "pretrained.py"),
                "--data-root",
                str(cfg["data_root"]),
                "--out-dir",
                odir,
                "--model-id",
                model_id,
                "--device",
                cfg["device_name"],
                "--image-size",
                str(cfg["image_size"]),
                "--num-viz",
                str(int(nviz_pt)),
            ]
            if cfg["use_fp16"]:
                cmd.append("--fp16")
            if tune:
                cmd.append("--tune-threshold")
            code, _log = stream_command(cmd)
            load_run_summaries.clear()
            if code == 0:
                st.success(f"Finished → {odir}")
                metrics_path = Path(odir) / "metrics.json"
                if metrics_path.exists():
                    payload = json.loads(metrics_path.read_text())
                    if "test" in payload:
                        show_metrics(payload["test"])
            else:
                st.error(f"Exited with code {code}")


def page_results() -> None:
    st.subheader("Saved runs")
    outputs = REPO / "outputs"
    df, runs = load_run_summaries(str(outputs))
    if df.empty:
        st.info("No outputs/*/metrics.json yet. Run inference, evaluate, or a baseline first.")
        return

    test_like = df[df["split"].isin(["test", "val"])]
    if not test_like.empty:
        st.markdown("Water IoU by run (val / test)")
        chart = test_like.pivot_table(index="run", columns="split", values="water_iou", aggfunc="first")
        st.bar_chart(chart)

    show = df.copy()
    for col in METRIC_COLS:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce")
    st.dataframe(
        show[["run", "method", "split"] + METRIC_COLS],
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    labels = [str(Path(r).relative_to(REPO)) if Path(r).is_relative_to(REPO) else r for r in runs]
    pick = st.selectbox("Inspect run", range(len(runs)), format_func=lambda i: labels[i])
    run_dir = Path(runs[pick])
    metrics_path = run_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text())
    st.json(payload, expanded=False)
    fig_dir = run_dir / "figures"
    pngs = sorted(fig_dir.glob("*.png")) if fig_dir.is_dir() else []
    if pngs:
        n = min(3, len(pngs))
        cols = st.columns(n)
        for i, png in enumerate(pngs):
            cols[i % n].image(str(png), caption=png.name, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Sen1Floods11 SegFormer", layout="wide")
    st.title("Sen1Floods11 SegFormer")
    st.caption(
        "Browse chips, run inference, evaluate splits, and compare baselines from the browser."
    )
    cfg = sidebar_config()
    tabs = st.tabs(["Browse", "Infer", "Evaluate", "Baselines", "Results"])
    with tabs[0]:
        page_browse(cfg)
    with tabs[1]:
        page_infer(cfg)
    with tabs[2]:
        page_evaluate(cfg)
    with tabs[3]:
        page_baselines(cfg)
    with tabs[4]:
        page_results()


if __name__ == "__main__":
    main()
