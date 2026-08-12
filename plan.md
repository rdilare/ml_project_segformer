# Project Plan: Flood Inundation Segmentation with SegFormer + Sen1Floods11

Build an end-to-end system that segments flood water from Sentinel imagery using SegFormer, trained and evaluated on **Sen1Floods11** (Cloud to Street / CVPR 2020).

**Hardware split:**

* **Google Colab (T4 GPU):** data prep, fine-tuning, evaluation, benchmarking.
* **M2 MacBook Air (8GB):** exploration, ONNX export, API, Streamlit UI, Docker.

---

## Feasibility verdict: Colab T4 — GO

**Yes — this project is feasible on free/pro Colab with a T4**, if v1 uses the hand-labeled subset + `mit-b0` + `fp16`. Existing Colab demos (TerraMind / TerraTorch Sen1Floods11) already target T4.

### Check results

| Constraint | Colab T4 | This project | Status |
|------------|----------|--------------|--------|
| **GPU VRAM** | ~16 GB (~15 usable) | `mit-b0` (~3.7M params), 512×512 chips, `fp16`, batch **4–8** | Fits |
| **Disk** | Often ~35–100 GB ephemeral | Full bucket ~**14 GB**; hand set (S1Hand + LabelHand) ~**1–3 GB** | Fits |
| **Sample count** | Session ~12h; weekly GPU quota | Hand: ~252 train / ~89 val / ~90 test → train in minutes–a few hours | Fits |
| **Data access** | GCS / Drive / tar mirrors | Official [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) — `gs://sen1floods11/` | OK |
| **Prior art** | T4 runtime | Public Sen1Floods11 Colab notebooks | Proven |

### Operating rules on T4

* **v1 scope:** download only `S1Hand` + `LabelHand` + official splits (~446 chips). Do **not** pull the full ~14–35 GB dump unless needed later.
* **Model:** `nvidia/mit-b0` only (not b5). `fp16=True`. Batch 4–8; use gradient accumulation for effective batch ~16.
* **Persistence:** mount Google Drive for checkpoints and processed caches — Colab local disk is wiped on disconnect.
* **Watch:** idle timeout, ~12h session cap, T4 not always available on free tier; don’t keep archive + extracted full dataset on disk at once.
* **Later (optional):** weakly labeled ~4k chips is still doable on T4, but plan extra disk and train time.

---

## Architecture & tech stack

| Area | Choice |
|------|--------|
| **Dataset** | [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) — start S1 hand-labeled; optional S2 / weak labels later |
| **Model** | `nvidia/mit-b0` via HF `SegformerForSemanticSegmentation` (adapt first conv for 2-band S1, or expand VV/VH→3ch) |
| **Training** | PyTorch, HF `Trainer` or Lightning, `fp16` on T4 |
| **Metrics** | mIoU, **water IoU**, Precision/Recall/F1 (water), Pixel Accuracy |
| **Tracking** | Weights & Biases or TensorBoard |
| **Baselines** | VH backscatter threshold; optional small U-Net |
| **Serving** | ONNX + FastAPI + Streamlit (M2) |

**Task:** binary semantic segmentation — `{0: "not_water", 1: "water"}`; ignore nodata (often `-1`).

**Hand-labeled splits (approx.):** train ~252 · val ~89 · test ~90 · Bolivia holdout for unseen-event test where applicable.

---

## Execution roadmap

### Phase 1: Data preparation (2–3 days)

*Goal: Reproducible loaders that fit Colab T4 disk/VRAM.*

* [ ] **Environment:** Colab + local — `torch`, `transformers`, `evaluate`, `albumentations`, `rasterio` / `tifffile`, `wandb`.
* [ ] **Acquire (minimal):** `S1Hand` + `LabelHand` + `flood_handlabeled` splits from GCS (`gsutil` / `gcsfs`) or a documented mirror; check `df -h` first.
* [ ] **Persist:** store raw + processed paths under Google Drive.
* [ ] **Splits:** use official train/val/test lists; keep Bolivia / country holdouts for the primary benchmark — no random reshuffle across countries.
* [ ] **Raster → tensors:** stack VV/VH, normalize, map labels to `{0,1}`, set ignore index for nodata.
* [ ] **Augmentation:** Flip / Rotate90; avoid color jitter on SAR; optional light intensity noise.
* [ ] **Dataset class:** yields `pixel_values` + `labels` for SegFormer (custom preprocess if 2-channel S1).
* [ ] **T4 smoke test:** one batch forward + `torch.cuda.max_memory_allocated()` → lock batch size.

### Phase 2: Fine-tuning (3–4 days)

*Goal: Fine-tune on Colab T4; save best checkpoint to Drive.*

* [ ] **Init:** `SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2, ...)`.
* [ ] **Input adapt:** 2-band proj for S1, or 3-channel VV/VH/VH hack as simpler baseline.
* [ ] **T4 training args:** `fp16=True`, batch 4–8, grad accumulation → effective ~16, AdamW, early stop on val water IoU / mIoU.
* [ ] **Loss:** CE with ignore index; optional class weights / Dice if water is sparse.
* [ ] **Checkpoint + log:** best-by-val metric → Drive / HF Hub; WandB for loss & metrics.
* [ ] **Export:** pull best weights to M2 when ready for ONNX/UI.

### Phase 3: Evaluation (1–2 days)

*Goal: Reproducible scores on official held-out test chips.*

* [ ] **Metrics:** mIoU; **water-class IoU** (primary); Precision, Recall, F1; Pixel Accuracy.
* [ ] **Ignore handling:** exclude nodata from all metrics.
* [ ] **Qualitative:** SAR composite / GT / pred / error map on a fixed hard-case set.
* [ ] **Geographic breakdown:** per event or Bolivia holdout where splits allow.
* [ ] **Failure notes:** FPs (permanent water) vs FNs (speckle, layover) → feed next train round.

### Phase 4: Benchmarking (1–2 days)

*Goal: Context for SegFormer numbers on Sen1Floods11.*

* [ ] **Baseline A:** VH (or VV) threshold; tune on **val only**.
* [ ] **Baseline B (optional):** small U-Net, same splits/augs/budget.
* [ ] **Fair table:** same chips, ignore rules, metrics (water IoU / mIoU).
* [ ] **T4 efficiency:** time/epoch, ms/chip, peak VRAM, param count.
* [ ] **Literature anchor:** compare to Sen1Floods11 paper / follow-ons on the **same hand-labeled subset**.
* [ ] **Ablations (optional):** S1 vs S2; 2ch vs 3ch; CE vs CE+Dice. Weak labels only after v1 is solid.

### Phase 5: Optimization for Apple Silicon (2 days)

* [ ] ONNX export from fine-tuned checkpoint.
* [ ] Optional INT8 dynamic quantization for 8GB M2.
* [ ] `onnxruntime` parity check vs PyTorch on a few chips.

### Phase 6: API & UI (3–4 days)

* [ ] FastAPI `POST /predict` — upload chip; return mask + latency.
* [ ] Dockerize (`python:3.11-slim`); stay within 8GB RAM.
* [ ] Streamlit: overlay flood mask; show water IoU if GT present + legend.

---

## Colab-first order of work

1. Runtime → **T4**; mount Drive; set cache dirs.
2. Download **hand-labeled S1 + LabelHand + official splits only** (~1–3 GB).
3. Dataset + one-batch VRAM smoke test → fix batch size.
4. Fine-tune `mit-b0` + `fp16`; log to WandB; checkpoint to Drive.
5. Evaluate on official test (water IoU + mIoU + plots).
6. VH threshold baseline → benchmarking table.
7. Pull checkpoint to M2 → ONNX / API / UI.

---

## Success criteria

* End-to-end Sen1Floods11 pipeline runs on **Colab T4** within the hand-labeled footprint.
* Documented **data preparation**, **fine-tuning**, **evaluation**, and **benchmarking**.
* Water IoU competitive with the VH threshold baseline under the same splits.
* Demo: upload chip → flood inundation overlay.
