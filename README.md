# Sen1Floods11 + SegFormer

Flood water segmentation on the Sen1Floods11 **hand-labeled** S1 set.

## Data layout

Point `--data-root` at a folder like this (local or Google Drive):

```text
DATA_ROOT/
  S1/                 *_S1Hand.tif
  Labels/             *_LabelHand.tif
  splits/
    flood_train_data.csv
    flood_valid_data.csv
    flood_test_data.csv
```

### Colab: attach dataset from Drive

`drive.mount` must run in the **notebook kernel** (not under `!python`).

**Option A — one cell (`%run`):**

```python
%run scripts/colab_attach_data.py
```

**Option B — mount, then verify:**

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!python scripts/colab_attach_data.py --no-mount
```

Checks `S1/` / `Labels/` / `splits/` and writes `/content/sen1floods11_data_root.txt`.
Default root: `/content/drive/MyDrive/sen1floods11_hand`

Optional download if the folder is empty:

```python
%run scripts/colab_attach_data.py --download
```

## Setup

```bash
pip install -r requirements.txt
```

On Colab, also clone/upload this repo and `cd` into it so `train/` and `eval/` import cleanly.

Drop a Hugging Face checkpoint at `models/best_hf/` (`config.json` + `model.safetensors`). Weights are gitignored.

## Local UI

Browse chips, run inference, compare chips × models, evaluate splits, and launch baselines from the browser:

```bash
streamlit run app.py
```

Set **Data root** (default `data/sen1floods11_hand`) and **Model dir** (default `models/best_hf/`) in the sidebar. The **Compare** tab runs several chips (or uploaded GeoTIFFs) through several Hugging Face checkpoints plus the VH baseline. The Results tab reads `outputs/*/metrics.json`.

## Fine-tune (Colab T4)

Smoke test (one step + peak VRAM):

```bash
python train/finetune_segformer.py \
  --data-root /content/drive/MyDrive/sen1floods11_hand \
  --out-dir /content/drive/MyDrive/sen1floods11_hand/runs/segformer_ft \
  --fp16 --smoke
```

Train (saves best checkpoint by **val water IoU**):

```bash
python train/finetune_segformer.py \
  --data-root /content/drive/MyDrive/sen1floods11_hand \
  --out-dir /content/drive/MyDrive/sen1floods11_hand/runs/segformer_ft \
  --fp16 \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 40 \
  --patience 8
```

Outputs under `--out-dir`:

- `last.pt` — full train state every epoch (use to resume)
- `best.pt` / `best_hf/` — best weights by val water IoU
- `metrics.json` — val + test scores (includes `small_water_iou`, `dry_fp_rate`)
- `history.json` — per-epoch log (updated each epoch)
- `figures/training_curves.png` — loss, IoU, precision/recall, dry-chip FP rate
- `figures/` — qualitative panels

Resume after Colab disconnect / interrupt: re-run the **same** command with the
same `--out-dir`. It auto-loads `last.pt` (else `best.pt`) when present; if
none exist, it starts from scratch. Force a fresh run with `--no-resume`.

This loss/sampling change is not compatible with old CE-only checkpoints — use
a new `--out-dir` or `--no-resume`.

Compare test **water IoU** to the VH threshold baseline (~0.53).

Copy the best Hugging Face folder into the repo for local inference / eval:

```bash
cp -r /content/drive/MyDrive/sen1floods11_hand/runs/segformer_ft/best_hf models/best_hf
```

## Eval

```bash
# Fine-tuned SegFormer (needs models/best_hf/)
python eval/finetuned.py --split test --num-viz 8

# VH threshold baseline
python eval/vh_threshold.py --data-root data/sen1floods11_hand

# Pretrained SegFormer (no fine-tune)
python eval/pretrained.py --data-root data/sen1floods11_hand
```
