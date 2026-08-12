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

### Colab: mount Drive (dataset already downloaded)

```python
from google.colab import drive
drive.mount("/content/drive")

DATA_ROOT = "/content/drive/MyDrive/sen1floods11_hand"
```

If you still need to download once:

```python
!mkdir -p {DATA_ROOT}/S1 {DATA_ROOT}/Labels {DATA_ROOT}/splits
%cd {DATA_ROOT}

!gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv splits/
!gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv splits/
!gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv splits/
!gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand S1
!gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand Labels
```

## Setup

```bash
pip install -r requirements.txt
```

On Colab, also clone/upload this repo and `cd` into it so `train/` and `baselines/` import cleanly.

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

- `best.pt` / `best_hf/` — best weights
- `metrics.json` — val + test scores
- `history.json` — per-epoch log
- `figures/` — qualitative panels

Compare test **water IoU** to the VH threshold baseline (~0.53).

## Baselines (already implemented)

```bash
# VH threshold
python baselines/vh_threshold.py --data-root data/sen1floods11_hand

# Pretrained SegFormer (no fine-tune)
python baselines/segformer_pretrained_eval.py --data-root data/sen1floods11_hand
```
