"""Sen1Floods11 hand-labeled Dataset for SegFormer fine-tuning.

Expected layout (local or Colab Drive)::

    DATA_ROOT/
      S1/                 *_S1Hand.tif   (bands: VV, VH)
      Labels/             *_LabelHand.tif  (-1 nodata, 0 not-water, 1 water)
      splits/
        flood_train_data.csv
        flood_valid_data.csv
        flood_test_data.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from baselines.common import (  # noqa: E402
    IGNORE_INDEX,
    load_s1_and_label,
    normalize_imagenet_chw,
    read_split_stems,
    resolve_pair,
    s1_to_pseudo_rgb,
)

try:
    import albumentations as A
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install albumentations: pip install albumentations") from e


def build_train_augs() -> A.Compose:
    """Geometric augs only + light SAR intensity noise (no color jitter)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.MultiplicativeNoise(
                multiplier=(0.95, 1.05),
                per_channel=True,
                elementwise=True,
                p=0.3,
            ),
        ],
    )


def build_eval_augs() -> A.Compose:
    return A.Compose([])


class Sen1Floods11SegDataset(Dataset):
    """Yields ``pixel_values`` (3,H,W) float32 and ``labels`` (H,W) int64."""

    def __init__(
        self,
        data_root: Path | str,
        split_csv: Path | str,
        *,
        s1_dir: str = "S1",
        label_dir: str = "Labels",
        image_size: int = 512,
        db_min: float = -30.0,
        db_max: float = 0.0,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.db_min = db_min
        self.db_max = db_max
        self.augs = build_train_augs() if augment else build_eval_augs()

        stems = read_split_stems(Path(split_csv))
        self.samples: list[tuple[str, Path, Path]] = []
        missing = 0
        for stem in stems:
            try:
                s1_path, lab_path = resolve_pair(
                    self.data_root, s1_dir, label_dir, stem
                )
            except FileNotFoundError:
                missing += 1
                continue
            self.samples.append((stem, s1_path, lab_path))
        if missing:
            print(f"  warning: skipped {missing}/{len(stems)} missing chips ({Path(split_csv).name})")
        if not self.samples:
            raise FileNotFoundError(
                f"No chips found for {split_csv} under {self.data_root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem, s1_path, lab_path = self.samples[idx]
        vv, vh, label = load_s1_and_label(s1_path, lab_path)

        # Pseudo-RGB in [0,1] before ImageNet norm (same as pretrained protocol A)
        rgb01 = s1_to_pseudo_rgb(vv, vh, db_min=self.db_min, db_max=self.db_max)
        label = label.astype(np.int16)

        if self.augs.transforms:
            out = self.augs(image=rgb01, mask=label)
            rgb01 = out["image"]
            label = out["mask"]

        # Clip after multiplicative noise; keep ignore pixels as IGNORE_INDEX
        rgb01 = np.clip(rgb01, 0.0, 1.0).astype(np.float32)
        rgb01 = np.nan_to_num(rgb01, nan=0.0, posinf=1.0, neginf=0.0)
        label = label.astype(np.int64)
        # Sen1Floods11: {-1,0,1}; anything else → ignore (avoids CE OOB → NaN)
        valid = (label == 0) | (label == 1) | (label == IGNORE_INDEX)
        label = np.where(valid, label, IGNORE_INDEX).astype(np.int64)

        chw = normalize_imagenet_chw(rgb01)
        pixel_values = torch.from_numpy(chw)
        labels = torch.from_numpy(label)

        if (
            pixel_values.shape[-2] != self.image_size
            or pixel_values.shape[-1] != self.image_size
        ):
            pixel_values = torch.nn.functional.interpolate(
                pixel_values.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            labels = (
                torch.nn.functional.interpolate(
                    labels.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "stem": stem,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "stem": [b["stem"] for b in batch],
    }
