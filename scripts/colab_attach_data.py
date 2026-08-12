#!/usr/bin/env python3
"""Attach Sen1Floods11 hand-set on Google Colab via Drive.

Default layout (already downloaded on Drive)::

    /content/drive/MyDrive/sen1floods11_hand/
      S1/
      Labels/
      splits/

Usage in a Colab cell::

    !python scripts/colab_attach_data.py
    # or with a custom path / one-time download:
    !python scripts/colab_attach_data.py --data-root /content/drive/MyDrive/sen1floods11_hand
    !python scripts/colab_attach_data.py --download

Prints DATA_ROOT and writes ``/content/sen1floods11_data_root.txt`` for later cells.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/sen1floods11_hand"
MARKER_FILE = Path("/content/sen1floods11_data_root.txt")
REQUIRED_SPLITS = (
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mount Drive and attach Sen1Floods11 hand set")
    p.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DRIVE_ROOT,
        help="Path to hand set on Drive (or local Colab disk)",
    )
    p.add_argument(
        "--mount-point",
        type=str,
        default="/content/drive",
        help="Google Drive mount point",
    )
    p.add_argument(
        "--no-mount",
        action="store_true",
        help="Skip drive.mount (use when data is already under /content)",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="If missing, pull S1Hand + LabelHand + splits from GCS (once)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-rsync from GCS even if folders already exist",
    )
    return p.parse_args()


def mount_drive(mount_point: str) -> None:
    try:
        from google.colab import drive  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "google.colab not available. Run this on Colab, or pass --no-mount "
            "with a local --data-root."
        ) from e

    print(f"Mounting Google Drive at {mount_point} ...")
    drive.mount(mount_point, force_remount=False)


def count_tifs(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for _ in folder.glob("*.tif")) + sum(1 for _ in folder.glob("*.tiff"))


def verify_layout(root: Path) -> list[str]:
    problems: list[str] = []
    s1 = root / "S1"
    labels = root / "Labels"
    splits = root / "splits"

    if not s1.is_dir():
        problems.append(f"missing directory: {s1}")
    if not labels.is_dir():
        problems.append(f"missing directory: {labels}")
    if not splits.is_dir():
        problems.append(f"missing directory: {splits}")

    for name in REQUIRED_SPLITS:
        p = splits / name
        if not p.is_file():
            problems.append(f"missing split CSV: {p}")

    n_s1, n_lab = count_tifs(s1), count_tifs(labels)
    if n_s1 == 0:
        problems.append(f"no .tif chips in {s1}")
    if n_lab == 0:
        problems.append(f"no .tif masks in {labels}")
    if n_s1 and n_lab and n_s1 != n_lab:
        problems.append(f"chip/mask count mismatch: S1={n_s1} Labels={n_lab}")

    return problems


def download_hand_set(root: Path) -> None:
    """Pull official hand-labeled S1 + labels + splits (needs gsutil + GCS access)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "S1").mkdir(exist_ok=True)
    (root / "Labels").mkdir(exist_ok=True)
    (root / "splits").mkdir(exist_ok=True)

    cmds = [
        [
            "gsutil",
            "cp",
            "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv",
            str(root / "splits" / "flood_train_data.csv"),
        ],
        [
            "gsutil",
            "cp",
            "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv",
            str(root / "splits" / "flood_valid_data.csv"),
        ],
        [
            "gsutil",
            "cp",
            "gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv",
            str(root / "splits" / "flood_test_data.csv"),
        ],
        [
            "gsutil",
            "-m",
            "rsync",
            "-r",
            "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand",
            str(root / "S1"),
        ],
        [
            "gsutil",
            "-m",
            "rsync",
            "-r",
            "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand",
            str(root / "Labels"),
        ],
    ]
    for cmd in cmds:
        print(">", " ".join(cmd))
        subprocess.run(cmd, check=True)


def write_marker(root: Path) -> None:
    MARKER_FILE.write_text(str(root.resolve()) + "\n", encoding="utf-8")
    # Also export for the current process / !bash children in same cell if sourced
    os.environ["SEN1FLOODS11_DATA_ROOT"] = str(root.resolve())


def main() -> None:
    args = parse_args()
    root = Path(args.data_root)

    if not args.no_mount:
        mount_drive(args.mount_point)

    if args.download or args.force_download:
        if args.force_download or verify_layout(root):
            print(f"Downloading hand set into {root} ...")
            download_hand_set(root)
        else:
            print(f"Layout OK — skip download (use --force-download to re-sync)")

    problems = verify_layout(root)
    if problems:
        print(f"DATA_ROOT not ready: {root}", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nFix: confirm Drive path, or re-run with --download",
            file=sys.stderr,
        )
        raise SystemExit(1)

    n_s1 = count_tifs(root / "S1")
    n_lab = count_tifs(root / "Labels")
    write_marker(root)

    print("=== Sen1Floods11 attached ===")
    print(f"DATA_ROOT={root}")
    print(f"S1 chips:     {n_s1}")
    print(f"Label masks:  {n_lab}")
    print(f"splits:       {root / 'splits'}")
    print(f"marker file:  {MARKER_FILE}")
    print()
    print("Train example:")
    print(
        f"  python train/finetune_segformer.py \\\n"
        f"    --data-root {root} \\\n"
        f"    --out-dir {root}/runs/segformer_ft \\\n"
        f"    --fp16"
    )


if __name__ == "__main__":
    main()
