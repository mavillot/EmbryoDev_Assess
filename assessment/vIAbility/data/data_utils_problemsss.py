"""
utils.py — shared helpers used by all three extraction scripts.

Provides:
  - logger setup
  - metadata / split loading
  - low-level frame reading (load_frames / extract_frames)
  - directory scaffolding
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from assessment.vIAbility.data.dataset import parse_video_name, load_metadata
from config.config import PATH_DATASET

# ── canonical paths ───────────────────────────────────────────────────────────
PATH_SOURCE       = Path(PATH_DATASET) / "Oslo_viability"
PATH_DST          = Path(PATH_DATASET) / "Oslo_viability"
PATH_SOURCE_VIDEOS = PATH_SOURCE / "videos"
PATH_EXCEL        = PATH_SOURCE / "Oslo_completo.xlsx"

CLASSMAP = {0: "no_FHB", 1: "FHB"}

# ── default sampling params ───────────────────────────────────────────────────
DEFAULT_N_FRAMES   = 214
DEFAULT_START_TIME = 0.0
DEFAULT_FREQ       = 1.0
DEFAULT_FRAME_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# Split loading  (train.csv / test.csv)
# ─────────────────────────────────────────────────────────────────────────────

def load_split(split: str) -> tuple[list[str], list[int]]:
    """
    Read train.csv or test.csv from PATH_DST.

    Returns
    -------
    filenames : list[str]
    labels    : list[int]   0 = no_FHB, 1 = FHB
    """
    csv_path = PATH_DST / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df["filename"].tolist(), df["label"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Frame I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_frames(
    video_path: str | Path,
    tse: float,
    frame_hour: float,
    time_window: list[float],
    frame_size: int,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Seek to each time-point in *time_window* and read one frame.

    Returns
    -------
    frames : list of (H, W, 3) uint8 RGB arrays — blank zeros if unavailable
    mask   : list of int   1 = real frame,  0 = blank/padded
    """
    frames, mask = [], []
    blank = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(str(video_path))
    try:
        for t in time_window:
            frame_idx = int((t - tse) * frame_hour)

            if frame_idx < 0 or not cap.isOpened():
                frames.append(blank.copy())
                mask.append(0)
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()

            if ret:
                img = cv2.resize(img, (frame_size, frame_size),
                                 interpolation=cv2.INTER_LINEAR)
                img = img[..., ::-1].copy()   # BGR → RGB
                frames.append(img)
                mask.append(1)
            else:
                frames.append(blank.copy())
                mask.append(0)
    finally:
        cap.release()

    return frames, mask


def extract_frames(
    video_filename: Path | str,
    metadata_df: pd.DataFrame,
    *,
    start_time: float = DEFAULT_START_TIME,
    freq: float       = DEFAULT_FREQ,
    n_frames: int     = DEFAULT_N_FRAMES,
    resize: int       = DEFAULT_FRAME_SIZE,
    log: logging.Logger | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """
    High-level wrapper: looks up metadata, builds the time window,
    calls load_frames, and returns (frames, mask).
    """
    time_window = [start_time + i * freq for i in range(n_frames)]
    video_filename = Path(video_filename)
    print(f"Extracting frames from {video_filename}...")
    vid_id, well = parse_video_name(video_filename.name)
    row = metadata_df[(metadata_df["ID"] == vid_id) & (metadata_df["Well"] == well)]
    if row.empty:
        raise ValueError(f"No metadata row for '{video_filename}'")

    tse        = float(row["Tse"].iloc[0])
    frame_hour = float(row["frame_hour"].iloc[0])
    log.info("%-40s  tse=%.2f  frame_hour=%.2f", video_filename, tse, frame_hour)

    video_path = PATH_SOURCE_VIDEOS / video_filename
    log.info("%-40s", video_path)
    frames, mask = load_frames(video_path, tse, frame_hour, time_window, resize)

    if log:
        log.info("%-40s  valid=%d/%d", video_filename, sum(mask), len(mask))

    return frames, mask


# ─────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_split_dirs(base_dir: Path) -> None:
    """Create  base_dir/{train,test}/{FHB,no_FHB}  trees."""
    for split in ("train", "test"):
        for cls in CLASSMAP.values():
            (base_dir / split / cls).mkdir(parents=True, exist_ok=True)


def out_path(base_dir: Path, split: str, label: int, stem: str, suffix: str) -> Path:
    """Return  base_dir/<split>/<class>/<stem><suffix>  as a Path."""
    return base_dir / split / CLASSMAP[label] / f"{stem}{suffix}"
