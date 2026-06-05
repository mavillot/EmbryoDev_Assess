"""
extract_markers.py — run the EmbryoVideo pipeline to extract developmental
markers (morphokinetic events + evolution curves) per video.

Output layout
-------------
<PATH_DST>/markers/
    extraction_config.json
    train/
        FHB/     <video_filename>.csv
        no_FHB/  <video_filename>.csv
    test/
        FHB/     <video_filename>.csv
        no_FHB/  <video_filename>.csv

Each .csv file is produced by EmbryoVideo.save_evolution().

Usage
-----
    python extract_markers.py
    python extract_markers.py --frame_size 224 --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from data_utils import (
    ROOT,
    PATH_DST, PATH_EXCEL,
    DEFAULT_N_FRAMES, DEFAULT_START_TIME, DEFAULT_FREQ, DEFAULT_FRAME_SIZE,
    get_logger, load_split, load_metadata, extract_frames,
    make_split_dirs, out_path,
)

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT))
import os
from assessment.embryo_utils.embryo_video import EmbryoVideo

log = get_logger(__name__)

MARKERS_DST = PATH_DST / "markers"


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline call
# ─────────────────────────────────────────────────────────────────────────────

def run_embryo_pipeline(
    frames,
    mask: list[int],
    output_path: Path,
) -> None:
    """
    Instantiate EmbryoVideo, compute evolution, and save to *output_path*.

    Parameters
    ----------
    frames      : list of (H, W, 3) uint8 RGB arrays
    mask        : list of int   1 = valid frame, 0 = blank
    output_path : destination .csv file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embryo = EmbryoVideo(frames=frames, mask=mask, cropped=True)
    embryo.get_evolution()
    embryo.save_evolution(filename=str(output_path))


# ─────────────────────────────────────────────────────────────────────────────
# Per-split extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def run_split(
    split: str,
    filenames: list[str],
    labels: list[int],
    metadata_df,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Run EmbryoVideo pipeline for one split. Returns (done, skipped, errors)."""
    done = skipped = errors = 0

    log.info("── %s  (%d videos) ──────────────────────────────────────", split.upper(), len(filenames))

    for video_file, label in tqdm(zip(filenames, labels), total=len(filenames),
                                  desc=f"Markers {split}", unit="video"):
        dst = out_path(MARKERS_DST, split, label, video_file, ".csv")

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        # ── load frames ───────────────────────────────────────────────────
        try:
            frames, mask = extract_frames(
                video_file, metadata_df,
                start_time=args.start_time,
                freq=args.frequency,
                n_frames=args.n_frames,
                resize=args.frame_size,
                log=log,
            )
        except Exception as exc:
            log.error("Frame extraction failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        n_valid = sum(mask)
        if n_valid == 0:
            log.warning("All frames blank for '%s' — skipping pipeline.", video_file)
            errors += 1
            continue

        # ── EmbryoVideo pipeline ──────────────────────────────────────────
        try:
            run_embryo_pipeline(frames, mask, dst)
        except Exception as exc:
            log.error("EmbryoVideo pipeline failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        done += 1

    log.info("%s done=%d  skipped=%d  errors=%d", split, done, skipped, errors)
    return done, skipped, errors


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── output dirs ───────────────────────────────────────────────────────
    make_split_dirs(MARKERS_DST)

    # ── save extraction config ────────────────────────────────────────────
    config_path = MARKERS_DST / "extraction_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Config saved → %s", config_path)

    # ── load metadata & splits ────────────────────────────────────────────
    metadata_df = load_metadata(PATH_EXCEL)
    log.info("Metadata: %d rows.", len(metadata_df))

    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")

    # ── run ───────────────────────────────────────────────────────────────
    total_done = total_skipped = total_errors = 0
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        d, s, e = run_split(split, X, y, metadata_df, args)
        total_done    += d
        total_skipped += s
        total_errors  += e

    log.info(
        "All done. done=%d  skipped=%d  errors=%d  markers → %s",
        total_done, total_skipped, total_errors, MARKERS_DST,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EmbryoVideo pipeline and save marker CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_frames",   type=int,   default=DEFAULT_N_FRAMES)
    parser.add_argument("--start_time", type=float, default=DEFAULT_START_TIME)
    parser.add_argument("--frequency",  type=float, default=DEFAULT_FREQ)
    parser.add_argument("--frame_size", type=int,   default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-run pipeline even if output file already exists")
    main(parser.parse_args())
