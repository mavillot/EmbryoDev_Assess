"""
run_experiments_multimodal.py
────────────────────────────────────────────────────────────────────────────────
Repeats the multimodal (R(2+1)D-18 + tabular) experiment N_REPEATS times, each
with a different seed, and logs every run's metrics to
multimodal_train.METRICS_CSV:

    • R(2+1)D-18  @ 15, 16, 20, 30, 60 frames   → 5 runs each   (25 runs total)

MViT is NOT included here yet — MultimodalClassifier currently only
implements an R(2+1)D-18 video encoder (see the note at the top of
multimodal_train.py). Once model.py is shared and a backbone switch is added,
extend EXPERIMENTS below with ("mvit", 16) the same way run_experiments.py
does for the single-modality model.

Resume support: skips any (backbone, n_frames, iteration) already present in
the results CSV, so you can kill and re-run this script freely.

Run from the same directory as multimodal_train.py:
    python run_experiments_multimodal.py
"""

import csv
import logging
import os

from train import TrainConfig, train, METRICS_CSV
from config.config import SEED

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

N_REPEATS = 5
SEEDS = [SEED + i for i in range(N_REPEATS)]

# R(2+1)D-18 only — see module docstring for why MViT isn't here yet.
EXPERIMENTS = [("r2plus1d_18", n) for n in (7,15, 16, 20, 30, 60)]


def already_done(backbone: str, n_frames: int, iteration: int) -> bool:
    if not os.path.exists(METRICS_CSV):
        return False
    with open(METRICS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if (
                row.get("backbone") == backbone
                and row.get("n_frames") not in (None, "")
                and int(row["n_frames"]) == n_frames
                and row.get("iteration") not in (None, "")
                and int(row["iteration"]) == iteration
            ):
                return True
    return False


def main():
    total = len(EXPERIMENTS) * N_REPEATS
    done = 0

    for backbone, n_frames in EXPERIMENTS:
        for i, seed in enumerate(SEEDS, start=1):
            done += 1

            if already_done(backbone, n_frames, i):
                log.info("[%2d/%2d] SKIP  %-11s n_frames=%-3d iter=%d  (already logged)", done, total, backbone, n_frames, i)
                continue

            log.info("[%2d/%2d] START %-11s n_frames=%-3d iter=%d seed=%d", done, total, backbone, n_frames, i, seed)

            cfg = TrainConfig(backbone=backbone, n_frames=n_frames, iteration=i, seed=seed)
            try:
                metrics = train(cfg)
                log.info(
                    "[%2d/%2d] DONE  %-11s n_frames=%-3d iter=%d  |  acc=%.3f  auroc=%.3f",
                    done, total, backbone, n_frames, i, metrics["accuracy"], metrics["auroc"],
                )
            except Exception:
                log.exception(
                    "[%2d/%2d] FAILED %s n_frames=%d iter=%d — continuing with next run",
                    done, total, backbone, n_frames, i,
                )
                continue

    log.info("Sweep complete. All results in: %s", METRICS_CSV)


if __name__ == "__main__":
    main()