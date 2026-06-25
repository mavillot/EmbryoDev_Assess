"""
run_experiments.py
────────────────────────────────────────────────────────────────────────────────
Repeats every experiment N_REPEATS times, each with a different seed, and logs
every run's metrics to train.METRICS_CSV:

    • MViT-V2-S   @ 16 frames                       → 5 runs
    • R(2+1)D-18  @ 15, 16, 20, 30, 60 frames        → 5 runs each

Total: 30 runs.

Resume support: before launching a run, this script checks whether a row for
that exact (backbone, n_frames, iteration) already exists in the results CSV
and skips it if so. So if the process dies partway through (OOM, machine
reboot, etc.) just re-run this script — completed runs are not redone.

Run from the same directory as train.py:
    python run_experiments.py
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
SEEDS = [SEED + i for i in range(N_REPEATS)]   # same 5 seeds reused for every config

EXPERIMENTS = [("mvit", 16)] + [("r2plus1d_18", n) for n in (7, 15, 16, 20, 30, 60)]


def already_done(backbone: str, n_frames: int, iteration: int) -> bool:
    """Check the results CSV for a row matching this exact run (resume support)."""
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
                log.info(
                    "[%3d/%3d] SKIP  %-11s n_frames=%-3d iter=%d  (already logged)",
                    done, total, backbone, n_frames, i,
                )
                continue

            log.info(
                "[%3d/%3d] START %-11s n_frames=%-3d iter=%d seed=%d",
                done, total, backbone, n_frames, i, seed,
            )

            cfg = TrainConfig(backbone=backbone, n_frames=n_frames, iteration=i, seed=seed)
            try:
                metrics = train(cfg)
                log.info("[%3d/%3d] DONE  %-11s n_frames=%-3d iter=%d  |  acc=%.3f  auroc=%.3f",done, total, backbone, n_frames, i,metrics["accuracy"], metrics["auroc"],)
            except Exception:
                # Log and move to the next run rather than losing the whole sweep
                # to one bad run (e.g. transient OOM on a particular n_frames).
                log.exception(
                    "[%3d/%3d] FAILED %s n_frames=%d iter=%d — continuing with next run",
                    done, total, backbone, n_frames, i,
                )
                continue

    log.info("Sweep complete. All results in: %s", METRICS_CSV)


if __name__ == "__main__":
    main()