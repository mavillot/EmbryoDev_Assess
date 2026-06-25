"""
run_repeats_markers_ploidy.py
────────────────────────────────────────────────────────────────────────────────
Runs the ENTIRE markers_ML_ploidy.py pipeline — frame subsampling →
clustering → train/val split → hyperparameter search → bagging → voting
search → refit on train+val → diagnostic 5-fold CV → evaluate on the fixed
held-out test set — for N_REPEATS independent OUTER repeats, each with its
own seed.

Same idea as run_repeats_markers.py for the viability version: no
architecture axis to sweep here (every classifier + bagging + voting combo
runs in a single pass), so n=5 simply means repeating that single run 5
times with different seeds. n_frames is held fixed at its CLI default (50)
across all repeats — pass --n_frames to change it for the whole sweep, or
edit EXPERIMENTS below if you also want to sweep multiple n_frames values.

Resumable: if a repeat's rows are already present in the master results CSV
(markers_ML_ploidy.METRICS_CSV), that repeat is loaded instead of retrained.

Usage:
    python run_repeats_markers_ploidy.py
    python run_repeats_markers_ploidy.py --n_frames 30
"""

import argparse
import os, sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from assessment.ploidIA.features.markers_ML import main, SEED

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

N_REPEATS = 5
SEEDS = [SEED + i for i in range(N_REPEATS)]


def run_all(cli_args):
    for dataset in ["Combined"]:
        for n_frames in [100]:
            log.info("Running dataset=%s n_frames=%d", dataset, n_frames)
            cli_args.dataset = dataset
            cli_args.n_frames = n_frames
            log.info("Dataset: %s, Frames: %d", dataset, n_frames)
            PATH_OUTPUT = Path(f"results/assessment/ploidIA/{cli_args.dataset}/markers/{cli_args.n_frames}/ML/")
            METRICS_CSV = str(PATH_OUTPUT / "experiment_results.csv")

            all_dfs = []

            for i, seed in enumerate(SEEDS, start=1):
                if os.path.exists(METRICS_CSV):
                    existing = pd.read_csv(METRICS_CSV)
                    if "iteration" in existing.columns and (existing["iteration"] == i).any():
                        log.info("[%d/%d] SKIP iter=%d  (already in %s)", i, N_REPEATS, i, METRICS_CSV)
                        all_dfs.append(existing[existing["iteration"] == i])
                        continue

                log.info("[%d/%d] RUN  iter=%d seed=%d", i, N_REPEATS, i, seed)
                run_args = argparse.Namespace(
                    dataset=cli_args.dataset,
                    n_frames=cli_args.n_frames,
                    n_clusters=cli_args.n_clusters,
                    # path_models=cli_args.path_models,
                    # out_dir=cli_args.out_dir,
                    vote_min=cli_args.vote_min,
                    vote_max=cli_args.vote_max,
                    voting=cli_args.voting,
                    vote_top_k=cli_args.vote_top_k,
                    cv_folds=cli_args.cv_folds,
                    val_size=cli_args.val_size,
                    iteration=i,
                    seed=seed,
                )
                try:
                    df = main(run_args)
                except Exception:
                    log.exception("[%d/%d] FAILED iter=%d — continuing with next repeat", i, N_REPEATS, i)
                    continue

                all_dfs.append(df)

            if not all_dfs:
                log.error("No repeats completed — nothing to aggregate.")
                return

            combined = pd.concat(all_dfs, ignore_index=True)

            out_root = os.path.dirname(METRICS_CSV)
            os.makedirs(out_root, exist_ok=True)

            combined_csv = os.path.join(out_root, "all_repeats_results.csv")
            combined.to_csv(combined_csv, index=False)
            log.info("All-repeats results saved to %s", combined_csv)

            # ── Aggregate: mean ± std of TEST metrics across the 5 outer repeats, per model.
            metric_cols = [c for c in combined.columns if c.startswith("test_") or c.startswith("cv_")]

            summary = combined.groupby("model")[metric_cols].agg(["mean", "std", "count"])
            summary.columns = [f"{m}_{s}" for m, s in summary.columns]
            sort_col = "test_fbeta_mean" if "test_fbeta_mean" in summary.columns else summary.columns[0]
            summary = summary.reset_index().sort_values(sort_col, ascending=False)

            summary_xlsx = os.path.join(out_root, "repeats_summary.xlsx")
            with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
                combined.sort_values(["model", "iteration"]).to_excel(writer, sheet_name="all_repeats", index=False)
                summary.to_excel(writer, sheet_name="summary", index=False)

            log.info("Summary written to %s", summary_xlsx)
            print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat markers_ML_ploidy.py's pipeline N=5 times with different seeds")
    parser.add_argument("--n_clusters", type=int, default=5)
    # parser.add_argument("--path_models", default="")
    # parser.add_argument("--out_dir", default="")
    parser.add_argument("--vote_min", type=int, default=2)
    parser.add_argument("--vote_max", type=int, default=None)
    parser.add_argument("--voting", default="both", choices=["hard", "soft", "both"])
    parser.add_argument("--vote_top_k", type=int, default=3)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.2)
    cli_args = parser.parse_args()
    
    run_all(cli_args)