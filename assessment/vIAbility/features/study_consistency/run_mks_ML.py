"""
run_repeats_markers.py
────────────────────────────────────────────────────────────────────────────────
Runs the ENTIRE markers_ML.py pipeline — clustering → train/val split →
hyperparameter search → bagging → voting search → refit on train+val →
diagnostic 5-fold CV → evaluate on the fixed held-out test set — for
N_REPEATS independent OUTER repeats, each with its own seed.

Unlike features_ML.py, this script has no dataset/strategy axis to sweep —
every classifier + bagging + voting combo is already evaluated in a single
run. So "n=5" here is simply: repeat that single run 5 times with different
seeds, then look at mean ± std of each model's TEST metrics across repeats.

Resumable: if a repeat's rows are already present in the master results CSV
(markers_ML.METRICS_CSV), that repeat is loaded instead of retrained — this
pipeline includes a full hyperparameter grid search + voting-combination
search, so a single repeat can be slow.

Usage (same flags as markers_ML.py's CLI, applied to every repeat):
    python run_repeats_markers.py
"""

import argparse
import os, sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from assessment.vIAbility.features.markers_ML import main, METRICS_CSV, SEED, PATH_MODELS, PATH_OUTPUT, CLASSIFIERS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

N_REPEATS = 5
SEEDS = [SEED + i for i in range(N_REPEATS)]


def _min_expected_rows_per_iteration() -> int:
    # Base + bagging are always produced; voting models may be zero depending on config.
    return 2 * len(CLASSIFIERS)


def _load_iteration_rows(existing: pd.DataFrame, iteration: int, seed: int) -> pd.DataFrame:
    rows = existing[existing["iteration"] == iteration].copy()
    if "seed" in rows.columns:
        rows = rows[rows["seed"] == seed].copy()
    if rows.empty:
        return rows

    # If duplicates exist for the same model in one iteration, keep the latest by timestamp.
    if "timestamp" in rows.columns and "model" in rows.columns:
        rows = rows.sort_values("timestamp").drop_duplicates(subset=["model"], keep="last")
    return rows


def run_all(cli_args):
    all_dfs = []
    min_expected = _min_expected_rows_per_iteration()

    for i, seed in enumerate(SEEDS, start=1):
        if os.path.exists(METRICS_CSV):
            existing = pd.read_csv(METRICS_CSV)
            if "iteration" in existing.columns:
                cached_rows = _load_iteration_rows(existing, i, seed)
                if len(cached_rows) >= min_expected:
                    log.info("[%d/%d] SKIP iter=%d seed=%d  (reusing %d rows from %s)", i, N_REPEATS, i, seed, len(cached_rows), METRICS_CSV)
                    all_dfs.append(cached_rows)
                    continue
                if not cached_rows.empty:
                    log.info("[%d/%d] RE-RUN iter=%d seed=%d  (found only %d/%d cached rows)", i, N_REPEATS, i, seed, len(cached_rows), min_expected)

        log.info("[%d/%d] RUN  iter=%d seed=%d", i, N_REPEATS, i, seed)
        run_args = argparse.Namespace(
            n_clusters=cli_args.n_clusters,
            path_models=cli_args.path_models,
            out_dir=cli_args.out_dir,
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

    summary_csv = os.path.join(out_root, "repeats_summary.csv")
    summary.to_csv(summary_csv, index=False)
    log.info("Summary CSV written to %s", summary_csv)

    summary_xlsx = os.path.join(out_root, "repeats_summary.xlsx")
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        combined.sort_values(["model", "iteration"]).to_excel(writer, sheet_name="all_repeats", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    log.info("Summary written to %s", summary_xlsx)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat markers_ML.py's pipeline N=5 times with different seeds")
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--path_models", default=str(PATH_MODELS))
    parser.add_argument("--out_dir", default=str(PATH_OUTPUT))
    parser.add_argument("--vote_min", type=int, default=2)
    parser.add_argument("--vote_max", type=int, default=None)
    parser.add_argument("--voting", default="none", choices=["hard", "soft", "both", "none"])
    parser.add_argument("--vote_top_k", type=int, default=3)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--val_size", type=float, default=0.2)
    cli_args = parser.parse_args()
    run_all(cli_args)