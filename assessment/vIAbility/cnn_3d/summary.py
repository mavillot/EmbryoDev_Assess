"""
summarize_results.py
────────────────────────────────────────────────────────────────────────────────
Reads the master results CSV produced by run_experiments.py and writes an
Excel workbook with two sheets:

    • "all_runs" : every individual run (one row per iteration), sorted by
                   backbone / n_frames / iteration
    • "summary"  : mean ± std of every metric, grouped by (backbone, n_frames)
                   — this is the table you'd report in the thesis.

Run from the same directory as train.py, after run_experiments.py has
finished (or partially finished):

    python summarize_results.py
"""

import os
import sys

import pandas as pd

from train import METRICS_CSV

OUT_XLSX = os.path.join(os.path.dirname(METRICS_CSV), "experiment_results_summary.xlsx")


def main():
    if not os.path.exists(METRICS_CSV):
        print(f"No results file found at {METRICS_CSV} — run run_experiments.py first.")
        sys.exit(1)

    df = pd.read_csv(METRICS_CSV)

    # Every metric column except the identifying / bookkeeping ones gets
    # mean + std in the summary sheet.
    id_cols = {"timestamp", "backbone", "n_frames", "iteration", "seed", "ckpt_path", "best_epoch"}
    metric_cols = [c for c in df.columns if c not in id_cols and not c.startswith("best_val_")]

    summary = (df.groupby(["backbone", "n_frames"])[metric_cols].agg(["mean", "std", "count"]))
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["backbone", "n_frames"])

    all_runs = df.sort_values(["backbone", "n_frames", "iteration"])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        all_runs.to_excel(writer, sheet_name="all_runs", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"{len(df)} runs found.")
    print(f"Summary written to: {OUT_XLSX}")


if __name__ == "__main__":
    main()