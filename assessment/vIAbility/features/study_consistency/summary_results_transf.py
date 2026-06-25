"""
summarize_results_transformer.py
────────────────────────────────────────────────────────────────────────────────
Reads the master results CSV produced by run_experiments_transformer.py and
writes an Excel workbook with "all_runs" and "summary" (mean ± std per
dataset/model/pos-encoding config, across the 5 repeats) sheets.

Run from the same directory as features_transformer.py:
    python summarize_results_transformer.py
"""

import os
import sys

import pandas as pd

from features_transformer import METRICS_CSV

OUT_XLSX = os.path.join(os.path.dirname(METRICS_CSV), "experiment_results_summary.xlsx")


def main():
    if not os.path.exists(METRICS_CSV):
        print(f"No results file found at {METRICS_CSV} — run run_experiments_transformer.py first.")
        sys.exit(1)

    df = pd.read_csv(METRICS_CSV)

    id_cols = {"timestamp", "dataset", "model", "loss", "use_pos_encoding", "iteration", "seed", "ckpt_path", "out_dir"}
    metric_cols = [c for c in df.columns if c not in id_cols]

    group_cols = ["dataset", "model", "use_pos_encoding"]
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values("fbeta_mean", ascending=False)

    all_runs = df.sort_values(group_cols + ["iteration"])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        all_runs.to_excel(writer, sheet_name="all_runs", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"{len(df)} runs found.")
    print(f"Summary written to: {OUT_XLSX}")


if __name__ == "__main__":
    main()