"""
run_repeats.py
────────────────────────────────────────────────────────────────────────────────
Runs the ENTIRE features_ML.py pipeline — train/val split → hyperparameter
search → inner 5-fold CV diagnostics → refit on train+val → evaluate on the
fixed held-out test set — for N_REPEATS independent OUTER repeats, each with
its own seed.


The held-out test set itself is the same fixed folder every repeat — only
the train/val split (and therefore which hyperparameters look best, and the
inner CV folds) varies across repeats.

    python run_repeats.py --dataset femi --strategies mean mean_std 4stat
"""

import argparse
import os, sys
import logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from assessment.ploidIA.features.features_ML import main, SEED

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

N_REPEATS = 5
SEEDS = [SEED + i for i in range(N_REPEATS)]


def expected_output_dir(dataset: str, feat: str, iteration: int, use_pca: bool, pca_components: int) -> str:
    base = f"results/assessment/ploidIA/{dataset}/{feat}/ML/iter{iteration}"
    return f"{base}_pca{pca_components}" if use_pca else base


def run_all(cli_args):
    all_dfs = []

    for i, seed in enumerate(SEEDS, start=1):
        out_dir = expected_output_dir(cli_args.dataset, cli_args.feat, i, cli_args.use_pca, cli_args.pca_components)
        csv_path = os.path.join(out_dir, "results.csv")

        if os.path.exists(csv_path):
            log.info("[%d/%d] SKIP iter=%d  (found existing %s)", i, N_REPEATS, i, csv_path)
            df = pd.read_csv(csv_path)
            df["iteration"] = i
            df["seed"] = seed
        else:
            log.info("[%d/%d] RUN  iter=%d seed=%d", i, N_REPEATS, i, seed)
            run_args = argparse.Namespace(
                dataset=cli_args.dataset,
                feat=cli_args.feat,
                strategies=cli_args.strategies,
                use_pca=cli_args.use_pca,
                pca_components=cli_args.pca_components,
                iteration=i,
                seed=seed,
            )
            try:
                df = main(run_args)
            except Exception:
                log.exception("[%d/%d] FAILED iter=%d — continuing with next repeat", i, N_REPEATS, i)
                continue

            if df is None:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    log.error("[%d/%d] iter=%d produced no DataFrame and no %s", i, N_REPEATS, i, csv_path)
                    continue

        if "iteration" not in df.columns:
            df["iteration"] = i
        if "seed" not in df.columns:
            df["seed"] = seed
        all_dfs.append(df)

    if not all_dfs:
        log.error("No repeats completed — nothing to aggregate.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    fold_cols = [c for c in combined.columns if c.endswith("_folds")]
    combined = combined.drop(columns=fold_cols, errors="ignore")

    out_root = f"results/assessment/ploidIA/{cli_args.dataset}/{cli_args.feat}/ML"
    os.makedirs(out_root, exist_ok=True)

    combined_csv = os.path.join(out_root, "all_repeats_results.csv")
    combined.to_csv(combined_csv, index=False)
    log.info("All-repeats results saved to %s", combined_csv)

    metric_cols = [
        "test_accuracy", "test_precision", "test_recall", "test_fbeta", "test_auroc",
        "cv_fbeta_mean", "cv_auroc_mean",
    ]
    metric_cols = [c for c in metric_cols if c in combined.columns]

    summary = (combined.groupby(["classifier", "strategy"])[metric_cols].agg(["mean", "std", "count"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    if "test_fbeta_mean" in summary.columns:
        summary = summary.sort_values("test_fbeta_mean", ascending=False)

    summary_xlsx = os.path.join(out_root, "repeats_summary.xlsx")
    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        combined.sort_values(["classifier", "strategy", "iteration"]).to_excel(writer, sheet_name="all_repeats", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    log.info("Summary written to %s", summary_xlsx)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeat features_ML.py's pipeline N=5 times with different seeds")
    parser.add_argument("--dataset", type=str, default="Combined", choices=["EasyFiv","Quiron", "Combined"])
    parser.add_argument("--feat", type=str, default="resnet34")
    parser.add_argument("--strategies", nargs="+", default=["mean", "mean_std", "4stat", "weighted_mean", "pos_encoding"])
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--pca-components", type=int, default=64)
    cli_args = parser.parse_args()
    run_all(cli_args)