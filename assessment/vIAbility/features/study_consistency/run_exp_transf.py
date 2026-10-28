"""
run_experiments_transformer.py
────────────────────────────────────────────────────────────────────────────────
Repeats features_transformer.py's training N_REPEATS times, each with a
different seed, across every requested config:

    • datasets : femi, markers, resnet34
    • models   : rnn, gru, lstm   (no positional-encoding variant)
                 transformer      (×2: with and without positional encoding)

→ 3 datasets × (3 plain models + 2 transformer variants) = 15 configs
  × 5 repeats = 75 runs total.

Every run's final test metrics (accuracy, precision, recall, F1, F-beta,
AUROC — already computed by EpochMetrics) are appended to
features_transformer.METRICS_CSV.

All other hyperparameters (epochs, batch_size, lr, loss="ce", dropout, etc.)
are left at features_transformer.py's own CLI defaults — only dataset /
model / use_pos_encoding / seed / iteration vary across runs. Edit BASE_ARGS
below if you want a different loss function, scheduler, etc. swept too.

Resumable: skips any (dataset, model, use_pos_encoding, iteration) combo
already present in the results CSV.

Run from the same directory as features_transformer.py:
    python run_experiments_transformer.py
"""

import csv, sys
from pathlib import Path
import logging
import os
from argparse import Namespace
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from assessment.vIAbility.features.features_transformer import main, METRICS_CSV, SEED, BASE_MODELS, BASE_OUTPUT


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

N_REPEATS = 5
SEEDS = [SEED + i for i in range(N_REPEATS)]

DATASETS = ["femi"]
PLAIN_MODELS = ["rnn", "gru", "lstm"]

# (model, use_pos_encoding) pairs — pos-encoding only meaningful for transformer
MODEL_VARIANTS = [(m, False) for m in PLAIN_MODELS] + [("transformer", False), ("transformer", True)]

EXPERIMENTS = [
    (dataset, model, pos_enc)
    for dataset in DATASETS
    for model, pos_enc in MODEL_VARIANTS
]

BASE_ARGS = dict(
    data_dir=None,
    checkpoint=str(BASE_MODELS),
    out_dir=str(BASE_OUTPUT),
    weight_loss=True,
    loss="ce",
    dropout=0.2,
    d_model=32, num_heads=4, ff_dim=64,
    hidden1=32, hidden2=16, fc_dim=16,
    epochs=200, batch_size=32, lr=1e-3, weight_decay=1e-4,
    patience=20, num_workers=4,
    scheduler="plateau",
    save_by="loss",
    beta=0.35,
)


def tag(dataset, model, pos_enc):
    suffix = "_pos_enc" if (pos_enc and model == "transformer") else ""
    return f"{dataset}/{model}{suffix}"


def already_done(dataset: str, model: str, pos_enc: bool, iteration: int) -> bool:
    if not os.path.exists(METRICS_CSV):
        return False
    with open(METRICS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if (
                row.get("dataset") == dataset
                and row.get("model") == model
                and str(row.get("use_pos_encoding")) == str(pos_enc)
                and row.get("iteration") not in (None, "")
                and int(row["iteration"]) == iteration
            ):
                return True
    return False


def run_all():
    total = len(EXPERIMENTS) * N_REPEATS
    done = 0

    for dataset, model, pos_enc in EXPERIMENTS:
        for i, seed in enumerate(SEEDS, start=1):
            done += 1
            label = tag(dataset, model, pos_enc)

            if already_done(dataset, model, pos_enc, i):
                log.info("[%2d/%2d] SKIP  %-22s iter=%d  (already logged)", done, total, label, i)
                continue

            log.info("[%2d/%2d] START %-22s iter=%d seed=%d", done, total, label, i, seed)

            args = Namespace(
                dataset=dataset, model=model, use_pos_encoding=pos_enc,
                iteration=i, seed=seed, **BASE_ARGS,
            )
            try:
                row = main(args)
                log.info(
                    "[%2d/%2d] DONE  %-22s iter=%d  |  acc=%.3f  fbeta=%.3f  auroc=%.3f",
                    done, total, label, i, row["acc"], row["fbeta"], row["auroc"],
                )
            except Exception:
                log.exception(
                    "[%2d/%2d] FAILED %s iter=%d — continuing with next run",
                    done, total, label, i,
                )
                continue

    log.info("Sweep complete. All results in: %s", METRICS_CSV)


if __name__ == "__main__":
    run_all()