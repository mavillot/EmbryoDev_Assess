import argparse
import os
import sys
import glob
import logging
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from dataclasses import dataclass, asdict

import torch

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.config import (
    SEED,
    PARAM_GRID,
    PATH_DATASET,
)

from assessment.vIAbility.models.stacking_ML import (
    aggregate_mean,
    aggregate_mean_std,
    aggregate_4stat,
    aggregate_concat_pca,
    aggregate_weighted_mean,
    aggregate_pos_encoding,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

classmap = {
    "no_FHB": 0,
    "FHB": 1,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier,
    "SVC": SVC,
    "GaussianNB": GaussianNB,
    "DecisionTree": DecisionTreeClassifier,
    "MLPClassifier": MLPClassifier,
    "LogisticRegression": LogisticRegression,
    "TabPFN": TabPFNClassifier,
}

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FEMIDataset:

    def __init__(self, files):

        self.files = files
        self.labels = []

        for f in files:

            class_name = Path(f).parent.name

            if class_name not in classmap:
                raise ValueError(
                    f"Unknown class folder: {class_name}"
                )

            self.labels.append(classmap[class_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = torch.load(
            self.files[idx],
            map_location="cpu",
            weights_only=True,
        )

        x = data["embeddings"].float()
        mask = data["mask"].bool()

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.float32,
        )

        return x, mask, y


def build_datasets(feature_dir: str):

    print(f"Loading features from: {feature_dir}")

    train_files = sorted(
        glob.glob(
            os.path.join(feature_dir, "train", "*", "*.pt")
        )
    )

    test_files = sorted(
        glob.glob(
            os.path.join(feature_dir, "test", "*", "*.pt")
        )
    )

    if not train_files:
        raise FileNotFoundError(
            f"No .pt files found under {feature_dir}/train/*/"
        )

    if not test_files:
        raise FileNotFoundError(
            f"No .pt files found under {feature_dir}/test/*/"
        )

    print(f"Found {len(train_files)} train files")
    print(f"Found {len(test_files)} test files")

    train_dataset = FEMIDataset(train_files)
    test_dataset = FEMIDataset(test_files)

    def _collate(dataset):

        X = []
        masks = []
        ys = []

        for i in range(len(dataset)):

            x, mask, y = dataset[i]

            X.append(x)
            masks.append(mask)
            ys.append(y)

        return (
            torch.stack(X),
            torch.stack(masks),
            torch.stack(ys),
        )

    return _collate(train_dataset), _collate(test_dataset)

# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

AGGREGATORS = {
    "mean": aggregate_mean,
    "mean_std": aggregate_mean_std,
    "4stat": aggregate_4stat,
    "concat_pca": aggregate_concat_pca,
    "weighted_mean": aggregate_weighted_mean,
    "pos_encoding": aggregate_pos_encoding,
}


def apply_aggregator(
    X_tensors,
    mask_tensors,
    strategy,
    pca_n_components=128,
    fit_pca=True,
    pca=None,
):

    if strategy not in AGGREGATORS:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "concat_pca":

        X_agg, fitted_pca = aggregate_concat_pca(
            X_tensors,
            mask_tensors,
            n_components=pca_n_components,
            fit=fit_pca,
            pca=pca,
        )

        return X_agg, fitted_pca

    fn = AGGREGATORS[strategy]

    X_agg = np.stack([
        fn(x, m)
        for x, m in zip(X_tensors, mask_tensors)
    ])

    return X_agg, None

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:

    classifier: str
    strategy: str
    params: dict

    accuracy: float
    precision: float
    recall: float
    f1: float

    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0

    cv_precision_mean: float = 0.0
    cv_precision_std: float = 0.0

    cv_recall_mean: float = 0.0
    cv_recall_std: float = 0.0

    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0

    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0


def evaluate_validation(
    clf,
    X_train,
    y_train,
    X_val,
    y_val,
    cv_folds=5,
):

    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)

    cv_results = cross_validate(
        clf,
        X_train,
        y_train,
        cv=StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=SEED,
        ),
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=-1,
        return_train_score=False,
    )

    metrics = {}

    for metric in ["accuracy", "precision", "recall", "f1"]:

        vals = cv_results[f"test_{metric}"]

        metrics[f"cv_{metric}_mean"] = vals.mean()
        metrics[f"cv_{metric}_std"] = vals.std()
        metrics[f"cv_{metric}_folds"] = vals.tolist()

    metrics["accuracy"] = accuracy_score(y_val, preds)

    metrics["precision"] = precision_score(
        y_val,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        y_val,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        y_val,
        preds,
        average="binary",
        zero_division=0,
    )

    metrics["predictions"] = preds

    return metrics


def evaluate_test_set(
    clf,
    X_test,
    y_test,
):

    preds = clf.predict(X_test)

    return {
        "test_accuracy": accuracy_score(y_test, preds),

        "test_precision": precision_score(
            y_test,
            preds,
            average="binary",
            zero_division=0,
        ),

        "test_recall": recall_score(
            y_test,
            preds,
            average="binary",
            zero_division=0,
        ),

        "test_f1": f1_score(
            y_test,
            preds,
            average="binary",
            zero_division=0,
        ),

        "predictions": preds,
    }


def save_cv_consistency_report(
    df,
    output_dir,
):

    lines = []

    header = (
        "\n"
        + "=" * 90
        + "\nCV vs TEST consistency report\n"
        + "=" * 90
    )

    lines.append(header)

    for _, row in df.iterrows():

        block = []

        block.append(
            f"\n[{row['classifier']} | {row['strategy']}]"
        )

        metrics = [
            ("accuracy", row["cv_accuracy_mean"], row["cv_accuracy_std"], row["test_accuracy"]),
            ("precision", row["cv_precision_mean"], row["cv_precision_std"], row["test_precision"]),
            ("recall", row["cv_recall_mean"], row["cv_recall_std"], row["test_recall"]),
            ("f1", row["cv_f1_mean"], row["cv_f1_std"], row["test_f1"]),
        ]

        for name, cv_mean, cv_std, test_score in metrics:

            diff = test_score - cv_mean

            threshold = max(1.5 * cv_std, 0.02)

            if diff > threshold:
                flag = "optimistic"
            elif diff < -threshold:
                flag = "pessimistic"
            else:
                flag = "consistent"

            block.append(
                f"{name:<10} "
                f"cv={cv_mean:.3f}±{cv_std:.3f} "
                f"test={test_score:.3f} "
                f"[{flag}]"
            )

        lines.append("\n".join(block))

    report_path = os.path.join(
        output_dir,
        "cv_consistency_report.txt",
    )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logging.info(f"Consistency report saved → {report_path}")


def _save_confusion_matrix(
    y_true,
    y_pred,
    output_dir,
    model_name,
    strategy,
):

    cm = confusion_matrix(y_true, y_pred)

    labels = ["no_FHB", "FHB"]

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(
        f"Test Confusion Matrix\n"
        f"{model_name} ({strategy})"
    )

    plt.ylabel("True")
    plt.xlabel("Predicted")

    path = os.path.join(
        output_dir,
        "best_model_confusion_matrix.png",
    )

    plt.savefig(path, dpi=150)
    plt.close()

    logging.info(f"Confusion matrix saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main experiments
# ──────────────────────────────────────────────────────────────────────────────

def run_experiments(
    X_train_tensors,
    mask_train,
    y_train,

    X_validation_tensors,
    mask_validation,
    y_validation,

    X_test_tensors,
    mask_test,
    y_test,

    strategies=("mean", "mean_std", "4stat"),

    classifiers=None,

    param_grid=None,

    use_scaling=True,

    use_pca=False,

    pca_components=64,

    cv_folds=5,

    output_dir="output/assessment/vIAbility/femi/ML",
):

    if classifiers is None:
        classifiers = list(CLASSIFIERS.keys())

    if param_grid is None:
        param_grid = PARAM_GRID

    if use_pca:
        output_dir = f"{output_dir}_pca{pca_components}"

    os.makedirs(output_dir, exist_ok=True)

    results = []

    best_overall_test_f1 = -1
    best_test_predictions = None
    best_model_info = {}

    for strategy in strategies:

        logging.info(
            f"\n{'='*70}\n"
            f"Aggregation strategy: {strategy}\n"
            f"{'='*70}"
        )

        X_tr, train_pca = apply_aggregator(
            X_train_tensors,
            mask_train,
            strategy,
            pca_n_components=128,
            fit_pca=True,
        )

        X_vl, _ = apply_aggregator(
            X_validation_tensors,
            mask_validation,
            strategy,
            pca_n_components=128,
            fit_pca=False,
            pca=train_pca,
        )

        X_ts, _ = apply_aggregator(
            X_test_tensors,
            mask_test,
            strategy,
            pca_n_components=128,
            fit_pca=False,
            pca=train_pca,
        )

        if use_scaling:

            scaler = StandardScaler()

            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)
            X_ts = scaler.transform(X_ts)

        if use_pca:

            pca = PCA(
                n_components=min(
                    pca_components,
                    X_tr.shape[0] - 1,
                    X_tr.shape[1],
                )
            )

            X_tr = pca.fit_transform(X_tr)
            X_vl = pca.transform(X_vl)
            X_ts = pca.transform(X_ts)

            logging.info(
                f"PCA explained variance: "
                f"{pca.explained_variance_ratio_.sum():.2%}"
            )

        for clf_name in classifiers:

            clf_cls = CLASSIFIERS[clf_name]

            grid = param_grid.get(clf_name, {})

            param_combinations = (
                [{}]
                if not grid
                else [
                    dict(zip(grid.keys(), v))
                    for v in itertools.product(*grid.values())
                ]
            )

            best_result = None

            for params in param_combinations:

                try:

                    clf = clf_cls(**params)

                    metrics = evaluate_validation(
                        clf,
                        X_tr,
                        y_train,
                        X_vl,
                        y_validation,
                        cv_folds,
                    )

                    result = RunResult(
                        classifier=clf_name,
                        strategy=strategy,
                        params=params,

                        accuracy=metrics["accuracy"],
                        precision=metrics["precision"],
                        recall=metrics["recall"],
                        f1=metrics["f1"],

                        cv_accuracy_mean=metrics["cv_accuracy_mean"],
                        cv_accuracy_std=metrics["cv_accuracy_std"],

                        cv_precision_mean=metrics["cv_precision_mean"],
                        cv_precision_std=metrics["cv_precision_std"],

                        cv_recall_mean=metrics["cv_recall_mean"],
                        cv_recall_std=metrics["cv_recall_std"],

                        cv_f1_mean=metrics["cv_f1_mean"],
                        cv_f1_std=metrics["cv_f1_std"],
                    )

                    if (
                        best_result is None
                        or result.f1 > best_result.f1
                    ):
                        best_result = result

                except Exception as e:

                    logging.warning(
                        f"[{clf_name}] "
                        f"params={params} failed → {e}"
                    )

            if best_result is not None:

                X_final = np.concatenate(
                    [X_tr, X_vl],
                    axis=0,
                )

                y_final = np.concatenate(
                    [y_train, y_validation],
                    axis=0,
                )

                final_clf = clf_cls(**best_result.params)

                final_clf.fit(X_final, y_final)

                test_metrics = evaluate_test_set(
                    final_clf,
                    X_ts,
                    y_test,
                )

                best_result.test_accuracy = test_metrics["test_accuracy"]
                best_result.test_precision = test_metrics["test_precision"]
                best_result.test_recall = test_metrics["test_recall"]
                best_result.test_f1 = test_metrics["test_f1"]

                results.append(best_result)

                logging.info(
                    f"{clf_name:20s} | "
                    f"val_f1={best_result.f1:.3f} | "
                    f"test_f1={best_result.test_f1:.3f} | "
                    f"cv_f1={best_result.cv_f1_mean:.3f}"
                    f"±{best_result.cv_f1_std:.3f}"
                )

                if best_result.test_f1 > best_overall_test_f1:

                    best_overall_test_f1 = best_result.test_f1

                    best_test_predictions = test_metrics["predictions"]

                    best_model_info = {
                        "name": clf_name,
                        "strategy": strategy,
                    }

    if best_test_predictions is not None:

        _save_confusion_matrix(
            y_test,
            best_test_predictions,
            output_dir,
            best_model_info["name"],
            best_model_info["strategy"],
        )

    df = pd.DataFrame([
        asdict(r)
        for r in results
    ])

    csv_path = os.path.join(
        output_dir,
        "results.csv",
    )

    df.to_csv(csv_path, index=False)

    cv_results_path = os.path.join(
        output_dir,
        "cv_results_detailed.csv",
    )

    df.to_csv(cv_results_path, index=False)

    logging.info(
        f"Detailed CV results saved → {cv_results_path}"
    )

    _plot_results(df, output_dir)

    save_cv_consistency_report(
        df,
        output_dir,
    )

    return df

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_results(
    df,
    output_dir,
):

    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]

    for metric, title in metrics:

        fig, ax = plt.subplots(
            figsize=(max(10, len(df) * 0.7), 6)
        )

        labels = [
            f"{r.classifier}\n({r.strategy})"
            for _, r in df.iterrows()
        ]

        cv_means = df[f"cv_{metric}_mean"] * 100
        cv_stds = df[f"cv_{metric}_std"] * 100
        test_scores = df[f"test_{metric}"] * 100

        x = np.arange(len(labels))

        width = 0.35

        ax.bar(
            x - width / 2,
            cv_means,
            width,
            yerr=cv_stds,
            capsize=4,
            label="CV mean ± std",
        )

        ax.bar(
            x + width / 2,
            test_scores,
            width,
            label="Test",
        )

        ax.set_ylabel(f"{title} (%)")

        ax.set_title(
            f"CV vs Test — {title}"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            labels,
            rotation=40,
            ha="right",
            fontsize=8,
        )

        ax.set_ylim(0, 110)

        ax.legend()

        fig.tight_layout()

        path = os.path.join(
            output_dir,
            f"cv_vs_test_{metric}.png",
        )

        plt.savefig(path, dpi=150)

        plt.close()

        logging.info(f"Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def tensors_to_lists(
    X,
    mask,
):

    return list(X), list(mask)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    (
        X_full,
        mask_full,
        y_full,
    ), (
        X_test,
        mask_test,
        y_test,
    ) = build_datasets(
        PATH_DATASET
        + f"Oslo_viability/{args.dataset}"
    )

    indices = np.arange(len(y_full))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y_full.numpy(),
        random_state=SEED,
    )

    X_tr = X_full[train_idx]
    mask_tr = mask_full[train_idx]
    y_tr = y_full[train_idx]

    X_vl = X_full[val_idx]
    mask_vl = mask_full[val_idx]
    y_vl = y_full[val_idx]

    print(f"Train samples: {len(y_tr)}")
    print(f"Val samples:   {len(y_vl)}")
    print(f"Test samples:  {len(y_test)}")

    X_tr_list, mask_tr_list = tensors_to_lists(
        X_tr,
        mask_tr,
    )

    X_vl_list, mask_vl_list = tensors_to_lists(
        X_vl,
        mask_vl,
    )

    X_test_list, mask_test_list = tensors_to_lists(
        X_test,
        mask_test,
    )

    df = run_experiments(
        X_tr_list,
        mask_tr_list,
        y_tr.numpy(),

        X_vl_list,
        mask_vl_list,
        y_vl.numpy(),

        X_test_list,
        mask_test_list,
        y_test.numpy(),

        strategies=args.strategies,

        use_scaling=True,

        use_pca=args.use_pca,

        pca_components=args.pca_components,

        cv_folds=5,

        output_dir=f"output/assessment/vIAbility/{args.dataset}/ML",
    )

    print(
        df.sort_values(
            "test_f1",
            ascending=False,
        ).to_string()
    )

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train ML models on FEMI embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="femi",
        help="Dataset folder",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "mean",
            "mean_std",
            "4stat",
            "weighted_mean",
            "pos_encoding",
        ],
    )

    parser.add_argument(
        "--use-pca",
        action="store_true",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    main(args)