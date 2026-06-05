import argparse
import os
import sys
import json
import glob
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PARAM_GRID, PATH_DATASET, PATH_MODELS
from assessment.vIAbility.models.stacking_ML import (aggregate_mean, aggregate_mean_std, aggregate_4stat, 
                                                     aggregate_concat_pca,aggregate_weighted_mean, aggregate_pos_encoding,)

classmap     = {"no_FHB": 0, "FHB": 1}
N_CLUSTERS   = 5

random.seed(SEED)

CLASSIFIERS = {
    # "RandomForest":       RandomForestClassifier,
    # "SVC":                SVC,
    # "GaussianNB":         GaussianNB,
    # "DecisionTree":       DecisionTreeClassifier,
    # "MLPClassifier":      MLPClassifier,
    # "LogisticRegression": LogisticRegression,
    "TabPFN":             TabPFNClassifier,
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FEMIDataset:
    """
    Expects files under:
        <split>/<class_name>/<file>.pt
    e.g.  femi/train/FHB/sample.pt
          femi/test/no_FHB/sample.pt

    The class label is derived from the parent folder name via `classmap`.
    """

    def __init__(self, files: list[str]):
        self.files = files
        # Derive label from the immediate parent directory name
        self.labels = []
        for f in files:
            class_name = Path(f).parent.name          # "FHB" or "no_FHB"
            if class_name not in classmap:
                raise ValueError(
                    f"Unknown class folder '{class_name}' in path: {f}\n"
                    f"Expected one of: {list(classmap.keys())}"
                )
            self.labels.append(classmap[class_name])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        x    = data["embeddings"].float()   # (T, embed_dim)
        mask = data["mask"].bool()          # (T,)  True = valid frame
        y    = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, mask, y


def build_datasets(
    feature_dir: str,
) -> tuple[tuple, tuple]:
    """
    Scans  <feature_dir>/train/<class>/*.pt  and
           <feature_dir>/test/<class>/*.pt
    Returns ((X_train, mask_train, y_train), (X_val, mask_val, y_val))
    as stacked tensors.
    """
    print(f"Loading features from: {feature_dir}")
    # Glob both class folders under train/ and test/
    train_files = sorted(glob.glob(os.path.join(feature_dir, "train", "*", "*.pt")))
    val_files   = sorted(glob.glob(os.path.join(feature_dir, "test",  "*", "*.pt")))

    if not train_files:
        raise FileNotFoundError(f"No .pt files found under {feature_dir}/train/*/")
    if not val_files:
        raise FileNotFoundError(f"No .pt files found under {feature_dir}/test/*/")

    print(f"Found {len(train_files)} train and {len(val_files)} val files.")

    train_dataset = FEMIDataset(train_files)
    val_dataset   = FEMIDataset(val_files)

    def _collate(dataset: FEMIDataset):
        X, masks, ys = [], [], []
        for i in range(len(dataset)):
            x, mask, y = dataset[i]
            X.append(x)
            masks.append(mask)
            ys.append(y)
        return (
            torch.stack(X),      # (N, T, embed_dim)
            torch.stack(masks),  # (N, T)
            torch.stack(ys),     # (N,)
        )

    return _collate(train_dataset), _collate(val_dataset)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation strategies
# ──────────────────────────────────────────────────────────────────────────────


AGGREGATORS = {
    "mean":     aggregate_mean,
    "mean_std": aggregate_mean_std,
    "4stat":    aggregate_4stat,
    "concat_pca": aggregate_concat_pca,
    "weighted_mean": aggregate_weighted_mean,
    "pos_encoding": aggregate_pos_encoding,
}


def apply_aggregator(
    X_tensors:    list[torch.Tensor],
    mask_tensors: list[torch.Tensor],
    strategy:     str,
    # concat_pca-only kwargs — ignored for all other strategies
    pca_n_components: int = 128,
    fit_pca:          bool = True,
    pca:              "PCA | None" = None,
) -> "tuple[np.ndarray, PCA | None]":
    """
    Apply an aggregation strategy to a list of (embedding, mask) pairs.

    Returns:
        (X_agg, fitted_pca)
        fitted_pca is None for all strategies except 'concat_pca'.
        For concat_pca call once with fit_pca=True on train, then again with
        fit_pca=False and pca=<train_pca> for val.

    FIX: concat_pca previously crashed because it expects the full dataset at once
    (not per-sample), and needs separate fit/transform calls for train vs val.
    """
    if strategy not in AGGREGATORS:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(AGGREGATORS)}")

    if strategy == "concat_pca":
        # FIX: concat_pca operates on the whole dataset, not per-sample
        X_agg, fitted_pca = aggregate_concat_pca(
            X_tensors, mask_tensors,
            n_components=pca_n_components,
            fit=fit_pca,
            pca=pca,
        )
        return X_agg, fitted_pca

    fn = AGGREGATORS[strategy]
    return np.stack([fn(x, m) for x, m in zip(X_tensors, mask_tensors)]), None


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    classifier:  str
    strategy:    str
    params:      dict
    accuracy:    float
    precision:   float
    recall:      float
    f1:          float
    cv_f1_mean:  float = 0.0
    cv_f1_std:   float = 0.0


def evaluate(
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    cv_folds: int = 5,
) -> dict:
    """
    Fit clf on X_train, evaluate on X_val, and compute cross-validated F1.

    FIX: previously this was called AFTER a manual clf.fit() in the loop, causing
    clf to be fitted twice. Now this is the only fit call — the caller must not
    pre-fit the classifier before calling this function.
    """
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="f1",
    )

    return dict(
        accuracy   = accuracy_score(y_val, preds),
        precision  = precision_score(y_val, preds, average="binary", zero_division=0),
        recall     = recall_score(y_val, preds, average="binary", zero_division=0),
        f1         = f1_score(y_val, preds, average="binary", zero_division=0),
        cv_f1_mean = cv_scores.mean(),
        cv_f1_std  = cv_scores.std(),
    )

def _save_confusion_matrix(y_true, y_pred, output_dir, model_name, strategy):
    cm = confusion_matrix(y_true, y_pred)
    # Usar los nombres de las clases definidos en tu classmap
    labels = ["no_FHB", "FHB"] 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix: {model_name} ({strategy})\nBest F1-Score Model")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    path = os.path.join(output_dir, "best_model_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Confusion Matrix saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def run_experiments(
    X_train_tensors: list[torch.Tensor],
    mask_train:      list[torch.Tensor],
    y_train:         np.ndarray,
    X_val_tensors:   list[torch.Tensor],
    mask_val:        list[torch.Tensor],
    y_val:           np.ndarray,
    strategies:      list[str] = ("mean", "mean_std", "4stat"),
    classifiers:     list[str] = None,
    param_grid:      dict      = None,
    use_scaling:     bool      = True,
    use_pca:         bool      = False,
    pca_components:  int       = 64,
    cv_folds:        int       = 5,
    output_dir:      str       = "output/assessment/vIAbility/femi/ML",
) -> pd.DataFrame:

    if classifiers is None:
        classifiers = list(CLASSIFIERS.keys())
    if param_grid is None:
        param_grid = PARAM_GRID
    if use_pca:
        output_dir = f'{output_dir}_pca{pca_components}'
    os.makedirs(output_dir, exist_ok=True)
    results: list[RunResult] = []

    best_overall_f1 = -1.0
    best_y_pred = None
    best_model_info = {"name": "", "strategy": ""}

    for strategy in strategies:
        logging.info(f"\n{'='*60}\nAggregation strategy: {strategy}\n{'='*60}")

        # FIX: concat_pca must fit on train and transform val with the same PCA object
        X_tr, train_pca = apply_aggregator(
            X_train_tensors, mask_train, strategy,
            pca_n_components=128, fit_pca=True,
        )
        X_vl, _ = apply_aggregator(
            X_val_tensors, mask_val, strategy,
            pca_n_components=128, fit_pca=False, pca=train_pca,
        )

        # Optional scaling + PCA (strongly recommended for SVC, LR, MLP)
        if use_scaling:
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_vl   = scaler.transform(X_vl)

        if use_pca:
            pca  = PCA(n_components=min(pca_components, X_tr.shape[0]-1, X_tr.shape[1]))
            X_tr = pca.fit_transform(X_tr)
            X_vl = pca.transform(X_vl)
            logging.info(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        for clf_name in classifiers:
            clf_cls = CLASSIFIERS[clf_name]
            grid    = param_grid.get(clf_name, {})
            param_combinations = [{}] if not grid else list(
                dict(zip(grid.keys(), v)) for v in
                __import__('itertools').product(*grid.values())
            )

            best: RunResult | None = None

            for params in param_combinations:
                try:
                    clf = clf_cls(**params)

                    # FIX: was calling clf.fit() here AND inside evaluate(), fitting twice.
                    # Now we fit once, get predictions, then run CV (which re-fits internally
                    # on folds but does not affect the already-fitted clf used for val preds).
                    metrics = evaluate(clf, X_tr, y_train, X_vl, y_val, cv_folds)
                    # After evaluate(), clf is fitted on the full X_tr (last fit inside evaluate)
                    preds = clf.predict(X_vl)   # FIX: predictions from the correctly fitted model

                    f1 = metrics["f1"]

                    # FIX: best_overall tracking now uses predictions from the correct model state
                    if f1 > best_overall_f1:
                        best_overall_f1 = f1
                        best_y_pred     = preds
                        best_model_info = {"name": clf_name, "strategy": strategy}

                    r = RunResult(classifier=clf_name, strategy=strategy, params=params, **metrics)
                    if best is None or r.f1 > best.f1:
                        best = r
                except Exception as e:
                    logging.warning(f"  [{clf_name}] params={params} failed: {e}")

            if best:
                results.append(best)
                logging.info(
                    f"  {clf_name:20s} | "
                    f"acc={best.accuracy:.3f} f1={best.f1:.3f} "
                    f"cv_f1={best.cv_f1_mean:.3f}±{best.cv_f1_std:.3f} | "
                    f"params={best.params}"
                )

    if best_y_pred is not None:
        _save_confusion_matrix(y_val, best_y_pred, output_dir, 
                               best_model_info["name"], best_model_info["strategy"])

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    _plot_results(df, output_dir)
    return df



# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_results(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot_f1 = df.pivot_table(index="classifier", columns="strategy", values="f1")
    sns.heatmap(
        pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu",
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Val F1 by classifier × aggregation strategy")
    axes[0].set_xlabel("")

    df_best = df.loc[df.groupby("classifier")["cv_f1_mean"].idxmax()].copy()
    df_best = df_best.sort_values("cv_f1_mean", ascending=True)
    axes[1].barh(
        df_best["classifier"] + "\n(" + df_best["strategy"] + ")",
        df_best["cv_f1_mean"],
        xerr=df_best["cv_f1_std"],
        color="#5D6BBF", ecolor="gray", capsize=4,
    )
    axes[1].set_xlabel("CV F1 (binary, ±1 std)")
    axes[1].set_title("Best CV F1 per classifier")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "results_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Plot saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def tensors_to_lists(
    X:    torch.Tensor,   # (N, T, D)
    mask: torch.Tensor,   # (N, T)
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Unstack batch dimension back to lists for the aggregators."""
    return list(X), list(mask)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    (X_tr, mask_tr, y_tr), (X_vl, mask_vl, y_vl) = build_datasets(PATH_DATASET + f"Oslo_viability/{args.dataset}")

    X_tr_list,  mask_tr_list  = tensors_to_lists(X_tr,  mask_tr)
    X_vl_list,  mask_vl_list  = tensors_to_lists(X_vl,  mask_vl)

    df = run_experiments(
        X_tr_list, mask_tr_list, y_tr.numpy(),
        X_vl_list, mask_vl_list, y_vl.numpy(),
        strategies     = args.strategies,
        use_scaling    = True,
        use_pca        = args.use_pca,
        pca_components = args.pca_components,
        cv_folds       = 5,
        output_dir     = f"output/assessment/vIAbility/{args.dataset}/ML",
    )
    print(df.sort_values("cv_f1_mean", ascending=False).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML for FEMI or Resnet34 features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",type=str,default="femi",help="Subfolder under PATH_DATASET containing train/ and test/ splits",)
    parser.add_argument("--strategies", nargs="+", default=["mean", "mean_std", "4stat", "weighted_mean", "pos_encoding"],
                        help="Aggregation strategies to evaluate. 'concat_pca' is available but handled separately.")
    parser.add_argument("--use-pca", action="store_true", help="Whether to apply PCA for dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=64, help="Number of PCA components if --use-pca is set")
    args = parser.parse_args()
    main(args)