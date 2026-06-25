import argparse, os, sys, glob, logging, random, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PARAM_GRID, PATH_DATASET
from assessment.vIAbility.features.stacking_ML import aggregate_mean, aggregate_mean_std, aggregate_4stat, aggregate_concat_pca, aggregate_weighted_mean, aggregate_pos_encoding

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
classmap = {"no_FHB": 0, "FHB": 1}
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASSIFIERS = {"RandomForest": RandomForestClassifier, "SVC": SVC, "GaussianNB": GaussianNB,"DecisionTree": DecisionTreeClassifier,
               "MLPClassifier": MLPClassifier,"LogisticRegression": LogisticRegression, "TabPFN": TabPFNClassifier}
AGGREGATORS = {"mean": aggregate_mean, "mean_std": aggregate_mean_std, "4stat": aggregate_4stat,"concat_pca": aggregate_concat_pca, 
               "weighted_mean": aggregate_weighted_mean, "pos_encoding": aggregate_pos_encoding}

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
                raise ValueError(f"Unknown class folder: {class_name}")
            self.labels.append(classmap[class_name])

    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        return data["embeddings"].float(), data["mask"].bool(), torch.tensor(self.labels[idx], dtype=torch.float32)

def build_datasets(feature_dir: str):
    print(f"Loading features from: {feature_dir}")
    train_files = sorted(glob.glob(os.path.join(feature_dir, "train", "*", "*.pt")))
    test_files = sorted(glob.glob(os.path.join(feature_dir, "test", "*", "*.pt")))
    if not train_files: 
        raise FileNotFoundError(f"No .pt files found under {feature_dir}/train/*/")
    if not test_files: 
        raise FileNotFoundError(f"No .pt files found under {feature_dir}/test/*/")
    print(f"Found {len(train_files)} train files\nFound {len(test_files)} test files")

    def _collate(dataset):
        X, masks, ys = [], [], []
        for i in range(len(dataset)):
            x, mask, y = dataset[i]
            X.append(x); masks.append(mask); ys.append(y)
        return torch.stack(X), torch.stack(masks), torch.stack(ys)
    return _collate(FEMIDataset(train_files)), _collate(FEMIDataset(test_files))

# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────
def apply_aggregator(X_tensors, mask_tensors, strategy, pca_n_components=128, fit_pca=True, pca=None):
    if strategy not in AGGREGATORS: raise ValueError(f"Unknown strategy: {strategy}")
    if strategy == "concat_pca":
        X_agg, fitted_pca = aggregate_concat_pca(X_tensors, mask_tensors, n_components=pca_n_components, fit=fit_pca, pca=pca)
        return X_agg, fitted_pca
    return np.stack([AGGREGATORS[strategy](x, m) for x, m in zip(X_tensors, mask_tensors)]), None

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RunResult:
    classifier: str; strategy: str; params: dict; iteration: int = 0; seed: int = SEED
    accuracy: float = 0.0; precision: float = 0.0; recall: float = 0.0; fbeta: float = 0.0; auroc: float = 0.0
    cv_accuracy_mean: float = 0.0; cv_accuracy_std: float = 0.0
    cv_precision_mean: float = 0.0; cv_precision_std: float = 0.0
    cv_recall_mean: float = 0.0; cv_recall_std: float = 0.0
    cv_fbeta_mean: float = 0.0; cv_fbeta_std: float = 0.0
    cv_auroc_mean: float = 0.0; cv_auroc_std: float = 0.0
    cv_accuracy_folds: object = None; cv_precision_folds: object = None; cv_recall_folds: object = None
    cv_fbeta_folds: object = None; cv_auroc_folds: object = None
    test_accuracy: float = 0.0; test_precision: float = 0.0; test_recall: float = 0.0; test_fbeta: float = 0.0; test_auroc: float = 0.0

def evaluate_validation(clf, X_train, y_train, X_val, y_val, cv_folds=5, seed=SEED):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    cv_results = cross_validate(
        clf, X_train, y_train, n_jobs=-1, return_train_score=False,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        scoring={
            "accuracy": "accuracy", "precision": "precision", "recall": "recall",
            "fbeta": make_scorer(fbeta_score, beta=0.5, pos_label=1, zero_division=0),
            "auroc": make_scorer(roc_auc_score, response_method="predict_proba"),
        }
    )
    metrics = {}
    for metric in ["accuracy", "precision", "recall", "fbeta", "auroc"]:
        vals = cv_results[f"test_{metric}"]
        metrics[f"cv_{metric}_mean"] = vals.mean()
        metrics[f"cv_{metric}_std"] = vals.std()
        metrics[f"cv_{metric}_folds"] = vals.tolist()

    metrics.update({
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, average="binary", zero_division=0),
        "recall": recall_score(y_val, preds, average="binary", zero_division=0),
        "fbeta": fbeta_score(y_val, preds, beta=0.5, average="binary", zero_division=0),
        "auroc": roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else preds),
        "predictions": preds
    })
    return metrics

def evaluate_test_set(clf, X_test, y_test):
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds
    return {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_precision": precision_score(y_test, preds, average="binary", zero_division=0),
        "test_recall": recall_score(y_test, preds, average="binary", zero_division=0),
        "test_fbeta": fbeta_score(y_test, preds, beta=0.5, average="binary", zero_division=0),
        "test_auroc": roc_auc_score(y_test, proba),
        "predictions": preds,
    }

def save_cv_consistency_report(df, output_dir):
    lines = ["\n" + "=" * 90 + "\nCV vs TEST consistency report\n" + "=" * 90]
    for _, row in df.iterrows():
        block = [f"\n[{row['classifier']} | {row['strategy']}]"]
        metrics = [
            ("accuracy", row["cv_accuracy_mean"], row["cv_accuracy_std"], row["test_accuracy"]),
            ("precision", row["cv_precision_mean"], row["cv_precision_std"], row["test_precision"]),
            ("recall", row["cv_recall_mean"], row["cv_recall_std"], row["test_recall"]),
            ("fbeta", row["cv_fbeta_mean"], row["cv_fbeta_std"], row["test_fbeta"]),
            ("auroc", row["cv_auroc_mean"], row["cv_auroc_std"], row["test_auroc"]),
        ]
        for name, cv_mean, cv_std, test_score in metrics:
            diff = test_score - cv_mean
            threshold = max(1.5 * cv_std, 0.02)
            flag = "optimistic" if diff > threshold else ("pessimistic" if diff < -threshold else "consistent")
            block.append(f"{name:<10} cv={cv_mean:.3f}±{cv_std:.3f} test={test_score:.3f} [{flag}]")
        lines.append("\n".join(block))

    report_path = os.path.join(output_dir, "cv_consistency_report.txt")
    with open(report_path, "w") as f: f.write("\n".join(lines))
    logging.info(f"Consistency report saved → {report_path}")

def _save_confusion_matrix(y_true, y_pred, output_dir, model_name, strategy):
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["no_FHB", "FHB"], yticklabels=["no_FHB", "FHB"])
    plt.title(f"Test Confusion Matrix\n{model_name} ({strategy})")
    plt.ylabel("True"); plt.xlabel("Predicted")
    path = os.path.join(output_dir, "best_model_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    logging.info(f"Confusion matrix saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main experiments
# ──────────────────────────────────────────────────────────────────────────────
def run_experiments(X_train_tensors, mask_train, y_train, X_validation_tensors, mask_validation, y_validation, X_test_tensors, mask_test, y_test, strategies=("mean", "mean_std", "4stat"), classifiers=None, param_grid=None, use_scaling=True, use_pca=False, pca_components=64, cv_folds=5, seed=SEED, iteration=0, output_dir="results/assessment/vIAbility/femi/ML"):
    if classifiers is None: classifiers = list(CLASSIFIERS.keys())
    if param_grid is None: param_grid = PARAM_GRID
    if use_pca: output_dir = f"{output_dir}_pca{pca_components}"
    os.makedirs(output_dir, exist_ok=True)

    results, best_overall_test_f1, best_test_predictions, best_model_info = [], -1, None, {}

    for strategy in strategies:
        logging.info(f"\n{'='*70}\nAggregation strategy: {strategy}\n{'='*70}")
        X_tr, train_pca = apply_aggregator(X_train_tensors, mask_train, strategy, pca_n_components=128, fit_pca=True)
        X_vl, _ = apply_aggregator(X_validation_tensors, mask_validation, strategy, pca_n_components=128, fit_pca=False, pca=train_pca)
        X_ts, _ = apply_aggregator(X_test_tensors, mask_test, strategy, pca_n_components=128, fit_pca=False, pca=train_pca)

        if use_scaling:
            scaler = StandardScaler()
            X_tr, X_vl, X_ts = scaler.fit_transform(X_tr), scaler.transform(X_vl), scaler.transform(X_ts)

        if use_pca:
            pca = PCA(n_components=min(pca_components, X_tr.shape[0] - 1, X_tr.shape[1]))
            X_tr, X_vl, X_ts = pca.fit_transform(X_tr), pca.transform(X_vl), pca.transform(X_ts)
            logging.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        for clf_name in classifiers:
            grid = param_grid.get(clf_name, {})
            param_combinations = [{}] if not grid else [dict(zip(grid.keys(), v)) for v in itertools.product(*grid.values())]
            best_result = None

            for params in param_combinations:
                try:
                    metrics = evaluate_validation(CLASSIFIERS[clf_name](**params), X_tr, y_train, X_vl, y_validation, cv_folds, seed=seed)
                    result = RunResult(
                        classifier=clf_name, strategy=strategy, params=params, iteration=iteration, seed=seed,
                        accuracy=metrics["accuracy"], precision=metrics["precision"], recall=metrics["recall"], fbeta=metrics["fbeta"], auroc=metrics["auroc"],
                        cv_accuracy_mean=metrics["cv_accuracy_mean"], cv_accuracy_std=metrics["cv_accuracy_std"],
                        cv_precision_mean=metrics["cv_precision_mean"], cv_precision_std=metrics["cv_precision_std"],
                        cv_recall_mean=metrics["cv_recall_mean"], cv_recall_std=metrics["cv_recall_std"],
                        cv_fbeta_mean=metrics["cv_fbeta_mean"], cv_fbeta_std=metrics["cv_fbeta_std"],
                        cv_auroc_mean=metrics["cv_auroc_mean"], cv_auroc_std=metrics["cv_auroc_std"],
                        cv_accuracy_folds=metrics["cv_accuracy_folds"], cv_precision_folds=metrics["cv_precision_folds"], cv_recall_folds=metrics["cv_recall_folds"], cv_fbeta_folds=metrics["cv_fbeta_folds"], cv_auroc_folds=metrics["cv_auroc_folds"]
                    )
                    if best_result is None or result.fbeta > best_result.fbeta: best_result = result
                except Exception as e:
                    logging.warning(f"[{clf_name}] params={params} failed → {e}")

            if best_result is not None:
                final_clf = CLASSIFIERS[clf_name](**best_result.params)
                final_clf.fit(np.concatenate([X_tr, X_vl], axis=0), np.concatenate([y_train, y_validation], axis=0))
                test_metrics = evaluate_test_set(final_clf, X_ts, y_test)

                for k, v in test_metrics.items():
                    if k != "predictions": setattr(best_result, k, v)
                results.append(best_result)
                logging.info(f"{clf_name:20s} | val_fbeta={best_result.fbeta:.3f} | test_fbeta={best_result.test_fbeta:.3f} | cv_fbeta={best_result.cv_fbeta_mean:.3f}±{best_result.cv_fbeta_std:.3f}")

                if best_result.test_fbeta > best_overall_test_f1:
                    best_overall_test_f1 = best_result.test_fbeta
                    best_test_predictions = test_metrics["predictions"]
                    best_model_info = {"name": clf_name, "strategy": strategy}

    if best_test_predictions is not None:
        _save_confusion_matrix(y_test, best_test_predictions, output_dir, best_model_info["name"], best_model_info["strategy"])

    df = pd.DataFrame([asdict(r) for r in results])
    fold_cols = [c for c in df.columns if c.endswith("_folds")]
    df.drop(columns=fold_cols, errors="ignore").to_csv(os.path.join(output_dir, "results.csv"), index=False)

    if fold_cols:
        cv_detail_rows = []
        for _, row in df.iterrows():
            base = {"classifier": row["classifier"], "strategy": row["strategy"]}
            for col in fold_cols:
                metric = col.replace("cv_", "").replace("_folds", "")
                for fold_i, val in enumerate(row[col] if row[col] is not None else []):
                    cv_detail_rows.append({**base, "metric": metric, "fold": fold_i, "score": val})
        pd.DataFrame(cv_detail_rows).to_csv(os.path.join(output_dir, "cv_results_detailed.csv"), index=False)

    _plot_results(df, output_dir)
    save_cv_consistency_report(df, output_dir)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def _plot_results(df, output_dir):
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("fbeta", "Fbeta(0.5)"), ("auroc", "AUROC")]
    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.7), 6))
        labels = [f"{r.classifier}\n({r.strategy})" for _, r in df.iterrows()]
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, df[f"cv_{metric}_mean"] * 100, width, yerr=df[f"cv_{metric}_std"] * 100, capsize=4, label="CV mean ± std")
        ax.bar(x + width/2, df[f"test_{metric}"] * 100, width, label="Test")
        ax.set_ylabel(f"{title} (%)"); ax.set_title(f"CV vs Test — {title}"); ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8); ax.set_ylim(0, 110); ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cv_vs_test_{metric}.png"), dpi=150); plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# Main & Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    iteration, seed = getattr(args, "iteration", 0), getattr(args, "seed", SEED)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    (X_full, mask_full, y_full), (X_test, mask_test, y_test) = build_datasets(PATH_DATASET + f"Oslo_viability/{args.dataset}")
    train_idx, val_idx = train_test_split(np.arange(len(y_full)), test_size=0.2, stratify=y_full.numpy(), random_state=seed)

    print(f"Train samples: {len(train_idx)}\nVal samples:   {len(val_idx)}\nTest samples:  {len(y_test)}")

    df = run_experiments(
        list(X_full[train_idx]), list(mask_full[train_idx]), y_full[train_idx].numpy(),
        list(X_full[val_idx]), list(mask_full[val_idx]), y_full[val_idx].numpy(),
        list(X_test), list(mask_test), y_test.numpy(),
        strategies=args.strategies, use_scaling=True, use_pca=args.use_pca, pca_components=args.pca_components,
        cv_folds=5, seed=seed, iteration=iteration,
        output_dir=f"results/assessment/vIAbility/features/{args.dataset}/ML/iter{iteration}"
    )
    print(df.sort_values("test_fbeta", ascending=False).to_string())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on FEMI embeddings", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="femi", help="Dataset folder")
    parser.add_argument("--strategies", nargs="+", default=["mean", "mean_std", "4stat", "weighted_mean", "pos_encoding"])
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--iteration", type=int, default=0, help="Outer repeat number")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed for split")
    main(parser.parse_args())