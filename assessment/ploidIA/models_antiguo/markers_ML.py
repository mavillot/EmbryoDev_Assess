import sys
import glob
import random
import argparse
import joblib
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans

from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier,
)

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from tabpfn import TabPFNClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.config import (
    SEED,
    PARAM_GRID,
    PATH_MODELS,
    PATH_DATASET,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

classmap = {
    "Aneuploides": 0,
    "Euploides": 1,
}

PATH_DATA = Path(PATH_DATASET) / "Quiron_ploidy/markers/"
PATH_MODELS = Path(PATH_MODELS) / "assessment/ploidIA/markers/ML/"
PATH_OUTPUT = Path("output/assessment/ploidIA/markers/ML/")

N_CLUSTERS = 5
CV_FOLDS = 5

random.seed(SEED)
np.random.seed(SEED)

CLASSIFIERS = {
    "RandomForest": RandomForestClassifier,
    "SVC": SVC,
    "GaussianNB": GaussianNB,
    "DecisionTree": DecisionTreeClassifier,
    "MLPClassifier": MLPClassifier,
    "LogisticRegression": LogisticRegression,
    "TabPFN": TabPFNClassifier,
}

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        pos_label=1,
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        pos_label=1,
        zero_division=0,
    ),
    "f1": make_scorer(
        f1_score,
        pos_label=1,
        zero_division=0,
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(path: Path, split: str = "train"):

    labels = []
    X = None

    files = sorted(glob.glob(str(path / split / "*/*.csv")))

    if not files:
        raise ValueError(f"No CSV files found in {path / split}")

    for i, file in enumerate(files):

        df = pd.read_csv(file)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.Categorical(df[col]).codes

        columns = df.columns[2:-1]

        if i == 0:
            X = {c: [] for c in columns}

        for c in columns:
            X[c].append(df[c].tolist())

        labels.append(classmap[Path(file).parent.stem])

    return (
        {c: np.array(X[c]) for c in columns},
        np.array(labels, dtype=int),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Time-series clustering
# ──────────────────────────────────────────────────────────────────────────────

def fit_cluster_models(
    X_train: dict,
    columns,
    n_clusters: int,
    seed: int,
):

    models = {}

    for v in columns:

        ts = X_train[v].reshape(
            X_train[v].shape[0],
            X_train[v].shape[1],
            1,
        )

        models[v] = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="euclidean",
            max_iter=100,
            random_state=seed,
        ).fit(ts)

        print(f"  Clustered feature '{v}'")

    return models


def apply_clusters(
    cluster_models: dict,
    X: dict,
    n_samples: int,
) -> np.ndarray:

    parts = []

    for v, km in cluster_models.items():

        ts = X[v].reshape(
            X[v].shape[0],
            X[v].shape[1],
            1,
        )

        ids = km.predict(ts).reshape(n_samples, 1)

        parts.append(ids)

    return np.concatenate(parts, axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter search
# ──────────────────────────────────────────────────────────────────────────────

def study_hyperparameters(
    model_cls,
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
):

    best_acc = -1
    best_params = None

    for params in ParameterGrid(param_grid):

        try:

            clf = model_cls(**params)

            clf.fit(X_train, y_train)

            preds = clf.predict(X_val)

            acc = accuracy_score(y_val, preds)

            print(f"  {params}  →  val_acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_params = params

        except Exception as e:
            print(f"  FAILED {params} → {e}")

    print(f"\n  Best params: {best_params}")
    print(f"  Best validation accuracy: {best_acc:.4f}\n")

    return best_params

# ──────────────────────────────────────────────────────────────────────────────
# Ensemble utilities
# ──────────────────────────────────────────────────────────────────────────────

def build_bagging_classifiers(
    best_params: dict,
    seed: int,
):

    bagging_models = {}

    for name, clf_cls in CLASSIFIERS.items():

        base = clf_cls(**best_params[name])

        bagging_models[f"Bagging_{name}"] = BaggingClassifier(
            estimator=base,
            n_estimators=10,
            random_state=seed,
            n_jobs=-1,
        )

    return bagging_models


def _make_estimator(
    name: str,
    best_params: dict,
):

    params = dict(best_params[name])

    if name == "SVC":
        params["probability"] = True

    return CLASSIFIERS[name](**params)


def search_voting_combinations(
    best_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    min_size: int = 2,
    max_size: int = None,
    voting: str = "both",
    top_k: int = 3,
):

    names = list(CLASSIFIERS.keys())

    max_size = max_size or len(names)

    strategies = (
        ["hard", "soft"]
        if voting == "both"
        else [voting]
    )

    results = {s: [] for s in strategies}

    total_combos = sum(
        len(list(itertools.combinations(names, r)))
        for r in range(min_size, max_size + 1)
    )

    print(
        f"  Evaluating {total_combos} combinations "
        f"(sizes {min_size}–{max_size})"
    )

    for size in range(min_size, max_size + 1):

        for combo in itertools.combinations(names, size):

            estimators = [
                (n, _make_estimator(n, best_params))
                for n in combo
            ]

            combo_tag = "+".join(combo)

            for strategy in strategies:

                try:

                    vc = VotingClassifier(
                        estimators=estimators,
                        voting=strategy,
                        n_jobs=-1,
                    )

                    vc.fit(X_train, y_train)

                    preds = vc.predict(X_val)

                    acc = accuracy_score(y_val, preds)

                    results[strategy].append(
                        (
                            acc,
                            combo_tag,
                            estimators,
                            strategy,
                        )
                    )

                    print(
                        f"    [{strategy}] "
                        f"{combo_tag:<60} "
                        f"val_acc={acc:.4f}"
                    )

                except Exception as e:

                    print(
                        f"    [{strategy}] "
                        f"{combo_tag} skipped ({e})"
                    )

    best_voters = {}

    for strategy, entries in results.items():

        entries.sort(key=lambda x: x[0], reverse=True)

        print(f"\nTop-{top_k} [{strategy}]:")

        for rank, (
            acc,
            tag,
            estimators,
            strategy_name,
        ) in enumerate(entries[:top_k], 1):

            key = f"Voting_{strategy}_{tag}"

            best_voters[key] = {
                "estimators": estimators,
                "strategy": strategy_name,
            }

            print(
                f"    #{rank} "
                f"{tag} "
                f"val_acc={acc:.4f}"
            )

    return best_voters

# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def cross_validate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = CV_FOLDS,
    seed: int = SEED,
):

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    cv_results = {}

    print("\n" + "=" * 70)
    print(f"Cross-validation ({n_folds}-fold)")
    print("=" * 70)

    for name, clf in models.items():

        print(f"\n[{name}]")

        try:

            scores = cross_validate(
                clf,
                X_train,
                y_train,
                cv=cv,
                scoring=CV_SCORING,
                n_jobs=-1,
                return_train_score=False,
            )

            summary = {}

            for metric in CV_SCORING:

                vals = scores[f"test_{metric}"]

                summary[metric] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "per_fold": vals.tolist(),
                }

                print(
                    f"  {metric:<10} "
                    f"mean={vals.mean()*100:.2f}% "
                    f"std={vals.std()*100:.2f}%"
                )

            cv_results[name] = summary

        except Exception as e:

            print(f"  CV failed → {e}")

            cv_results[name] = None

    return cv_results

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_and_save(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    path_model: str,
    out_dir: str,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["Aneuploid", "Euploid"]

    summary_lines = []
    test_metrics = {}

    for name, clf in models.items():

        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        prec = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        rec = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        test_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

        report = classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )

        print("\nClassification report:")
        print(report)

        result_str = (
            f"{name}\n\n"
            f"Accuracy : {acc:.4f}\n"
            f"Precision: {prec:.4f}\n"
            f"Recall   : {rec:.4f}\n"
            f"F1-score : {f1:.4f}\n\n"
            f"{report}"
        )

        (out_dir / f"test_results_{name}.txt").write_text(result_str)

        summary_lines.append(
            f"{name:<50} "
            f"acc={acc:.4f} "
            f"prec={prec:.4f} "
            f"rec={rec:.4f} "
            f"f1={f1:.4f}"
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)

        fig.tight_layout()

        fig.savefig(
            out_dir / f"confusion_matrix_{name}.png",
            dpi=150,
        )

        plt.close(fig)

        # Save model
        model_path = Path(path_model) / f"{name}.joblib"

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(clf, model_path)

        print(f"Saved model → {model_path}")

    summary = "\n".join(summary_lines)

    (out_dir / "all_results.txt").write_text(summary)

    print("\nFinal summary")
    print(summary)

    return test_metrics

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):

    print("Loading data...")

    X_full_train, y_full_train = load_data(
        PATH_DATA,
        split="train",
    )

    X_test, y_test = load_data(
        PATH_DATA,
        split="test",
    )

    columns = list(X_full_train.keys())

    print(f"Features: {columns}")

    # ──────────────────────────────────────────────────────────────────────
    # Clustering
    # ──────────────────────────────────────────────────────────────────────

    print("\nFitting clustering models...")

    cluster_models = fit_cluster_models(
        X_full_train,
        columns,
        args.n_clusters,
        SEED,
    )

    X_full_train_cl = apply_clusters(
        cluster_models,
        X_full_train,
        len(y_full_train),
    )

    X_test_cl = apply_clusters(
        cluster_models,
        X_test,
        len(y_test),
    )

    print(f"\nClustered shape: {X_full_train_cl.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # Validation split
    # ──────────────────────────────────────────────────────────────────────

    X_train_cl, X_val_cl, y_train, y_val = train_test_split(
        X_full_train_cl,
        y_full_train,
        test_size=0.2,
        stratify=y_full_train,
        random_state=SEED,
    )

    print("\nDataset split:")
    print(f"Train: {len(y_train)}")
    print(f"Val  : {len(y_val)}")
    print(f"Test : {len(y_test)}")

    # ──────────────────────────────────────────────────────────────────────
    # Hyperparameter search
    # ──────────────────────────────────────────────────────────────────────

    print("\nSearching hyperparameters...")

    best_params = {}

    for name, clf_cls in CLASSIFIERS.items():

        print(f"\n--- {name} ---")

        best_params[name] = study_hyperparameters(
            clf_cls,
            PARAM_GRID[name],
            X_train_cl,
            y_train,
            X_val_cl,
            y_val,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Refit on train + validation
    # ──────────────────────────────────────────────────────────────────────

    X_final_train = np.concatenate(
        [X_train_cl, X_val_cl],
        axis=0,
    )

    y_final_train = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Base models
    # ──────────────────────────────────────────────────────────────────────

    print("\nTraining final base models...")

    best_models = {}

    for name in CLASSIFIERS:

        clf = CLASSIFIERS[name](**best_params[name])

        clf.fit(X_final_train, y_final_train)

        best_models[name] = clf

    # ──────────────────────────────────────────────────────────────────────
    # Bagging
    # ──────────────────────────────────────────────────────────────────────

    print("\nTraining bagging models...")

    bagging_models = build_bagging_classifiers(
        best_params,
        SEED,
    )

    for name, clf in bagging_models.items():

        print(f"  Fitting {name}")

        clf.fit(X_final_train, y_final_train)

    # ──────────────────────────────────────────────────────────────────────
    # Voting search
    # ──────────────────────────────────────────────────────────────────────

    print("\nSearching voting combinations...")

    voting_configs = search_voting_combinations(
        best_params,
        X_train_cl,
        y_train,
        X_val_cl,
        y_val,
        min_size=args.vote_min,
        max_size=args.vote_max,
        voting=args.voting,
        top_k=args.vote_top_k,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Final voting models
    # ──────────────────────────────────────────────────────────────────────

    final_voting_models = {}

    print("\nRetraining selected voting models...")

    for name, cfg in voting_configs.items():

        clf = VotingClassifier(
            estimators=cfg["estimators"],
            voting=cfg["strategy"],
            n_jobs=-1,
        )

        clf.fit(X_final_train, y_final_train)

        final_voting_models[name] = clf

    # ──────────────────────────────────────────────────────────────────────
    # Collect all models
    # ──────────────────────────────────────────────────────────────────────

    all_models = {
        **best_models,
        **bagging_models,
        **final_voting_models,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Cross-validation
    # ──────────────────────────────────────────────────────────────────────

    cv_results = cross_validate_models(
        all_models,
        X_train_cl,
        y_train,
        n_folds=args.cv_folds,
        seed=SEED,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Final test evaluation
    # ──────────────────────────────────────────────────────────────────────

    test_metrics = evaluate_and_save(
        all_models,
        X_test_cl,
        y_test,
        path_model=args.path_models,
        out_dir=args.out_dir,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Ploidy classification via "
            "time-series clustering + ML"
        )
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=N_CLUSTERS,
    )

    parser.add_argument(
        "--path_models",
        default=PATH_MODELS,
    )

    parser.add_argument(
        "--out_dir",
        default=PATH_OUTPUT,
    )

    parser.add_argument(
        "--vote_min",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--vote_max",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--voting",
        default="both",
        choices=["hard", "soft", "both"],
    )

    parser.add_argument(
        "--vote_top_k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=CV_FOLDS,
    )

    main(parser.parse_args())