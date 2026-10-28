import sys
import glob
import shap
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tslearn.clustering import TimeSeriesKMeans
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, make_scorer, roc_auc_score)
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_DATASET

classmap = {"Aneuploid": 0, "Euploid": 1}
PATH_DATA = Path(PATH_DATASET + "Oslo_viability/markers/")

def _align_time_dim(X_dicts: list) -> list:
    """
    X_dicts: list of dicts, each column -> array of shape (N, T).
    Downsamples every dict's arrays to the minimum T across all dicts,
    using evenly spaced frame indices (np.linspace), so datasets with
    different time-series lengths can be concatenated.
    """
    target_T = min(arr.shape[1] for X in X_dicts for arr in X.values())

    aligned = []
    for X in X_dicts:
        new_X = {}
        for col, arr in X.items():
            T = arr.shape[1]
            if T > target_T:
                idx = np.linspace(0, T - 1, target_T).round().astype(int)
                arr = arr[:, idx]
            new_X[col] = arr
        aligned.append(new_X)

    return aligned

def tensors_to_lists(X, mask):
    return list(X), list(mask)

def load_data(path: Path, split: str = "train"):
    labels = []
    X = None
    files = sorted(glob.glob(str(path / split / "*/*.csv")))

    if not files:
        raise ValueError(f"No CSV files found in {path / split}")

    for i, file in enumerate(files):
        df = pd.read_csv(file)
        df = df.drop(columns=["expansion", "te_grading"], errors="ignore")
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

    return {c: np.array(X[c]) for c in columns}, np.array(labels, dtype=int)

def fit_cluster_models(X_train: dict, columns, n_clusters: int, seed: int):
    models = {}
    for v in columns:
        ts = X_train[v].reshape(X_train[v].shape[0], X_train[v].shape[1], 1)
        models[v] = TimeSeriesKMeans(
            n_clusters=n_clusters, metric="euclidean", max_iter=100, random_state=seed,
        ).fit(ts)
        print(f"  Clustered feature '{v}'")
    return models

def apply_clusters(cluster_models: dict, X: dict, n_samples: int) -> np.ndarray:
    parts = []
    for v, km in cluster_models.items():
        ts  = X[v].reshape(X[v].shape[0], X[v].shape[1], 1)
        ids = km.predict(ts).reshape(n_samples, 1)
        parts.append(ids)
    return np.concatenate(parts, axis=1)


def main():
    seed  =  SEED
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    PATH_DATA   = Path(PATH_DATASET + "EasyFiv/markers/")
    PATH_DATA1  = Path(PATH_DATASET + "Quiron_ploidy/markers/")
    X_full_train, y_full_train   = load_data(PATH_DATA, split="train")
    X_full_train1, y_full_train1 = load_data(PATH_DATA1, split="train")
    X_test, y_test               = load_data(PATH_DATA, split="test")
    X_test1, y_test1             = load_data(PATH_DATA1, split="test")

    # Datasets can have different numbers of timepoints per feature (e.g. 200 vs 50),
    # so align them to a common length before concatenating.
    X_full_train, X_full_train1, X_test, X_test1 = _align_time_dim(
        [X_full_train, X_full_train1, X_test, X_test1]
    )

    for key in X_full_train:
        X_full_train[key] = np.concatenate([X_full_train[key], X_full_train1[key]], axis=0)
        X_test[key]       = np.concatenate([X_test[key], X_test1[key]], axis=0)


    y_full_train = np.concatenate([y_full_train, y_full_train1], axis=0)
    y_test = np.concatenate([y_test, y_test1], axis=0)
    columns = list(X_full_train.keys())


    cluster_models  = fit_cluster_models(X_full_train, columns, 5, seed)
    #save clusters centroids
    for v, km in cluster_models.items():
        centroids = km.cluster_centers_.reshape(km.n_clusters, km.cluster_centers_.shape[1])
        plt.figure(figsize=(8, 6))
        for i in range(km.n_clusters):
            plt.plot(centroids[i], label=f'Cluster {i}')
        plt.title(f'Cluster Centroids for Feature: {v}')
        plt.xlabel('Time Points')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f"centroids_ploidy/cluster_centroids_{v}.png", dpi=300, bbox_inches='tight')
        plt.close()
    X_full_train_cl = apply_clusters(cluster_models, X_full_train, len(y_full_train))
    X_test_cl       = apply_clusters(cluster_models, X_test, len(y_test))

    print(f"\nClustered shape: {X_full_train_cl.shape}")

    X_full_train_cl_df = pd.DataFrame(X_full_train_cl, columns=columns)
    X_test_cl_df       = pd.DataFrame(X_test_cl, columns=columns)

    X_test_cl_df.to_csv("X_clustered_ploidy.csv", index=False)

    GaussianNB_model = GaussianNB().fit(X_full_train_cl_df, y_full_train)

    def predict_proba_pipeline(x):
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=columns)
        return GaussianNB_model.predict_proba(x)

    background_data = shap.sample(X_full_train_cl, 200, random_state=42)
    explainer = shap.KernelExplainer(predict_proba_pipeline, background_data)
    shap_values = explainer.shap_values(X_test_cl)

    # Newer SHAP versions return a single 3D array instead of a list of
    # per-class arrays, so branch consistently for both the plot and the CSV.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1]

    shap.summary_plot(sv, X_test_cl, feature_names=columns)
    plt.savefig("shap_summary_ploidy.png", dpi=300, bbox_inches='tight')

    shap_values_df = pd.DataFrame(sv, columns=columns)
    shap_values_df.to_csv("shap_values_ploidy.csv", index=False)


if __name__ == "__main__":
    main()