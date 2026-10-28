import sys, glob, random, argparse, joblib, itertools, json, csv
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import shap
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from tslearn.clustering import TimeSeriesKMeans
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split
from tabpfn import TabPFNClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, fbeta_score, make_scorer, roc_auc_score


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_DATASET

torch.cuda.set_device(0)

classmap = {"no_FHB": 0, "FHB": 1}
PATH_DATA = Path(PATH_DATASET + "Oslo_viability/markers/")

def load_data(path: Path, split: str = "train"):
    files = sorted(glob.glob(str(path / split / "*/*.csv")))
    if not files: raise ValueError(f"No CSV files found in {path / split}")
    labels, X = [], None
    for i, file in enumerate(files):
        df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan).fillna(0)
        df = df.drop(columns=["expansion", "te_grading"], errors="ignore")
        print(df.columns)
        for col in df.columns:
            if df[col].dtype == object: df[col] = pd.Categorical(df[col]).codes
        cols = df.columns[2:-1]
        if i == 0: X = {c: [] for c in cols}
        for c in cols: X[c].append(df[c].tolist())
        labels.append(classmap[Path(file).parent.stem])
        print(cols)
    return {c: np.array(X[c]) for c in cols}, np.array(labels, dtype=int)

def fit_cluster_models(X_train: dict, columns, n_clusters: int, seed: int):
    models = {}
    for v in columns:
        ts = X_train[v].reshape(X_train[v].shape[0], X_train[v].shape[1], 1)
        print(ts.shape)
        models[v] = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=100, random_state=seed).fit(ts)
        for model in models:
            joblib.dump(models[model], f"models/ts_kmeans_{model}.pkl")
    return models

def apply_clusters(cluster_models: dict, X: dict, n_samples: int) -> np.ndarray:
    return np.concatenate([km.predict(X[v].reshape(X[v].shape[0], X[v].shape[1], 1)).reshape(n_samples, 1) for v, km in cluster_models.items()], axis=1)

def main():
    seed  =  SEED
    random.seed(seed); np.random.seed(seed)
    path_models = 'models/'

    X_full_train, y_full_train = load_data(PATH_DATA, "train")
    X_test, y_test = load_data(PATH_DATA, "test")
    columns = list(X_full_train.keys())

    cluster_models = fit_cluster_models(X_full_train, columns, 5, seed)
    #save cluster models centroids
    # for v, km in cluster_models.items():
    #     centroids = km.cluster_centers_.reshape(km.n_clusters, km.cluster_centers_.shape[1])
    #     plt.figure(figsize=(10, 6))
    #     for i in range(km.n_clusters):
    #         plt.plot(centroids[i], label=f"Cluster {i}")
    #     plt.title(f"Cluster Centroids for {v}")
    #     plt.xlabel("Time")
    #     plt.ylabel("Value")
    #     plt.legend()
    #     plt.savefig(f"centroids/ts_kmeans_{v}_centroids.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    X_full_train_cl = apply_clusters(cluster_models, X_full_train, len(y_full_train))
    X_test_cl = apply_clusters(cluster_models, X_test, len(y_test))

    # Use DataFrames so the fitted feature names match what SHAP passes back in,
    # avoiding the "X has feature names, but ... was fitted without feature names" warning.
    X_full_train_cl_df = pd.DataFrame(X_full_train_cl, columns=columns)
    X_test_cl_df       = pd.DataFrame(X_test_cl, columns=columns)

    X_test_cl_df.to_csv("X_clustered_fhb.csv", index=False)

    clf = TabPFNClassifier(device="cuda:1").fit(X_full_train_cl_df, y_full_train)
    preds = clf.predict(X_test_cl_df)
    score = np.mean(preds == y_test)
    print(f"Test accuracy = {score:.4f}")
    #metrics

    # --- SHAP explanation ---
    def predict_proba_pipeline(x):
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=columns)
        return clf.predict_proba(x)

    background_data = shap.sample(X_full_train_cl_df, 200, random_state=42)
    explainer = shap.KernelExplainer(predict_proba_pipeline, background_data)
    shap_values = explainer.shap_values(X_test_cl_df)

    # Newer SHAP versions return a single 3D array instead of a list of
    # per-class arrays, so branch consistently for both the plot and the CSV.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values[:, :, 1]

    shap.summary_plot(sv, X_test_cl_df, feature_names=columns)
    plt.savefig("shap_summary_fhb.png", dpi=300, bbox_inches='tight')

    shap_values_df = pd.DataFrame(sv, columns=columns)
    shap_values_df.to_csv("shap_values_fhb.csv", index=False)


if __name__ == "__main__":
    main()