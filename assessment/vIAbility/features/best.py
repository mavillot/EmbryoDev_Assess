import sys, glob, random, argparse, joblib, itertools, json, csv
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from tslearn.clustering import TimeSeriesKMeans
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split
from tabpfn import TabPFNClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, fbeta_score, make_scorer, roc_auc_score


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_DATASET

classmap = {"no_FHB": 0, "FHB": 1}
PATH_DATA = Path(PATH_DATASET + "Oslo_viability/markers/")

def load_data(path: Path, split: str = "train"):
    files = sorted(glob.glob(str(path / split / "*/*.csv")))
    if not files: raise ValueError(f"No CSV files found in {path / split}")
    labels, X = [], None
    for i, file in enumerate(files):
        df = pd.read_csv(file).replace([np.inf, -np.inf], np.nan).fillna(0)
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
    X_full_train_cl = apply_clusters(cluster_models, X_full_train, len(y_full_train))
    X_test_cl = apply_clusters(cluster_models, X_test, len(y_test))

    clf = TabPFNClassifier().fit(X_full_train_cl, y_full_train)
    preds = clf.predict(X_test_cl)
    score = np.mean(preds == y_test)
    print(f"Test accuracy = {score:.4f}")
    #metrics

    
if __name__ == "__main__":
    main()