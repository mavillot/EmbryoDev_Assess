import argparse
import os
import sys
import glob
import logging
import random
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, asdict
import torch
from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_validate)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, make_scorer, roc_auc_score)
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.config import (SEED, PARAM_GRID, PATH_DATASET)

from assessment.ploidIA.features.stacking_ML import (aggregate_mean, aggregate_mean_std, aggregate_4stat, aggregate_concat_pca, aggregate_weighted_mean, aggregate_pos_encoding)



sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_DATASET

classmap = {"Aneuploid": 0, "Euploid": 1}
PATH_DATA = Path(PATH_DATASET + "Oslo_viability/markers/")

def _subsample_time_dim(parts):
    """
    parts: list of (X, mask, y) tuples, where X is (N, T, D) and mask is (N, T).
    Downsamples every part to the minimum T across parts by selecting evenly
    spaced frame indices via np.linspace (instead of padding).
    """
    target_T = min(p[0].shape[1] for p in parts)

    subsampled = []

    for X, mask, y in parts:

        T = X.shape[1]

        if T > target_T:

            idx = np.linspace(0, T - 1, target_T).round().astype(int)
            idx = torch.from_numpy(idx).long()

            X = X[:, idx, :]
            mask = mask[:, idx]

        subsampled.append((X, mask, y))

    return subsampled

def tensors_to_lists(X, mask):

    return list(X), list(mask)

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
        x = data["embeddings"].float()
        mask = data["mask"].bool()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, mask, y

def build_datasets(feature_dir: str):

    print(f"Loading features from: {feature_dir}")

    train_files = sorted(glob.glob(os.path.join(feature_dir, "train", "*", "*.pt")))
    test_files = sorted(glob.glob(os.path.join(feature_dir, "test", "*", "*.pt")))

    if not train_files:
        raise FileNotFoundError(f"No .pt files found under {feature_dir}/train/*/")

    if not test_files:
        raise FileNotFoundError(f"No .pt files found under {feature_dir}/test/*/")

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
        return (torch.stack(X), torch.stack(masks), torch.stack(ys))
    return _collate(train_dataset), _collate(test_dataset)

def apply_aggregator(X_tensors, mask_tensors):
    def aggregate_mean(x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """Mean pool over valid frames. Shape: (embed_dim,)"""
        valid = x[mask]
        return valid.mean(dim=0).numpy()
    
    fn = aggregate_mean

    X_agg = np.stack([fn(x, m) for x, m in zip(X_tensors, mask_tensors)])

    return X_agg, None

def main():
    seed  =  SEED
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
    path_models = 'models/'
    
    train_parts, test_parts = [], []

    for dataset in ['EasyFiv', 'Quiron_ploidy']:

        train_part, test_part = build_datasets(PATH_DATASET + f"{dataset}/resnet34")

        train_parts.append(train_part)
        test_parts.append(test_part)

        train_parts = _subsample_time_dim(train_parts)
        test_parts = _subsample_time_dim(test_parts)

        X_full = torch.cat([p[0] for p in train_parts], dim=0)
        mask_full = torch.cat([p[1] for p in train_parts], dim=0)
        y_full = torch.cat([p[2] for p in train_parts], dim=0)

        X_test = torch.cat([p[0] for p in test_parts], dim=0)
        mask_test = torch.cat([p[1] for p in test_parts], dim=0)
        y_test = torch.cat([p[2] for p in test_parts], dim=0)

    

    X_tr_list, mask_tr_list = tensors_to_lists(X_full, mask_full)
    X_test_list, mask_test_list = tensors_to_lists(X_test, mask_test)

    X_tr, train_pca = apply_aggregator(X_tr_list, mask_tr_list)
    X_ts, _ = apply_aggregator(X_test_list, mask_test_list)

    GaussianNB_model = GaussianNB().fit(X_tr, y_full.numpy())
    preds = GaussianNB_model.predict(X_ts)

    print("Test Accuracy:", accuracy_score(y_test.numpy(), preds))
    print("Test Precision:", precision_score(y_test.numpy(), preds))
    print("Test Recall:", recall_score(y_test.numpy(), preds))
    print("Test F1 Score:", f1_score(y_test.numpy(), preds))
    print("Test AUC Score:", roc_auc_score(y_test.numpy(), preds))

    #save the model
    joblib.dump(GaussianNB_model, 'gaussian_nb_ploidy.pkl')

    
if __name__ == "__main__":
    main()