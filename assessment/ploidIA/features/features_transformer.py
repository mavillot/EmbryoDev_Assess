"""
train.py
─────────────────────────────────────────────────────────────────────────────
Unified training script for viability classification.

Datasets  : markers   — per-frame .pt files under train/<class>/ test/<class>/
            femi      — pre-extracted FEMI embeddings under train/<class>/ test/<class>/
            resnet34  — ResNet-34 features under train/<class>/ test/<class>/
            (any folder following the same layout is accepted via --data_dir)

Models    : rnn         — stacked vanilla RNN with masked mean-pool
            gru         — stacked GRU with masked mean-pool
            lstm        — stacked LSTM with masked mean-pool
            transformer — lightweight sinusoidal / CLS Transformer encoder

Loss      : ce          — standard CrossEntropyLoss (default)
            weighted_ce — CE with positive-class frequency weighting (--weight_loss)
            fbeta       — differentiable Fbeta loss, beta=0.5 (--loss fbeta)

Labels are inferred from the class subfolder name — no Excel metadata needed.
History, metrics, confusion matrices and training curves are always saved.
─────────────────────────────────────────────────────────────────────────────

Usage examples
──────────────
# Markers dataset, GRU
python train.py --dataset markers --model gru

# Markers dataset, Transformer
python train.py --dataset markers --model transformer --d_model 64 --num_heads 4

# FEMI dataset, LSTM with FbetaLoss
python train.py --dataset femi --model lstm --loss fbeta

# FEMI dataset, Transformer (default)
python train.py --dataset femi --model transformer

# ResNet-34 dataset, GRU
python train.py --dataset resnet34 --model gru

# Any custom folder
python train.py --dataset mydata --data_dir /path/to/mydata --model transformer
"""

import os
import sys
import glob
import json
import random
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_MODELS, PATH_DATASET
from assessment.ploidIA.models.models import build_model

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_MODELS  = os.path.join(PATH_MODELS,  "assessment/ploidIA")
BASE_OUTPUT  = "results/assessment/ploidIA"
METRICS_CSV = os.path.join(BASE_OUTPUT, "experiment_results.csv")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)



# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CLASSMAP     = {"Aneuploid": 0, "Euploid": 1}
NUM_CLASSES  = len(CLASSMAP)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1.  Datasets
# =============================================================================

def _linspace_subsample(x: torch.Tensor, mask: torch.Tensor, target_T: int):
    """Evenly subsample the time dim of (T, D) embeddings / (T,) mask to target_T. No-op if T <= target_T."""
    T = x.shape[0]
    if T <= target_T:
        return x, mask
    idx = np.linspace(0, T - 1, target_T).round().astype(int)
    idx = torch.from_numpy(idx).long()
    return x[idx], mask[idx]


def _linspace_subsample_np(x: np.ndarray, mask: np.ndarray, target_T: int):
    T = x.shape[0]
    if T <= target_T:
        return x, mask
    idx = np.linspace(0, T - 1, target_T).round().astype(int)
    return x[idx], mask[idx]


def _infer_native_seq_len(root: str, split: str) -> int:
    """Peek at one .pt file to find a root/split's native T before any subsampling."""

    files = sorted(glob.glob(os.path.join(root, split, "*", "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found at {root}/{split}/*/*")
    data = torch.load(files[0], map_location="cpu", weights_only=True)
    return data["embeddings"].shape[0]

# ── 1a.  Shared .pt embedding dataset (femi / resnet34 / any pre-extracted) ──

class EmbeddingDataset(Dataset):
    """
    Loads .pt files whose label is encoded in the parent folder name.

    Expected layout:
        <root>/<split>/<class_name>/<file>.pt

    Each .pt file must contain:
        {"embeddings": Tensor(T, D),  "mask": Tensor(T,)}
    """
    def __init__(self, root: str, split: str, target_seq_len: int | None = None):
        pattern = os.path.join(root, split, "*", "*.pt")
        files   = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No .pt files found at {pattern!r}.")

        self.files = files
        self.target_seq_len = target_seq_len   # <-- new

        self.labels = []
        for f in files:
            class_name = Path(f).parent.name
            if class_name not in CLASSMAP:
                raise ValueError(f"Unknown class folder '{class_name}' in: {f}")
            self.labels.append(CLASSMAP[class_name])

        counts = np.bincount(self.labels)
        log.info("  %s split — %d files | class counts: %s", split, len(files), counts.tolist())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        x    = data["embeddings"].float()
        mask = data["mask"].bool()
        if self.target_seq_len is not None:
            x, mask = _linspace_subsample(x, mask, self.target_seq_len)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, mask, y


def build_embedding_loaders(root: "str | list[str]", args):
    roots = [root] if isinstance(root, str) else list(root)

    native_lens = [
        _infer_native_seq_len(r, split) for r in roots for split in ("train", "test")
    ]
    target_seq_len = min(native_lens)

    if len(set(native_lens)) > 1:
        log.info("Mismatched native seq lengths %s — subsampling all to seq_len=%d via linspace.",
                  native_lens, target_seq_len)

    train_ds = ConcatDataset([EmbeddingDataset(r, "train", target_seq_len) for r in roots])
    val_ds   = ConcatDataset([EmbeddingDataset(r, "test", target_seq_len) for r in roots])

    x0, _, _ = train_ds[0]
    seq_len, embed_dim = x0.shape[0], x0.shape[1]
    log.info("Combined seq_len=%d embed_dim=%d (train=%d, val=%d)",
              seq_len, embed_dim, len(train_ds), len(val_ds))

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    train_labels = torch.tensor(
        [lbl for ds in train_ds.datasets for lbl in ds.labels], dtype=torch.float32
    )
    n_neg, n_pos = (train_labels == 0).sum(), (train_labels == 1).sum()
    pos_weight = (n_neg / n_pos).unsqueeze(0) if n_pos > 0 else None

    return train_loader, val_loader, seq_len, embed_dim, pos_weight


# ── 1b.  Markers CSV dataset (ported from ploidy pipeline) ──────────────────

def _load_single_csv(path: str, target_T: int | None = None):
    df = pd.read_csv(path)
    df.drop(columns=['grading', 'frame', 'time_hours', 'expansion', 'te_grading', 'icm_grading'], errors='ignore', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes
    x = df.to_numpy(dtype=np.float32)
    mask = np.ones(len(df), dtype=bool)
    if target_T is not None:
        x, mask = _linspace_subsample_np(x, mask, target_T)
    lbl = CLASSMAP[Path(path).parent.stem]
    return x, mask, lbl


def _load_markers_split(markers_dirs: "str | list[str]", split: str, target_T: int | None = None):
    dirs = [markers_dirs] if isinstance(markers_dirs, str) else list(markers_dirs)

    files = []
    for d in dirs:
        files.extend(glob.glob(str(Path(d) / split / "*" / "*")))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {dirs}/{split}/*/")

    files = random.sample(files, len(files))

    X, M, labels = [], [], []
    for f in files:
        x, mask, lbl = _load_single_csv(f, target_T=target_T)
        X.append(x); M.append(mask); labels.append(lbl)

    shapes = {arr.shape for arr in X}
    if len(shapes) > 1:
        raise ValueError(f"Shapes still inconsistent after subsampling: {shapes}. "
                          "Pass a target_T matching the shortest sequence across all sources.")

    counts = np.bincount(labels)
    log.info("  %s split — %d files | class counts: %s | shape: %s", split, len(files), counts.tolist(), X[0].shape)

    return np.stack(X), np.stack(M), np.array(labels, dtype=np.int64)


class MarkersDataset(Dataset):
    """In-memory dataset for pre-stacked marker arrays."""

    def __init__(self, X: np.ndarray, M: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()   # (N, T, F)
        self.M = torch.from_numpy(M).bool()    # (N, T)
        self.y = torch.from_numpy(y)           # (N,) int64 for CE

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx]


def build_markers_loaders(markers_dir: "str | list[str]", args):
    dirs = [markers_dir] if isinstance(markers_dir, str) else list(markers_dir)
    if args.n_frames:
        log.info("Subsampling all marker sequences to T=%d frames", args.n_frames)
        target_T = args.n_frames
    else:
        target_T = None
        if len(dirs) > 1:
            sample_files = [
                f for d in dirs for split in ("train", "test")
                for f in glob.glob(str(Path(d) / split / "*" / "*"))[:1]
            ]
            native_Ts = [len(pd.read_csv(f, usecols=[0])) for f in sample_files]
            target_T = min(native_Ts)
            if len(set(native_Ts)) > 1:
                log.info("Mismatched row counts %s — subsampling all to T=%d via linspace.", native_Ts, target_T)

    X_tr, M_tr, y_tr = _load_markers_split(dirs, "train", target_T=target_T)
    X_va, M_va, y_va = _load_markers_split(dirs, "test", target_T=target_T)

    log.info("Markers — train: %s | val: %s", X_tr.shape, X_va.shape)

    seq_len   = int(X_tr.shape[1])   # T
    n_features = int(X_tr.shape[2])  # F

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        MarkersDataset(X_tr, M_tr, y_tr),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        MarkersDataset(X_va, M_va, y_va),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    train_labels = torch.tensor(y_tr, dtype=torch.float32)
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    pos_weight = (n_neg / n_pos).unsqueeze(0) if n_pos > 0 else None

    return train_loader, val_loader, seq_len, n_features, pos_weight


# ── 1c.  Unified entry point ─────────────────────────────────────────────────

def build_loaders(args):
    source = args.dataset.lower()

    if source == "combined":
        roots = [os.path.join(PATH_DATASET, 'EasyFiv', args.feat), os.path.join(PATH_DATASET, "Quiron_ploidy", args.feat),]
    else:
        roots = args.data_dir or os.path.join(PATH_DATASET, args.dataset, args.feat)
    log.info("Loading dataset from: %s", roots)
    if args.feat == "markers":
        return build_markers_loaders(roots, args)
    else:
        return build_embedding_loaders(roots, args)


# =============================================================================
# 2.  FbetaLoss
# =============================================================================

class FbetaLoss(nn.Module):
    """
    Differentiable Fbeta loss for binary classification.

    Source - https://stackoverflow.com/a/57740822
    Posted by Jinglesting, modified by community. See post 'Timeline' for change history.
    Retrieved 2026-05-23, License - CC BY-SA 4.0
    """

    def __init__(self, beta: float = 0.35, epsilon: float = 1e-7):
        super().__init__()
        self.beta    = beta
        self.epsilon = epsilon
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_logits, y_true):
        # y_logits: (B, 2) class logits  OR  (B,) positive-class logit
        # y_true:   (B,) long labels {0, 1}
        if y_logits.dim() == 2:
            # Take logit of positive class; convert to float
            y_logits = y_logits[:, 1]
        y_true = y_true.float()
        y_pred = self.sigmoid(y_logits)
        # Operate over the batch dimension (dim=0)
        TP = (y_pred * y_true).sum()
        FP = ((1 - y_pred) * y_true).sum()
        FN = (y_pred * (1 - y_true)).sum()
        fbeta = (
            (1 + self.beta ** 2) * TP
            / ((1 + self.beta ** 2) * TP + (self.beta ** 2) * FN + FP + self.epsilon)
        )
        fbeta = fbeta.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - fbeta



# =============================================================================
# 3.  Metrics
# =============================================================================

@dataclass
class EpochMetrics:
    loss:   float      = 0.0
    acc:    float      = 0.0
    prec:   float      = 0.0
    rec:    float      = 0.0
    f1:     float      = 0.0
    fbeta:  float      = 0.0
    auroc:  float      = 0.0
    preds:  np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in ("preds", "labels")}

    def log_str(self, prefix: str = "") -> str:
        return (
            f"{prefix}loss={self.loss:.4f} acc={self.acc:.4f} "
            f"prec={self.prec:.4f} rec={self.rec:.4f} f1={self.f1:.4f} "
            f"fbeta={self.fbeta:.4f} auroc={self.auroc:.4f}"
        )


# =============================================================================
# 4.  Train / eval loop
# =============================================================================

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: "torch.optim.Optimizer | None" = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss, all_preds, all_labels, all_scores = 0.0, [], [], []

    with torch.set_grad_enabled(training):
        for X, M, y in loader:
            X, M, y = X.to(DEVICE), M.to(DEVICE), y.to(DEVICE)
            logits = model(X, M)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
            # Soft scores for AUROC: probability of positive class
            all_scores.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())

    y_pred  = np.concatenate(all_preds)
    y_true  = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")  # single class in batch

    return EpochMetrics(
        loss   = total_loss / len(y_true),
        acc    = accuracy_score(y_true, y_pred),
        prec   = precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        rec    = recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1     = f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        fbeta  = fbeta_score(y_true, y_pred, pos_label=1, beta=0.35, zero_division=0),
        auroc  = auroc,
        preds  = y_pred,
        labels = y_true,
    )


def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    args,
    pos_weight:   "torch.Tensor | None" = None,
) -> tuple[nn.Module, dict]:

    # Loss — optionally weight the positive class for imbalanced datasets
    if args.loss == "fbeta":
        criterion = FbetaLoss(beta=args.beta)
        log.info("Using FbetaLoss (beta=%f)", args.beta)
    elif args.weight_loss and pos_weight is not None:
        w    = torch.ones(NUM_CLASSES, device=DEVICE)
        w[1] = pos_weight.item()
        criterion = nn.CrossEntropyLoss(weight=w)
        log.info("Using weighted CE loss: pos_weight=%.3f", w[1].item())
    else:
        criterion = nn.CrossEntropyLoss()
        log.info("Using CrossEntropyLoss")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
        )

    iteration = getattr(args, "iteration", 0)
    ckpt_path = Path(args.checkpoint) / args.dataset / args.loss/args.model/f"iter{iteration}"  / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_val_prec  = 0.0
    best_val_f1    = 0.0
    best_val_fbeta = 0.0
    best_val_auroc = 0.0
    patience_cnt   = 0

    history: dict[str, list] = {
        "train_loss": [], "val_loss":  [],
        "train_acc":  [], "val_acc":   [],
        "val_f1":     [], "val_prec":  [], "val_fbeta": [], "val_auroc": [],
    }

    log.info("Starting training — up to %d epochs (patience=%d)", args.epochs, args.patience)

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, criterion, optimizer)
        val = run_epoch(model, val_loader,   criterion)

        if np.isnan(val.loss) or np.isnan(tr.loss):
            raise RuntimeError(
                f"NaN loss at epoch {epoch}. "
                "Check data for inf/nan and verify masks are not all-False."
            )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val.loss)
        else:
            scheduler.step()

        history["train_loss"].append(tr.loss)
        history["val_loss"].append(val.loss)
        history["train_acc"].append(tr.acc)
        history["val_acc"].append(val.acc)
        history["val_prec"].append(val.prec)
        history["val_f1"].append(val.f1)
        history["val_fbeta"].append(val.fbeta)
        history["val_auroc"].append(val.auroc)

        lr_now   = optimizer.param_groups[0]["lr"]
        improved = (val.loss  < best_val_loss)  if args.save_by == "loss"  \
               else (val.prec  > best_val_prec)  if args.save_by == "prec"  \
               else (val.f1    > best_val_f1)    if args.save_by == "f1"    \
               else (val.fbeta > best_val_fbeta) if args.save_by == "fbeta" \
               else (val.auroc > best_val_auroc) if args.save_by == "auroc" \
               else (val.acc   > best_val_acc)

        log.info(
            "Epoch %3d/%d | %s | %s | lr=%.2e%s",
            epoch, args.epochs,
            tr.log_str("train "), val.log_str("val "),
            lr_now, "  ✓" if improved else "",
        )

        if improved:
            best_val_loss  = val.loss
            best_val_acc   = val.acc
            best_val_f1    = val.f1
            best_val_fbeta = val.fbeta
            best_val_prec  = val.prec
            best_val_auroc = val.auroc
            patience_cnt   = 0
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss":  val.loss,
                    "val_acc":   val.acc,
                    "val_f1":    val.f1,
                    "val_fbeta": val.fbeta,
                    "val_auroc": val.auroc,
                    "val_prec":  val.prec,
                    "args":      vars(args),
                },
                ckpt_path,
            )
            log.info("  Checkpoint saved → %s", ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        log.info("Loaded best checkpoint (epoch %d)", ckpt["epoch"])
    else:
        log.warning("No checkpoint found — returning last-epoch weights.")

    return model, history


# =============================================================================
# 5.  Plotting helpers
# =============================================================================

def plot_confusion_matrix(
    y_true, y_pred, title: str, path: str, normalize: "str | None" = None
) -> None:
    class_names = list(CLASSMAP.keys())
    cm  = confusion_matrix(y_true, y_pred, normalize=normalize)
    fmt = ".2f" if normalize is not None else "d"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", path)


def plot_training_curves(history: dict, out_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend();          axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"],   label="val acc")
    axes[1].plot(epochs, history["val_f1"],    label="val f1")
    axes[1].plot(epochs, history["val_fbeta"], label="val fbeta")
    axes[1].set_title("Val metrics"); axes[1].set_xlabel("Epoch")
    axes[1].legend();                 axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_auroc"], label="val auroc", color="darkorange")
    axes[2].set_title("Val AUROC"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1);        axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Training curves saved → %s", path)


# =============================================================================
# 6.  Main
# =============================================================================

def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Output directory — suffix with _pos_enc when positional encoding is on
    out_dir = Path(args.out_dir) / 'features' /args.dataset /args.feat / args.model / args.loss / f"iter{args.iteration}"
    if args.use_pos_encoding and args.model == "transformer":
        out_dir = Path(str(out_dir) + "_pos_enc")
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Run config saved → %s", config_path)

    log.info(
        "Dataset: %s | Model: %s | Pos-enc: %s | Device: %s",
        args.dataset, args.model, args.use_pos_encoding, DEVICE,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, seq_len, n_feat, pos_weight = build_loaders(args)
    log.info("seq_len=%d  num_features=%d", seq_len, n_feat)

    # ── Model ─────────────────────────────────────────────────────────────
    model    = build_model(args, seq_len, n_feat, num_classes=NUM_CLASSES, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s  |  Trainable params: %s", type(model).__name__, f"{n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────
    model, history = train(model, train_loader, val_loader, args, pos_weight)

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved → %s", history_path)
    plot_training_curves(history, str(out_dir))

    # ── Final evaluation ──────────────────────────────────────────────────
    # Use unweighted loss for final eval so metrics are not penalised by imbalance weight
    val_metrics = run_epoch(model, val_loader, nn.CrossEntropyLoss())

    results_str = (
        f"Test accuracy:  {val_metrics.acc  * 100:.2f}%\n"
        f"Test precision: {val_metrics.prec * 100:.2f}%\n"
        f"Test recall:    {val_metrics.rec  * 100:.2f}%\n"
        f"Test F1:        {val_metrics.f1   * 100:.2f}%\n"
        f"Test F0.5:      {val_metrics.fbeta * 100:.2f}%\n"
        f"Test AUROC:     {val_metrics.auroc * 100:.2f}%\n"
    )
    log.info("\n===== FINAL RESULTS =====\n%s", results_str)
    (out_dir / "test_results.txt").write_text(results_str)

    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(val_metrics.to_dict(), f, indent=2)

    plot_confusion_matrix(
        val_metrics.labels, val_metrics.preds,
        title=f"{type(model).__name__} ({args.dataset})",
        path=str(out_dir / "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        val_metrics.labels, val_metrics.preds,
        title=f"{type(model).__name__} ({args.dataset}) — normalized",
        path=str(out_dir / "confusion_matrix_normalized.png"),
        normalize="all",
    )

    log.info("All outputs saved to %s", out_dir)


# =============================================================================
# 7.  CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GRU or Transformer for FHB viability classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", default="EasyFiv", choices=["EasyFiv", "Quiron_ploidy", "Combined"],
        help=(
            "Dataset name. Built-in: EasyFiv, Quiron, Combined. "
            "Any name is accepted — just point --data_dir at the right folder."
        ),
    )

    parser.add_argument(
        "--feat", default="femi", choices=["femi", "markers", "resnet34"],
        help=(
            "Feature set to use. Built-in: femi, markers, resnet34. "
            "Any name is accepted — just point --data_dir at the right folder."
        ),
    )

    # ── Data path ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_dir", default=None,
        help=(
            "Root directory containing train/<class>/*.pt  (femi/resnet34)  "
            "or  train/<class>/*.csv  (markers). "
            "Defaults to PATH_DATASET/<dataset> so you rarely need to set this."
        ),
    )

    # ── Output paths ──────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=str(BASE_MODELS),
                        help="Root directory for model checkpoints.")
    parser.add_argument("--out_dir",    default=str(BASE_OUTPUT),
                        help="Root directory for results, plots, and logs.")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Subsample all sequences to this number of frames (markers only).")
    parser.add_argument("--model", choices=["rnn", "gru", "lstm", "transformer"], default="transformer")
    parser.add_argument("--weight_loss", action="store_true",
                        help="Apply class-frequency weighting to the CE loss.")
    parser.add_argument("--loss", choices=["ce", "fbeta"], default="ce",
                        help="Loss function: 'ce' = CrossEntropy, 'fbeta' = differentiable Fbeta(beta=0.5).")
    parser.add_argument("--dropout",  type=float, default=0.2)
    parser.add_argument("--use_pos_encoding", action="store_true",
                        help="Use sinusoidal positional encoding (Transformer only).")

    # ── Transformer hyper-params ──────────────────────────────────────────
    parser.add_argument("--d_model",   type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim",    type=int, default=64)

    # ── GRU hyper-params ─────────────────────────────────────────────────
    parser.add_argument("--hidden1", type=int, default=32,
                        help="[GRU] First GRU hidden size.")
    parser.add_argument("--hidden2", type=int, default=16,
                        help="[GRU] Second GRU hidden size.")
    parser.add_argument("--fc_dim",  type=int, default=16,
                        help="[GRU] FC layer size before classifier head.")

    # ── Training hyper-params ─────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--scheduler",    choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--save_by", choices=["loss", "acc", "prec", "f1", "fbeta", "auroc"], default="loss")
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument('--beta',         type=float, default=0.5, help="Beta value for FbetaLoss and Fbeta metric.")
    parser.add_argument("--iteration",    type=int,   default=0,
                        help="Repeat number (0 for a standalone run). Only affects output paths and the iteration column in the results CSV.")

    main(parser.parse_args())