"""
train_ploidy.py  —  Unified ploidy classifier training script
=============================================================
Supports two dataset pipelines (femi / markers) and two architectures
(gru / transformer) via CLI flags. Both architectures are identical
regardless of dataset; only the data loading differs.

Valid combinations
------------------
  --dataset femi    --model transformer  [--use_pos_encoding]
  --dataset femi    --model gru
  --dataset markers --model transformer  [--use_pos_encoding]
  --dataset markers --model gru

Usage examples
--------------
python train_ploidy.py --dataset femi    --model transformer --use_pos_encoding
python train_ploidy.py --dataset femi    --model gru --hidden1 64 --hidden2 32
python train_ploidy.py --dataset markers --model transformer --d_model 64 --nhead 4
python train_ploidy.py --dataset markers --model gru
"""

import os
import sys
import glob
import json
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_MODELS, PATH_DATASET

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSMAP    = {"Aneuploides": 0, "Euploides": 1}
NUM_CLASSES = len(CLASSMAP)

PATH_MODELS_FEMI    = PATH_MODELS + "assessment/ploidIA/femi/"
PATH_MODELS_MARKERS = PATH_MODELS + "assessment/ploidIA/markers/"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified ploidy classifier — FEMI or Markers x GRU or Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Pipeline & architecture -------------------------------------------
    p.add_argument(
        "--dataset", choices=["femi", "markers"], required=True,
        help="Dataset pipeline: 'femi' (.pt embedding files) or 'markers' (CSV files).",
    )
    p.add_argument(
        "--model", choices=["gru", "transformer"], default="transformer",
        help="Model architecture. Both work with either dataset.",
    )
    p.add_argument(
        "--use_pos_encoding", action="store_true",
        help="[Transformer only] Add sinusoidal positional encoding.",
    )

    # -- Paths -------------------------------------------------------------
    p.add_argument(
        "--feature_dir",
        default=os.path.join(PATH_DATASET, "Quiron_ploidy/femi"),
        help="[FEMI] Root dir with train/*/*.pt and test/*/*.pt feature files.",
    )
    p.add_argument(
        "--markers_dir",
        default=str(Path("../Datasets/Processed/Quiron_ploidy/markers/")),
        help="[Markers] Root dir with train/*/*.csv and test/*/*.csv files.",
    )
    p.add_argument("--model_dir",   default=None,
                   help="Checkpoint directory. Auto-set if omitted.")
    p.add_argument("--metrics_dir", default=None,
                   help="Metrics/plots directory. Auto-set if omitted.")

    # -- FEMI data ---------------------------------------------------------
    p.add_argument("--embed_dim", type=int, default=None,
                   help="[FEMI] Embedding width — inferred from files if omitted.")
    p.add_argument("--n_frames",  type=int, default=1400,
                   help="[FEMI] Max frames per video (used for PE buffer size).")

    # -- Shared model hyper-params -----------------------------------------
    p.add_argument("--d_model",    type=int,   default=32,
                   help="[Transformer] Internal hidden dimension.")
    p.add_argument("--nhead",      type=int,   default=4,
                   help="[Transformer] Number of attention heads.")
    p.add_argument("--num_layers", type=int,   default=2,
                   help="[Transformer] Number of encoder layers.")
    p.add_argument("--ff_dim",     type=int,   default=64,
                   help="[Transformer] Feed-forward hidden size.")
    p.add_argument("--dropout",    type=float, default=0.2)

    # -- GRU hyper-params --------------------------------------------------
    p.add_argument("--hidden1", type=int, default=32,
                   help="[GRU] First GRU hidden size.")
    p.add_argument("--hidden2", type=int, default=16,
                   help="[GRU] Second GRU hidden size.")
    p.add_argument("--fc_dim",  type=int, default=16,
                   help="[GRU] FC layer size before classifier head.")

    # -- Training ----------------------------------------------------------
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=50,
                   help="Early-stopping patience in epochs.")
    p.add_argument("--num_workers",  type=int,   default=4)

    args = p.parse_args()

    # -- Default output paths ----------------------------------------------
    base = PATH_MODELS_FEMI if args.dataset == "femi" else PATH_MODELS_MARKERS
    if args.model_dir is None:
        if args.use_pos_encoding:
            args.model_dir = base + args.model + "_pos_emb/"
        else:
            args.model_dir = base + args.model + "/"
    if args.metrics_dir is None:
        if args.use_pos_encoding:
            args.metrics_dir = f"output/assessment/ploidIA/{args.dataset}/{args.model}_pos_emb/"
        else:
            args.metrics_dir = f"output/assessment/ploidIA/{args.dataset}/{args.model}/"

    return args


# =============================================================================
# Data — FEMI pipeline
# =============================================================================

def infer_embed_dim(feature_dir: str) -> int:
    files = glob.glob(os.path.join(feature_dir, "train/*/*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {feature_dir}")
    sample = torch.load(files[0], map_location="cpu", weights_only=True)
    emb = sample["embeddings"]
    if emb.ndim != 2:
        raise ValueError(f"Expected (T, D) embeddings, got {tuple(emb.shape)}")
    return int(emb.shape[-1])


class FEMIDataset(Dataset):
    def __init__(self, files: list[str]):
        self.files  = files
        self.labels = [CLASSMAP[Path(f).parent.stem] for f in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        x    = data["embeddings"].float()                        # (T, D)
        mask = data["mask"].bool()                               # (T,)
        y    = torch.tensor(self.labels[idx], dtype=torch.long) # int64 for CE
        return x, mask, y


def build_femi_loaders(
    feature_dir: str, batch_size: int, num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_files = glob.glob(os.path.join(feature_dir, "train/*/*.pt"))
    val_files   = glob.glob(os.path.join(feature_dir, "test/*/*.pt"))
    log.info("FEMI — %d train | %d val files", len(train_files), len(val_files))
    train_loader = DataLoader(
        FEMIDataset(train_files),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        FEMIDataset(val_files),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# =============================================================================
# Data — Markers pipeline
# =============================================================================

def _load_single_csv(path: str):
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes
    x    = df.to_numpy(dtype=np.float32)
    mask = np.ones(len(df), dtype=bool)
    lbl  = CLASSMAP[Path(path).parent.stem]
    return x, mask, lbl


def _load_markers_split(markers_dir: str, split: str):
    files = glob.glob(str(Path(markers_dir) / split / "*/*.csv"))
    X, M, labels = [], [], []
    for f in random.sample(files, len(files)):
        x, mask, lbl = _load_single_csv(f)
        X.append(x); M.append(mask); labels.append(lbl)
    return np.array(X), np.array(M), np.array(labels, dtype=np.int64)


class MarkersDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.M = torch.from_numpy(M)
        self.y = torch.from_numpy(y)  # int64 for CE

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.M[idx], self.y[idx]


def build_markers_loaders(
    markers_dir: str, batch_size: int, num_workers: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    X_tr, M_tr, y_tr = _load_markers_split(markers_dir, "train")
    X_va, M_va, y_va = _load_markers_split(markers_dir, "test")
    log.info("Markers — train: %s | val: %s", X_tr.shape, X_va.shape)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(MarkersDataset(X_tr, M_tr, y_tr),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(MarkersDataset(X_va, M_va, y_va),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, int(X_tr.shape[1]), int(X_tr.shape[2])


# =============================================================================
# Models  (single implementation — shared by both pipelines)
# =============================================================================

def _sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    ang = pos / torch.pow(10_000, (2 * (dim // 2)) / d_model)
    ang[:, 0::2] = torch.sin(ang[:, 0::2])
    ang[:, 1::2] = torch.cos(ang[:, 1::2])
    return ang.unsqueeze(0)   # (1, T, d_model)


class GRUModel(nn.Module):
    """
    Two-layer stacked GRU with masked mean-pooling.
    Accepts any (B, T, input_dim) sequence — works for FEMI and Markers.
    """
    def __init__(
        self,
        input_dim:   int,
        hidden1:     int   = 32,
        hidden2:     int   = 16,
        fc_dim:      int   = 16,
        dropout:     float = 0.2,
        num_classes: int   = 2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru1 = nn.GRU(input_dim, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1,   hidden2, batch_first=True)
        self.norm = nn.LayerNorm(hidden2)
        self.fc   = nn.Linear(hidden2, fc_dim)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(fc_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:   nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias" in name:      nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)   mask: (B, T) bool — True = valid frame
        x = self.input_norm(x)
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        mask_f = mask.float().unsqueeze(-1)                        # (B, T, 1)
        pooled = (out * mask_f).sum(1) / (mask_f.sum(1) + 1e-9)   # (B, H2)
        pooled = self.norm(pooled)
        h = F.silu(self.fc(pooled))
        return self.out(self.drop(h))                              # (B, num_classes)


class TransformerModel(nn.Module):
    """
    Multi-layer Transformer encoder with masked mean-pooling.
    Accepts any (B, T, input_dim) sequence — works for FEMI and Markers.
    Optional sinusoidal positional encoding.
    """
    def __init__(
        self,
        input_dim:         int,
        d_model:           int   = 32,
        nhead:             int   = 4,
        num_layers:        int   = 2,
        ff_dim:            int   = 64,
        dropout:           float = 0.2,
        num_classes:       int   = 2,
        max_len:           int   = 1400,
        use_pos_encoding: bool  = False,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.use_pos_encoding = use_pos_encoding

        self.input_norm = nn.LayerNorm(input_dim)
        self.proj       = nn.Linear(input_dim, d_model)

        if use_pos_encoding:
            self.register_buffer("pos_enc", _sinusoidal_encoding(max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc   = nn.Linear(d_model, d_model // 2)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(d_model // 2, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)   mask: (B, T) bool — True = valid frame
        B, T, _ = x.shape
        x = self.input_norm(x)
        x = self.proj(x)                                           # (B, T, d_model)
        if self.use_pos_encoding:
            x = x + self.pos_enc[:, :T, :]
        # TransformerEncoder: key_padding_mask True = IGNORE position
        x = self.transformer(x, src_key_padding_mask=~mask)
        mask_f = mask.float().unsqueeze(-1)
        pooled = (x * mask_f).sum(1) / (mask_f.sum(1) + 1e-9)    # (B, d_model)
        h = F.silu(self.fc(pooled))
        return self.out(self.drop(h))                              # (B, num_classes)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class EpochMetrics:
    loss:   float      = 0.0
    acc:    float      = 0.0
    prec:   float      = 0.0
    rec:    float      = 0.0
    f1:     float      = 0.0
    preds:  np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in ("preds", "labels")}


# =============================================================================
# Train / Eval loop  (single, shared by all dataset x model combinations)
# =============================================================================

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    str,
    training:  bool,
) -> EpochMetrics:
    model.train(training)
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)       # (B, num_classes)
            loss   = criterion(logits, y) # y is int64

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(y.cpu())

    p = torch.cat(all_preds).numpy()
    l = torch.cat(all_labels).numpy()
    return EpochMetrics(
        loss=total_loss / len(loader),
        acc=accuracy_score(l, p),
        prec=precision_score(l, p, zero_division=0),
        rec=recall_score(l, p, zero_division=0),
        f1=f1_score(l, p, zero_division=0),
        preds=p, labels=l,
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, title: str, path: str, normalize: str = None) -> None:
    cm     = confusion_matrix(y_true, y_pred, normalize=normalize)
    fmt = ".2f" if normalize is not None else "d"
    labels = list(CLASSMAP.keys())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.savefig(path, dpi=150); plt.close()
    log.info("Confusion matrix saved -> %s", path)


def plot_training_curves(history: dict, metrics_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["val_f1"],  label="val f1")
    axes[1].set_title("Val metrics"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(metrics_dir, "training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    log.info("Training curves saved -> %s", path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Dataset: %s | Model: %s | Pos embedding: %s | Device: %s",
             args.dataset, args.model, args.use_pos_encoding, device)

    os.makedirs(args.model_dir,   exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    with open(os.path.join(args.metrics_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -- Data --------------------------------------------------------------
    if args.dataset == "femi":
        inferred = infer_embed_dim(args.feature_dir)
        if args.embed_dim is None:
            args.embed_dim = inferred
            log.info("Inferred embed_dim=%d", args.embed_dim)
        elif args.embed_dim != inferred:
            log.warning("embed_dim=%d != inferred %d — using inferred.",
                        args.embed_dim, inferred)
            args.embed_dim = inferred

        train_loader, val_loader = build_femi_loaders(
            args.feature_dir, args.batch_size, args.num_workers)
        input_dim = args.embed_dim
        max_len   = args.n_frames

    else:  # markers
        train_loader, val_loader, seq_len, n_feat = build_markers_loaders(
            args.markers_dir, args.batch_size, args.num_workers)
        input_dim = n_feat
        max_len   = seq_len

    # -- Model -------------------------------------------------------------
    if args.model == "gru":
        model = GRUModel(
            input_dim   = input_dim,
            hidden1     = args.hidden1,
            hidden2     = args.hidden2,
            fc_dim      = args.fc_dim,
            dropout     = args.dropout,
            num_classes = NUM_CLASSES,
        ).to(device)
    else:
        model = TransformerModel(
            input_dim         = input_dim,
            d_model           = args.d_model,
            nhead             = args.nhead,
            num_layers        = args.num_layers,
            ff_dim            = args.ff_dim,
            dropout           = args.dropout,
            num_classes       = NUM_CLASSES,
            max_len           = max_len,
            use_pos_encoding = args.use_pos_encoding,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s | Trainable params: %s", type(model).__name__, f"{n_params:,}")

    # -- Loss, optimiser, scheduler ----------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.dataset == "femi":
        scheduler      = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler_step = lambda val_loss: scheduler.step()
    else:
        scheduler      = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                           patience=7, min_lr=1e-6)
        scheduler_step = lambda val_loss: scheduler.step(val_loss)

    # -- Training loop -----------------------------------------------------
    ckpt_path    = os.path.join(args.model_dir, "best_model.pt")
    best_acc     = 0.0
    patience_cnt = 0
    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    log.info("Training for up to %d epochs (patience=%d) ...", args.epochs, args.patience)

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val = run_epoch(model, val_loader,   criterion, None,      device, training=False)
        scheduler_step(val.loss)

        if np.isnan(val.loss) or np.isnan(tr.loss):
            raise RuntimeError(
                f"NaN loss at epoch {epoch}. "
                "Check data for inf/nan and that masks are not all-False.")

        history["train_loss"].append(tr.loss)
        history["val_loss"].append(val.loss)
        history["val_acc"].append(val.acc)
        history["val_f1"].append(val.f1)

        lr_now = (scheduler.get_last_lr()[0]
                  if hasattr(scheduler, "get_last_lr")
                  else optimizer.param_groups[0]["lr"])

        log.info(
            "Epoch %3d/%d | train loss=%.4f acc=%.3f | "
            "val loss=%.4f acc=%.3f prec=%.3f rec=%.3f f1=%.3f | lr=%.2e",
            epoch, args.epochs, tr.loss, tr.acc,
            val.loss, val.acc, val.prec, val.rec, val.f1, lr_now)

        if val.acc > best_acc:
            best_acc     = val.acc
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val.loss, "val_acc": val.acc,
                        "val_f1": val.f1, "args": vars(args)}, ckpt_path)
            plot_confusion_matrix(
                val.labels, val.preds,
                title=f"Epoch {epoch} — val (best)",
                path=os.path.join(args.metrics_dir, "best_cm.png"))
            plot_confusion_matrix(
                val.labels, val.preds,
                title=f"Epoch {epoch} — val (best)",
                path=os.path.join(args.metrics_dir, "best_cm_normalized.png"),
                normalize="all")
            log.info("  checkpointed (val_acc=%.4f)", best_acc)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    # -- Post-training -----------------------------------------------------
    plot_training_curves(history, args.metrics_dir)
    with open(os.path.join(args.metrics_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    log.info("Loading best checkpoint for final evaluation ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    val = run_epoch(model, val_loader, criterion, None, device, training=False)
    log.info("\n===== FINAL RESULTS =====")
    for k, v in val.to_dict().items():
        log.info("  %-12s %.4f", k, v)

    with open(os.path.join(args.metrics_dir, "final_metrics.json"), "w") as f:
        json.dump(val.to_dict(), f, indent=2)

    plot_confusion_matrix(
        val.labels, val.preds,
        title="Final confusion matrix",
        path=os.path.join(args.metrics_dir, "final_cm.png"))

    log.info("All outputs -> %s", args.metrics_dir)
    log.info("Best checkpoint -> %s", ckpt_path)


if __name__ == "__main__":
    main()