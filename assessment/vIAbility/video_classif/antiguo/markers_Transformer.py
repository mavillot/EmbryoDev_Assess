import sys
import glob
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PATH_MODELS

# ── Constants ─────────────────────────────────────────────────────────────────
classmap   = {"no_HB": 0, "HB": 1}
PATH_DATA   = Path("../Datasets/Processed/Viability_markers/")
PATH_MODELS = PATH_MODELS + "assessment/vIAbility/"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Data ──────────────────────────────────────────────────────────────────────

def single_data(path: str):
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes
    x_values = df.to_numpy(dtype=np.float32)
    # All positions are valid (no padding) — True = valid throughout this codebase
    mask = np.ones(len(df), dtype=bool)
    lbl  = classmap[Path(path).parent.stem]
    return x_values, mask, lbl


def load_data(path: Path, split: str = "train"):
    X, M, labels = [], [], []
    files = glob.glob(str(path / "*/*.csv"))
    X_train, X_test, _, _ = train_test_split(files, [0 for f in files], test_size=0.2, random_state=SEED)
    if split == "train":
        selected_files = X_train
    else:        
        selected_files = X_test
    for file in selected_files:
        x, mask, lbl = single_data(file)
        X.append(x)
        M.append(mask)
        labels.append(lbl)
    return np.array(X), np.array(M), np.array(labels, dtype=np.int64)



class VideoDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)          # (N, T, F)  float32
        self.M = torch.from_numpy(M)          # (N, T)     bool
        self.y = torch.from_numpy(y)          # (N,)       int64

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx]


# ── Models ────────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    """
    Stacked GRU encoder with LayerNorm + swish head.
    Mask convention: True = valid position (invert for pack_padded if needed).
    """
    def __init__(self, seq_len: int, num_features: int,
                 hidden1: int = 32, hidden2: int = 16,
                 fc_dim: int = 16, dropout: float = 0.2,
                 num_classes: int = 2):
        super().__init__()
        self.gru1 = nn.GRU(num_features, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1,      hidden2, batch_first=True)
        self.norm = nn.LayerNorm(hidden2)
        self.fc   = nn.Linear(hidden2, fc_dim)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(fc_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)  mask: (B, T) bool, True = valid
        out, _ = self.gru1(x)                      # (B, T, H1)
        out, _ = self.gru2(out)                    # (B, T, H2)

        # Masked mean-pool: average over valid (non-padded) time steps
        mask_f = mask.float().unsqueeze(-1)        # (B, T, 1)
        pooled = (out * mask_f).sum(1) / (mask_f.sum(1) + 1e-9)  # (B, H2)

        pooled = self.norm(pooled)
        h = F.silu(self.fc(pooled))                # swish = silu in PyTorch
        h = self.drop(h)
        return self.out(h)                         # (B, num_classes) — raw logits


def _sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pos  = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim  = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
    ang  = pos / torch.pow(10_000, (2 * (dim // 2)) / d_model)
    ang[:, 0::2] = torch.sin(ang[:, 0::2])
    ang[:, 1::2] = torch.cos(ang[:, 1::2])
    return ang.unsqueeze(0)                        # (1, T, d_model)


class TransformerModel(nn.Module):
    """
    Lightweight Transformer encoder:
      projection → sinusoidal pos-enc → MHA + LN → FFN + LN → masked mean-pool → head
    """
    def __init__(self, seq_len: int, num_features: int,
                 d_model: int = 32, num_heads: int = 4, ff_dim: int = 64,
                 dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.proj    = nn.Linear(num_features, d_model)
        self.register_buffer("pos_enc", _sinusoidal_encoding(seq_len, d_model))

        self.attn    = nn.MultiheadAttention(d_model, num_heads,
                                             dropout=0.1, batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)

        self.ff      = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2   = nn.LayerNorm(d_model)

        self.fc      = nn.Linear(d_model, 16)
        self.drop    = nn.Dropout(dropout)
        self.out     = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)   mask: (B, T) bool, True = VALID position
        # PyTorch MHA key_padding_mask: True = IGNORE → invert
        pad_mask = ~mask                           # (B, T): True = padded

        x = self.proj(x) + self.pos_enc           # (B, T, d_model)

        # Self-attention block (pre-norm style)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=pad_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward block
        x = self.norm2(x + self.ff(x))

        # Masked mean-pool
        mask_f = mask.float().unsqueeze(-1)        # (B, T, 1)
        pooled = (x * mask_f).sum(1) / (mask_f.sum(1) + 1e-9)   # (B, d_model)

        h = F.silu(self.fc(pooled))
        h = self.drop(h)
        return self.out(h)                         # (B, num_classes) — raw logits


# ── Metrics ───────────────────────────────────────────────────────────────────

class RunningMetrics:
    """Accumulates predictions over an epoch, then computes all metrics at once."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._preds  = []
        self._labels = []
        self._loss   = 0.0
        self._n      = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float):
        preds = logits.argmax(dim=-1).cpu().numpy()
        self._preds.append(preds)
        self._labels.append(labels.cpu().numpy())
        self._loss += loss * len(labels)
        self._n    += len(labels)

    def compute(self) -> dict:
        y_pred = np.concatenate(self._preds)
        y_true = np.concatenate(self._labels)
        return {
            "loss":      self._loss / self._n,
            "accuracy":  (y_pred == y_true).mean(),
            "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall":    recall_score(   y_true, y_pred, pos_label=1, zero_division=0),
            "f1":        f1_score(       y_true, y_pred, pos_label=1, zero_division=0),
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None):
    """Single train or eval pass. Pass optimizer=None for eval."""
    training = optimizer is not None
    model.train(training)
    metrics = RunningMetrics()

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

            metrics.update(logits.detach(), y, loss.item())

    return metrics.compute()


def train(model, train_loader, val_loader, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=7, min_lr=1e-6)

    best_val_loss = float("inf")
    patience_cnt  = 0
    checkpoint    = Path(args.checkpoint)/args.model/ "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, criterion, optimizer)
        val = run_epoch(model, val_loader,   criterion)

        # Abort early if loss exploded — indicates a data or mask bug
        if np.isnan(val["loss"]) or np.isnan(tr["loss"]):
            raise RuntimeError(
                f"NaN loss at epoch {epoch}. "
                "Check your data for inf/nan and verify the mask is not all-False."
            )

        scheduler.step(val["loss"])

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f}  |  "
            f"val loss={val['loss']:.4f} acc={val['accuracy']:.4f} "
            f"prec={val['precision']:.4f} rec={val['recall']:.4f} f1={val['f1']:.4f}"
        )

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            patience_cnt  = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  ✓ saved checkpoint → {checkpoint}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Restore best weights (file is guaranteed to exist: NaN raises before first save)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    else:
        print("Warning: no checkpoint found — returning model from last epoch.")
    return model


# ── Evaluation & output ───────────────────────────────────────────────────────

def evaluate_and_save(model, loader, out_dir: str = "."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    metrics   = run_epoch(model, loader, criterion)   # eval mode, no optimizer

    results = (
        f"Test accuracy:  {metrics['accuracy']  * 100:.2f}%\n"
        f"Test precision: {metrics['precision'] * 100:.2f}%\n"
        f"Test recall:    {metrics['recall']    * 100:.2f}%\n"
        f"Test F1:        {metrics['f1']        * 100:.2f}%\n"
    )
    print("\n" + results)
    (out_dir / "test_results.txt").write_text(results)

    # Collect all predictions for confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, M, y in loader:
            logits = model(X.to(DEVICE), M.to(DEVICE))
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    class_names = list(classmap.keys())

    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{type(model).__name__} — confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Results saved to {out_dir}/")


# ── Inference ─────────────────────────────────────────────────────────────────

def sequence_prediction(path: str, model: nn.Module):
    class_vocab = list(classmap.keys())
    x, mask, _  = single_data(path)
    X    = torch.from_numpy(x   ).unsqueeze(0).to(DEVICE)   # (1, T, F)
    M    = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)   # (1, T)
    model.eval()
    with torch.no_grad():
        logits = model(X, M)
        probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()
    for i in np.argsort(probs)[::-1]:
        print(f"  {class_vocab[i]}: {probs[i] * 100:5.2f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args):
    # Data
    X_train, M_train, y_train = load_data(PATH_DATA, split="train")
    X_test,  M_test,  y_test  = load_data(PATH_DATA, split="test")
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    seq_len = X_train.shape[1]
    n_feat  = X_train.shape[2]
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    train_ds = VideoDataset(X_train, M_train, y_train)
    test_ds  = VideoDataset(X_test,  M_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=DEVICE.type == "cuda")

    # Model
    if args.model == "gru":
        model = GRUModel(seq_len, n_feat).to(DEVICE)
    elif args.model == "transformer":
        model = TransformerModel(seq_len, n_feat,
                                 d_model=args.d_model,
                                 num_heads=args.num_heads,
                                 ff_dim=args.ff_dim).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__}  |  Trainable params: {total_params:,}")

    # Train
    model = train(model, train_loader, test_loader, args)

    # Evaluate
    out_dir = Path(args.out_dir) / args.model
    evaluate_and_save(model, test_loader, out_dir=out_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GRU or Transformer for ploidy classification (PyTorch)")

    parser.add_argument("--model", choices=["gru", "transformer"], default="gru",
                        help="Architecture to train (default: gru)")
    parser.add_argument("--checkpoint", default=PATH_MODELS,
                        help="Path to save best model weights (.pt)")
    parser.add_argument("--out_dir", default="output/assessment/vIAbility/",
                        help="Directory for results and confusion matrix")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int, default=10,
                        help="Early-stopping patience (epochs)")

    # Transformer hyper-params
    parser.add_argument("--d_model",   type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim",    type=int, default=128)

    main(parser.parse_args())