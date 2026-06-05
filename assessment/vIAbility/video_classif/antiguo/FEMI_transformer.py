"""
train_femi.py
─────────────────────────────────────────────────────────────────────────────
FEMI-based fetal heartbeat classifier.

Architecture : Frozen FEMI (ViT-MAE) frame embeddings
               + lightweight Transformer temporal head
Task         : Binary classification  (Fetal_Heart_Beat  0 / 1)
─────────────────────────────────────────────────────────────────────────────

Usage examples
──────────────
# Train with defaults
python train_femi.py

# Custom paths
python train_femi.py \
    --feature_dir /data/femi_features \
    --excel_path  /data/Oslo_completo.xlsx \
    --model_dir   models/femi/ \
    --metrics_dir results/femi/

# Tune hyperparameters
python train_femi.py --epochs 300 --lr 5e-5 --batch_size 8 --patience 30
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.data.dataset import build_labels
from config.config import PATH_DATASET, SEED


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def infer_embed_dim(feature_dir: str) -> int:
    """Read one feature file and return its embedding width."""
    files = sorted(
        os.path.join(feature_dir, f)
        for f in os.listdir(feature_dir)
        if f.endswith(".pt")
    )
    if not files:
        raise FileNotFoundError(f"No .pt feature files found in: {feature_dir}")

    sample = torch.load(files[0], map_location="cpu", weights_only=True)
    if "embeddings" not in sample:
        raise KeyError(f"Missing 'embeddings' key in feature file: {files[0]}")

    emb = sample["embeddings"]
    if emb.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (T, D), got {tuple(emb.shape)} in {files[0]}"
        )

    return int(emb.shape[-1])


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLI args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FEMI heartbeat classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--feature_dir",
        default=os.path.join(PATH_DATASET, "Oslo_processed/femi_features"),
        help="Directory containing pre-extracted .pt FEMI features",
    )
    p.add_argument(
        "--excel_path",
        default=os.path.join(PATH_DATASET, "Oslo_processed/Oslo_completo.xlsx"),
        help="Path to the Excel metadata file",
    )
    p.add_argument(
        "--model_dir",
        default="models/assessment/vIAbility/femi_transformer/",
        help="Directory where model checkpoints are saved",
    )
    p.add_argument(
        "--metrics_dir",
        default="output/assessment/vIAbility/femi_transformer/",
        help="Directory where metrics, plots, and JSON logs are saved",
    )

    # ── Data ───────────────────────────────────────────────────────────────
    p.add_argument("--embed_dim",  type=int,   default=None,
                   help="FEMI embedding dimension. If omitted, inferred from feature files")
    p.add_argument("--n_frames",   type=int,   default=214,
                   help="Number of frames extracted per video (must match extractor)")
    p.add_argument("--test_size",  type=float, default=0.2,
                   help="Fraction of data held out for validation")


    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--d_model",    type=int,   default=64)
    p.add_argument("--nhead",      type=int,   default=4)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.3)

    # ── Training ───────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=50,
                   help="Early-stopping patience (epochs)")
    p.add_argument("--num_workers", type=int,   default=4)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────
class FEMIDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int]):
        self.files  = files
        self.labels = labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        x    = data["embeddings"].float()   # (T, embed_dim)
        mask = data["mask"].bool()          # (T,)  True = valid frame
        y    = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, mask, y


def build_loaders(
    feature_dir: str,
    excel_path:  str,
    test_size:   float,
    batch_size:  int,
    num_workers: int,
    seed:        int,
) -> tuple[DataLoader, DataLoader]:
    df        = pd.read_excel(excel_path)
    all_files = sorted([
        os.path.join(feature_dir, f)
        for f in os.listdir(feature_dir) if f.endswith(".pt")
    ])
    
    namefiles = [os.path.basename(f) for f in all_files]
    labels    = build_labels(namefiles, df)

    train_files, val_files, y_train, y_val = train_test_split(
        all_files, labels,
        test_size=test_size,
        random_state=seed,
    )

    with open("femi_files.txt", "w") as f:
        for file in train_files:
            f.write(str(file) + "\n")

    log.info("Train: %d  |  Val: %d", len(train_files), len(val_files))
    log.info(
        "Class dist — train: %s  val: %s",
        np.bincount(y_train).tolist(),
        np.bincount(y_val).tolist(),
    )

    train_loader = DataLoader(
        FEMIDataset(train_files, y_train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        FEMIDataset(val_files, y_val),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Return pos_weight for loss function alongside loaders
    label_t    = torch.tensor(labels, dtype=torch.float32)
    pos_weight = ((len(label_t) - label_t.sum()) / label_t.sum()).unsqueeze(0)

    return train_loader, val_loader, pos_weight


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────────────────
class FEMIClassifier(nn.Module):
    """
    Lightweight Transformer over frozen FEMI per-frame embeddings.

    Input  : (B, T, embed_dim) + mask (B, T)  True = valid frame
    Output : (B,) raw logit
    """

    def __init__(
        self,
        embed_dim:  int = 768,
        d_model:    int = 64,
        nhead:      int = 4,
        num_layers: int = 2,
        max_len:    int = 214,
        dropout:    float = 0.3,
    ):
        super().__init__()

        self.input_proj    = nn.Linear(embed_dim, d_model)
        # +1 position for the CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len + 1, d_model) * 0.02
        )
        self.cls_token     = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation="gelu", norm_first=True,   # pre-norm → more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        x         = self.input_proj(x)                           # (B, T, d_model)
        cls       = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x         = torch.cat([cls, x], dim=1)                   # (B, T+1, d_model)
        x         = x + self.pos_embedding[:, :T + 1]

        cls_mask  = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)           # (B, T+1)

        # TransformerEncoder key_padding_mask: True = IGNORE → invert valid mask
        x = self.transformer(x, src_key_padding_mask=~full_mask)

        return self.head(x[:, 0]).squeeze(-1)                    # CLS → (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EpochMetrics:
    loss:  float = 0.0
    acc:   float = 0.0
    prec:  float = 0.0
    rec:   float = 0.0
    f1:    float = 0.0
    preds:  np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in ("preds", "labels")}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / eval loop
# ─────────────────────────────────────────────────────────────────────────────
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

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return EpochMetrics(
        loss  = total_loss / len(loader),
        acc   = accuracy_score(all_labels, all_preds),
        prec  = precision_score(all_labels, all_preds, zero_division=0),
        rec   = recall_score(all_labels, all_preds, zero_division=0),
        f1    = f1_score(all_labels, all_preds, zero_division=0),
        preds = all_preds,
        labels= all_labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title: str, path: str) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", path)


def plot_training_curves(history: dict, metrics_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy + F1
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["val_f1"],  label="val f1")
    axes[1].set_title("Val metrics"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(metrics_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved training curves → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Directories ───────────────────────────────────────────────────────
    os.makedirs(args.model_dir,   exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # Save run config for reproducibility
    config_path = os.path.join(args.metrics_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Run config saved → %s", config_path)

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Loading features from %s …", args.feature_dir)
    train_loader, val_loader, pos_weight = build_loaders(
        feature_dir = args.feature_dir,
        excel_path  = args.excel_path,
        test_size   = args.test_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = SEED,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    inferred_embed_dim = infer_embed_dim(args.feature_dir)
    if args.embed_dim is None:
        args.embed_dim = inferred_embed_dim
        log.info("Inferred embed_dim=%d from feature files", args.embed_dim)
    elif args.embed_dim != inferred_embed_dim:
        log.warning(
            "embed_dim=%d does not match features (%d). Using inferred value.",
            args.embed_dim,
            inferred_embed_dim,
        )
        args.embed_dim = inferred_embed_dim

    model = FEMIClassifier(
        embed_dim  = args.embed_dim,
        d_model    = args.d_model,
        nhead      = args.nhead,
        num_layers = args.num_layers,
        max_len    = args.n_frames,
        dropout    = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable params: %s", f"{n_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    ckpt_path     = os.path.join(args.model_dir, "best_model.pt")
    best_val_loss = float("inf")
    best_f1       = 0.0
    best_acc      = 0.0
    patience_cnt  = 0

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "val_acc": [],    "val_f1":  [],
    }

    log.info("Starting training for up to %d epochs …", args.epochs)

    for epoch in range(1, args.epochs + 1):
        tr  = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val = run_epoch(model, val_loader,   criterion, None,      device, training=False)
        scheduler.step()

        history["train_loss"].append(tr.loss)
        history["val_loss"].append(val.loss)
        history["val_acc"].append(val.acc)
        history["val_f1"].append(val.f1)

        log.info(
            "Epoch %3d/%d | train loss=%.4f acc=%.3f | "
            "val loss=%.4f acc=%.3f prec=%.3f rec=%.3f f1=%.3f | "
            "lr=%.2e",
            epoch, args.epochs,
            tr.loss,  tr.acc,
            val.loss, val.acc, val.prec, val.rec, val.f1,
            scheduler.get_last_lr()[0],
        )

        #if val.loss < best_val_loss:
        if val.acc > best_acc:  # Save best by F1 score
            best_val_loss = val.loss
            best_f1       = val.f1
            best_acc      = val.acc
            patience_cnt  = 0
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss":  val.loss,
                    "val_acc":   val.acc,
                    "val_f1":    val.f1,
                    "args":      vars(args),
                },
                ckpt_path,
            )
            plot_confusion_matrix(
                val.labels, val.preds,
                title=f"Epoch {epoch} — val (best)",
                path=os.path.join(args.metrics_dir, "best_cm.png"),
            )
            log.info("  ✓ saved checkpoint  (val_loss=%.4f)", best_val_loss)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                log.info("Early stopping at epoch %d.", epoch)
                break

    # ── Save training curves ──────────────────────────────────────────────
    plot_training_curves(history, args.metrics_dir)

    # Save history as JSON
    history_path = os.path.join(args.metrics_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved → %s", history_path)

    # ── Final evaluation with best checkpoint ─────────────────────────────
    log.info("Loading best checkpoint for final evaluation …")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    val = run_epoch(model, val_loader, criterion, None, device, training=False)

    final_metrics = val.to_dict()
    log.info("\n===== FINAL RESULTS =====")
    for k, v in final_metrics.items():
        log.info("  %-12s %.4f", k, v)

    # Save final metrics JSON
    metrics_path = os.path.join(args.metrics_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    log.info("Final metrics saved → %s", metrics_path)

    # Save final confusion matrix
    plot_confusion_matrix(
        val.labels, val.preds,
        title="Final confusion matrix",
        path=os.path.join(args.metrics_dir, "final_cm.png"),
    )

    log.info("All outputs saved to %s", args.metrics_dir)
    log.info("Best checkpoint saved to %s", ckpt_path)


if __name__ == "__main__":
    main()