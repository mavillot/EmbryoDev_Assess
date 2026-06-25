"""
train.py
────────────────────────────────────────────────────────────────────────────────
R(2+1)D / MViT binary classifier for fetal-heartbeat detection.

Architecture  : R(2+1)D-18  (factorised spatial + temporal convolutions)
                or MViT-V2-S
Task          : Binary classification  (Fetal_Heart_Beat  0 / 1)
Extras        : Mixed-precision (AMP) · Early stopping · Padded-frame masking
                · Repeatable runs (backbone / n_frames / iteration / seed are
                  now parameters of TrainConfig, not hard-coded globals) ·
                  Every run's metrics (accuracy, precision, recall, F1,
                  F-beta(0.5), AUROC) are appended to a master results CSV.
────────────────────────────────────────────────────────────────────────────────
"""

import csv
import json
import os
import sys
import logging
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, MViT_V2_S_Weights, mvit_v2_s

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score, fbeta_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.data.dataset import get_dataloaders, mixup_collate
from assessment.vIAbility.models.video import MViTClassifier, R2Plus1DClassifier
from config.config import PATH_DATASET, SEED


VIDEO_DIR = os.path.join(PATH_DATASET, "Oslo_viability/videos")
EXCEL_PATH = os.path.join(PATH_DATASET, "Oslo_viability/Oslo_completo.xlsx")

# Master CSV that accumulates ONE ROW PER RUN across every backbone /
# n_frames / iteration combination. run_experiments.py reads this file to
# decide what's already been done (resume support).
METRICS_CSV = "results/assessment/vIAbility/3d_cnn/experiment_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # data
    frame_h: int        = 112       # R(2+1)D expects 112 x 112
    frame_w: int        = 112
    n_frames: int       = 16        # temporal depth (TIME_WINDOW length)
    num_workers: int    = 4
    backbone: str       = "mvit"   # "r2plus1d_18" or "mvit"

    # ── Repetition bookkeeping ────────────────────────────────────────────
    # iteration: 1-indexed repeat number (used only for naming ckpts/outputs
    #            and for the "iteration" column in the results CSV).
    # seed:      RNG seed for THIS run. Repeats must use different seeds or
    #            every "repeat" would be bit-for-bit identical.
    iteration: int      = 0
    seed: int           = SEED
    metrics_csv: str    = METRICS_CSV

    # ── Two-stage fine-tuning ─────────────────────────────────────────────
    # Stage 1: backbone frozen → only the classification head is trained.
    # Stage 2: full model unfrozen → backbone gets a much smaller LR.
    weight_loss: float  = 1.48    # weight for positive class in BCE loss (if using weights)
    freeze_epochs: int  = 30        # how many epochs to keep backbone frozen
    lr_head: float      = 3e-4     # head LR (both stages)
    lr_backbone: float  = 1e-5     # backbone LR (stage 2 only; ~30× smaller)
    warmup_epochs: int  = 3        # linear LR warmup at the start of each stage
    weight_decay: float = 1e-4
    beta: float         = 0.5      # for F-beta metric (precision-recall tradeoff)
    save_by: str       = "val_fbeta"   # metric to determine best checkpoint

    # ── Regularisation ────────────────────────────────────────────────────
    label_smoothing: float = 0.2   # prevents overconfident logits
    dropout_head: float    = 0.5   # dropout before first linear in head
    dropout_mid: float     = 0.5   # dropout between linear layers

    # training budget
    epochs: int         = 31
    batch_size: int     = 8

    # early stopping
    patience: int       = 10
    min_delta: float    = 1e-4

    #data aumegmentation
    augmentation = True
    oversample = False
    factor1 = 2
    factor0 = 1.5

    # checkpointing
    ckpt_dir: str       = "models/assessment/vIAbility/3d_cnn/"
    ckpt_name: str      = field(init=False)

    # evaluation outputs
    out_dir: str        = "results/assessment/vIAbility/3d_cnn"

    # misc
    device: str         = field(default_factory=lambda: ("cuda:2" if torch.cuda.is_available() else "cpu"))
    amp: bool           = False      # mixed-precision; auto-disabled on CPU

    def __post_init__(self):
        self.ckpt_name = f"best_{self.backbone}_{self.n_frames}_iter{self.iteration}.pt"
        if self.backbone == "mvit":
            # MViT_V2_S was pretrained on 224×224 frames
            self.frame_h = 224
            self.frame_w = 224


CFG = TrainConfig()


class EarlyStopping:
    """Stops training when val loss stops improving for `patience` epochs."""

    def __init__(self, patience: int = CFG.patience, min_delta: float = CFG.min_delta):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            log.info("EarlyStopping counter: %d / %d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LR warmup helper
# ─────────────────────────────────────────────────────────────────────────────
def linear_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    """Returns a scalar in (0, 1] that linearly ramps from 0 to 1."""
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    cfg:       TrainConfig,
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler:    Optional[GradScaler],
    device:    str,
    training:  bool,
) -> tuple[float, float, float]:
    """One full pass through *loader*. Returns (avg_loss, accuracy, fbeta)."""
    model.train() if training else model.eval()

    total_loss, correct, total, fbeta = 0.0, 0, 0, 0.0
    amp_enabled = cfg.amp and device != "cpu"

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, mask, labels in loader:
            frames = frames.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)

            with autocast("cuda", enabled=amp_enabled):
                # MViTClassifier ignores the mask argument; pass it anyway — it's a no-op.
                logits = model(frames, mask)
                loss = criterion(logits.view(-1), labels.float())

            if training:

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * frames.size(0)
            preds       = (torch.sigmoid(logits) >= 0.5).long()
            correct    += (preds == labels).sum().item()
            total      += frames.size(0)
            fbeta += fbeta_score(labels.cpu().numpy().astype(int), preds.cpu().numpy().astype(int), pos_label=1, beta=cfg.beta, zero_division=0)

    return total_loss / total, correct / total, fbeta / total


# ─────────────────────────────────────────────────────────────────────────────
# 6b.  Metrics logging (one row per run, across the whole sweep)
# ─────────────────────────────────────────────────────────────────────────────
def append_metrics_row(csv_path: str, row: dict) -> None:
    """Append one run's metrics as a row to the master results CSV.

    Creates the file (with a header) the first time it's called, so the
    sweep is crash-safe — every completed run is on disk immediately,
    nothing is held in memory until the end.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: Optional[TrainConfig] = None):
    cfg = cfg or CFG

    # ── Seed THIS run ───────────────────────────────────────────────────────
    # Each repeat must use a different seed, otherwise "repeating" the
    # experiment 5 times just trains the identical model 5 times.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device
    log.info(
        "Backbone: %s | n_frames: %d | iteration: %d | seed: %d | Device: %s | AMP: %s",
        cfg.backbone, cfg.n_frames, cfg.iteration, cfg.seed, device, cfg.amp and device != "cpu",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Building dataset …")
    train_loader, val_loader = get_dataloaders(
        video_dir   = VIDEO_DIR,
        excel_path  = EXCEL_PATH,
        frame_h     = cfg.frame_h,
        frame_w     = cfg.frame_w,
        n_frames    = cfg.n_frames,
        augment     = cfg.augmentation,
        oversample  = cfg.oversample,
        factor1     = cfg.factor1,
        factor0     = cfg.factor0,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        device      = device,
    )

    log.info("Train: %d  |  Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    # ── Model ─────────────────────────────────────────────────────────────────
    if cfg.backbone == "mvit":
        model = MViTClassifier(num_classes=1, pretrained=True, dropout_head=cfg.dropout_head, dropout_mid=cfg.dropout_mid).to(device)
    elif cfg.backbone == "r2plus1d_18":
        model = R2Plus1DClassifier(num_classes=1, pretrained=True, dropout_head=cfg.dropout_head, dropout_mid=cfg.dropout_mid).to(device)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    # Stage 1: freeze backbone — only head is active
    model.freeze_backbone()
    log.info(
        "Stage 1 — backbone frozen. Trainable params: %.2fM / %.2fM",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # ── Loss: BCEWithLogitsLoss + label smoothing ──────────────────────────────
    def smooth_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = cfg.label_smoothing
        targets_smooth = targets * (1.0 - eps) + 0.5 * eps
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth, weight=(torch.tensor([cfg.weight_loss], device=logits.device) if torch.cuda.is_available() else None))

    criterion = smooth_bce

    # ── Stage-1 optimiser (head only) ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.head_parameters(),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    scaler  = GradScaler() if (cfg.amp and device == "cuda") else None
    stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # Per-run checkpoint + output dirs include the iteration number so
    # repeats never overwrite each other.
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.backbone, str(cfg.n_frames), f"iter{cfg.iteration}")
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_path = os.path.join(ckpt_path, cfg.ckpt_name)

    if cfg.save_by == "val_loss":
        best_val = float("inf")
    elif cfg.save_by in {"val_acc", "val_fbeta"}:
        best_val = 0.0
    else:
        raise ValueError(f"Invalid save_by metric: {cfg.save_by}")
    TRAIN_LOSS, VALID_LOSS = [], []
    TRAIN_ACC, VALID_ACC = [], []
    TRAIN_fbeta, VALID_fbeta = [], []
    # ── Training loop ─────────────────────────────────────────────────────────
    stage = 1
    log.info("Starting training for up to %d epochs …", cfg.epochs)
    saved_checkpoint = False

    for epoch in range(1, cfg.epochs + 1):
        train_loader.collate_fn = mixup_collate if epoch < (cfg.epochs - 10) else default_collate
        # ── Stage transition ───────────────────────────────────────────────
        if epoch == cfg.freeze_epochs + 1:
            stage = 2
            model.unfreeze_backbone()
            # Rebuild optimiser with two param groups (different LRs)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.head_parameters(),     "lr": cfg.lr_head},
                    {"params": model.backbone_parameters(), "lr": cfg.lr_backbone},
                ],
                weight_decay=cfg.weight_decay,
            )
            stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
            log.info(
                "── Stage 2: backbone unfrozen. "
                "head lr=%.0e  backbone lr=%.0e ──",
                cfg.lr_head, cfg.lr_backbone,
            )

        # ── Linear warmup within each stage ───────────────────────────────
        stage_epoch = epoch if stage == 1 else epoch - cfg.freeze_epochs
        warmup      = linear_warmup_factor(stage_epoch - 1, cfg.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup
        # Store initial_lr on first pass so warmup has a reference
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])

        train_loss, train_acc, train_fbeta = run_epoch(
            cfg, model, train_loader, criterion, optimizer, scaler, device, training=True
        )
        val_loss, val_acc, val_fbeta = run_epoch(
            cfg, model, val_loader, criterion, None, None, device, training=False
        )

        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d [S%d]  |  "
            "train loss: %.4f  acc: %.3f  fbeta: %.3f  |  "
            "val loss: %.4f  acc: %.3f  fbeta: %.3f  |  "
            "lr: %.2e",
            epoch, cfg.epochs, stage,
            train_loss, train_acc, train_fbeta,
            val_loss,   val_acc, val_fbeta,
            current_lr,
        )
        TRAIN_LOSS.append(train_loss)
        VALID_LOSS.append(val_loss)
        TRAIN_ACC.append(train_acc)
        VALID_ACC.append(val_acc)
        TRAIN_fbeta.append(train_fbeta)
        VALID_fbeta.append(val_fbeta)

        if cfg.save_by == "val_loss":
            metric_to_check = val_loss
        elif cfg.save_by == "val_acc":
            metric_to_check = val_acc
        elif cfg.save_by == "val_fbeta":
            metric_to_check = val_fbeta
        else:
            raise ValueError(f"Invalid save_by metric: {cfg.save_by}")

        if metric_to_check < best_val if cfg.save_by == "val_loss" else metric_to_check > best_val:
            best_val = metric_to_check
            torch.save(
                {
                    "epoch":     epoch,
                    "stage":     stage,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc":  train_acc,
                    "train_fbeta": train_fbeta,
                    "val_loss":  val_loss,
                    "val_acc":   val_acc,
                    "val_fbeta": val_fbeta,
                    "config":    vars(cfg),   # plain dict — safe with weights_only=True
                },
                ckpt_path,
            )
            saved_checkpoint = True
            log.info(f"  ✓ Saved best checkpoint  (val_{cfg.save_by}={best_val:.4f})")

        if stopper.step(val_loss):
            log.info("Early stopping triggered at epoch %d (stage %d).", epoch, stage)
            break
    if not saved_checkpoint:
        log.warning("No checkpoint was saved during training. ")
        torch.save(
                {
                    "epoch":     epoch,
                    "stage":     stage,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc":  train_acc,
                    "train_fbeta": train_fbeta,
                    "val_loss":  val_loss,
                    "val_acc":   val_acc,
                    "val_fbeta": val_fbeta,
                    "config":    vars(cfg),   # plain dict — safe with weights_only=True
                },
                ckpt_path,
            )

    log.info("Training complete. Best val %s: %.4f", cfg.save_by, best_val)
    log.info("Best checkpoint saved to: %s", ckpt_path)

    def save_training_plots(out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(TRAIN_LOSS, label="Train Loss")
        plt.plot(VALID_LOSS, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(TRAIN_ACC, label="Train Accuracy")
        plt.plot(VALID_ACC, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        curves_path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(curves_path)   # savefig BEFORE show — show clears the figure
        plt.show()
        plt.close()
        log.info("Training curves saved to: %s", curves_path)

    # Per-run output dir — also keyed by iteration so repeats don't collide.
    out_dir = os.path.join(cfg.out_dir, cfg.backbone, str(cfg.n_frames), f"iter{cfg.iteration}")
    save_training_plots(out_dir)
    save_training_config(cfg, out_dir)

    # ── Load best checkpoint and run full evaluation ───────────────────────
    log.info("Loading best checkpoint for evaluation …")
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model"])
    metrics = evaluate(model, val_loader, device, out_dir, cfg)

    # ── Log this run's row to the master results CSV ───────────────────────
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backbone": cfg.backbone,
        "n_frames": cfg.n_frames,
        "iteration": cfg.iteration,
        "seed": cfg.seed,
        "best_epoch": best_ckpt["epoch"],
        f"best_val_{cfg.save_by}": best_val,
        **metrics,
        "ckpt_path": ckpt_path,
    }
    append_metrics_row(cfg.metrics_csv, row)
    log.info("Logged metrics row to %s", cfg.metrics_csv)

    # free GPU memory before the next run in a sweep
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader,
             device: str, out_dir: str, cfg: TrainConfig) -> dict:
    """
    Run inference on *loader* with the best checkpoint, then save to *out_dir*:
      - confusion_matrix.png
      - classification_report.txt   (precision / recall / F1 / accuracy / F-beta / AUROC)
      - metrics.json                 (same numbers, machine-readable)

    Returns a flat dict of metrics that the caller (train()) logs to the
    master results CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for frames, mask, labels in loader:
            frames = frames.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)

            logits = model(frames, mask).view(-1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true  = np.array(all_labels, dtype=int)
    y_pred  = np.array(all_preds,  dtype=int)
    y_probs = np.array(all_probs,  dtype=float)

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No HB (0)", "HB (1)"],
        yticklabels=["No HB (0)", "HB (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    plt.close()
    log.info("Confusion matrix saved to: %s", cm_path)

    # ── Core classification metrics ────────────────────────────────────────
    acc                    = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    fbeta_1 = fbeta_score(y_true, y_pred, pos_label=1, beta=cfg.beta, zero_division=0)

    # AUROC needs both classes present in y_true; guard against the
    # degenerate case where a tiny val split is single-class.
    try:
        auroc = roc_auc_score(y_true, y_probs)
    except ValueError:
        log.warning("AUROC undefined (only one class present in this validation split) — recording NaN.")
        auroc = float("nan")

    report_str = classification_report(
        y_true, y_pred,
        target_names=["No Heartbeat (0)", "Heartbeat (1)"],
        zero_division=0,
    )
    log.info("\n%s", report_str)
    log.info("F-beta(beta=%.2f, pos=1): %.4f  |  AUROC: %.4f", cfg.beta, fbeta_1, auroc)

    metrics = {
        "accuracy":    acc,
        "precision_0": prec[0], "recall_0": rec[0], "f1_0": f1[0],
        "precision_1": prec[1], "recall_1": rec[1], "f1_1": f1[1],
        f"fbeta{cfg.beta}_1": fbeta_1,
        "auroc":       auroc,
    }

    # ── Human-readable report ──────────────────────────────────────────────
    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(f"Validation results — {cfg.backbone} ({cfg.n_frames} frames, iter {cfg.iteration}, seed {cfg.seed})\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(report_str)
        fh.write(f"\nOverall accuracy        : {acc:.4f}\n")
        fh.write(f"F-beta (beta={cfg.beta}, class 1): {fbeta_1:.4f}\n")
        fh.write(f"AUROC                   : {auroc:.4f}\n\n")
        fh.write("Per-class breakdown:\n")
        for i, name in enumerate(["No Heartbeat (0)", "Heartbeat (1)"]):
            fh.write(
                f"  {name:<22}  "
                f"precision={prec[i]:.4f}  "
                f"recall={rec[i]:.4f}  "
                f"f1={f1[i]:.4f}  "
                f"support={int(support[i])}\n"
            )
    log.info("Classification report saved to: %s", report_path)

    # ── Machine-readable copy ──────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=4)
    log.info("Metrics JSON saved to: %s", metrics_path)

    return metrics


def load_model(ckpt_path: str, device: str = CFG.device) -> nn.Module:
    """Load a saved checkpoint and return an eval-mode model."""
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = R2Plus1DClassifier(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(
        "Loaded checkpoint from epoch %d  (val_loss=%.4f  val_acc=%.3f)",
        ckpt["epoch"], ckpt["val_loss"], ckpt["val_acc"],
    )
    return model

def save_training_config(config: TrainConfig, out_dir: str) -> None:
    """Save the training config as a JSON file in the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "training_config.json")
    with open(config_path, "w") as fh:
        json.dump(vars(config), fh, indent=4)
    log.info("Training config saved to: %s", config_path)

if __name__ == "__main__":
    train(CFG)