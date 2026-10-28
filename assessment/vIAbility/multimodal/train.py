"""
multimodal_train.py
────────────────────────────────────────────────────────────────────────────────
Multimodal (R(2+1)D-18 video embeddings + tabular clinical features) binary
classifier for fetal heartbeat detection.

NOTE on backbone: as currently written, MultimodalClassifier is hard-wired to
an R(2+1)D-18 video encoder (video_embed_dim=512 is R(2+1)D-18's penultimate
feature size, and there's no backbone switch in the model). This refactor
makes the experiment REPEATABLE (n_frames / iteration / seed are now
parameters) and logs every run's metrics, but it does NOT add an MViT path —
that requires seeing model.py so the encoder swap + embed-dim change can be
wired in without guessing at the architecture. See run_experiments_multimodal.py
for the active sweep (R(2+1)D @ 15/16/20/30/60 frames × 5 repeats).
────────────────────────────────────────────────────────────────────────────────
"""

import csv
import json
import os
import time
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from matplotlib import pyplot as plt

from sklearn.metrics import (
    fbeta_score, precision_recall_fscore_support, accuracy_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.multimodal.dataloader import get_dataloaders, mixup_collate
from assessment.vIAbility.multimodal.model import MultimodalClassifier
from assessment.vIAbility.multimodal.eval import evaluate as external_evaluate
from config.config import PATH_DATASET, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S",)
log = logging.getLogger(__name__)

# Master CSV — one row per run, across every n_frames / iteration combo.
METRICS_CSV = "results/assessment/vIAbility/multimodal/experiment_results.csv"


@dataclass
class TrainConfig:

    VIDEO_DIR: str  = os.path.join(PATH_DATASET, "Oslo_viability/videos")
    EXCEL_PATH: str = os.path.join(PATH_DATASET, "Oslo_viability/all_patient_morphk.csv")

    class_names: list[str] = field(default_factory=lambda: ["No Heartbeat (0)", "Heartbeat (1)"])

    frame_h: int     = 224
    frame_w: int     = 224
    n_frames: int    = 15
    num_workers: int = 4
    tabular_dim: int = 13

    # backbone is informational/for naming only right now — MultimodalClassifier
    # only implements r2plus1d_18. Kept as a field so the results CSV schema
    # already matches the other experiment's CSV, ready for when MViT is added.
    backbone: str = "r2plus1d_18"

    # ── Repetition bookkeeping ────────────────────────────────────────────
    iteration: int   = 0
    seed: int        = SEED
    metrics_csv: str = METRICS_CSV

    # ── Two-stage fine-tuning ─────────────────────────────────────────────
    weight_loss: float  = 1.48

    freeze_epochs: int  = 5
    warmup_epochs: int  = 3

    lr_head: float      = 3e-4
    lr_backbone: float  = 1e-5

    weight_decay: float = 1e-4
    beta: float         = 0.5

    save_by: str        = "val_fbeta"

    # ── Architecture ─────────────────────────────────────────────────────────
    video_embed_dim: int   = 512   # R(2+1)D-18 penultimate feature size
    tabular_hidden_dim: int = 128  # tabular encoder output dim
    # fusion_dim = video_embed_dim + tabular_hidden_dim  (computed in model)

    # ── Regularisation ────────────────────────────────────────────────────
    label_smoothing: float = 0.2
    dropout_head: float    = 0.5
    dropout_mid: float     = 0.3

    # ── Training budget ───────────────────────────────────────────────────
    epochs: int     = 60
    batch_size: int = 16

    # ── Early stopping ────────────────────────────────────────────────────
    patience: int    = 10
    min_delta: float = 1e-2

    # ── Data augmentation ─────────────────────────────────────────────────
    augmentation: bool = True
    oversample: bool   = False
    factor1: float     = 2.0
    factor0: float     = 1.5
    mixup: bool       = False

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir: str  = "models/assessment/vIAbility/multimodal/"
    ckpt_name: str = field(init=False)

    # ── Evaluation outputs ────────────────────────────────────────────────
    out_dir: str = "output/assessment/vIAbility/multimodal"

    # ── Misc ─────────────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: ("cuda:1" if torch.cuda.is_available() else "cpu"))
    amp: bool = False

    def __post_init__(self):
        self.ckpt_name = f"best_multimodal_{self.backbone}_{self.n_frames}_iter{self.iteration}.pt"


CFG = TrainConfig()


class EarlyStopping:
    def __init__(self, patience: int = CFG.patience, min_delta: float = CFG.min_delta):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
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


def make_smooth_bce(cfg: TrainConfig):
    """Returns a BCE-with-label-smoothing loss function bound to *cfg*."""
    def smooth_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps            = cfg.label_smoothing
        targets_smooth = targets * (1.0 - eps) + 0.5 * eps
        pos_weight     = (torch.tensor([cfg.weight_loss], device=logits.device) if torch.cuda.is_available() else None)
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth, weight=pos_weight)
    return smooth_bce


def linear_warmup_factor(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / warmup_epochs)


def run_epoch(cfg: TrainConfig, model: nn.Module, loader: DataLoader, criterion, optimizer: Optional[torch.optim.Optimizer],
              scaler: Optional[GradScaler], device: str, training: bool,) -> tuple[float, float, float]:
    """One full pass through *loader*. Returns (avg_loss, accuracy, fbeta)."""
    model.train() if training else model.eval()

    total_loss, correct, total, fbeta_sum = 0.0, 0, 0, 0.0
    amp_enabled = cfg.amp and device != "cpu"

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for frames, mask, tabular, labels in loader:
            frames  = frames.to(device)
            mask    = mask.to(device)
            tabular = tabular.to(device)
            labels  = labels.to(device)

            with autocast("cuda", enabled=amp_enabled):
                logits = model(frames, mask, tabular)
                loss   = criterion(logits.view(-1), labels.float())


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
            preds       = (torch.sigmoid(logits.view(-1)) >= 0.5).long()
            correct    += (preds == labels).sum().item()
            total      += frames.size(0)
            fbeta_sum  += fbeta_score(labels.cpu().numpy().astype(int), preds.cpu().numpy().astype(int), pos_label=1, beta=cfg.beta, zero_division=0,)

    return total_loss / total, correct / total, fbeta_sum / total


def save_training_config(config: TrainConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "training_config.json")
    with open(config_path, "w") as fh:
        json.dump(vars(config), fh, indent=4)
    log.info("Training config saved to: %s", config_path)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics logging — independent of the external eval.py, since we don't know
# what (if anything) that module returns. Computed fresh here so the master
# CSV always has accuracy / precision / recall / f1 / F-beta(0.5) / AUROC.
# ─────────────────────────────────────────────────────────────────────────────
def compute_full_metrics(model: nn.Module, loader: DataLoader, device: str, cfg: TrainConfig) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for frames, mask, tabular, labels in loader:
            frames  = frames.to(device)
            mask    = mask.to(device)
            tabular = tabular.to(device)
            labels  = labels.to(device)

            logits = model(frames, mask, tabular).view(-1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true  = np.array(all_labels, dtype=int)
    y_pred  = np.array(all_preds,  dtype=int)
    y_probs = np.array(all_probs,  dtype=float)

    acc                    = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    fbeta_1                = fbeta_score(y_true, y_pred, pos_label=1, beta=cfg.beta, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_probs)
    except ValueError:
        log.warning("AUROC undefined (only one class present in this validation split) — recording NaN.")
        auroc = float("nan")

    log.info(
        "Eval — acc: %.4f | P0/R0/F1_0: %.4f/%.4f/%.4f | P1/R1/F1_1: %.4f/%.4f/%.4f | F-beta(%.2f): %.4f | AUROC: %.4f",
        acc, prec[0], rec[0], f1[0], prec[1], rec[1], f1[1], cfg.beta, fbeta_1, auroc,
    )

    return {
        "accuracy":    acc,
        "precision_0": prec[0], "recall_0": rec[0], "f1_0": f1[0],
        "precision_1": prec[1], "recall_1": rec[1], "f1_1": f1[1],
        f"fbeta{cfg.beta}_1": fbeta_1,
        "auroc":       auroc,
    }


def append_metrics_row(csv_path: str, row: dict) -> None:
    """Append one run's metrics row, creating the file+header on first call."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: Optional[TrainConfig] = None):
    cfg = cfg or CFG

    # ── Seed THIS run — repeats need different seeds or they're identical ──
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device
    log.info(
        "Backbone: %s | n_frames: %d | iteration: %d | seed: %d | Device: %s | AMP: %s",
        cfg.backbone, cfg.n_frames, cfg.iteration, cfg.seed, device, cfg.amp and device != "cpu",
    )

    log.info("Building dataset …")
    train_loader, val_loader = get_dataloaders(
        video_dir=cfg.VIDEO_DIR, excel_path=cfg.EXCEL_PATH, frame_h=cfg.frame_h, frame_w=cfg.frame_w,
        n_frames=cfg.n_frames, augment=cfg.augmentation, oversample=cfg.oversample, factor1=cfg.factor1,
        factor0=cfg.factor0, mixup=cfg.mixup, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        device=device, log=log,
    )
    log.info("Train: %d  |  Val: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = MultimodalClassifier(
        tabular_dim=cfg.tabular_dim, tabular_hidden=cfg.tabular_hidden_dim, video_embed_dim=cfg.video_embed_dim,
        num_classes=1, pretrained=True, dropout_head=cfg.dropout_head, dropout_mid=cfg.dropout_mid,
    ).to(device)

    # STAGE 1: freeze backbone
    model.freeze_backbone()
    log.info(
        "Stage 1 — backbone frozen. Trainable params: %.2fM / %.2fM",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    criterion = make_smooth_bce(cfg)

    optimizer = torch.optim.AdamW(model.head_parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    scaler    = GradScaler(device=device) if (cfg.amp and device == "cuda") else None
    stopper   = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # Per-run checkpoint dir keyed by iteration so repeats never collide.
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.backbone, str(cfg.n_frames), f"iter{cfg.iteration}")
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_path = os.path.join(ckpt_path, cfg.ckpt_name)

    best_val = float("inf") if cfg.save_by == "val_loss" else 0.0
    loss_hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_fbeta': [], 'val_fbeta': []}

    saved_checkpoint = False
    stage = 1

    log.info("Starting training for up to %d epochs …", cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        train_loader.collate_fn = (mixup_collate if epoch < (cfg.epochs - 10) else default_collate)

        if epoch == cfg.freeze_epochs + 1:
            stage = 2
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                [{"params": model.head_parameters(), "lr": cfg.lr_head},
                 {"params": model.backbone_parameters(), "lr": cfg.lr_backbone}],
                weight_decay=cfg.weight_decay,
            )
            stopper = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
            log.info("── Stage 2: backbone unfrozen. head lr=%.0e  backbone lr=%.0e ──", cfg.lr_head, cfg.lr_backbone)

        stage_epoch = epoch if stage == 1 else epoch - cfg.freeze_epochs
        warmup      = linear_warmup_factor(stage_epoch - 1, cfg.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])

        train_loss, train_acc, train_fbeta = run_epoch(cfg, model, train_loader, criterion, optimizer, scaler, device, training=True)
        val_loss, val_acc, val_fbeta = run_epoch(cfg, model, val_loader, criterion, None, None, device, training=False)
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %3d/%d [S%d]  |  train loss: %.4f  acc: %.3f  fbeta: %.3f  |  val loss: %.4f  acc: %.3f  fbeta: %.3f  |  lr: %.2e",
            epoch, cfg.epochs, stage, train_loss, train_acc, train_fbeta, val_loss, val_acc, val_fbeta, current_lr,
        )

        loss_hist['train_loss'].append(train_loss)
        loss_hist['val_loss'].append(val_loss)
        loss_hist['train_acc'].append(train_acc)
        loss_hist['val_acc'].append(val_acc)
        loss_hist['train_fbeta'].append(train_fbeta)
        loss_hist['val_fbeta'].append(val_fbeta)

        if cfg.save_by == "val_loss":
            metric_to_check = val_loss
        elif cfg.save_by == "val_acc":
            metric_to_check = val_acc
        elif cfg.save_by == "val_fbeta":
            metric_to_check = val_fbeta
        else:
            raise ValueError(f"Invalid save_by metric: {cfg.save_by}")

        is_best = (metric_to_check < best_val if cfg.save_by == "val_loss" else metric_to_check > best_val)
        if is_best:
            best_val = metric_to_check
            torch.save(
                {
                    "epoch": epoch, "stage": stage, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss, "train_acc": train_acc, "train_fbeta": train_fbeta,
                    "val_loss": val_loss, "val_acc": val_acc, "val_fbeta": val_fbeta, "config": vars(cfg),
                },
                ckpt_path,
            )
            saved_checkpoint = True
            log.info("  ✓ Saved best checkpoint  (val_%s=%.4f)", cfg.save_by, best_val)

        if stopper.step(val_loss):
            log.info("Early stopping triggered at epoch %d (stage %d).", epoch, stage)
            break

    if not saved_checkpoint:
        log.warning("No checkpoint saved during training — saving last epoch.")
        torch.save(
            {
                "epoch": epoch, "stage": stage, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "train_loss": train_loss, "train_acc": train_acc, "train_fbeta": train_fbeta,
                "val_loss": val_loss, "val_acc": val_acc, "val_fbeta": val_fbeta, "config": vars(cfg),
            },
            ckpt_path,
        )

    log.info("Training complete. Best val %s: %.4f", cfg.save_by, best_val)
    log.info("Best checkpoint saved to: %s", ckpt_path)

    def save_training_plots(out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_hist['train_loss'], label="Train Loss")
        plt.plot(loss_hist['val_loss'], label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Training and Validation Loss"); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(loss_hist['train_acc'], label="Train Accuracy")
        plt.plot(loss_hist['val_acc'], label="Val Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy"); plt.legend()
        plt.tight_layout()
        curves_path = os.path.join(out_dir, "training_curves.png")
        plt.savefig(curves_path)
        plt.show()
        plt.close()
        log.info("Training curves saved to: %s", curves_path)

    # Per-run output dir keyed by iteration so repeats don't overwrite.
    out_dir = os.path.join(cfg.out_dir, cfg.backbone, str(cfg.n_frames), f"iter{cfg.iteration}")
    save_training_plots(out_dir)
    save_training_config(cfg, out_dir)

    log.info("Loading best checkpoint for evaluation …")
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model"])

    # Preserve existing behaviour: still produce whatever plots/report eval.py makes.
    external_evaluate(model, val_loader, device, out_dir, class_names=cfg.class_names, log=log)

    # Independent metrics computation for the master results CSV.
    metrics = compute_full_metrics(model, val_loader, device, cfg)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=4)

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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


if __name__ == "__main__":
    train(CFG)