"""
train.py
────────────────────────────────────────────────────────────────────────────────
R(2+1)D binary classifier for fetal-heartbeat detection.

Architecture  : R(2+1)D  (factorised spatial + temporal convolutions)
Task          : Binary classification  (Fetal_Heart_Beat  0 / 1)
Extras        : Mixed-precision (AMP) · Early stopping · Padded-frame masking
────────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import sys
import logging
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
    precision_recall_fscore_support, accuracy_score, fbeta_score
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.data.dataset import get_dataloaders, mixup_collate
from config.config import PATH_DATASET, SEED


torch.manual_seed(SEED)
np.random.seed(SEED)
VIDEO_DIR = os.path.join(PATH_DATASET, "Oslo_viability/videos")
EXCEL_PATH = os.path.join(PATH_DATASET, "Oslo_viability/Oslo_completo.xlsx")

#PATH_DATASET = os.path.join(PATH_DATASET, f"Oslo_processed/dataset_{n_frames}.pt")

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
    out_dir: str        = "output/assessment/vIAbility/3d_cnn"
 
    # misc
    device: str         = field(default_factory=lambda: ("cuda:2" if torch.cuda.is_available() else "cpu"))
    amp: bool           = False      # mixed-precision; auto-disabled on CPU

    def __post_init__(self):
        self.ckpt_name = f"best_{self.backbone}_{self.n_frames}.pt"
        if self.backbone == "mvit":
            # MViT_V2_S was pretrained on 224×224 frames
            self.frame_h = 224
            self.frame_w = 224
 
 
CFG = TrainConfig()
 

class MViTClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True,
                 dropout_head=CFG.dropout_head, dropout_mid=CFG.dropout_mid):
        super().__init__()
        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        backbone = mvit_v2_s(weights=weights)

        # Strip the original classifier head — find in_features dynamically.
        # MViT_V2_S head is Sequential(LayerNorm, Linear); grab the last Linear.
        head_linear = None
        for m in backbone.head.children():
            if isinstance(m, nn.Linear):
                head_linear = m
        if head_linear is None:
            raise RuntimeError("Could not find Linear layer in MViT head")
        in_features = head_linear.in_features   # 768

        self.backbone = backbone
        backbone.head = nn.Identity()           # remove original head

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_head),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, num_classes),
        )

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return list(self.head.parameters())

    def freeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad_(True)

    def forward(self, x, mask=None):
        # MViT expects (B, C, T, H, W) — same as R(2+1)D, no change needed
        features = self.backbone(x)   # (B, 768)
        return self.head(features).squeeze(1)  # (B,) raw logit
    
class R2Plus1DClassifier(nn.Module):
    """
    Pre-trained R(2+1)D-18 backbone with a custom binary classification head.
 
    The temporal mask is used to zero-out padded frame features before
    the global average pool, so the classifier is never influenced by
    blank padding frames.
    """
 
    def __init__(self, num_classes: int = 1, pretrained: bool = True,
                 dropout_head: float = CFG.dropout_head,
                 dropout_mid:  float = CFG.dropout_mid, backbone_name: str = CFG.backbone):
        super().__init__()
        
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        backbone = r2plus1d_18(weights=weights)
                
        self.stem    = backbone.stem
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.avgpool = backbone.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_head),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_mid),
            nn.Linear(128, num_classes),
            )
 
    def backbone_parameters(self):
        """All parameters that belong to the pretrained backbone."""
        return (
            list(self.stem.parameters())  +
            list(self.layer1.parameters()) +
            list(self.layer2.parameters()) +
            list(self.layer3.parameters()) +
            list(self.layer4.parameters()) +
            list(self.avgpool.parameters())
        )
 
    def head_parameters(self):
        return list(self.head.parameters())
 
    def freeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad_(False)
 
    def unfreeze_backbone(self):
        for p in self.backbone_parameters():
            p.requires_grad_(True)
 
    def forward(
        self,
        x:    torch.Tensor,                 # (B, C, T, H, W)
        mask: Optional[torch.Tensor] = None # (B, T) bool — True = padded
    ) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                  # (B, 512, T', H', W')
 
        # Zero out padded temporal positions before pooling
        if mask is not None:
            # Downsample mask to match feature-map temporal dim
            T_feat = x.shape[2]
            T_orig = mask.shape[1]
            step   = T_orig // T_feat
            # Take every `step`-th mask entry; keep if ALL sub-frames padded
            ds_mask = mask[:, ::step][:, :T_feat]          # (B, T')
            ds_mask = ds_mask.float().unsqueeze(1)          # (B, 1, T')
            ds_mask = ds_mask.unsqueeze(-1).unsqueeze(-1)   # (B,1,T',1,1)
            x = x * (1.0 - ds_mask)                        # zero padded slots
 
        x = self.avgpool(x)
        return self.head(x).squeeze(1)      # (B,)  raw logit
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.  Early stopping
# ─────────────────────────────────────────────────────────────────────────────
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
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler:    Optional[GradScaler],
    device:    str,
    training:  bool,
) -> tuple[float, float]:
    """One full pass through *loader*. Returns (avg_loss, accuracy)."""
    model.train() if training else model.eval()
 
    total_loss, correct, total, fbeta = 0.0, 0, 0, 0.0
    amp_enabled = CFG.amp and device != "cpu"
 
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
            fbeta += fbeta_score(labels.cpu().numpy().astype(int), preds.cpu().numpy().astype(int), pos_label=1, beta=CFG.beta, zero_division=0)

 
    return total_loss / total, correct / total, fbeta / total
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────
def train():
    device = CFG.device
    log.info("Device: %s  |  AMP: %s", device, CFG.amp and device != "cpu")
 
    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Building dataset …")
    # NEW
    train_loader, val_loader = get_dataloaders(
        video_dir   = VIDEO_DIR,
        excel_path  = EXCEL_PATH,
        frame_h     = CFG.frame_h,
        frame_w     = CFG.frame_w,
        n_frames    = CFG.n_frames,
        augment     = CFG.augmentation,
        oversample  = CFG.oversample,
        factor1     = CFG.factor1,
        factor0     = CFG.factor0,
        batch_size  = CFG.batch_size,
        num_workers = CFG.num_workers,
        device      = device,  
    )
 
    log.info("Train: %d  |  Val: %d", len(train_loader.dataset), len(val_loader.dataset))
 
    # ── Model ─────────────────────────────────────────────────────────────────
    if CFG.backbone == "mvit":
        model = MViTClassifier(num_classes=1, pretrained=True).to(device)
    elif CFG.backbone == "r2plus1d_18":
        model = R2Plus1DClassifier(num_classes=1, pretrained=True).to(device)
 
    # Stage 1: freeze backbone — only head is active
    model.freeze_backbone()
    log.info(
        "Stage 1 — backbone frozen. Trainable params: %.2fM / %.2fM",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
 
    # ── Loss: BCEWithLogitsLoss + label smoothing ──────────────────────────────
    # Label smoothing turns hard 0/1 targets into (ε/2, 1-ε/2),
    # preventing the model from becoming overconfident on small datasets.
    def smooth_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = CFG.label_smoothing
        targets_smooth = targets * (1.0 - eps) + 0.5 * eps
        return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth,weight=(torch.tensor([CFG.weight_loss], device=logits.device) if torch.cuda.is_available() else None))
 
    criterion = smooth_bce
 
    # ── Stage-1 optimiser (head only) ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.head_parameters(),
        lr=CFG.lr_head,
        weight_decay=CFG.weight_decay,
    )
    scaler  = GradScaler() if (CFG.amp and device == "cuda") else None
    stopper = EarlyStopping(patience=CFG.patience, min_delta=CFG.min_delta)
    ckpt_path = os.path.join(CFG.ckpt_dir, CFG.backbone , str(CFG.n_frames))
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_path = os.path.join(ckpt_path, CFG.ckpt_name)
    
    if CFG.save_by == "val_loss":
        best_val = float("inf")
    elif CFG.save_by in {"val_acc", "val_fbeta"}:
        best_val = 0.0
    else:
        raise ValueError(f"Invalid save_by metric: {CFG.save_by}")
    TRAIN_LOSS, VALID_LOSS = [], []
    TRAIN_ACC, VALID_ACC = [], []
    TRAIN_fbeta, VALID_fbeta = [], []
    # ── Training loop ─────────────────────────────────────────────────────────
    stage = 1
    log.info("Starting training for up to %d epochs …", CFG.epochs)
    saved_checkpoint = False
 
    for epoch in range(1, CFG.epochs + 1):
        train_loader.collate_fn = mixup_collate if epoch < (CFG.epochs - 10) else default_collate
        # ── Stage transition ───────────────────────────────────────────────
        if epoch == CFG.freeze_epochs + 1:
            stage = 2
            model.unfreeze_backbone()
            # Rebuild optimiser with two param groups (different LRs)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.head_parameters(),     "lr": CFG.lr_head},
                    {"params": model.backbone_parameters(), "lr": CFG.lr_backbone},
                ],
                weight_decay=CFG.weight_decay,
            )
            stopper = EarlyStopping(patience=CFG.patience, min_delta=CFG.min_delta)
            log.info(
                "── Stage 2: backbone unfrozen. "
                "head lr=%.0e  backbone lr=%.0e ──",
                CFG.lr_head, CFG.lr_backbone,
            )
 
        # ── Linear warmup within each stage ───────────────────────────────
        stage_epoch = epoch if stage == 1 else epoch - CFG.freeze_epochs
        warmup      = linear_warmup_factor(stage_epoch - 1, CFG.warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup
        # Store initial_lr on first pass so warmup has a reference
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
 
        train_loss, train_acc, train_fbeta = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, training=True
        )
        val_loss, val_acc, val_fbeta = run_epoch(
            model, val_loader, criterion, None, None, device, training=False
        )
 
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d [S%d]  |  "
            "train loss: %.4f  acc: %.3f  fbeta: %.3f  |  "
            "val loss: %.4f  acc: %.3f  fbeta: %.3f  |  "
            "lr: %.2e",
            epoch, CFG.epochs, stage,
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

        if CFG.save_by == "val_loss":
            metric_to_check = val_loss
        elif CFG.save_by == "val_acc":
            metric_to_check = val_acc
        elif CFG.save_by == "val_fbeta":
            metric_to_check = val_fbeta
        else:
            raise ValueError(f"Invalid save_by metric: {CFG.save_by}")

        if metric_to_check < best_val if CFG.save_by == "val_loss" else metric_to_check > best_val:
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
                    "config":    vars(CFG),   # plain dict — safe with weights_only=True
                },
                ckpt_path,
            )
            saved_checkpoint = True
            log.info(f"  ✓ Saved best checkpoint  (val_{CFG.save_by}={best_val:.4f})")
 
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
                    "config":    vars(CFG),   # plain dict — safe with weights_only=True
                },
                ckpt_path,
            )

    log.info("Training complete. Best val %s: %.4f", CFG.save_by, best_val)
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
        log.info("Training curves saved to: %s", curves_path)

    out_dir = os.path.join(CFG.out_dir, CFG.backbone,str(CFG.n_frames))
    save_training_plots(out_dir)
    save_training_config(CFG, out_dir)

    # ── Load best checkpoint and run full evaluation ───────────────────────
    log.info("Loading best checkpoint for evaluation …")
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model"])
    evaluate(model, val_loader, device, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader,
             device: str, out_dir: str) -> None:
    """
    Run inference on *loader* with the best checkpoint, then save to *out_dir*:
      - confusion_matrix.png
      - classification_report.txt   (precision / recall / F1 / accuracy)
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

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds,  dtype=int)

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
    log.info("Confusion matrix saved to: %s", cm_path)

    # ── Classification report ─────────────────────────────────────────────
    
    acc                    = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=["No Heartbeat (0)", "Heartbeat (1)"],
        zero_division=0,
    )

    log.info("\n%s", report_str)

    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write("Validation results — R(2+1)D binary classifier\n")
        fh.write("=" * 52 + "\n\n")
        fh.write(report_str)
        fh.write(f"\nOverall accuracy : {acc:.4f}\n\n")
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
        import json
        json.dump(vars(config), fh, indent=4)
    log.info("Training config saved to: %s", config_path)

if __name__ == "__main__":
    train()