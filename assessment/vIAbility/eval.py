"""
evaluate.py
────────────────────────────────────────────────────────────────────────────────
Architecture-agnostic evaluation script.

Usage
-----
python evaluate.py --ckpt checkpoints/best_r2plus1d.pt [--batch_size 4]

The checkpoint must contain:
    {
        "model":   <state_dict>,
        "config":  <TrainConfig dataclass or plain dict>,   # optional but recommended
        ...
    }

The model architecture is re-instantiated from the checkpoint's "config" field.
If your checkpoint was saved with a different structure, see the NOTE in
`load_model_from_ckpt()` below.
────────────────────────────────────────────────────────────────────────────────
"""
import os
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data.dataset import load_dataset
from assessment.vIAbility.models.cnn_3d import VideoDataset,R2Plus1DClassifier, CFG   # swap CFG / model class as needed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import SEED, PATH_DATASET


PATH_DATASET = os.path.join(PATH_DATASET, "Oslo_processed/dataset_30.pt")
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
torch.manual_seed(SEED)
np.random.seed(SEED)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1.  Checkpoint loader  (architecture-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def load_model_from_ckpt(
    ckpt_path: str,
    model_class: type = R2Plus1DClassifier,
    device: str = "cpu",
    **model_kwargs,
) -> tuple[nn.Module, dict]:
    """
    Load *any* model from a checkpoint file.
 
    Parameters
    ----------
    ckpt_path   : path to the .pt / .pth checkpoint
    model_class : the nn.Module subclass to instantiate
                  → swap this to evaluate a different architecture
    device      : 'cuda' or 'cpu'
    **model_kwargs : forwarded to model_class(...)
                  e.g. num_classes=1, pretrained=False
 
    Returns
    -------
    model : nn.Module in eval mode, weights loaded
    meta  : dict with epoch / val_loss / val_acc if present in checkpoint
    """
    # ── PyTorch 2.6+: weights_only=True is the new default.
    # Checkpoints saved with a dataclass (e.g. TrainConfig) require either:
    #   (a) allowlisting the class, or
    #   (b) falling back to weights_only=False (safe when you own the file).
    # We try (a) first; if the config class isn't importable we fall back to (b).
    try:
        from train import TrainConfig          # allowlist our own dataclass
        torch.serialization.add_safe_globals([TrainConfig])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        log.warning(
            "weights_only=True load failed — retrying with weights_only=False. "
            "Only do this with checkpoints you trust."
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
 
    # ── Reconstruct model ──────────────────────────────────────────────────
    # Priority: kwargs passed by caller > config stored in checkpoint
    if not model_kwargs:
        cfg = ckpt.get("config", None)
        if cfg is not None:
            # Support both dataclass and plain dict
            if hasattr(cfg, "__dict__"):
                cfg = vars(cfg)
            model_kwargs = {k: cfg[k] for k in cfg if k in _model_init_params(model_class)}
 
    model = model_class(**model_kwargs)
 
    # ── Load weights ────────────────────────────────────────────────────────
    # NOTE: if your checkpoint stores the full model (torch.save(model, ...))
    # instead of state_dict, replace the two lines below with:
    #   model = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)   # fall back to raw dict if no "model" key
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
 
    meta = {k: ckpt[k] for k in ("epoch", "val_loss", "val_acc") if k in ckpt}
    if meta:
        log.info(
            "Checkpoint loaded │ epoch=%d  val_loss=%.4f  val_acc=%.3f",
            meta.get("epoch", -1),
            meta.get("val_loss", float("nan")),
            meta.get("val_acc", float("nan")),
        )
    else:
        log.info("Checkpoint loaded (no training metadata found).")
 
    return model, meta
 
 
def _model_init_params(cls: type) -> set[str]:
    """Return the parameter names accepted by cls.__init__."""
    import inspect
    return set(inspect.signature(cls.__init__).parameters) - {"self"}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2.  Inference pass
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over the full loader.
 
    Returns
    -------
    labels  : (N,)  int   ground-truth
    probs   : (N,)  float sigmoid probabilities
    preds   : (N,)  int   hard predictions at threshold 0.5
    """
    all_labels, all_probs = [], []
 
    for frames, mask, labels in loader:
        frames = frames.to(device)
        mask   = mask.to(device)
 
        # ── Forward pass ────────────────────────────────────────────────────
        # Works with any model that accepts (frames, mask) or just (frames).
        try:
            logits = model(frames, mask)
        except TypeError:
            logits = model(frames)          # architecture doesn't use mask
 
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)
 
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)
    preds  = (probs >= 0.5).astype(int)
    return labels, probs, preds
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3.  Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict:
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
 
    metrics = {
        "accuracy":    accuracy_score(labels, preds),
        "roc_auc":     roc_auc_score(labels, probs),
        "f1":          f1_score(labels, preds, zero_division=0),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,   # recall / TPR
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,   # TNR
        "ppv":         tp / (tp + fp) if (tp + fp) > 0 else 0.0,   # precision
        "npv":         tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "confusion_matrix": cm,
    }
    return metrics
 
 
def print_metrics(metrics: dict) -> None:
    log.info("─" * 52)
    log.info("  Accuracy     : %.4f", metrics["accuracy"])
    log.info("  ROC-AUC      : %.4f", metrics["roc_auc"])
    log.info("  F1 Score     : %.4f", metrics["f1"])
    log.info("  Sensitivity  : %.4f  (recall / TPR)", metrics["sensitivity"])
    log.info("  Specificity  : %.4f  (TNR)",           metrics["specificity"])
    log.info("  PPV          : %.4f  (precision)",     metrics["ppv"])
    log.info("  NPV          : %.4f",                  metrics["npv"])
    log.info("─" * 52)
    log.info("Classification report:\n%s", classification_report(
        *_labels_preds_from_metrics(metrics), target_names=["No HB", "HB"]
    ))
 
 
def _labels_preds_from_metrics(_):
    # placeholder: classification_report is called with real arrays in evaluate()
    pass

def save_metrics(metrics: dict, save_dir: str = "eval_results") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "metrics.txt"
    with open(out, "w") as f:
        for k, v in metrics.items():
            if k != "confusion_matrix":
                f.write(f"{k}: {v:.4f}\n")
    log.info("Metrics saved to %s", out)
 
# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(
    labels:  np.ndarray,
    probs:   np.ndarray,
    preds:   np.ndarray,
    metrics: dict,
    save_dir: str = "eval_results",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
 
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
 
    # ── Confusion matrix ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    cm  = metrics["confusion_matrix"]
    im  = ax0.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_xticks([0, 1]); ax0.set_xticklabels(["No HB", "HB"])
    ax0.set_yticks([0, 1]); ax0.set_yticklabels(["No HB", "HB"])
    ax0.set_xlabel("Predicted"); ax0.set_ylabel("True")
    ax0.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=14, fontweight="bold")
 
    # ── ROC curve ─────────────────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(labels, probs)
    ax1.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
 
    # ── Probability distribution ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.hist(probs[labels == 0], bins=20, alpha=0.6, label="No HB (0)", color="steelblue")
    ax2.hist(probs[labels == 1], bins=20, alpha=0.6, label="HB (1)",    color="tomato")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution")
    ax2.legend()
 
    out = Path(save_dir) / "evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Plots saved to %s", out)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(ckpt_path: str, batch_size: int = CFG.batch_size, save_dir: str = "eval_results") -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
 
    # ── Load model ────────────────────────────────────────────────────────
    model, meta = load_model_from_ckpt(
        ckpt_path,
        model_class=R2Plus1DClassifier,   # ← swap for any other architecture
        device=device,
        num_classes=1,
        pretrained=False,
    )
 
    # ── Build test set ────────────────────────────────────────────────────
    log.info("Building dataset …")
    ds       = load_dataset(PATH_DATASET, map_location='cpu')
    test_set = VideoDataset(ds["X_test"], ds["mask_test"], ds["y_test"])
    loader   = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device == "cuda"),
    )
    log.info("Test samples: %d", len(test_set))
 
    # ── Inference ─────────────────────────────────────────────────────────
    log.info("Running inference …")
    labels, probs, preds = run_inference(model, loader, device)
 
    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = compute_metrics(labels, probs, preds)
 
    log.info("─" * 52)
    log.info("  Accuracy     : %.4f", metrics["accuracy"])
    log.info("  ROC-AUC      : %.4f", metrics["roc_auc"])
    log.info("  F1 Score     : %.4f", metrics["f1"])
    log.info("  Sensitivity  : %.4f  (recall / TPR)", metrics["sensitivity"])
    log.info("  Specificity  : %.4f  (TNR)",           metrics["specificity"])
    log.info("  PPV          : %.4f  (precision)",     metrics["ppv"])
    log.info("  NPV          : %.4f",                  metrics["npv"])
    log.info("─" * 52)
    log.info(
        "Classification report:\n%s",
        classification_report(labels, preds, target_names=["No HB", "HB"]),
    )
    save_metrics(metrics, save_dir)
 
    # ── Plots ─────────────────────────────────────────────────────────────
    plot_results(labels, probs, preds, metrics, save_dir)
 
    return metrics
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a binary video classifier.")
    parser.add_argument("--ckpt",       required=True,        help="Path to checkpoint .pt file")
    parser.add_argument("--batch_size", type=int, default=4,  help="Inference batch size")
    parser.add_argument('--save_dir',  type=str, default='output/assessment/vIAbility/', help='Directory to save evaluation plots')
    args = parser.parse_args()
 
    evaluate(ckpt_path=args.ckpt, batch_size=args.batch_size, save_dir=args.save_dir)