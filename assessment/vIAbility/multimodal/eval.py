import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score, fbeta_score,)

def evaluate(model: nn.Module, loader: DataLoader, device: str, out_dir: str, class_names: list, log = None) -> None:
    if log is None:
        import logging
        log = logging.getLogger(__name__)
    
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for frames, mask, tabular, labels in loader:
            frames  = frames.to(device)
            mask    = mask.to(device)
            tabular = tabular.to(device)

            logits = model(frames, mask, tabular).view(-1)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).long()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds,  dtype=int)
    end_time = time.time()
    log.info("Evaluation completed in %.4f seconds", end_time - start_time)

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Validation Set (Multimodal)")
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
        target_names=class_names,
        zero_division=0,
    )
    log.info("\n%s", report_str)

    report_path = os.path.join(out_dir, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write("Validation results — Multimodal R(2+1)D + Tabular classifier\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(report_str)
        fh.write(f"\nOverall accuracy : {acc:.4f}\n\n")
        fh.write("Per-class breakdown:\n")
        for i, name in enumerate(class_names):
            fh.write(
                f"  {name:<22}  "
                f"precision={prec[i]:.4f}  "
                f"recall={rec[i]:.4f}  "
                f"f1={f1[i]:.4f}  "
                f"support={int(support[i])}\n"
            )
    log.info("Classification report saved to: %s", report_path)