#!/usr/bin/env python3

"""
Evaluation script for multitask blastocyst grading model.

Runs inference on the labeled test set and saves per-head:
  - Confusion matrix (PNG)
  - Classification report (txt)
  - Summary metrics CSV

Post-processing rule:
    If predicted Expansion ∈ NOT_GRADABLE_EXP → TE and ICM forced to MASK_LABEL.
    Masked samples (ground-truth MASK_LABEL) are excluded from TE/ICM metrics.

Usage:
    python evaluate.py                               # auto-detect checkpoint
    python evaluate.py --checkpoint path/to/file.pt  # explicit
    python evaluate.py --output results/run1          # custom output folder
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET

from train import (
    MultitaskBlastoModel,
    build_kg_token_ids,
    DATA_MAP,
    KG_TRIPLES, KG_EMBED_DIM, KG_MAX_LEN,
    IMG_SIZE, CKPT_DIR,
    NOT_GRADABLE_EXP, MASK_LABEL,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_best_checkpoint(ckpt_dir: Path) -> Path:
    candidates = list(ckpt_dir.glob("best_mean*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No 'best_mean*.pt' checkpoint found in {ckpt_dir}.\n"
            "Pass --checkpoint explicitly."
        )
    def _acc(p: Path) -> float:
        try:
            return float(p.stem.split("best_mean")[1].split("_")[0])
        except Exception:
            return -1.0
    best = max(candidates, key=_acc)
    print(f"Auto-selected checkpoint: {best}")
    return best


# ---------------------------------------------------------------------------
# Test dataset  (labeled — returns image + all 3 ground-truth labels)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, transform=None):
        self.df         = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(self.image_root / row["Image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        gt_exp = int(row[DATA_MAP["exp"]["test"]])   # NaN rows already dropped in main

        # TE / ICM: NaN → MASK_LABEL (same treatment as "not gradable")
        raw_te  = row[DATA_MAP["te"]["test"]]
        raw_icm = row[DATA_MAP["icm"]["test"]]
        gt_te   = MASK_LABEL if pd.isna(raw_te) or isinstance(raw_te,str) else int(raw_te)
        gt_icm  = MASK_LABEL if pd.isna(raw_icm) or isinstance(raw_icm,str) else int(raw_icm)

        return image, gt_exp, gt_te, gt_icm


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Inference — returns arrays of predictions and ground truths
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    all_gt_exp,  all_pred_exp  = [], []
    all_gt_te,   all_pred_te   = [], []
    all_gt_icm,  all_pred_icm  = [], []

    for images, gt_exp, gt_te, gt_icm in loader:
        images = images.to(device)
        lo_exp, lo_te, lo_icm = model(images)

        pred_exp = lo_exp.argmax(1).cpu().tolist()
        pred_te  = lo_te.argmax(1).cpu().tolist()
        pred_icm = lo_icm.argmax(1).cpu().tolist()

        gt_exp = gt_exp.tolist()
        gt_te  = gt_te.tolist()
        gt_icm = gt_icm.tolist()

        for i in range(len(pred_exp)):
            exp_pred = pred_exp[i]

            # Post-processing: force MASK_LABEL if expansion not gradable
            forced   = exp_pred in NOT_GRADABLE_EXP
            te_pred  = MASK_LABEL if forced else pred_te[i]
            icm_pred = MASK_LABEL if forced else pred_icm[i]

            all_gt_exp.append(gt_exp[i])
            all_pred_exp.append(exp_pred)
            all_gt_te.append(gt_te[i])
            all_pred_te.append(te_pred)
            all_gt_icm.append(gt_icm[i])
            all_pred_icm.append(icm_pred)

    return (
        np.array(all_gt_exp),  np.array(all_pred_exp),
        np.array(all_gt_te),   np.array(all_pred_te),
        np.array(all_gt_icm),  np.array(all_pred_icm),
    )


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, classes, title, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(len(classes) + 2, len(classes) + 2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(title,        fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def evaluate_head(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        head_name: str,
        classes: list,
        output_dir: Path,
        mask_val: int = None,
) -> dict:
    """
    Compute and save metrics for one head.
    If mask_val is set, samples where y_true == mask_val are excluded.
    """
    if mask_val is not None:
        valid        = y_true != mask_val
        y_true       = y_true[valid]
        y_pred       = y_pred[valid]
        eval_classes = [c for c in classes if c != mask_val]
        note         = f" (excluding ground-truth class {mask_val})"
    else:
        eval_classes = classes
        note         = ""

    print(f"\n{'='*60}")
    print(f"  {head_name}{note}")
    print(f"{'='*60}")

    report = classification_report(
        y_true, y_pred,
        labels=eval_classes,
        zero_division=0,
    )
    print(report)

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")

    # Save classification report as txt
    report_path = output_dir / f"report_{head_name.lower()}.txt"
    with open(report_path, "w") as f:
        f.write(f"{head_name}{note}\n\n")
        f.write(report)
        f.write(f"\nAccuracy         : {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
    print(f"  Saved: {report_path.name}")

    # Save confusion matrix
    save_confusion_matrix(
        y_true, y_pred,
        classes=eval_classes,
        title=f"Confusion Matrix — {head_name}{note}",
        save_path=output_dir / f"cm_{head_name.lower()}.png",
    )

    return {"head": head_name, "accuracy": acc, "balanced_accuracy": bal_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                find_best_checkpoint(CKPT_DIR)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dataset_path = Path(PATH_DATASET) / "Blastocyst_Dataset"
    test_csv     = pd.read_csv(
        dataset_path / "Gardner_test_gold_onlyGardnerScores.csv", sep=";")
    image_root   = dataset_path / "Images"

    # Drop rows where Expansion label is NaN — we cannot evaluate without it.
    # TE / ICM NaNs are handled inside the Dataset (__getitem__) as MASK_LABEL.
    exp_col  = DATA_MAP["exp"]["test"]
    n_before = len(test_csv)
    test_csv = test_csv.dropna(subset=[exp_col]).reset_index(drop=True)
    n_dropped = n_before - len(test_csv)
    if n_dropped:
        print(f"Dropped {n_dropped} rows with NaN in '{exp_col}'")
    print(f"Test samples: {len(test_csv)}")

    test_ds = TestDataset(test_csv, image_root, get_test_transform())
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    token_ids, vocab_size = build_kg_token_ids(KG_TRIPLES, KG_MAX_LEN)
    model = MultitaskBlastoModel(
        vocab_size=vocab_size,
        token_ids=token_ids,
        embed_dim=KG_EMBED_DIM,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded weights from: {ckpt_path}")

    # Inference
    print("\nRunning inference...")
    gt_exp, pred_exp, gt_te, pred_te, gt_icm, pred_icm = \
        run_inference(model, test_loader, device)

    n_forced = sum(1 for p in pred_exp if p in NOT_GRADABLE_EXP)
    print(f"Forced mask ({MASK_LABEL}): {n_forced} samples "
          f"({100*n_forced/len(pred_exp):.1f}%)")

    # Per-head evaluation
    # TE and ICM: append MASK_LABEL to the class list so the CM axis shows it,
    # but it is excluded from metrics inside evaluate_head via mask_val
    exp_classes = DATA_MAP["exp"]["classes"]
    te_classes  = DATA_MAP["te"]["classes"] + [MASK_LABEL]
    icm_classes = DATA_MAP["icm"]["classes"] + [MASK_LABEL]

    results = []
    results.append(evaluate_head(
        gt_exp, pred_exp,
        head_name="Expansion",
        classes=exp_classes,
        output_dir=output_dir,
        mask_val=None,
    ))
    results.append(evaluate_head(
        gt_te, pred_te,
        head_name="TE",
        classes=te_classes,
        output_dir=output_dir,
        mask_val=MASK_LABEL,
    ))
    results.append(evaluate_head(
        gt_icm, pred_icm,
        head_name="ICM",
        classes=icm_classes,
        output_dir=output_dir,
        mask_val=MASK_LABEL,
    ))

    # Summary CSV
    summary = pd.DataFrame(results)
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multitask blastocyst grading — evaluation")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--output",      type=str,
                        default="results/evaluation",
                        help="Folder where CMs, reports and summary are saved")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    main(args)