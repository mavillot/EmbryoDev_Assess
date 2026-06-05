#!/usr/bin/env python3

"""
Evaluation script for blastocyst quality model.

Runs inference on the test set and saves predictions to a CSV with columns:
    ID, Class

Usage:
    # Auto-detect best checkpoint in default folder:
    python evaluate.py

    # Explicit checkpoint:
    python evaluate.py --checkpoint models/markers/blastocyst_quality/best_val0.8912_ep34.pt

    # Full options:
    python evaluate.py --checkpoint <path> --output results/predictions.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import PATH_DATASET

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

DEFAULT_CKPT_DIR = Path("models/markers/blastocyst_quality")


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """
    Scan ckpt_dir for files matching best_val*.pt and return the one
    with the highest val accuracy encoded in the filename.
    """
    candidates = list(ckpt_dir.glob("best_val*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files matching 'best_val*.pt' found in {ckpt_dir}.\n"
            f"Pass --checkpoint <path> explicitly."
        )

    def _acc_from_name(p: Path) -> float:
        # filename format: best_val{acc:.4f}_ep{epoch}.pt
        try:
            return float(p.stem.split("best_val")[1].split("_ep")[0])
        except Exception:
            return -1.0

    best = max(candidates, key=_acc_from_name)
    print(f"Auto-selected checkpoint: {best}")
    return best


# ---------------------------------------------------------------------------
# Dataset (test — no labels)
# ---------------------------------------------------------------------------

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_root / row["Image"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(row["ID"])


# ---------------------------------------------------------------------------
# Transform (same val transform used during training)
# ---------------------------------------------------------------------------

def get_val_transform(val_crop=299, val_resize=342):
    return transforms.Compose([
        transforms.Resize(val_resize, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(val_crop),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name, checkpoint_path, num_classes, device):
    constructor = models.__dict__[model_name]
    model = constructor(weights=None)       # architecture only, no pretrained weights

    # Replace head — mirrors train_best.py build_model()
    if model_name.startswith("inception"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
        model.AuxLogits = None
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(
                model.classifier.in_features, num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded weights from: {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    all_ids, all_preds, all_probs = [], [], []

    for images, ids in loader:
        images = images.to(device)
        outputs = model(images)                         # (B, num_classes)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_ids.extend(ids)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())   # prob of class 1 (poor)

    return all_ids, all_preds, all_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- resolve checkpoint ----
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        ckpt_path = find_best_checkpoint(DEFAULT_CKPT_DIR)

    # ---- data ----
    dataset_path = Path(PATH_DATASET) / "hvwc23"
    test_csv = pd.read_csv(dataset_path / "test.csv")
    print(f"\nTest set: {len(test_csv)} samples")

    image_root = dataset_path / "test"
    transform  = get_val_transform()

    test_ds = TestDataset(test_csv, image_root, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- model ----
    model = load_model(args.model_name, ckpt_path, num_classes=2, device=device)

    # ---- inference ----
    print("\nRunning inference...")
    ids, preds, probs = run_inference(model, test_loader, device)

    # ---- save results ----
    results = pd.DataFrame({
        "ID":    ids,
        "Class": preds,
        #"Prob_class1": [round(p, 4) for p in probs],  # confidence score (optional)
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\nSaved {len(results)} predictions → {output_path}")
    print(f"\nPrediction breakdown:")
    print(results["Class"].value_counts().rename({0: "Class 0 (good)", 1: "Class 1 (poor)"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blastocyst quality inference — outputs ID + predicted Class")

    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to .pt checkpoint. If omitted, auto-selects "
                             f"best_val*.pt from {DEFAULT_CKPT_DIR}")
    parser.add_argument("--output",      type=str,
                        default="results/predictions.csv",
                        help="Where to save the output CSV")
    parser.add_argument("--model_name",  type=str, default="inception_v3",
                        help="Must match the architecture used during training")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)