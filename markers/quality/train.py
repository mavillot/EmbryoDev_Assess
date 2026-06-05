#!/usr/bin/env python3
 
"""
Training script for blastocyst quality models (good/poor classification).
 
Usage:
    python train_best.py --epochs 50 --batch_size 16 --lr 0.001
 
"""
 
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
 
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
 
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
 
from config.config import PATH_DATASET
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class Dataset_Blasto_Grades(torch.utils.data.Dataset):
    def __init__(self, csv, image_root, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.imgs = self.csv["Image"].values
        self.labels = self.csv["Class"].values.astype(int)
 
    def __len__(self):
        return len(self.imgs)
 
    def __getitem__(self, idx):
        img_path = self.image_root / self.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
 
    def get_data_info(self, name="Dataset"):
        print(f"\n{name} size: {len(self)}")
        counts = pd.Series(self.labels).value_counts().sort_index()
        for cls, cnt in counts.items():
            print(f"  Class {cls}: {cnt} samples")
 
 
# ---------------------------------------------------------------------------
# Transforms  (conservative augmentation for medical images)
# ---------------------------------------------------------------------------
 
def get_transforms(train_crop=299, val_crop=299, val_resize=342):
    interp = InterpolationMode.BILINEAR
 
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(train_crop, scale=(0.8, 1.0),
                                     interpolation=interp),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
 
    val_tf = transforms.Compose([
        transforms.Resize(val_resize, interpolation=interp),
        transforms.CenterCrop(val_crop),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
 
    return train_tf, val_tf
 
 
# ---------------------------------------------------------------------------
# Dataloaders with stratified train/val split + WeightedRandomSampler
# ---------------------------------------------------------------------------
 
def build_dataloaders(batch_size, num_workers, val_fraction=0.15, seed=42):
    path = Path(PATH_DATASET) / "hvwc23"
    image_root = path / "train"
 
    csv = pd.read_csv(path / "train.csv")
 
    # Stratified split so both classes appear in val
    train_csv, val_csv = train_test_split(
        csv,
        test_size=val_fraction,
        stratify=csv["Class"],
        random_state=seed,
    )
 
    train_tf, val_tf = get_transforms()
 
    train_ds = Dataset_Blasto_Grades(train_csv, image_root, transform=train_tf)
    val_ds   = Dataset_Blasto_Grades(val_csv,   image_root, transform=val_tf)
 
    train_ds.get_data_info("Train")
    val_ds.get_data_info("Val")
 
    # ---- WeightedRandomSampler: oversample minority class ----
    labels = train_ds.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts          # weight per class
    sample_weights = class_weights[labels]       # weight per sample
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
 
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
 
    return train_loader, val_loader, class_counts
 
 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
 
def build_model(model_name, num_classes, device, freeze_backbone=False):
    """
    Load a pretrained model and replace the head.
    Uses the new torchvision weights API (weights='DEFAULT').
    """
    # New API: weights='DEFAULT' loads best available pretrained weights
    constructor = models.__dict__[model_name]
    model = constructor(weights="DEFAULT")
 
    if model_name.startswith("inception"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.aux_logits = False
        model.AuxLogits = None
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        # e.g. EfficientNet, VGG
        if isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, num_classes)
        else:
            model.classifier = nn.Linear(
                model.classifier.in_features, num_classes)
 
    if freeze_backbone:
        # Only train the new head — useful when data is very small
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen — training head only.")
 
    model.to(device)
    return model
 
 
# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
 
def evaluate(model, loader, criterion, device, num_classes=2):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    per_class_correct = np.zeros(num_classes)
    per_class_total   = np.zeros(num_classes)
 
    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device)
            targets = targets.to(device)
 
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
 
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            n += images.size(0)
 
            for c in range(num_classes):
                mask = targets == c
                per_class_correct[c] += (preds[mask] == targets[mask]).sum().item()
                per_class_total[c]   += mask.sum().item()
 
    avg_loss = total_loss / n
    avg_acc  = correct / n
    per_class_acc = per_class_correct / (per_class_total + 1e-8)
    return avg_loss, avg_acc, per_class_acc
 
 
def train(model, train_loader, val_loader, criterion,
          optimizer, scheduler, epochs, device, save_dir):
 
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_ckpt = None
 
    print("\n{:<8} {:<12} {:<12} {:<12} {:<16} {:<16}".format(
        "Epoch", "Train Loss", "Val Loss", "Val Acc",
        "Acc Class-0", "Acc Class-1"))
    print("-" * 80)
 
    for epoch in range(epochs):
        # ---- train ----
        model.train()
        running_loss = 0.0
        n = 0
        for images, targets in train_loader:
            images  = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            n += images.size(0)
 
        train_loss = running_loss / n
 
        # ---- validate ----
        val_loss, val_acc, per_cls = evaluate(
            model, val_loader, criterion, device)
 
        print("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<16.4f} {:<16.4f}".format(
            epoch, train_loss, val_loss, val_acc,
            per_cls[0], per_cls[1]))
 
        scheduler.step()
 
        # ---- checkpoint best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = save_dir / f"best_val{val_acc:.4f}_ep{epoch}.pt"
            torch.save(model.state_dict(), best_ckpt)
            print(f"  → Saved best model: {best_ckpt.name}")
 
    print(f"\nBest val accuracy: {best_val_acc:.4f}  ({best_ckpt})")
    return best_ckpt
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    train_loader, val_loader, class_counts = build_dataloaders(
        args.batch_size, args.num_workers, args.val_fraction)
 
    model = build_model(args.model_name, num_classes=2,
                        device=device,
                        freeze_backbone=args.freeze_backbone)
 
    # Weighted cross-entropy: down-weight the majority class
    total = class_counts.sum()
    ce_weights = torch.tensor(
        total / (len(class_counts) * class_counts), dtype=torch.float
    ).to(device)
    print(f"\nCross-entropy class weights: {ce_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
 
    # Only pass trainable parameters to the optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
 
    train(model, train_loader, val_loader, criterion,
          optimizer, scheduler, args.epochs, device,
          save_dir="models/markers/blastocyst_quality")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blastocyst quality training (good/poor)")
 
    parser.add_argument("--model_name",      type=str,   default="inception_v3")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=16,
                        help="Smaller batch recommended for small dataset")
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--lr",              type=float, default=0.001,
                        help="Lower LR recommended when fine-tuning ImageNet weights")
    parser.add_argument("--momentum",        type=float, default=0.9)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--val_fraction",    type=float, default=0.15,
                        help="Fraction of training data to use for validation")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the encoder and only train the classification head")
 
    args = parser.parse_args()
    main(args)