"""
ResNet34 Transfer Learning — Embryo Stage Classification
Classes: empty | 1cell | 2cell | 4cell

Usage:
    python train_resnet34.py --data_dir ./stage_classif
    python train_resnet34.py --data_dir ./stage_classif --epochs 30 --batch_size 32
"""

import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CLASSES      = ["empty", "1cell", "2cell", "+3cell", "+8cell","morula" ,"blastocyst"] 
IMG_SIZE     = 224          # ResNet canonical input
MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats
STD          = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
def get_transforms(img_size: int = IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def build_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.15,
                      num_workers: int = 4):
    train_tf, val_tf = get_transforms()

    # Load full dataset with training transforms first (for splitting)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tf)

    # Validate expected classes are present
    found = set(full_dataset.classes)
    expected = set(CLASSES)
    missing = expected - found
    if missing:
        print(f"[WARNING] Expected classes not found in dataset: {missing}")
        print(f"          Found classes: {found}")

    n_total  = len(full_dataset)
    n_val    = max(1, int(n_total * val_split))
    n_train  = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Override val transform (random_split keeps the parent dataset's transform)
    val_subset.dataset = copy.deepcopy(full_dataset)
    val_subset.dataset.transform = val_tf

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"\nDataset  : {data_dir}")
    print(f"Classes  : {full_dataset.classes}")
    print(f"Total    : {n_total}  |  Train: {n_train}  |  Val: {n_val}")
    print(f"Batches  : {len(train_loader)} train  /  {len(val_loader)} val\n")

    return train_loader, val_loader, full_dataset.classes


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def build_model(num_classes: int, freeze_backbone: bool = False):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model    : ResNet34 (pretrained ImageNet)")
    print(f"Params   : {total:,} total  |  {trainable:,} trainable\n")
    return model


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train(model, train_loader, val_loader, device, args):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: warm-up — only train the new head ──
    print("=" * 55)
    print(f"Phase 1 — Head warm-up ({args.warmup_epochs} epochs, backbone frozen)")
    print("=" * 55)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    run_phase(model, train_loader, val_loader, optimizer, criterion,
              device, args.warmup_epochs, phase=1)

    # ── Phase 2: fine-tune entire network ──
    print("\n" + "=" * 55)
    print(f"Phase 2 — Full fine-tune ({args.epochs} epochs, all layers)")
    print("=" * 55)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {"params": list(model.parameters())[:-2], "lr": args.lr * 0.1},  # backbone
        {"params": model.fc.parameters(),          "lr": args.lr},        # head
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_model = run_phase(model, train_loader, val_loader, optimizer, criterion,
                           device, args.epochs, scheduler=scheduler, phase=2)

    return best_model


def run_phase(model, train_loader, val_loader, optimizer, criterion, device,
              epochs, scheduler=None, phase=1):
    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += imgs.size(0)

        # ── validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs  = model(imgs)
                loss     = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds    = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total * 100
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total * 100
        lr_now = optimizer.param_groups[-1]["lr"]
        elapsed = time.time() - t0

        print(f"[P{phase}] Ep {epoch:>3}/{epochs}  "
              f"train {t_loss:.4f}/{t_acc:.1f}%  "
              f"val {v_loss:.4f}/{v_acc:.1f}%  "
              f"lr {lr_now:.2e}  {elapsed:.1f}s")

        if v_acc > best_acc:
            best_acc   = v_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"            ✔ best val acc → {best_acc:.1f}%")

        if scheduler:
            scheduler.step()

    model.load_state_dict(best_state)
    print(f"\nPhase {phase} done — best val acc: {best_acc:.1f}%")
    return model


# ──────────────────────────────────────────────
# Evaluation (per-class accuracy)
# ──────────────────────────────────────────────
def evaluate(model, val_loader, class_names, device):
    model.eval()
    n_classes = len(class_names)
    correct   = torch.zeros(n_classes)
    total     = torch.zeros(n_classes)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for c in range(n_classes):
                mask       = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                total[c]   += mask.sum().item()

    print("\n── Per-class accuracy ──────────────────────────")
    for c, name in enumerate(class_names):
        acc = correct[c] / total[c] * 100 if total[c] > 0 else float("nan")
        print(f"  {name:<10}  {int(correct[c]):>4}/{int(total[c]):<4}  {acc:.1f}%")
    overall = correct.sum() / total.sum() * 100
    print(f"  {'OVERALL':<10}  {int(correct.sum()):>4}/{int(total.sum()):<4}  {overall:.1f}%")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="ResNet34 transfer learning — embryo stage classification")
    p.add_argument("--data_dir",      default="./stage_classif",  help="Root of dataset")
    p.add_argument("--model_path",    default="./checkpoints",    help="Where to save best model")
    p.add_argument("--epochs",        type=int, default=25,        help="Fine-tune epochs (phase 2)")
    p.add_argument("--warmup_epochs", type=int, default=5,         help="Head-only warm-up epochs (phase 1)")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--lr",            type=float, default=3e-4,    help="Head learning rate")
    p.add_argument("--val_split",     type=float, default=0.15)
    p.add_argument("--num_workers",   type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    os.makedirs(args.model_path, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers
    )
    num_classes = len(class_names)

    model = build_model(num_classes).to(device)
    model = train(model, train_loader, val_loader, device, args)

    evaluate(model, val_loader, class_names, device)

    # Save final best model
    ckpt_path = os.path.join(args.model_path, "resnet34_embryo_best.pth")
    torch.save(model, ckpt_path)

    print(f"\nModel saved → {ckpt_path}")


if __name__ == "__main__":
    main()