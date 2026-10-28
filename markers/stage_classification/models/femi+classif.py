"""
FEMI features — Embryo Stage Classification
Classes: empty | 1cell | 2cell | 4cell

Usage:
    python train_femi.py --data_dir ./stage_classif
    python train_femi.py --data_dir ./stage_classif --epochs 30 --batch_size 32
"""

import argparse
import copy
import glob
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
FEATURE_DIM  = 1024
MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats
STD          = [0.229, 0.224, 0.225]




# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def build_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.15,
                      num_workers: int = 4):

    full_dataset = datasets.DatasetFolder(
        root=data_dir,
        loader=torch.load,
        extensions=".pt",
        transform=None, 
    )
    n_total  = len(full_dataset)
    n_val    = max(1, int(n_total * val_split))
    n_train  = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\nDataset  : {data_dir}")
    print(f"Classes  : {full_dataset.classes}")
    print(f"Total    : {n_total}  |  Train: {n_train}  |  Val: {n_val}")
    print(f"Batches  : {len(train_loader)} train  /  {len(val_loader)} val\n")

    return train_loader, val_loader, full_dataset.classes


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def build_model(in_features:int, num_classes: int):
    
    model = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters())
    print(f"Model    : FEMI (not trained) + Sequential (trainable)")
    print(f"Params   : {total:,} total  |  {trainable:,} trainable\n")
    return model


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train(model, train_loader, val_loader, device, args):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase 1: warm-up — only train the new head ──
    print("=" * 55)
    print(f"Phase 1 — Head warm-up ({args.epochs} epochs, backbone frozen)")
    print("=" * 55)

    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-3, weight_decay=1e-4)
    best_model = run_phase(model, train_loader, val_loader, optimizer, criterion,
              device, args.epochs, phase=1)

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
            imgs, labels = imgs['embeddings'].to(device), labels.to(device)
            imgs = imgs.squeeze(1) 
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
                imgs, labels = imgs['embeddings'].to(device), labels.to(device)
                imgs = imgs.squeeze(1) 
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
            imgs, labels = imgs['embeddings'].to(device), labels.to(device)
            imgs = imgs.squeeze(1)
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
    p = argparse.ArgumentParser(description="FEMI + classifier — embryo stage classification")
    p.add_argument("--data_dir",      default="./stage_classif",  help="Root of dataset")
    p.add_argument("--model_path",    default="./checkpoints",    help="Where to save best model")
    p.add_argument("--epochs",        type=int, default=5,         help="Head-only warm-up epochs (phase 1)")
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

    model = build_model(FEATURE_DIM, num_classes).to(device)
    model = train(model, train_loader, val_loader, device, args)

    evaluate(model, val_loader, class_names, device)

    # Save final best model
    ckpt_path = os.path.join(args.model_path, "femi_embryo_best.pth")
    torch.save(model, ckpt_path)
    print(f"\nModel saved → {ckpt_path}")


if __name__ == "__main__":
    main()