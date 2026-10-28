import argparse
import copy
import glob
import os, sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET

PATH_PROCESSED = os.path.join(PATH_DATASET, "stage_classif/EVALUATE")

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
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return val_tf

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def build_dataloaders(data_dir: str, input: str = 'images', batch_size: int=32, num_workers: int = 4):

    if input == 'images':
        val_tf = get_transforms()
        full_dataset = datasets.ImageFolder(root=data_dir, transform=val_tf)

    if input == 'femi':
        full_dataset = datasets.DatasetFolder(
            root=data_dir,
            loader=torch.load,
            extensions=".pt",
            transform=None,  # already preprocessed
        )
    

    data_loader   = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\nDataset  : {data_dir}")
    print(f"Classes  : {full_dataset.classes}")
    print(f"Total    : {len(full_dataset)} ")


    return  data_loader, full_dataset.classes

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────

def load_model(model_path: str, model, device: str='cuda'):
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# ──────────────────────────────────────────────
# Evaluation (per-class accuracy)
# ──────────────────────────────────────────────
def save_confusion_matrix(cm, data, filepath):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.classes, yticklabels=data.classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

def save_accuracy_report(correct, total, class_names, filepath):
    """Write per-class and overall accuracy to a plain text file."""
    lines = ["── Per-class accuracy ──────────────────────────"]
    for c, name in enumerate(class_names):
        acc = correct[c] / total[c] * 100 if total[c] > 0 else float("nan")
        lines.append(f"  {name:<10}  {int(correct[c]):>4}/{int(total[c]):<4}  {acc:.1f}%")
    overall = correct.sum() / total.sum() * 100
    lines.append(f"  {'OVERALL':<10}  {int(correct.sum()):>4}/{int(total.sum()):<4}  {overall:.1f}%")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAccuracy report saved → {filepath}")

def evaluate(model, data_loader, class_names, device):
    model.eval()
    n_classes = len(class_names)
    all_labels = [] 
    all_preds  = [] 
    incorrect_images = []
    incorrect_labels = []
    correct    = torch.zeros(n_classes)
    total      = torch.zeros(n_classes)

    with torch.no_grad():
        for imgs, batch_labels in data_loader: # Rename the loop variable
            if isinstance(imgs, dict) and 'embeddings' in imgs:
                imgs = imgs['embeddings'].to(device).squeeze(1)
            else:
                imgs = imgs.to(device)
            batch_labels = batch_labels.to(device)
            batch_preds = model(imgs).argmax(dim=1)
            
            # Now you are extending the LIST with the values from the TENSOR
            all_labels.extend(batch_labels.cpu().tolist())
            all_preds.extend(batch_preds.cpu().tolist())
            incorrect_images.extend(imgs[batch_preds != batch_labels].cpu())  
            incorrect_labels.extend(batch_preds[batch_preds != batch_labels].cpu())

            for c in range(n_classes):
                mask       = batch_labels == c
                correct[c] += (batch_preds[mask] == batch_labels[mask]).sum().item()
                total[c]   += mask.sum().item()

    print("\n── Per-class accuracy ──────────────────────────")
    for c, name in enumerate(class_names):
        acc = correct[c] / total[c] * 100 if total[c] > 0 else float("nan")
        print(f"  {name:<10}  {int(correct[c]):>4}/{int(total[c]):<4}  {acc:.1f}%")
    overall = correct.sum() / total.sum() * 100
    print(f"  {'OVERALL':<10}  {int(correct.sum()):>4}/{int(total.sum()):<4}  {overall:.1f}%")
    return  all_labels, all_preds, incorrect_images, incorrect_labels, correct, total

def save_incorrect_images(incorrect_images, incorrect_labels, class_names, output_dir):
    os.makedirs(os.path.join(output_dir, "incorrect"), exist_ok=True)
    
    for idx, (img_tensor, label) in enumerate(zip(incorrect_images, incorrect_labels)):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CxHxW → HxWxC
        img = (img * np.array(STD) + np.array(MEAN)) * 255  # Denormalize
        img = np.clip(img, 0, 255).astype(np.uint8)
        class_name = class_names[label]
        plt.imsave(os.path.join(output_dir, "incorrect", f"incorrect_{idx}_{class_name}.png"), img)
                                 
def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build dataloader ─────────────────────────────────────────────────
    data_loader, class_names = build_dataloaders(args.data_dir, input=args.input, batch_size=args.batch_size)

    # ── Load model ───────────────────────────────────────────────────────
    model = load_model(args.model_path, device)

    # ── Evaluate ─────────────────────────────────────────────────────────
    labels, preds, incorrect_images, incorrect_labels, correct, total = evaluate(model, data_loader, class_names, device)
    cm = confusion_matrix(labels, preds)
    save_confusion_matrix(cm, data_loader.dataset, os.path.join(args.output_dir, "confusion_matrix.png"))
    save_accuracy_report(correct, total, class_names, os.path.join(args.output_dir, "accuracy_report.txt"))
    if args.save_incorrect:
        save_incorrect_images(incorrect_images, incorrect_labels, class_names, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embryo stage classification")
    parser.add_argument("--data_dir",   default=os.path.join(PATH_PROCESSED, "Femi/test"), help="Root of dataset (should contain 'images/' and 'femi/')")
    parser.add_argument("--model_path", default="models/markers/stage_classif/FEMI/best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--input",      default="femi", choices=["images", "femi"], help="Whether to evaluate on raw images or FEMI embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device",     default='cuda', help="Device to run evaluation on (e.g. cpu, cuda, cuda:1)")
    parser.add_argument("--output_dir", default="output/markers/stage_classification/", help="Where to save evaluation results (e.g. confusion matrix)")
    parser.add_argument("--save_incorrect", action="store_true", help="Whether to save incorrectly classified images for analysis")

    args = parser.parse_args()
    main(args)