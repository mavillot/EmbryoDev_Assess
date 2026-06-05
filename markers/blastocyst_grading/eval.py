#!/usr/bin/env python3

"""
Evaluation script for blastocyst grading models.

For running the script: python -m grading.eval_best EXP --checkpoint path/to/checkpoint.pth
"""


import os
import sys
from pathlib import Path
import argparse
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET


DATA = {
    "EXP": {"test": "EXP_gold", "classes": [0, 1, 2, 3, 4], "class_map":{0:'1',1:'2',2:'3',3:'4',4:'5'}},
    "ICM": {"test": "ICM_gold", "classes": [0, 1, 2], "class_map":{0:'A',1:'B',2:'C'}},
    "TE":  {"test": "TE_gold",  "classes": [0, 1, 2], "class_map":{0:'A',1:'B',2:'C'}},
}


class DatasetBlastoGrades(Dataset):
    def __init__(self, csv: pd.DataFrame, images_dir: Path, label_column: str, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.images_dir = images_dir
        self.label_column = label_column
        self.transform = transform
        self.images = self.csv["Image"].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.images_dir / img_name

        image = Image.open(img_path).convert("RGB")
        label = int(self.csv.loc[idx, self.label_column])

        if self.transform:
            image = self.transform(image)

        return image, label


def build_test_transform():
    return transforms.Compose([
        transforms.Resize(342, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(299),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(num_classes: int,model_name: str, checkpoint_path: Path, device: torch.device):
    model = torchvision.models.__dict__[model_name](init_weights=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.aux_logits = False
    model.AuxLogits = None

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device).eval()
    return model


def evaluate(model, dataloader, criterion, device):
    all_targets = []
    all_preds = []
    running_loss = 0.0

    with torch.inference_mode():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            all_targets.append(targets)
            all_preds.append(outputs.argmax(dim=1))

    avg_loss = running_loss / len(dataloader)
    all_targets = torch.cat(all_targets).cpu()
    all_preds = torch.cat(all_preds).cpu()

    return avg_loss, all_targets, all_preds


def save_confusion_matrix(cm, task, filepath):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=DATA[task]["class_map"].values(), yticklabels=DATA[task]["class_map"].values())
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()

def main(args):

    print(f"Evaluating task: {args.task}")
    print(f"Using checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    dataset_root = Path(PATH_DATASET) / "Blastocyst_Dataset"
    images_dir = dataset_root / "Images"
    labels_path = dataset_root / "Gardner_test_gold_onlyGardnerScores.csv"

    # Load labels
    df = pd.read_csv(labels_path, sep=";")
    label_col = DATA[args.task]["test"]

    df = df[df[label_col] != "ND"]
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)

    # Dataset & loader
    dataset = DatasetBlastoGrades(
        csv=df,
        images_dir=images_dir,
        label_column=label_col,
        transform=build_test_transform(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model
    model = load_model(
        num_classes=len(DATA[args.task]["classes"]),
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Evaluation
    print("Starting evaluation...")
    loss, targets, preds = evaluate(model, dataloader, criterion, device)
    report = classification_report(targets, preds, digits=4)
    cm = confusion_matrix(targets, preds)
    #save figure of confusion matrix
    

    if args.save_predictions:
        results_path = Path("output/markers/blastocyst_grading")
        results_path.mkdir(parents=True, exist_ok=True)

        with open(results_path / f"{args.task}_classification_report.txt", "w") as f:
            f.write(report)

        save_confusion_matrix(cm, args.task,
                              results_path / f"{args.task}_confusion_matrix.png")

    print(f"Test loss: {loss:.4f}")
    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate blastocyst grading model")
    parser.add_argument("task", choices=["EXP", "ICM", "TE"], help="Grading task")
    parser.add_argument("--model_name", type=str, default="inception_v3",
                        help="Model architecture to use")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_predictions", type=bool, default=True,
                        help="Whether to save the metrics")
    args = parser.parse_args()
    main(args)