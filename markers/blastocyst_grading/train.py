#!/usr/bin/env python3

"""
Training script for blastocyst grading models.

For running the script: python -m grading.train_best EXP --epochs 50 --batch_size 32 --lr 0.01 --momentum 0.9 --weight_decay 1e-4
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import PATH_DATASET

DATA = {
    'EXP': {'train': 'EXP_silver', 'test': 'EXP_gold', 'classes': [0, 1, 2, 3, 4]},
    'ICM': {'train': 'ICM_silver', 'test': 'ICM_gold', 'classes': [0, 1, 2]},
    'TE':  {'train': 'TE_silver',  'test': 'TE_gold',  'classes': [0, 1, 2]},
}

class Dataset_Blasto_Grades(torch.utils.data.Dataset):
    def __init__(self, csv, grade_column, image_root, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.grade_column = grade_column
        self.image_root = image_root
        self.transform = transform
        self.imgs = self.csv["Image"].values

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = self.image_root / img_name

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = int(self.csv.loc[self.csv["Image"] == img_name,
                                 self.grade_column].values[0])

        return image, label
    
    def get_data_info(self):
        print(f"Dataset size: {len(self)}")
        print(f"Classes: {self.csv[self.grade_column].unique()}")
        print(self.csv[self.grade_column].value_counts())

def get_transforms(train_crop=299, val_crop=299, val_resize=342):
    interpolation = InterpolationMode.BILINEAR

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(train_crop, interpolation=interpolation),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_tf = transforms.Compose([
        transforms.Resize(val_resize, interpolation=interpolation),
        transforms.CenterCrop(val_crop),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tf, test_tf

def build_dataloaders(task, batch_size, num_workers):
    path = Path(PATH_DATASET) / "Blastocyst_Dataset"
    image_root = path / "Images"

    train_csv = pd.read_csv(path / "Gardner_train_silver.csv",  encoding="latin-1", sep=";")
    test_csv = pd.read_csv(path / "Gardner_test_gold_onlyGardnerScores.csv", sep=";")

    test_col = DATA[task]["test"]
    train_col = DATA[task]["train"]

    if task == "TE" or task == "ICM":
        train_csv = train_csv[train_csv[train_col] != 3]

    test_csv = test_csv[test_csv[test_col] != "ND"]
    test_csv = test_csv.dropna(subset=[test_col])
    test_csv[test_col] = test_csv[test_col].astype(int)

    train_tf, test_tf = get_transforms()

    train_ds = Dataset_Blasto_Grades(train_csv, train_col, image_root, transform=train_tf)
    test_ds = Dataset_Blasto_Grades(test_csv, test_col, image_root, transform=test_tf)

    #PRINT DATA INFO
    print("Train dataset info:")
    train_ds.get_data_info()
    print("\nTest dataset info:")
    test_ds.get_data_info()

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, len(DATA[task]["classes"])

def build_model(model_name, num_classes, device):
    model = torchvision.models.__dict__[model_name](init_weights=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Match your training behavior
    model.aux_logits = False
    model.AuxLogits = None

    model.to(device)
    return model

def train(model, loader, criterion, optimizer, scheduler, epochs, device):
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        n = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n += images.size(0)

        epoch_loss = running_loss / n
        print(f"Epoch [{epoch}] Train loss: {epoch_loss:.4f}")

        scheduler.step()

    return epoch_loss


def main(args):
    task = args.task

    print(f"Running task: {task}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, num_classes = build_dataloaders(
        task, args.batch_size, args.num_workers
    )

    model = build_model(args.model_name, num_classes, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    final_loss = train(
        model, train_loader, criterion, optimizer, scheduler,
        args.epochs, device
    )

    ckpt_path = f"models/markers/blastocyst_grading/{task}_{final_loss:.4f}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blastocyst grading training script")

    parser.add_argument("task", type=str, choices=["EXP", "ICM", "TE"],
                        help="Task to be graded")
    parser.add_argument("--model_name", type=str, default="inception_v3",
                        help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    main(args)
