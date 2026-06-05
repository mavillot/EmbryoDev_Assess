#!/usr/bin/env python3

"""
Training script for blastocyst segmentation using HRNet.

Usage: python train.py --data_path ../dataset --epochs_frozen 20 --epochs_unfrozen 20 --batch_size 2 --lr_frozen 1e-4 --lr_unfrozen 1e-7
"""

import argparse
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import PIL
from PIL import Image

from fastai.basics import *
from fastai.vision import models
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *
from fastai.callback import *

# SemTorch
from semtorch import get_segmentation_learner

# Albumentations for data augmentation
from albumentations import (Compose, OneOf, ElasticTransform, GridDistortion, 
                            OpticalDistortion, HorizontalFlip, Rotate, Transpose, 
                            CLAHE, ShiftScaleRotate)


# ========== CUSTOM TRANSFORMS ==========

class SegmentationAlbumentationsTransform(ItemTransform):
    """Custom transform to apply Albumentations augmentations in fastai pipeline."""
    split_idx = 0
    
    def __init__(self, aug): 
        self.aug = aug
        
    def encodes(self, x):
        img, mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])


class TargetMaskConvertTransform(ItemTransform):
    """Convert ground truth mask values to class indices."""
    def __init__(self): 
        pass
    
    def encodes(self, x):
        img, mask = x
        
        # Convert to array
        mask = np.array(mask)
        # {'TE':255, 'ICM':150,'ZP':75, 'Background':0}
        mask[mask == 0] = 0
        mask[mask == 75] = 1
        mask[mask == 255] = 2
        mask[mask == 150] = 3     
        
        # Back to PILMask
        mask = PILMask.create(mask)
        return img, mask


# ========== METRICS ==========

def msa(input, target):
    """Mean Segmentation Accuracy."""
    target = target.squeeze(1)
    mask = target != -1
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def background(input, target):
    """Background accuracy."""
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def zp(input, target):
    """Zona Pellucida accuracy."""
    target = target.squeeze(1)
    mask = target != 1
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def te(input, target):
    """Trophectoderm accuracy."""
    target = target.squeeze(1)
    mask = target != 2
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


def icm(input, target):
    """Inner Cell Mass accuracy."""
    target = target.squeeze(1)
    mask = target != 3
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


# ========== MAIN TRAINING FUNCTION ==========

def train(args):
    """Main training function."""
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    set_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Setup paths
    data_path = Path(args.data_path)
    path_train = data_path / 'train'
    
    if not path_train.exists():
        raise FileNotFoundError(f"Training data path not found: {path_train}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get function for masks
    get_y_fn = lambda x: Path(str(x).replace("train", "GT_Mask"))
    
    # Define class codes
    codes = np.array(['Background', 'ZP', 'TE', 'ICM'])
    
    # Data augmentation transforms
    transforms_aug = Compose([
        HorizontalFlip(p=0.5),
        Rotate(p=0.40, limit=10)
    ], p=1)
    transform_pipeline = SegmentationAlbumentationsTransform(transforms_aug)
    
    # Create DataBlock
    print("Creating DataBlock...")
    trainDB = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes)),
        get_items=get_image_files,
        get_y=get_y_fn,
        splitter=RandomSplitter(valid_pct=args.valid_pct),
        item_tfms=[
            Resize((args.image_size, args.image_size)), 
            TargetMaskConvertTransform(), 
            transform_pipeline
        ],
        batch_tfms=Normalize.from_stats(*imagenet_stats)
    )
    
    # Create DataLoaders
    trainDLS = trainDB.dataloaders(path_train, bs=args.batch_size)
    print(f"Training samples: {len(trainDLS.train_ds)}")
    print(f"Validation samples: {len(trainDLS.valid_ds)}")
    
    # Create learner with HRNet
    print(f"Creating HRNet model (backbone: {args.backbone})...")
    learn = get_segmentation_learner(
        dls=trainDLS, 
        number_classes=4, 
        segmentation_type="Semantic Segmentation",
        architecture_name="hrnet", 
        backbone_name=args.backbone, 
        metrics=[background, zp, te, icm, msa],
        wd=args.weight_decay,
        pretrained=True,
        normalize=True
    ).to_fp16()
    
    # Find learning rate
    if args.find_lr:
        print("Finding optimal learning rate...")
        learn.lr_find()
    
    # Training callbacks (frozen)
    monitor_training = "valid_loss"
    comp_training = np.less
    patience = args.patience
    
    fname_frozen = str(output_dir / "hrnet-before-unfreeze")
    callbacks_frozen = [
        ShowGraphCallback(), 
        EarlyStoppingCallback(monitor=monitor_training, comp=comp_training, patience=patience),
        SaveModelCallback(monitor=monitor_training, comp=comp_training, every_epoch=False, fname=fname_frozen)
    ]
    
    # Train frozen layers
    print(f"\nTraining frozen model for {args.epochs_frozen} epochs...")
    learn.fit_one_cycle(
        args.epochs_frozen, 
        slice(args.lr_frozen, args.lr_frozen),
        cbs=callbacks_frozen
    )
    
    # Unfreeze and continue training
    print("\nUnfreezing model...")
    learn.load(fname_frozen)
    learn.unfreeze()
    
    # Training callbacks (unfrozen)
    fname_unfrozen = str(output_dir / "hrnet-after-unfreeze")
    callbacks_unfrozen = [
        ShowGraphCallback(),
        EarlyStoppingCallback(monitor=monitor_training, comp=comp_training, patience=patience),
        SaveModelCallback(monitor=monitor_training, comp=comp_training, every_epoch=False, fname=fname_unfrozen)
    ]
    
    # Train unfrozen model
    print(f"\nTraining unfrozen model for {args.epochs_unfrozen} epochs...")
    learn.fit_one_cycle(
        args.epochs_unfrozen, 
        slice(args.lr_unfrozen, args.lr_unfrozen),
        cbs=callbacks_unfrozen
    )
    
    # Validation
    print("\nFinal validation:")
    learn.validate()
    
    # Save the final model
    print("\nSaving final model...")
    model = learn.model.cpu()
    traced_model = torch.jit.script(model)
    final_model_path = output_dir / "hrnet_final.pth"
    traced_model.save(str(final_model_path))
    print(f"Model saved to: {final_model_path}")
    
    print("\nTraining complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HRNet for blastocyst segmentation")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="../dataset",
                        help="Path to dataset directory containing train folder")
    parser.add_argument("--output_dir", type=str, default="models/markers/blastocyst_segmentation",
                        help="Directory to save model checkpoints")
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="hrnet_w30",
                        choices=["hrnet_w18", "hrnet_w30", "hrnet_w32", "hrnet_w48"],
                        help="HRNet backbone architecture")
    
    # Training parameters
    parser.add_argument("--epochs_frozen", type=int, default=20,
                        help="Number of epochs to train with frozen backbone")
    parser.add_argument("--epochs_unfrozen", type=int, default=20,
                        help="Number of epochs to train with unfrozen backbone")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--lr_frozen", type=float, default=1e-4,
                        help="Learning rate for frozen training")
    parser.add_argument("--lr_unfrozen", type=float, default=1e-7,
                        help="Learning rate for unfrozen training")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for optimizer")
    parser.add_argument("--valid_pct", type=float, default=0.2,
                        help="Percentage of data to use for validation")
    parser.add_argument("--image_size", type=int, default=480,
                        help="Image size for training")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=2020,
                        help="Random seed for reproducibility")
    parser.add_argument("--find_lr", action="store_true",
                        help="Run learning rate finder before training")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
