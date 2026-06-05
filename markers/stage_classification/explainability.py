import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET

PATH_PROCESSED = os.path.join(PATH_DATASET, "stage_classif/EVALUATE")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CLASSES  = ["empty", "1cell", "2cell", "4cell"]
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]


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
def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    n_per_class: int = 15,
):

    val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=val_tf
    )

    # Collect indices per class
    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    # Randomly select a few images from each class
    selected_indices = []

    for label, indices in class_indices.items():
        chosen = random.sample(
            indices,
            min(n_per_class, len(indices))
        )
        selected_indices.extend(chosen)

    # Create subset
    subset_dataset = Subset(full_dataset, selected_indices)

    data_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"\nDataset : {data_dir}")
    print(f"Classes : {full_dataset.classes}")
    print(f"Selected images : {len(subset_dataset)}")

    return data_loader, full_dataset.classes


def load_model(model_path: str, device: str = "cuda"):
    """Load a fully serialised model (torch.save(model, path))."""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to an HWC uint8 RGB numpy array."""
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)   # CHW → HWC
    img  = (img * std + mean).clip(0, 1)
    return img.astype(np.float32)


def get_gradcam(model, input_tensor: torch.Tensor, rgb_img: np.ndarray,
                target_class: int | None = None) -> np.ndarray:
    """
    Compute a GradCAM heatmap and overlay it on rgb_img.

    Args:
        model        : the ResNet model (must have a .layer4 attribute)
        input_tensor : (1, C, H, W) float tensor on the correct device
        rgb_img      : (H, W, 3) float32 numpy array in [0, 1]
        target_class : class index to explain; None → argmax (highest-scoring class)

    Returns:
        visualization: (H, W, 3) uint8 numpy array with the CAM overlay
    """
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)   # (1, H, W)
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
    return visualization


def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build dataloader ──────────────────────────────────────────────────
    data_loader, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(args.model_path, device)
    model.to(device)

    # ── Target class index (optional) ────────────────────────────────────
    target_class = None
    if args.target_class is not None:
        if args.target_class in class_names:
            target_class = class_names.index(args.target_class)
        else:
            raise ValueError(f"--target_class '{args.target_class}' not in {class_names}")

    # ── Run GradCAM over every batch ──────────────────────────────────────
    img_idx = 0
    for batch_tensors, batch_labels in data_loader:
        batch_tensors = batch_tensors.to(device)

        for i in range(batch_tensors.size(0)):
            single_tensor = batch_tensors[i].unsqueeze(0)   # (1, C, H, W)
            rgb_img       = denormalize(batch_tensors[i])    # (H, W, 3) float32

            vis = get_gradcam(model, single_tensor, rgb_img, target_class=target_class)

            label_name = class_names[batch_labels[i].item()]
            save_path  = os.path.join(args.output_dir, f"gradcam_{img_idx:04d}_{label_name}.png")
            plt.imsave(save_path, vis)
            print(f"Saved: {save_path}")
            img_idx += 1

    print(f"\nDone. {img_idx} GradCAM images saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM visualisation for embryo stage classification")
    parser.add_argument("--data_dir",     default=os.path.join(PATH_PROCESSED, "Images"),
                        help="Root of dataset (ImageFolder layout)")
    parser.add_argument("--model_path",   default="models/markers/stage_classif/resnet34/resnet34_we_100.pth",
                        help="Path to trained model checkpoint (full model saved with torch.save)")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--device",       default="cuda",
                        help="Device (cpu / cuda / cuda:1 …)")
    parser.add_argument("--output_dir",   default="output/markers/stage_classification/gradcam/",
                        help="Directory where GradCAM overlay images are saved")
    parser.add_argument("--target_class", default=None,
                        help="Class name to explain (e.g. '2cell'). Defaults to the model's top prediction.")

    args = parser.parse_args()
    main(args)