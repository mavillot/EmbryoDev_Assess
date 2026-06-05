#!/usr/bin/env python3

"""
Evaluation script for blastocyst segmentation using HRNet.

Usage: python markers/blastocyst_segmentation/eval.py --model_path models/markers/blastocyst_segmentation/hrnet_final.pth --data_path ../dataset --output_csv results.csv
"""

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import torch
import cv2
import PIL
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# ========== METRICS ==========

def accuracy(target, prediction):
    """Calculate pixel-wise accuracy."""
    true_detec = np.logical_not(np.logical_xor(target, prediction))
    return np.sum(true_detec) / np.sum(np.ones_like(target))


def precision(target, prediction):
    """Calculate precision."""
    intersection = np.logical_and(target, prediction)
    pred_sum = np.sum(prediction)
    return np.sum(intersection) / pred_sum if pred_sum > 0 else 0.0


def recall(target, prediction):
    """Calculate recall (sensitivity)."""
    intersection = np.logical_and(target, prediction)
    target_sum = np.sum(target)
    return np.sum(intersection) / target_sum if target_sum > 0 else 0.0


def jaccard(target, prediction):
    """Calculate Jaccard index (IoU)."""
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    union_sum = np.sum(union)
    return np.sum(intersection) / union_sum if union_sum > 0 else 0.0


def dice(target, prediction):
    """Calculate Dice coefficient."""
    intersection = np.logical_and(target, prediction)
    denominator = np.sum(target) + np.sum(prediction)
    return 2 * np.sum(intersection) / denominator if denominator > 0 else 0.0


def metrics(target, prediction):
    """Calculate all metrics for a single image."""
    return {
        'accuracy': accuracy(target, prediction),
        'precision': precision(target, prediction),
        'recall': recall(target, prediction),
        'specificity': recall(1 - target, 1 - prediction),
        'jaccard': jaccard(target, prediction),
        'dice': dice(target, prediction)
    }


def summary_metrics(y_test, predictions):
    """Calculate average metrics across all test images."""
    a, p, r, s, j, d = 0., 0., 0., 0., 0., 0.
    n = len(predictions)
    
    for i in range(n):
        preds = predictions[i] == 255
        gt = y_test[i] == 255
        metricas = metrics(gt, preds)
        a += metricas['accuracy']
        p += metricas['precision']
        r += metricas['recall']
        s += metricas['specificity']
        j += metricas['jaccard']
        d += metricas['dice']
    
    return {
        'accuracy': a / n,
        'precision': p / n,
        'recall': r / n,
        'specificity': s / n,
        'jaccard': j / n,
        'dice': d / n
    }


# ========== MODEL LOADING AND PREDICTION ==========

def transform_image(image, device):
    """Transform image for model input."""
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image_aux = image
    return my_transforms(image_aux).unsqueeze(0).to(device)


def prediction(model, path, device, image_size=480):
    """Predict segmentation mask for a single image."""
    img = PIL.Image.open(path).convert('RGB')
    original_size = img.size
    
    # Resize to model input size
    image = transforms.Resize((image_size, image_size))(img)
    tensor = transform_image(image, device)
    
    model.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    
    outputs = torch.argmax(outputs, 1)
    mask = np.array(outputs.cpu())
    
    # Convert class indices back to original mask values
    # {'Background':0, 'ZP':1, 'TE':2, 'ICM':3}
    mask[mask == 1] = 75
    mask[mask == 2] = 255
    mask[mask == 3] = 150
    mask = np.reshape(mask, (image_size, image_size))
    
    # Resize back to original size
    mask_resized = np.array(
        transforms.Resize(original_size[::-1])(
            Image.fromarray(mask.astype('uint8'))
        )
    )
    
    return mask_resized


def get_mask(path, string, path_gt):
    """Load ground truth mask for a specific structure."""
    file = f'GT_{string}/' + path.stem + f' {string}_Mask.bmp'
    path_gt_file = str(path_gt / file)
    return cv2.imread(path_gt_file, cv2.IMREAD_GRAYSCALE)


# ========== EVALUATION FUNCTION ==========

def evaluate(args):
    """Main evaluation function."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = torch.jit.load(args.model_path)
    model = model.to(device)
    model.eval()
    
    # Setup paths
    data_path = Path(args.data_path)
    path_test = data_path / 'test'
    
    if not path_test.exists():
        raise FileNotFoundError(f"Test data path not found: {path_test}")
    
    # Get test files
    files = [path_test / f for f in os.listdir(path_test) if f.endswith(('.bmp', '.BMP', '.jpg', '.png'))]
    print(f"Found {len(files)} test images")
    
    # Load ground truth masks
    print("Loading ground truth masks...")
    gtruth_zp = [get_mask(f, 'ZP', data_path) for f in files]
    gtruth_te = [get_mask(f, 'TE', data_path) for f in files]
    gtruth_icm = [get_mask(f, 'ICM', data_path) for f in files]
    
    # Generate predictions
    print("Generating predictions...")
    preds = [prediction(model, str(f), device, args.image_size) for f in files]
    
    # Separate predictions by structure
    preds_zp = [255 * (p == 75).astype('uint8') for p in preds]
    preds_te = [255 * (p == 255).astype('uint8') for p in preds]
    preds_icm = [255 * (p == 150).astype('uint8') for p in preds]
    
    # Save predictions if requested
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving predictions to: {output_dir}")
        
        for i, f in enumerate(files):
            cv2.imwrite(str(output_dir / f.name), preds[i])
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n--- Zona Pellucida (ZP) ---")
    metrics_zp = summary_metrics(gtruth_zp, preds_zp)
    for key, value in metrics_zp.items():
        print(f"{key.capitalize():15s}: {value:.4f}")
    
    print("\n--- Trophectoderm (TE) ---")
    metrics_te = summary_metrics(gtruth_te, preds_te)
    for key, value in metrics_te.items():
        print(f"{key.capitalize():15s}: {value:.4f}")
    
    print("\n--- Inner Cell Mass (ICM) ---")
    metrics_icm = summary_metrics(gtruth_icm, preds_icm)
    for key, value in metrics_icm.items():
        print(f"{key.capitalize():15s}: {value:.4f}")
    
    print("\n" + "="*60)
    
    # Save results to CSV
    if args.output_csv:
        print(f"\nSaving results to: {args.output_csv}")
        with open(args.output_csv, 'w', newline='') as csvfile:
            fieldnames = ['structure', 'accuracy', 'precision', 'recall', 'specificity', 'jaccard', 'dice']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            row_zp = {'structure': 'ZP'}
            row_zp.update(metrics_zp)
            writer.writerow(row_zp)
            
            row_te = {'structure': 'TE'}
            row_te.update(metrics_te)
            writer.writerow(row_te)
            
            row_icm = {'structure': 'ICM'}
            row_icm.update(metrics_icm)
            writer.writerow(row_icm)
    
    # Visualize some predictions if requested
    if args.visualize:
        print("\nVisualizing predictions...")
        num_samples = min(args.num_visualize, len(files))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Original image
            img = cv2.imread(str(files[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Original: {files[i].name}')
            axes[i, 0].axis('off')
            
            # Prediction
            axes[i, 1].imshow(preds[i], cmap='gray')
            axes[i, 1].set_title('Prediction')
            axes[i, 1].axis('off')
            
            # Combined ground truth (for visualization)
            combined_gt = np.zeros_like(gtruth_zp[i])
            combined_gt[gtruth_zp[i] == 255] = 75
            combined_gt[gtruth_te[i] == 255] = 255
            combined_gt[gtruth_icm[i] == 255] = 150
            axes[i, 2].imshow(combined_gt, cmap='gray')
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        if args.save_visualization:
            plt.savefig(args.save_visualization, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {args.save_visualization}")
        else:
            plt.show()
    
    print("\nEvaluation complete!")
    
    return metrics_zp, metrics_te, metrics_icm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HRNet for blastocyst segmentation")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.pth file)")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="../dataset",
                        help="Path to dataset directory containing test folder")
    parser.add_argument("--image_size", type=int, default=480,
                        help="Image size for model input")
    
    # Output parameters
    parser.add_argument("--output_csv", type=str, default="",
                        help="Path to save results CSV file")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save predicted masks")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Directory to save predicted masks")
    
    # Visualization parameters
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize predictions")
    parser.add_argument("--num_visualize", type=int, default=5,
                        help="Number of samples to visualize")
    parser.add_argument("--save_visualization", type=str, default="",
                        help="Path to save visualization figure")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
