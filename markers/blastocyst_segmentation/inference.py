#!/usr/bin/env python3

"""
Inference script for blastocyst segmentation using HRNet.

Usage: 
    Single image: python inference.py --model_path model.pth --input image.png --output mask.png
    Directory:    python inference.py --model_path model.pth --input images/ --output_dir results/
"""

import argparse
import os
from pathlib import Path
from glob import glob

import numpy as np
import torch
import cv2
import PIL
from PIL import Image
import torchvision.transforms as transforms


def transform_image(image, device):
    """Transform image for model input with ImageNet normalization."""
    my_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    return my_transforms(image).unsqueeze(0).to(device)


def predict_mask(model, image_path, device, image_size=480):
    """
    Predict segmentation mask for a single image.
    
    Args:
        model: Loaded PyTorch model
        image_path: Path to input image
        device: torch device (cuda or cpu)
        image_size: Size to resize image for model input
        
    Returns:
        mask: Numpy array with segmentation mask (original image size)
    """
    # Load and convert image
    img = PIL.Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    
    # Resize to model input size
    image = transforms.Resize((image_size, image_size))(img)
    tensor = transform_image(image, device)
    
    # Run inference
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(tensor)
    
    # Convert predictions to class indices
    outputs = torch.argmax(outputs, 1)
    mask = np.array(outputs.cpu())
    
    # Convert class indices back to mask values
    # Class 0 (Background) -> 0
    # Class 1 (ZP) -> 75
    # Class 2 (TE) -> 255
    # Class 3 (ICM) -> 150
    mask_converted = np.zeros_like(mask, dtype=np.uint8)
    mask_converted[mask == 0] = 0
    mask_converted[mask == 1] = 75
    mask_converted[mask == 2] = 255
    mask_converted[mask == 3] = 150
    
    mask_converted = np.reshape(mask_converted, (image_size, image_size))
    
    # Resize back to original size
    mask_resized = np.array(
        transforms.Resize(original_size[::-1])(
            Image.fromarray(mask_converted)
        )
    )
    
    return mask_resized


def get_image_paths(path):
    """Get list of image paths from file or directory."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        # Support common image extensions
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.BMP', '*.tif', '*.tiff')
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(path, ext)))
        return sorted(files)
    else:
        raise ValueError(f"Invalid path: {path}")


def create_overlay(image_path, mask, alpha=0.5):
    """
    Create an overlay visualization of mask on original image.
    
    Args:
        image_path: Path to original image
        mask: Segmentation mask (grayscale)
        alpha: Transparency of mask overlay (0-1)
        
    Returns:
        overlay: RGB image with colored mask overlay
    """
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize mask to match image if needed
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Assign colors to each structure
    # Background: black (no color)
    # ZP (75): blue
    # TE (255): green
    # ICM (150): red
    colored_mask[mask == 75] = [0, 0, 255]    # ZP - Blue
    colored_mask[mask == 255] = [0, 255, 0]   # TE - Green
    colored_mask[mask == 150] = [255, 0, 0]   # ICM - Red
    
    # Create overlay
    overlay = cv2.addWeighted(img, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def main(args):
    """Main inference function."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    model = torch.jit.load(args.model_path)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Get input image paths
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s) to process")
    
    if len(image_paths) == 0:
        print("No images found to process!")
        return
    
    # Single image output
    if args.output and len(image_paths) == 1:
        print(f"\nProcessing: {os.path.basename(image_paths[0])}")
        mask = predict_mask(model, image_paths[0], device, args.image_size)
        
        # Save mask
        cv2.imwrite(args.output, mask)
        print(f"Saved mask to: {args.output}")
        
        # Save overlay if requested
        if args.save_overlay:
            overlay_path = args.output.replace('.', '_overlay.')
            overlay = create_overlay(image_paths[0], mask, args.overlay_alpha)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved overlay to: {overlay_path}")
    
    # Multiple images or directory output
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.save_overlay:
            overlay_dir = output_dir / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to: {output_dir}")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
            
            try:
                # Predict mask
                mask = predict_mask(model, img_path, device, args.image_size)
                
                # Save mask
                output_path = output_dir / os.path.basename(img_path)
                cv2.imwrite(str(output_path), mask)
                
                # Save overlay if requested
                if args.save_overlay:
                    overlay = create_overlay(img_path, mask, args.overlay_alpha)
                    overlay_path = overlay_dir / os.path.basename(img_path)
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"  Error processing {img_path}: {str(e)}")
                continue
        
        print(f"\nProcessing complete! Results saved to {output_dir}")
        if args.save_overlay:
            print(f"Overlays saved to {overlay_dir}")
    
    # Print mask value legend
    print("\n" + "="*50)
    print("Mask value legend:")
    print("  0   = Background")
    print("  75  = Zona Pellucida (ZP)")
    print("  255 = Trophectoderm (TE)")
    print("  150 = Inner Cell Mass (ICM)")
    print("="*50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference script for blastocyst segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python inference.py --model_path model.pth --input image.bmp --output result.png
  
  # Directory of images
  python inference.py --model_path model.pth --input images/ --output_dir results/
  
  # With overlay visualization
  python inference.py --model_path model.pth --input image.bmp --output result.png --save_overlay
        """
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.pth file)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="",
                        help="Output path for single image (ignored for directories)")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Output directory for multiple images")
    
    # Visualization arguments
    parser.add_argument("--save_overlay", action="store_true",
                        help="Save overlay visualization (mask on original image)")
    parser.add_argument("--overlay_alpha", type=float, default=0.5,
                        help="Transparency of mask overlay (0-1)")
    
    # Model arguments
    parser.add_argument("--image_size", type=int, default=480,
                        help="Image size for model input")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
