# 🔬 Blastocyst Segmentation with HRNet

A powerful toolkit for training and evaluating semantic segmentation models to identify key structures in blastocyst embryo images! This project uses HRNet (High-Resolution Net) to segment four important classes: Background, Zona Pellucida (ZP), Trophectoderm (TE), and Inner Cell Mass (ICM).

---

## 📋 What's Inside?

This module provides tools to:
- **Train** HRNet segmentation models with data augmentation
- **Evaluate** model performance with comprehensive metrics
- **Predict** segmentation masks for new blastocyst images

Segmented structures:
- **Background**: Non-embryo regions
- **ZP (Zona Pellucida)**: Protective outer shell
- **TE (Trophectoderm)**: Outer cell layer forming placenta
- **ICM (Inner Cell Mass)**: Inner cells forming the embryo proper

---

## 🛠️ Setup & Requirements

### Dependencies
```bash
pip install torch torchvision fastai semtorch albumentations opencv-python pillow numpy matplotlib pandas
```

**Key packages:**
- `fastai`: High-level deep learning framework
- `semtorch`: Semantic segmentation models for fastai
- `albumentations`: Advanced data augmentation library
- `opencv-python`: Image processing
- `matplotlib`: Visualization

### Dataset Structure

Your dataset should be organized as:
```
dataset/
├── train/                    # Training images
│   ├── image_001.bmp
│   ├── image_002.bmp
│   └── ...
├── GT_Mask/                  # Training masks (combined)
│   ├── image_001.bmp
│   ├── image_002.bmp
│   └── ...
├── test/                     # Test images
│   ├── test_001.bmp
│   ├── test_002.bmp
│   └── ...
├── GT_ZP/                    # ZP ground truth masks
│   ├── test_001 ZP_Mask.bmp
│   └── ...
├── GT_TE/                    # TE ground truth masks
│   ├── test_001 TE_Mask.bmp
│   └── ...
└── GT_ICM/                   # ICM ground truth masks
    ├── test_001 ICM_Mask.bmp
    └── ...
```

**Mask encoding:**
- `0` = Background
- `75` = Zona Pellucida (ZP)
- `150` = Inner Cell Mass (ICM)
- `255` = Trophectoderm (TE)


---

## 🚀 Quick Start

### 1️⃣ Training

Train an HRNet model with two-stage training (frozen then unfrozen):

```bash
python train.py --data_path ../dataset --epochs_frozen 20 --epochs_unfrozen 20 --batch_size 2
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | `../dataset` | Path to dataset directory |
| `--output_dir` | str | `../../models/blastocyst_segmentation` | Model save directory |
| `--backbone` | str | `hrnet_w30` | HRNet backbone (`hrnet_w18`, `hrnet_w30`, `hrnet_w32`, `hrnet_w48`) |
| `--epochs_frozen` | int | `20` | Epochs with frozen backbone |
| `--epochs_unfrozen` | int | `20` | Epochs with unfrozen backbone |
| `--batch_size` | int | `2` | Training batch size |
| `--lr_frozen` | float | `1e-4` | Learning rate (frozen) |
| `--lr_unfrozen` | float | `1e-7` | Learning rate (unfrozen) |
| `--weight_decay` | float | `1e-2` | L2 regularization |
| `--valid_pct` | float | `0.2` | Validation split percentage |
| `--image_size` | int | `480` | Input image size |
| `--patience` | int | `2` | Early stopping patience |
| `--seed` | int | `2020` | Random seed |
| `--find_lr` | flag | - | Run learning rate finder |

**Outputs:**
- `hrnet-before-unfreeze.pth`: Checkpoint after frozen training
- `hrnet-after-unfreeze.pth`: Checkpoint after unfrozen training
- `hrnet_final.pth`: Final traced model for inference

**Two-stage training:**
1. **Frozen stage**: Train only the segmentation head while keeping HRNet backbone frozen
2. **Unfrozen stage**: Fine-tune the entire model including backbone

---

### 2️⃣ Evaluation

Evaluate model performance on the test set:

```bash
python eval.py --model_path ../../models/blastocyst_segmentation/hrnet_final.pth --data_path ../dataset --output_csv results.csv --visualize --save_predictions
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | ✓ Required | Path to trained model (.pth) |
| `--data_path` | str | `../dataset` | Path to dataset directory |
| `--image_size` | int | `480` | Model input size |
| `--output_csv` | str | `""` | Save results to CSV |
| `--save_predictions` | flag | - | Save predicted masks |
| `--output_dir` | str | `predictions` | Prediction save directory |
| `--visualize` | flag | - | Show visualizations |
| `--num_visualize` | int | `5` | Number of samples to visualize |
| `--save_visualization` | str | `""` | Save visualization figure |

**Outputs:**
- **Console**: Metrics for ZP, TE, and ICM (accuracy, precision, recall, specificity, Jaccard, Dice)
- **CSV file**: Structured results table
- **Predictions**: Segmentation masks for test images
- **Visualizations**: Side-by-side comparison of original, prediction, and ground truth

**Metrics explained:**
- **Accuracy**: Overall pixel-wise correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **Jaccard (IoU)**: Intersection over Union
- **Dice coefficient**: 2 × (Intersection) / (Sum of both masks)

---

### 3️⃣ Inference

Generate segmentation masks for new images:

```bash
python inference.py --model_path ../../models/blastocyst_segmentation/hrnet_final.pth --input image.bmp --output result.png --save_overlay
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | ✓ Required | Path to trained model (.pth) |
| `--input` | str | ✓ Required | Image file or directory |
| `--output` | str | `""` | Output path (single image only) |
| `--output_dir` | str | `predictions` | Output directory (multiple images) |
| `--save_overlay` | flag | - | Save colored overlay visualization |
| `--overlay_alpha` | float | `0.5` | Overlay transparency (0-1) |
| `--image_size` | int | `480` | Model input size |

**Outputs:**
- **Segmentation masks**: Grayscale images with structure encoding (0, 75, 150, 255)
- **Colored overlays** (optional): Original image with colored masks
  - Blue = Zona Pellucida (ZP)
  - Green = Trophectoderm (TE)
  - Red = Inner Cell Mass (ICM)

**Examples:**
```bash
# Single image with overlay
python inference.py --model_path model.pth --input image.bmp --output mask.png --save_overlay

# Process entire directory
python inference.py --model_path model.pth --input test_images/ --output_dir results/ --save_overlay

# Custom overlay transparency
python inference.py --model_path model.pth --input image.bmp --output mask.png --save_overlay --overlay_alpha 0.3
```

---

## 💡 Example Workflow

```bash
# 1. Train with default settings
python train.py --data_path ../dataset --batch_size 2

# 2. Evaluate and save everything
python eval.py \
    --model_path ../../models/blastocyst_segmentation/hrnet_final.pth \
    --data_path ../dataset \
    --output_csv results.csv \
    --save_predictions \
    --visualize \
    --save_visualization evaluation_results.png

# 3. Run inference on new images
python inference.py \
    --model_path ../../models/blastocyst_segmentation/hrnet_final.pth \
    --input new_images/ \
    --output_dir predictions/ \
    --save_overlay

# 4. Train with larger backbone and more epochs
python train.py \
    --data_path ../dataset \
    --backbone hrnet_w48 \
    --epochs_frozen 30 \
    --epochs_unfrozen 30 \
    --batch_size 4 \
    --find_lr
```

---

## ⚙️ Technical Details

### Model Architecture
- **Base model**: HRNet (High-Resolution Network)
- **Backbones**: hrnet_w18, hrnet_w30 (default), hrnet_w32, hrnet_w48
- **Task**: Semantic segmentation (4 classes)
- **Framework**: fastai + semtorch

### Data Augmentation
Applied during training:
- Horizontal flip (50% probability)
- Random rotation (±10°, 40% probability)
- Random resized crop
- ImageNet normalization

### Training Strategy
1. **Frozen training**: Train segmentation head only
   - Higher learning rate (1e-4)
   - Fast convergence on basic features
   
2. **Unfrozen training**: Fine-tune entire network
   - Lower learning rate (1e-7)
   - Refinement of learned features

### Optimization
- **Optimizer**: Built-in fastai optimizer (Adam-based)
- **Learning rate**: One-cycle policy
- **Weight decay**: 1e-2 (default)
- **Mixed precision**: FP16 for faster training

---

## 🎨 Mask Visualization

Predicted masks use the following encoding:
- Dark gray (0): Background
- Light gray (75): Zona Pellucida
- White (255): Trophectoderm  
- Mid gray (150): Inner Cell Mass

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch_size` to 1 |
| Import error for `semtorch` | `pip install semtorch` |
| Import error for `albumentations` | `pip install albumentations` |
| Dataset path not found | Check `--data_path` points to correct location |
| Mask values incorrect | Ensure masks use encoding: 0, 75, 150, 255 |
| Model loading fails | Verify `--model_path` points to correct .pth file |
| Poor convergence | Try `--find_lr` to find optimal learning rate |

---

## 📊 Understanding Results

**Good segmentation metrics (typical ranges):**
- Dice coefficient: > 0.85 (excellent), > 0.70 (good)
- Jaccard index: > 0.75 (excellent), > 0.60 (good)
- Precision/Recall: > 0.80 (good balance)

**Structure-specific considerations:**
- **ZP**: Usually easiest to segment (clear boundary)
- **TE**: Moderate difficulty (larger region)
- **ICM**: Most challenging (small, variable appearance)

---

## 📝 Notes

- **GPU strongly recommended**: Training on CPU will be very slow
- **Batch size**: Default is 2 due to memory constraints; increase if you have more VRAM
- **Image size**: 480×480 is the default; larger sizes need more memory
- **Reproducibility**: Set `--seed` for consistent results
- **Early stopping**: Training stops automatically if validation loss doesn't improve
- **Model checkpoints**: Best models are saved automatically during training
- **Input formats**: Supports .bmp, .jpg, .png image formats

---

## 📖 File Structure

```
blastocyst_segmentation/
├── train.py          # Training script
├── eval.py           # Evaluation script
├── inference.py      # Inference script for new images
├── HRnet.ipynb       # Original notebook (reference)
└── README.md         # This file
```

---

## 🔗 References

**HRNet Paper:**
> Deep High-Resolution Representation Learning for Visual Recognition
> Wang et al., CVPR 2019

**Key advantages:**
- Maintains high-resolution representations throughout
- Parallel multi-resolution subnetworks
- Repeated multi-scale fusions
- Excellent for dense prediction tasks like segmentation

Happy segmenting! 🔬✨
