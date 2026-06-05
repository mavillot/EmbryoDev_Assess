# Fragmentation Segmentation

Semantic segmentation of embryo fragmentation (cleavage-stage). This directory includes scripts to create masks, train a model, evaluate it, and run inference. The dataset comes without masks; you must generate them first.

---

## 📁 What’s Here

- [mask_extraction.py](markers/fragmentation/mask_extraction.py) — build binary masks from COCO annotations
- [train.py](markers/fragmentation/train.py) — train HRNet (or other backbones) on the generated masks
- [eval.py](markers/fragmentation/eval.py) — evaluate trained models
- [inference.py](markers/fragmentation/inference.py) — run predictions on new images

---

## 📊 Dataset

- Source: CleavageEmbryo dataset on Kaggle — https://www.kaggle.com/datasets/austin012/cleavageembryo-dataset/data
- Classes: Background, Fragmentation (binary)
- Images: RGB `.jpg`
- Masks: binary `.png` (0 background, 255 fragmentation) — **must be generated**

### Citation

```bibtex
@article{Zhang_2024,
		author = {Zhang, Chensheng and Shi, Xintong and Yin, Xinyue and Sun, Jiayi and Zhao, Jianhui and Zhang, Yi},
		title = {Cleavage-stage embryo segmentation using SAM-based dual branch pipeline: development and evaluation with the CleavageEmbryo dataset},
		journal = {Bioinformatics},
		volume = {41},
		number = {4},
		pages = {btae617},
		year = {2024},
		month = {10},
		issn = {1367-4811},
		doi = {10.1093/bioinformatics/btae617},
		url = {https://doi.org/10.1093/bioinformatics/btae617},
		eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/4/btae617/59875783/btae617.pdf},
}
```

---

## 🧭 Mask Extraction (required)

The dataset does **not** ship masks. Generate them first with [mask_extraction.py](markers/fragmentation/mask_extraction.py).

### How it works
- Reads COCO annotations (`instances_val.json` by default) under `PATH_DATASET/CleavageEmbryo/annotations/`
- Keeps only `category_id == 2` (fragmentation class)
- Decodes polygons/RLE to binary masks
- Saves to `PATH_DATASET/CleavageEmbryo/masks/val/` as `.png` (0 background, 255 fragmentation)

### Run
```bash
pip install pycocotools opencv-python
python markers/fragmentation/mask_extraction.py
```

### Train split
If you also need training masks, edit `json_path` and `output_dir` in the script to point to `instances_train.json` and `masks/train/`, then re-run.

---

## 🛠️ Setup

Install main dependencies (GPU recommended):
```bash
pip install torch torchvision fastai albumentations semtorch opencv-python pillow matplotlib numpy pandas
```

Set your dataset path in [config/config.py](config/config.py):
```python
PATH_DATASET = "/abs/path/to/datasets"
```

Expected structure after masks are generated:
```
PATH_DATASET/CleavageEmbryo/
├── images/
│   ├── train/
│   └── val/
├── masks/
│   ├── train/   # if generated
│   └── val/
└── annotations/
		├── instances_train.json
		└── instances_val.json
```

---

## 🎯 Training (train.py)

Two-stage training (frozen, then unfrozen backbone) using FastAI + SemTorch HRNet.

Basic run:
```bash
python markers/fragmentation/train.py
```

Common options:
```bash
python markers/fragmentation/train.py \
	--dataset CleavageEmbryo \
	--size 480 \
	--batch_size 16 \
	--epochs_before_unfreeze 20 \
	--epochs_after_unfreeze 10 \
	--lr 0.001 \
	--weight_decay 1e-4 \
	--model_name hrnet \
	--backbone_name hrnet_w30
```

Outputs (default):
- Models: `models/fragmentation/` (`hrnet_before_unfreeze.pth`, `hrnet_after_unfreeze.pth`, `fragmentation_hrnet.pth` TorchScript)
- Preview: `output/fragmentation/train_batch_augmentation.png`

---

## 📈 Evaluation (eval.py)

Evaluate a trained model on validation/test data:
```bash
python markers/fragmentation/eval.py
```

Metrics typically include pixel accuracy, background accuracy, fragmentation accuracy, IoU, and Dice.

---

## 🔮 Inference (inference.py)

Run predictions on new images:
```python
import torch
from markers.fragmentation.inference import predict_fragmentation, color_inference

model = torch.jit.load("models/fragmentation/fragmentation_hrnet.pth").eval()
mask = predict_fragmentation(model, "path/to/image.jpg", shape=(480, 480))

# Optional overlay
from PIL import Image
import cv2
img = Image.open("path/to/image.jpg").convert("RGB")
overlay = color_inference(img, mask)
cv2.imwrite("out.png", overlay)
```

`mask` is a binary numpy array (0 background, 255 fragmentation). You can also batch over image files with the same functions.

---

## ✅ Repro Steps (quick checklist)

1) Download dataset from Kaggle and place under `PATH_DATASET/CleavageEmbryo/`
2) Generate masks: `python markers/fragmentation/mask_extraction.py`
3) Configure `PATH_DATASET` in [config/config.py](config/config.py)
4) Train: `python markers/fragmentation/train.py`
5) Evaluate: `python markers/fragmentation/eval.py`
6) Infer on new images using [inference.py](markers/fragmentation/inference.py)

---

## 📚 References & Links

- Dataset: https://www.kaggle.com/datasets/austin012/cleavageembryo-dataset/data
- Paper: https://doi.org/10.1093/bioinformatics/btae617
- HRNet: https://arxiv.org/abs/1908.07919
- FastAI docs: https://docs.fast.ai/
- SemTorch: https://github.com/mrm8488/semtorch

---

Last updated: January 2026
