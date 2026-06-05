# 🧬 Blastocyst Grading with Deep Learning

A friendly toolkit for training, evaluating, and running inference on blastocyst embryo grading models! This project uses InceptionV3 to classify blastocyst images across three important grading tasks.

---

## 📋 What's Inside?

This module provides tools to:
- **Train** models on blastocyst grading tasks
- **Evaluate** model performance on test datasets
- **Predict** grades for new blastocyst images

Supported grading tasks:
- **EXP** (Expansion stage): Classes 1-5
- **ICM** (Inner Cell Mass): Classes A, B, C, ND - Not Determined
- **TE** (Trophectoderm): Classes A, B, C, ND - Not Determined

---

## 📚 Dataset

We use the publicly available **blastocyst dataset**:

**Source:** [Blastocyst dataset](https://figshare.com/articles/figure/Blastocyst_dataset_zip/20123153/3?file=39348899)

**Citation:**

```bibtex
@article{Kromp2022,
author = "Florian Kromp and Raphael Wagner and Basak Balaban and Veronique Cottin and Irene Cuevas Saiz and Clara Dacho and Peter Fancsovits and Mohamed Fawzy and Lukas Fischer and Necati Findikli and Borut Kovacic and Dejan Ljiljak and Iris Martinez Rodero and Lodovico Parmegiani and Omar Shebl and Xie Min and Thomas Ebner",
title = "{A human blastocyst dataset including clinical annotations to benchmark deep learning architectures for in vitro fertilization}",
year = "2022",
month = "6",
url = "https://figshare.com/articles/figure/Blastocyst_dataset_zip/20123153",
doi = "10.6084/m9.figshare.20123153.v3"
}
```

After downloading and unzipping, your dataset should look like:
```
Blastocyst_Dataset/
├── Images/                              # All embryo images
├── Gardner_train_silver.csv             # Training labels
└── Gardner_test_gold_onlyGardnerScores.csv  # Test labels
```

---

## 🛠️ Setup & Requirements

### Dependencies
```bash
pip install torch torchvision pandas pillow scikit-learn
```

### Configuration
Update `PATH_DATASET` in `config/config.py` to point to the folder where the Blastocyst_Dataset folder is.

---

## 🚀 Quick Start

## 📖 Folder Structure

```
blastocyst_grading/
├── train.py          # Training script
├── eval.py           # Evaluation script
├── inference.py      # Inference script
└── README.md         # This file
```

Scripts are designed to be run from the root directory.

### 1️⃣ Training

Train a model on your chosen grading task:

```bash
python -m markers.blastocyst_grading.train.py TASK [--model_name MODEL_NAME] [--epochs EPOCHS] [--batch_size BATCH] [--lr LEARNING_RATE] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]

```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `task` | str | ✓ Required | `EXP`, `ICM`, or `TE` |
| `--model_name` | str | `inception_v3` | Model architecture |
| `--epochs` | int | `50` | Number of training epochs |
| `--batch_size` | int | `32` | Batch size |
| `--num_workers` | int | `0` | DataLoader workers |
| `--lr` | float | `0.01` | Learning rate |
| `--momentum` | float | `0.9` | SGD momentum |
| `--weight_decay` | float | `1e-4` | L2 regularization |

**Output:** Trained model weights saved to `models/{task}_{loss:.4f}.pt`

---

### 2️⃣ Evaluation

Evaluate model performance on the test set:

```bash
python -m markers.blastocyst_grading.eval.py TASK [--model_name MODEL_NAME] [--checkpoint CHECKPOINT] [--batch_size BATCH] 
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `task` | str | ✓ Required | `EXP`, `ICM`, or `TE` |
| `--model_name` | str | `inception_v3` | Model architecture |
| `--checkpoint` | Path | ✓ Required | Path to model checkpoint file |
| `--batch_size` | int | `32` | Batch size |

**Output:**
- Test loss
- Classification report (precision, recall, F1-score)
- Confusion matrix

---

### 3️⃣ Inference

Generate predictions for new images:

```bash
python -m markers.blastocyst_grading.inference.py TASK [--input INPUT] [--model_name MODEL_NAME] [--checkpoint CHECKPOINT] [--output OUTPUT] 
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `task` | str | ✓ Required | `EXP`, `ICM`, or `TE` |
| `--input` | str | ✓ Required | Single image file or directory |
| `--checkpoint` | str | ✓ Required | Path to model weights |
| `--output` | str | `output/blastocyst_grading/predictions.csv` | Output CSV filename |

**Output:** CSV with columns:
- `filename`: Image filename
- `prediction`: Predicted grade
- `prob_{class}`: Probability for each class

---

## 💡 Example Workflow

```bash
# 1. Train a model on expansion grading
python -m markers.blastocyst_grading.train.py EXP --epochs 50 --batch_size 32 --lr 0.01 --momentum 0.9 --weight_decay 1e-4

# 2. Evaluate on test set
python -m markers.blastocyst_grading.eval.py EXP --weights models/EXP_0.1234.pt

# 3. Make predictions on new images
python -m markers.blastocyst_grading.inference.py --input new_images/ --task EXP --checkpoint models/EXP_0.1234.pt
```

---

## ⚙️ Technical Details

- **Architecture:** InceptionV3 with custom output layer
- **Input size:** 299×299 pixels
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Optimizer:** SGD with momentum
- **Loss function:** Cross-entropy
- **Device:** GPU (if available) or CPU fallback

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Invalid path" in inference | Check that `--input` points to a valid file or directory |
| Model loading fails | Ensure `--checkpoint` path is correct |
| CUDA out of memory | Reduce `--batch_size` or use CPU |
| Missing dataset | Configure `PATH_DATASET` in `config/config.py` |

---

## 📝 Notes

- Trained models are saved in `models/blastocyst_grading` directory

---




