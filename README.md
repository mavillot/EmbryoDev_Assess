# 🧫 Embryo Development Assessment 🧫

EmbryoDev Assess is a research-oriented repository for embryo assessment pipelines.

This README documents the source code that is part of the tracked project structure and focuses on the multi-task deep learning workflow under `assessment/MultiTask`.
This respository is thought as repository where all the assessments task will be included in the assessment (e.g. future viability prediction)

## Repository Scope

Main documented modules:

- `assessment/MultiTask`: multi-task training, evaluation, inference, dataset loading, augmentations, and model definitions.
- `config`: shared paths and constants used by scripts.

## 🗂️ Project Structure

```text
.
├── README.md
├── assessment/
│   └── MultiTask/
│       ├── train.py
│       ├── eval.py
│       ├── inference.py
│       ├── data/
│       │   ├── augmentation.py
│       │   ├── dataset.py
│       │   └── managment.py
│       └── models/
│           ├── multi_task.py
│           ├── encoder/
│           │   └── visual_encoder.py
│           ├── heads/
│           │   ├── classification.py
│           │   └── segmentation.py
│           └── temporal/
│               └── temporal_transformer.py
└── config/
		├── __init__.py
		└── config.py
```

## 🔍 What The MultiTask Pipeline Predicts
A single model, multiple tasks 🚀

The multi-task model combines image-based and optional video-based outputs:

### Segmentation 
- Blastocyst structure segmentation (`blasto_str`)
- Fragmentation segmentation (`fragmentation`)
### Classification
- Morphological grading classification (`exp`, `icm`, `te`)
### Temporal 🎥
- Ploidy classification from video (`ploidy`, when transformer/video branch is enabled)

## ⚙️ Configuration

Global paths and constants are defined in `config/config.py`.

🔑 Important keys:

- `PATH_DATASET`: base dataset directory (default: `../Datasets/Processed/`)
- `OUTPUT_PATH`: folder used by evaluation/inference outputs
- `TMP_PATH`: temporary files folder
- label value maps and image-size defaults

⚠️ Update these values before running experiments in a new environment.

## 🚀 Quick Start

### 1️⃣ Environment

Use Python 3.10+ and install the common dependencies used across scripts:

```bash
pip install torch torchvision numpy pandas opencv-python matplotlib seaborn scikit-learn pillow tqdm
```

### 2️⃣ Training 🏋️‍♂️

From the repository root:

```bash
python assessment/MultiTask/train.py \
	--encoder resnet34 \
	--device cuda \
	--batch_size 4 \
	--transformer
```

Useful optional flags:

- `--multi_gpu`: enables multiGPU
- `--weighted_loss`: class-loss weighting
- `--data_augmentation`: enable augmentation pipeline
- `--path_model`: output/checkpoint directory
- `--baseline`: run a baseline model mode

### 3️⃣ Evaluation 📊

```bash
python assessment/MultiTask/eval.py \
	--path ../Datasets/Processed/ \
	--checkpoint /path/to/checkpoint.pth \
	--encoder resnet34 \
```

Useful optional flags:

- `--batch_size`
- `--multi_gpu`
- `--transformer`
- `--output_dir`

### 4️⃣ Inference 🔮

```bash
python assessment/MultiTask/inference.py \
	--input_images /path/to/images_or_image \
	--checkpoint /path/to/checkpoint.pth \
	--encoder resnet34
```

🎥 For video inference, provide `--input_videos` and enable `--transformer`.

Additional options:

- `--batch_size`
- `--image_size`
- `--n_frames`
- `--multi_gpu`
- `--output_dir`
- `--diagnose --diagnose_n 4` (BatchNorm train/eval gap diagnosis)

## Data Notes

The training/evaluation scripts use dataset management utilities in `assessment/MultiTask/data/managment.py` and custom samplers/collate logic in `assessment/MultiTask/data/dataset.py`.

To avoid runtime path issues:

- keep dataset folder structure consistent with what `prepare_datasets` expects
- validate that `PATH_DATASET` points to the correct root before launching runs

## Reproducibility

- Seed defaults are configured in `config/config.py`.
- Keep experiment arguments and checkpoints logged per run.
- Prefer fixed train/val/test splits when comparing model variants.

## Current Focus

The maintained, documented code path in this repository is the multi-task embryo assessment workflow under `assessment/MultiTask`.