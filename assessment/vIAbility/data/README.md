# Feature Extraction Scripts

Three independent scripts for extracting features from embryo videos, plus a shared utilities module.

## Directory layout

```
extraction/
├── utils.py              # shared: paths, frame I/O, logging, split loading
├── extract_femi.py       # FEMI (ViT-MAE) → .pt  (T, 768)
├── extract_resnet34.py   # ResNet-34      → .pt  (T, 512)
└── extract_markers.py    # EmbryoVideo    → .csv
```

All three scripts read the same `train.csv` / `test.csv` from `<PATH_DST>/` and write outputs under:

```
<PATH_DST>/
├── femi/
│   ├── extraction_config.json
│   ├── train/{FHB,no_FHB}/<video>.pt
│   └── test/ {FHB,no_FHB}/<video>.pt
├── resnet34/
│   ├── extraction_config.json
│   ├── train/{FHB,no_FHB}/<video>.pt
│   └── test/ {FHB,no_FHB}/<video>.pt
└── markers/
    ├── extraction_config.json
    ├── train/{FHB,no_FHB}/<video>.csv
    └── test/ {FHB,no_FHB}/<video>.csv
```

---

## Running each extractor

All scripts are resume-friendly: already-processed files are skipped unless `--overwrite` is passed.

### FEMI
```bash
python extract_femi.py                        # defaults
python extract_femi.py \
    --model_id   ihlab/FEMI \
    --device     cuda \
    --batch_frames 32 \
    --overwrite
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `ihlab/FEMI` | HuggingFace model ID |
| `--hf_token` | `$HF_TOKEN` | Auth token for gated models |
| `--n_frames` | 214 | Frames sampled per video |
| `--start_time` | 0.0 | Sampling window start (hours) |
| `--frequency` | 1.0 | Time step between frames (hours) |
| `--frame_size` | 224 | Resize each frame to N×N |
| `--batch_frames` | 16 | Forward-pass batch size (↓ if OOM) |
| `--device` | auto | `cpu` / `cuda` / `cuda:1` |
| `--overwrite` | off | Re-extract existing files |

### ResNet-34
```bash
python extract_resnet34.py                    # defaults
python extract_resnet34.py \
    --model_path models/markers/stage_classif/resnet34/resnet34_we15_e100.pth \
    --device cuda \
    --overwrite
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `models/…/resnet34_we15_e100.pth` | Path to saved `.pth` |
| `--n_frames` | 214 | |
| `--start_time` | 0.0 | |
| `--frequency` | 1.0 | |
| `--frame_size` | 224 | |
| `--batch_frames` | 16 | |
| `--device` | auto | |
| `--overwrite` | off | |

### EmbryoVideo markers
```bash
python extract_markers.py                     # defaults
python extract_markers.py --overwrite
```

| Argument | Default | Description |
|---|---|---|
| `--n_frames` | 214 | |
| `--start_time` | 0.0 | |
| `--frequency` | 1.0 | |
| `--frame_size` | 224 | |
| `--overwrite` | off | |

---

## Output format

### `.pt` files (FEMI & ResNet-34)
```python
data = torch.load("video.pt")
data["embeddings"]  # FloatTensor  (T, 768) for FEMI  /  (T, 512) for ResNet-34
data["mask"]        # LongTensor   (T,)   1 = valid frame, 0 = blank
data["video_file"]  # str          original filename
data["n_valid"]     # int          number of non-blank frames
data["label"]       # int          0 = no_FHB, 1 = FHB  (ResNet-34 only)
```

### `.csv` files (markers)
Each row corresponds to one frame; columns are defined by `EmbryoVideo.save_evolution()`.

---

## Key design decisions

- **`utils.py` is the single source of truth** for paths, frame I/O, split loading, and directory helpers — none of that logic is duplicated across scripts.
- **train/test split is always read from `train.csv` / `test.csv`** — no random splitting inside extraction scripts.
- **All scripts are resume-friendly** — skip already-done files by default; use `--overwrite` to redo.
- **An `extraction_config.json` is written to each output directory** for full reproducibility.
