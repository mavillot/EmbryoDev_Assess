# EmbryoDev_Assess

Research codebase for automated embryo assessment experiments from time-lapse imaging and derived features.

> **Research use only (non-clinical).**
> This repository is intended for research, reproducibility, and method development. It is **not** validated for clinical use, diagnosis, treatment decisions, or regulatory use.

## Project purpose and scope

This repository consolidates multiple embryo assessment workflows, including:

- **Multi-task embryo assessment** (joint segmentation/classification/video branch) in `assessment/MultiTask/`
- **Ploidy workflows** in `assessment/ploidIA/`
- **Viability workflows** in `assessment/vIAbility/`
- **Marker-specific workflows** (grading, segmentation, fragmentation, cell count, stage, quality) in `markers/`
- Shared embryo utilities in `assessment/embryo_utils/`

The codebase contains both actively maintained paths and legacy/experimental scripts. The MultiTask path is the primary documented end-to-end pipeline.

## Repository structure

```text
.
├── assessment/
│   ├── MultiTask/           # primary multi-task training/eval/inference scripts
│   ├── embryo_utils/        # reusable embryo image/video utility code
│   ├── ploidIA/             # ploidy-oriented workflows
│   └── vIAbility/           # viability-oriented workflows
├── markers/                 # marker-specific modeling/evaluation workflows
├── config/                  # shared configuration values
├── docs/                    # lightweight documentation for public use
├── environment.yml          # conda environment definition
├── LICENSE                  # MIT license
└── README.md
```

## Multi-task architecture/workflow overview

`assessment/MultiTask` scripts orchestrate a shared multi-head model pipeline:

1. **Dataset assembly** from configured dataset root (`config/config.py`)
2. **Task selection** (`--tasks`) and optional temporal branch (`--transformer`)
3. **Staged training** in `train.py` (frozen encoder, optional progressive unfreezing phases)
4. **Evaluation** in `eval.py` with per-task metrics
5. **Inference** in `inference.py` for image and/or video inputs
6. **Consistency studies** with repeated runs (`consistency.py`, `consistency_eval.py`)

## Environment setup

Use the existing Conda environment file:

```bash
conda env create -f environment.yml
conda activate viability3d
```

### GPU/CUDA notes

- `environment.yml` is configured for **PyTorch + CUDA 11.8** (`pytorch-cuda=11.8`).
- Ensure your NVIDIA driver is compatible.
- If your system uses a different CUDA stack, adapt `environment.yml` accordingly before creating the environment.

### CPU-only note

- CPU-only execution is possible for many scripts, but substantially slower.
- Some workloads may still assume CUDA availability unless script flags are set to CPU.

## Data, checkpoints, and privacy responsibilities

- **Datasets and model checkpoints are not included.**
- You must provide your own data in the expected folder structures used by each workflow.
- You are responsible for dataset licensing, consent/privacy compliance, and secure handling of sensitive medical data.
- Do not commit private datasets, patient-identifiable metadata, checkpoints, or generated predictions to this repository.

## Reproducible CLI commands (verified against argparse interfaces)

From repository root:

### MultiTask training

```bash
python assessment/MultiTask/train.py \
  --encoder resnet34 \
  --device cuda \
  --batch_size 4
```

### MultiTask evaluation

```bash
python assessment/MultiTask/eval.py \
  --checkpoint /path/to/checkpoint.pth \
  --encoder resnet34
```

### MultiTask inference (images)

```bash
python assessment/MultiTask/inference.py \
  --checkpoint /path/to/checkpoint.pth \
  --input_images /path/to/image_or_dir \
  --encoder resnet34
```

### MultiTask consistency runs

```bash
python assessment/MultiTask/consistency.py --encoder resnet34
python assessment/MultiTask/consistency_eval.py --runs_dir /path/to/consistency_output
```

### Additional major workflow entry points

```bash
python assessment/vIAbility/eval.py --ckpt /path/to/checkpoint.pt
python assessment/vIAbility/inference.py /path/to/video.mp4
python markers/evaluate_multitask.py --path /path/to/dataset --checkpoint /path/to/checkpoint.pth
```

> Note: successful execution also depends on external data layout and some model modules/assets that are not packaged in this repository clone.

## Configuration

Shared defaults live in `config/config.py` and can be overridden with environment variables:

- `EMBRYODEV_DATASET_PATH`
- `EMBRYODEV_MODELS_PATH`
- `EMBRYODEV_TMP_PATH`
- `EMBRYODEV_OUTPUT_PATH`

## Known limitations

- Several scripts rely on local data conventions that are not fully standardized across all subprojects.
- Some workflows reference optional pretrained assets not bundled in the repository.
- Legacy/experimental folders are kept for research traceability and may require additional setup.

## Troubleshooting (concise)

- **Import/path errors (`ModuleNotFoundError`)**: run from repository root and verify project-relative imports.
- **CUDA errors**: confirm driver/CUDA compatibility with your installed PyTorch build.
- **Missing checkpoints/data**: provide valid paths and ensure expected split/file naming conventions.
- **Out-of-memory**: reduce batch size/frames and use CPU mode if necessary.

## Citation and reuse

If you use this repository in academic work, please cite your corresponding paper/project and reference this repository URL and commit hash.
A starter citation file is provided in `CITATION.cff` and should be updated with publication metadata when available.

## Contributing and issues

- See `CONTRIBUTING.md` for contribution workflow.
- Use GitHub Issues to report bugs, reproducibility blockers, or documentation gaps.
- For security-sensitive disclosures, see `SECURITY.md`.

## License

MIT License (see `LICENSE`).
