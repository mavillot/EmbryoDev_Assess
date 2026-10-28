"""
Consistency test for the MultiTask training pipeline.

Runs the full train.py pipeline (Phase 1 / 2 / 3) n=5 times with different
random seeds, keeping every run's outputs in its own subfolder, logging
each run's result to a master CSV as it finishes, and producing an
aggregated Excel summary (mean / std / min / max per metric) at the end.

Usage: same flags as train.py, plus --n_runs (default 5) and --seed_base.
Example:
    python consistency_test.py --encoder resnet34 --transformer --phase2 --phase3
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train import Trainer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET, PATH_MODELS

from models.multi_task import MultiTaskModel
from data.dataset import MultiTaskDataset
from data.managment import prepare_datasets
from data.augmentation import get_train_transforms, get_val_transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    if args.encoder == "resnet34":
        from models.encoder.visual_encoder import ResNetEncoder
        encoder = ResNetEncoder(pretrained=True)
    elif args.encoder == "efficientnet_b3":
        from models.encoder.visual_encoder import EfficientNetB3Encoder
        encoder = EfficientNetB3Encoder(pretrained=True)
    elif args.encoder == "femi":
        from models.encoder.femi_encoder import FEMIEncoder
        encoder = FEMIEncoder(freeze=False, embedding_dim=512)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    if args.transformer:
        from models.temporal.temporal_transformer import VideoTransformer
        transformer = VideoTransformer(args.frames)
    else:
        transformer = None

    return MultiTaskModel(encoder, transformer, embedding_dim=encoder.embedding_dim,
                           weighted_loss=args.weighted_loss)


def run_once(args, run_idx, run_dir):
    """One full Phase1/2/3 training run. Mirrors train.py's main()."""
    seed = args.seed_base + run_idx
    set_seed(seed)
    print("=" * 60)
    print(f"RUN {run_idx + 1}/{args.n_runs}  (seed={seed})")
    print("=" * 60)

    path = Path(PATH_DATASET)
    image_samples, video_samples = prepare_datasets(path, args.transformer, 'train', args.tasks, args.frames)

    val_ratio = 0.15
    all_samples = image_samples
    random.shuffle(all_samples)
    split = int(len(all_samples) * (1 - val_ratio))
    train_images, val_images = all_samples[:split], all_samples[split:]

    random.shuffle(video_samples)
    vid_split = int(len(video_samples) * (1 - val_ratio))
    train_videos, val_videos = video_samples[:vid_split], video_samples[vid_split:]

    size = (224, 224)
    train_dataset = MultiTaskDataset(
        image_samples=train_images, video_samples=train_videos, img_size=size,
        transform=get_train_transforms(size, normalize=args.encoder != "femi") if args.data_augmentation else None,
        use_femi_processor=args.encoder == "femi",
    )
    val_dataset = MultiTaskDataset(
        image_samples=val_images, video_samples=val_videos, img_size=size,
        transform=get_val_transforms(normalize=args.encoder != "femi"),
        use_femi_processor=args.encoder == "femi",
    )

    model = build_model(args)

    _task_weights = json.loads(args.task_weights) if args.task_weights else None

    run_dir.mkdir(parents=True, exist_ok=True)

    phase_boundary_epochs = []
    t0 = time.time()

    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=1e-3,
        device=args.device,
        multi_gpu=True,
        freeze_encoder=True,
        best_model_path=run_dir / "best_model_phase1.pth",
        oversample=args.oversample,
        oversample_factor=args.oversample_factor,
        task_weights=_task_weights,
    )
    best_loss_phase1 = trainer.train(epochs=20, patience=5)
    phase_boundary_epochs.append(len(trainer.history['train_loss']))

    best_loss_phase2 = None
    if args.phase2:
        trainer.best_model_path = run_dir / "best_model_phase2.pth"
        trainer.unfreeze_encoder(layers=1, new_lr=1e-4)
        best_loss_phase2 = trainer.train(epochs=5, patience=5)
        phase_boundary_epochs.append(len(trainer.history['train_loss']))

    best_loss_phase3 = None
    if args.phase3:
        trainer.best_model_path = run_dir / "best_model_phase3.pth"
        trainer.unfreeze_encoder(layers=2, new_lr=1e-5)
        best_loss_phase3 = trainer.train(epochs=5, patience=7)

    trainer.save_history(run_dir / "loss_history.json")
    trainer.plot_history(path=run_dir / "loss_curves.png", phase_epochs=phase_boundary_epochs)

    final_loss = best_loss_phase3 if best_loss_phase3 is not None else \
        (best_loss_phase2 if best_loss_phase2 is not None else best_loss_phase1)
    final_model_path = run_dir / f"multitask_model_final_{final_loss:.4f}.pth"
    trainer.save_checkpoint(final_model_path)

    elapsed_min = (time.time() - t0) / 60

    return {
        "run": run_idx + 1,
        "seed": seed,
        "phase1_best_loss": best_loss_phase1,
        "phase2_best_loss": best_loss_phase2,
        "phase3_best_loss": best_loss_phase3,
        "final_loss": final_loss,
        "n_epochs_total": len(trainer.history['train_loss']),
        "elapsed_min": round(elapsed_min, 2),
    }


def append_to_master_csv(master_csv, row):
    """Log this run's result immediately, so partial progress survives a crash."""
    file_exists = master_csv.exists()
    with open(master_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged run {row['run']} to {master_csv}")


def write_summary_excel(master_csv, summary_xlsx):
    """Aggregate all runs into mean/std/min/max per metric."""
    df = pd.read_csv(master_csv)
    metric_cols = ["phase1_best_loss", "phase2_best_loss", "phase3_best_loss",
                    "final_loss", "n_epochs_total", "elapsed_min"]

    summary = df[metric_cols].agg(['mean', 'std', 'min', 'max']).T
    summary.index.name = "metric"
    summary = summary.reset_index()

    with pd.ExcelWriter(summary_xlsx, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='runs', index=False)
        summary.to_excel(writer, sheet_name='summary', index=False)

    print(f"Consistency summary saved to {summary_xlsx}")
    print(summary.to_string(index=False))


def main(args):
    tasks_tag = "_".join(sorted(args.tasks)) if args.tasks else "all"
    base_path = Path(args.path_model) /"consistency_test"/ args.encoder / ("transformer" if args.transformer else "no_transformer") / tasks_tag
    consistency_root = Path("results/assessment/MultiTask/consistency_test") / args.encoder / ("transformer" if args.transformer else "no_transformer") / tasks_tag
    base_path.mkdir(parents=True, exist_ok=True)

    master_csv = consistency_root / "consistency_master.csv"
    summary_xlsx = consistency_root / "consistency_summary.xlsx"

    for run_idx in range(args.n_runs):
        run_dir = consistency_root / f"run_{run_idx + 1}"
        result = run_once(args, run_idx, run_dir)
        append_to_master_csv(master_csv, result)

    write_summary_excel(master_csv, summary_xlsx)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consistency test (n repeated runs) for MultiTask training")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--path_model', type=str, default=PATH_MODELS+'assessment/Multitask/')
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--oversample_factor', type=float, default=1.5)
    parser.add_argument('--task_weights', type=str, default=None)
    parser.add_argument('--phase2', action='store_true')
    parser.add_argument('--phase3', action='store_true')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of repeated runs for the consistency test (default 5)')
    parser.add_argument('--seed_base', type=int, default=0, help='Base seed; run i uses seed_base + i')
    args = parser.parse_args()
    main(args)