"""
Consistency evaluation: runs MultiTaskEvaluator (from eval.py) once per
checkpoint produced by consistency_test.py, and writes one CSV per task —
each row is an iteration (run), each column a metric.

Expects the folder layout produced by consistency_test.py:
    <runs_dir>/run_1/multitask_model_final_*.pth
    <runs_dir>/run_2/multitask_model_final_*.pth
    ...

Output:
    <output_dir>/exp.csv, icm.csv, te.csv, stage.csv, ploidy.csv, fhb.csv,
    fragmentation.csv, blasto_str.csv   (only for tasks present in the eval)
    each with columns: iteration, <metrics...>, plus trailing mean/std rows.

Usage:
    python consistency_eval.py --runs_dir .../consistency_test --n_runs 5 \
        --encoder resnet34 --transformer
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.multi_task import MultiTaskModel
from data.dataset import MultiTaskDataset, multitask_collate, MultiTaskBatchSampler
from data.managment import prepare_datasets
from eval import MultiTaskEvaluator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET


def build_model(args):
    if args.encoder == "resnet34":
        from models.encoder.visual_encoder import ResNetEncoder
        encoder = ResNetEncoder(pretrained=True)
    elif args.encoder == "efficientnet_b3":
        from models.encoder.visual_encoder import EfficientNetB3Encoder
        encoder = EfficientNetB3Encoder(pretrained=True)
    elif args.encoder == "femi":
        from models.encoder.femi_encoder import FEMIEncoder
        encoder = FEMIEncoder(freeze=True, embedding_dim=512)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    if args.transformer:
        from models.temporal.temporal_transformer import VideoTransformer
        transformer = VideoTransformer(args.frames)
    else:
        transformer = None

    return MultiTaskModel(encoder, transformer, embedding_dim=encoder.embedding_dim,
                           weighted_loss=args.weighted_loss)


def find_checkpoint(run_dir):
    """Prefer the final saved model; fall back to the latest phase checkpoint."""
    final_candidates = sorted(run_dir.glob("multitask_model_final_*.pth"))
    print(f"Found {len(final_candidates)} final model candidates in {run_dir}")
    if final_candidates:
        return final_candidates[0]
    for name in ["best_model_phase3.pth", "best_model_phase2.pth", "best_model_phase1.pth"]:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def summary_to_task_rows(summaries):
    """Group each run's flat/nested summary dict into one row per task."""
    task_rows = defaultdict(list)

    for it, summary in enumerate(summaries, start=1):
        # Classification tasks: exp_metrics, icm_metrics, te_metrics, stage_metrics, ploidy_metrics, fhb_metrics
        for key, val in summary.items():
            if key.endswith('_metrics') and isinstance(val, dict):
                task = key[: -len('_metrics')]
                row = {'iteration': it}
                row.update(val)
                task_rows[task].append(row)

        # Fragmentation segmentation (flat frag_* keys)
        frag_keys = {k: v for k, v in summary.items() if k.startswith('frag_')}
        if frag_keys:
            row = {'iteration': it}
            row.update({k[len('frag_'):]: v for k, v in frag_keys.items()})
            task_rows['fragmentation'].append(row)

        # Blastocyst structure segmentation (seg_class_X_metric keys)
        seg_keys = {k: v for k, v in summary.items() if k.startswith('seg_class_')}
        if seg_keys:
            row = {'iteration': it}
            row.update(seg_keys)
            task_rows['blasto_str'].append(row)

    return task_rows


def write_task_csvs(task_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for task, rows in task_rows.items():
        df = pd.DataFrame(rows).set_index('iteration')
        # Trailing mean/std rows for a quick consistency check at a glance
        mean_row = df.mean(numeric_only=True)
        std_row = df.std(numeric_only=True)
        df.loc['mean'] = mean_row
        df.loc['std'] = std_row

        csv_path = output_dir / f"{task}.csv"
        df.to_csv(csv_path)
        print(f"{task}: {csv_path}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = Path(args.path)
    image_samples, video_samples = prepare_datasets(dataset_path, args.transformer, split="test", tasks=args.tasks)
    test_dataset = MultiTaskDataset(
        image_samples=image_samples,
        video_samples=video_samples,
        img_size=(224, 224),
        use_femi_processor=args.encoder == "femi",
    )
    print(f"Total test samples: {len(test_dataset)}\n")

    test_loader = DataLoader(
        test_dataset,
        batch_sampler=MultiTaskBatchSampler(test_dataset, batch_size=args.batch_size),
        collate_fn=multitask_collate,
        num_workers=args.num_workers,
    )

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir / "consistency_eval"

    summaries = []
    for i in range(1, args.n_runs + 1):
        run_dir = runs_dir / f"run_{i}"
        checkpoint = find_checkpoint(run_dir)
        print("=" * 60)
        print(f"EVALUATING RUN {i}/{args.n_runs}: {checkpoint}")
        print("=" * 60)

        model = build_model(args)
        evaluator = MultiTaskEvaluator(
            model=model,
            checkpoint_path=str(checkpoint),
            device=device,
            multi_gpu=args.multi_gpu,
            use_transformer=args.transformer,
            output_dir=output_dir / f"run_{i}",
        )
        summary = evaluator.evaluate(test_loader)
        summaries.append(summary)

    task_rows = summary_to_task_rows(summaries)
    write_task_csvs(task_rows, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency evaluation across n repeated training runs")
    parser.add_argument("--path", type=str, default=PATH_DATASET, help="Path to dataset directory")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--tasks", nargs="+", default=None, help=(
            "Tasks to include in evaluation. Pass one or more of: "
            "exp icm te seg frag stage ploidy fhb. "
            "Omit to include all tasks (default behaviour)."
        ),
    )
    parser.add_argument("--runs_dir", type=str, required=True, help="Path to consistency_test's output root (contains run_1..run_N)")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to write per-task CSVs (default: <runs_dir>/consistency_eval)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--transformer", action="store_true")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--weighted_loss", action="store_true")
    args = parser.parse_args()
    main(args)