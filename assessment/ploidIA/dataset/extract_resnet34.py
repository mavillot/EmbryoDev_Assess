"""
extract_resnet34.py — extract ResNet-34 penultimate-layer features per video.

Output layout
-------------
<PATH_DST>/resnet34/
    extraction_config.json
    train/
        FHB/     <video_filename>.pt
        no_FHB/  <video_filename>.pt
    test/
        FHB/     <video_filename>.pt
        no_FHB/  <video_filename>.pt

Each .pt file contains:
    {
        "embeddings": FloatTensor (T, 512),
        "mask":       LongTensor  (T,)      1 = valid frame, 0 = blank
        "video_file": str,
        "label":      int,
        "n_valid":    int,
    }

Usage
-----
    python extract_resnet34.py
    python extract_resnet34.py --model_path path/to/resnet34.pth --device cuda --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

# ── shared helpers ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assessment.vIAbility.data.data_utils import (
    PATH_DST, PATH_EXCEL,
    DEFAULT_N_FRAMES, DEFAULT_START_TIME, DEFAULT_FREQ, DEFAULT_FRAME_SIZE,
    get_logger, load_split, load_metadata, extract_frames,
    make_split_dirs, out_path,
)

log = get_logger(__name__)

# ── ImageNet normalisation ────────────────────────────────────────────────────
_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

RESNET34_DST = PATH_DST / "resnet34"

# ── default model path ────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "models/markers/stage_classif/resnet34/resnet34_we15_e100.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_resnet34(model_path: str, device: str) -> torch.nn.Module:
    """
    Load a saved ResNet-34, strip the final FC layer, and set eval mode.
    Returns a feature extractor with output shape (B, 512).
    """
    log.info("Loading ResNet-34 from '%s' onto %s …", model_path, device)
    model = torch.load(model_path, weights_only=False, map_location=device)
    model = torch.nn.Sequential(*list(model.children())[:-1])   # drop classifier
    model = model.to(device)
    model.eval()
    log.info("ResNet-34 loaded.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batched(
    frames,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Run ResNet-34 in mini-batches and return (T, 512) feature tensor.

    Returns
    -------
    features : FloatTensor (T, 512) on CPU
    """
    transformed  = [_PREPROCESS(frame) for frame in frames]
    all_features = []

    with torch.no_grad():
        for i in range(0, len(transformed), batch_size):
            batch = torch.stack(transformed[i: i + batch_size]).to(device)
            feats = model(batch).squeeze(-1).squeeze(-1)   # (B, 512)
            all_features.append(feats.cpu())
            del batch, feats
            torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0)   # (T, 512)


# ─────────────────────────────────────────────────────────────────────────────
# Per-split extraction loop
# ─────────────────────────────────────────────────────────────────────────────

def run_split(
    split: str,
    filenames: list[str],
    labels: list[int],
    metadata_df,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Extract and save ResNet-34 features for one split. Returns (done, skipped, errors)."""
    done = skipped = errors = 0

    log.info("── %s  (%d videos) ──────────────────────────────────────", split.upper(), len(filenames))

    for video_file, label in tqdm(zip(filenames, labels), total=len(filenames),
                                  desc=f"ResNet34 {split}", unit="video"):
        dst = out_path(RESNET34_DST, split, label, video_file, ".pt")

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        # ── load frames ───────────────────────────────────────────────────
        try:
            frames, mask = extract_frames(
                video_file, metadata_df,
                start_time=args.start_time,
                freq=args.frequency,
                n_frames=args.n_frames,
                resize=args.frame_size,
                log=log,
            )
        except Exception as exc:
            log.error("Frame extraction failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        n_valid = sum(mask)
        if n_valid == 0:
            log.warning("All frames blank for '%s' — saving anyway.", video_file)

        # ── ResNet-34 inference ───────────────────────────────────────────
        try:
            features = extract_features_batched(
                frames, model,
                device=args.device,
                batch_size=args.batch_frames,
            )
        except RuntimeError as exc:
            log.error("ResNet-34 inference failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        # ── save ──────────────────────────────────────────────────────────
        torch.save(
            {
                "embeddings": features,             # (T, 512) float32
                "mask":       torch.tensor(mask),   # (T,)     int
                "video_file": video_file,
                "label":      label,
                "n_valid":    n_valid,
            },
            dst,
        )
        done += 1

    log.info("%s done=%d  skipped=%d  errors=%d", split, done, skipped, errors)
    return done, skipped, errors


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── output dirs ───────────────────────────────────────────────────────
    make_split_dirs(RESNET34_DST)

    # ── save extraction config ────────────────────────────────────────────
    config_path = RESNET34_DST / "extraction_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Config saved → %s", config_path)

    # ── load model ────────────────────────────────────────────────────────
    model = load_resnet34(args.model_path, args.device)

    # ── load metadata & splits ────────────────────────────────────────────
    metadata_df = load_metadata(PATH_EXCEL)
    log.info("Metadata: %d rows.", len(metadata_df))

    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")

    # ── run ───────────────────────────────────────────────────────────────
    total_done = total_skipped = total_errors = 0
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        d, s, e = run_split(split, X, y, metadata_df, model, args)
        total_done    += d
        total_skipped += s
        total_errors  += e

    log.info(
        "All done. done=%d  skipped=%d  errors=%d  features → %s",
        total_done, total_skipped, total_errors, RESNET34_DST,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ResNet-34 features from embryo videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",   default=DEFAULT_MODEL_PATH,
                        help="Path to saved ResNet-34 .pth file")
    parser.add_argument("--n_frames",     type=int,   default=DEFAULT_N_FRAMES)
    parser.add_argument("--start_time",   type=float, default=DEFAULT_START_TIME)
    parser.add_argument("--frequency",    type=float, default=DEFAULT_FREQ)
    parser.add_argument("--frame_size",   type=int,   default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--batch_frames", type=int,   default=16,
                        help="Frames per forward pass (reduce if OOM)")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Re-extract even if output file already exists")
    main(parser.parse_args())
