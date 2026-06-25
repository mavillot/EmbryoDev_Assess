"""
extract_resnet34.py — extract ResNet-34 penultimate-layer features per video.

Frame sampling matches EmbryoVideo (embryo_video.py) and extract_femi.py:
evenly spaced indices 0, step, 2*step, ... where step = n_total // n_frames,
read directly via cv2.VideoCapture (no metadata/Excel dependency).

Folder layout expected under --video_dir (mirrors extract_femi.py):
    <video_dir>/train/FHB/*.mp4
    <video_dir>/train/no_FHB/*.mp4
    <video_dir>/test/FHB/*.mp4
    <video_dir>/test/no_FHB/*.mp4

Output layout
-------------
<out_dir>/
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
        "label":      int,   # 1 = FHB, 0 = no_FHB
        "n_valid":    int,
    }

Usage
-----
    python extract_resnet34.py
    python extract_resnet34.py --model_path path/to/resnet34.pth --device cuda --overwrite
"""

import argparse
import json
import logging
import os,sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET

PATH_PROCESSED = Path(PATH_DATASET) / 'Quiron_ploidy'

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── ImageNet normalisation ────────────────────────────────────────────────────
_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Class label mapping inferred from folder name (mirrors Euploid / Aneuploid dirs)
LABEL_MAP = {"Euploid": 1, "Aneuploid": 0}

DEFAULT_MODEL_PATH = "models/markers/stage_classif/resnet34/resnet34_we15_e100.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Frame loading — same sampling pattern as EmbryoVideo / extract_femi.py
# ─────────────────────────────────────────────────────────────────────────────
def load_frames(video_path: str, max_frames: int, frame_size: int, n_total: int,) -> tuple[list[np.ndarray], list[int]]:
    """
    Sample `max_frames` evenly across the first `n_total` frames of the video,
    matching the sampling pattern used by EmbryoVideo (embryo_video.py) and
    extract_femi.py: indices 0, step, 2*step, ... where
    step = max(1, n_total // max_frames).

    Returns
    -------
    frames : list of (H, W, 3) uint8 RGB arrays — blank if unavailable
    mask   : list of int  1 = real frame,  0 = padded blank
    """
    frames, mask = [], []
    blank = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    step = max(1, n_total // max_frames)

    cap = cv2.VideoCapture(video_path)
    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (frame_size, frame_size),interpolation=cv2.INTER_LINEAR)
        img = img[..., ::-1].copy()  # BGR -> RGB
        frames.append(img)
        mask.append(1)

    if len(frames) < max_frames:
        frames.extend([blank.copy()] * (max_frames - len(frames)))
        mask.extend([0] * (max_frames - len(mask)))

    cap.release()
    return frames, mask


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
def extract_features_batched(frames, model: torch.nn.Module, device: str, batch_size: int,) -> torch.Tensor:
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
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0)   # (T, 512)


# ─────────────────────────────────────────────────────────────────────────────
# Per-split extraction loop
# ─────────────────────────────────────────────────────────────────────────────
def run_split(split: str, video_dir: Path, out_dir: Path, model: torch.nn.Module, args: argparse.Namespace,) -> tuple[int, int, int]:
    """Extract and save ResNet-34 features for one split. Returns (done, skipped, errors)."""
    done = skipped = errors = 0

    video_files = []
    for class_name, label in LABEL_MAP.items():
        class_dir = video_dir / split / class_name
        log.info("Scanning %s …", class_dir)
        for ext in ("*.mp4", "*.avi"):
            for vf in sorted(class_dir.glob(ext)):
                video_files.append((vf, class_name, label))

    log.info("── %s  (%d videos) ──────────────────────────────────────", split.upper(), len(video_files))

    for video_file, class_name, label in tqdm(video_files, desc=f"ResNet34 {split}", unit="video"):
        dst_dir = out_dir / split / class_name
        os.makedirs(dst_dir, exist_ok=True)
        dst = dst_dir / f"{video_file.stem}.pt"

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        # ── load frames ───────────────────────────────────────────────────
        try:
            frames, mask = load_frames(
                video_path=str(video_file),
                max_frames=args.n_frames,
                frame_size=args.frame_size,
                n_total=args.n_total,
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
                "video_file": str(video_file),
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
    video_dir = Path(args.video_dir)
    out_dir   = Path(args.out_dir)

    # ── output dirs ───────────────────────────────────────────────────────
    for split in ("train", "test"):
        for class_name in LABEL_MAP:
            os.makedirs(out_dir / split / class_name, exist_ok=True)

    # ── save extraction config ────────────────────────────────────────────
    config_path = out_dir / "extraction_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Config saved → %s", config_path)

    # ── load model ────────────────────────────────────────────────────────
    model = load_resnet34(args.model_path, args.device)

    # ── run ───────────────────────────────────────────────────────────────
    total_done = total_skipped = total_errors = 0
    for split in ("train", "test"):
        d, s, e = run_split(split, video_dir, out_dir, model, args)
        total_done    += d
        total_skipped += s
        total_errors  += e

    log.info("All done. done=%d  skipped=%d  errors=%d  features → %s", total_done, total_skipped, total_errors, out_dir,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ResNet-34 features from embryo videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_dir",    default=os.path.join(PATH_PROCESSED, "videos"), help="Root directory containing train/<class>/*.mp4 and test/<class>/*.mp4")
    parser.add_argument("--out_dir",      default=os.path.join(PATH_PROCESSED, "resnet34"), help="Output directory for .pt feature files")
    parser.add_argument("--model_path",   default=DEFAULT_MODEL_PATH, help="Path to saved ResNet-34 .pth file")
    parser.add_argument("--n_frames",     type=int,   default=200, help="Number of frames to extract per video (n_MAX)")
    parser.add_argument("--n_total",      type=int,   default=1300, help="Total number of source frames to sample across (n_total, matches EmbryoVideo)")
    parser.add_argument("--frame_size",   type=int,   default=224)
    parser.add_argument("--batch_frames", type=int,   default=16,help="Frames per forward pass (reduce if OOM)")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite",    action="store_true", help="Re-extract even if output file already exists")
    main(parser.parse_args())