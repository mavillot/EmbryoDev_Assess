"""
extract_femi.py — extract FEMI (ViT-MAE) CLS-token embeddings per video.

Output layout
-------------
<PATH_DST>/femi/
    extraction_config.json
    train/
        FHB/     <video_filename>.pt
        no_FHB/  <video_filename>.pt
    test/
        FHB/     <video_filename>.pt
        no_FHB/  <video_filename>.pt

Each .pt file contains:
    {
        "embeddings": FloatTensor (T, 768),
        "mask":       LongTensor  (T,)      1 = valid frame, 0 = blank
        "video_file": str,
        "n_valid":    int,
    }

Usage
-----
    python extract_femi.py
    python extract_femi.py --model_id ihlab/FEMI --device cuda --overwrite
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from huggingface_hub import login
from transformers import AutoImageProcessor, ViTMAEModel

# ── shared helpers ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assessment.vIAbility.data.data_utils import (
    PATH_DST, PATH_EXCEL,
    DEFAULT_N_FRAMES, DEFAULT_START_TIME, DEFAULT_FREQ, DEFAULT_FRAME_SIZE,
    get_logger, load_split, load_metadata, extract_frames,
    make_split_dirs, out_path,
)

log = get_logger(__name__)

# ── ImageNet normalisation (used before feeding frames to FEMI) ───────────────
_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

FEMI_DST = PATH_DST / "femi"


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def embed_frames(
    frames,
    processor,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """
    Run FEMI in mini-batches and return CLS-token embeddings.

    Returns
    -------
    embeddings : FloatTensor (T, 768)
    """
    all_embeddings = []

    for i in range(0, len(frames), batch_size):
        batch        = frames[i: i + batch_size]
        pil_batch    = [Image.fromarray(f) for f in batch]
        tensor_batch = torch.stack([_PREPROCESS(img) for img in pil_batch])
        inputs       = processor(images=tensor_batch, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()   # (B, 768)

        all_embeddings.append(cls_emb)

    return torch.cat(all_embeddings, dim=0)   # (T, 768)


# ─────────────────────────────────────────────────────────────────────────────
# Per-split extraction loop  (shared by train and test)
# ─────────────────────────────────────────────────────────────────────────────

def run_split(
    split: str,
    filenames: list[str],
    labels: list[int],
    metadata_df,
    processor,
    femi: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Extract and save FEMI features for one split. Returns (done, skipped, errors)."""
    done = skipped = errors = 0

    log.info("── %s  (%d videos) ──────────────────────────────────────", split.upper(), len(filenames))

    for video_file, label in tqdm(zip(filenames, labels), total=len(filenames),
                                  desc=f"FEMI {split}", unit="video"):
        dst = out_path(FEMI_DST, split, label, video_file, ".pt")

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

        # ── FEMI inference ────────────────────────────────────────────────
        try:
            embeddings = embed_frames(
                frames, processor, femi,
                device=args.device,
                batch_size=args.batch_frames,
            )
        except RuntimeError as exc:
            log.error("FEMI inference failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        # ── save ──────────────────────────────────────────────────────────
        torch.save(
            {
                "embeddings": embeddings,           # (T, 768) float32
                "mask":       torch.tensor(mask),   # (T,)     int
                "video_file": video_file,
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
    # ── HuggingFace auth ──────────────────────────────────────────────────
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
        log.info("Logged in to HuggingFace.")
    else:
        log.info("No HF token provided — using cached credentials.")

    # ── output dirs ───────────────────────────────────────────────────────
    make_split_dirs(FEMI_DST)

    # ── save extraction config ────────────────────────────────────────────
    config_path = FEMI_DST / "extraction_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Config saved → %s", config_path)

    # ── load model ────────────────────────────────────────────────────────
    log.info("Loading FEMI '%s' onto %s …", args.model_id, args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    femi      = ViTMAEModel.from_pretrained(args.model_id).to(args.device)
    femi.eval()
    log.info("FEMI loaded.")

    # ── load metadata & splits ────────────────────────────────────────────
    metadata_df = load_metadata(PATH_EXCEL)
    log.info("Metadata: %d rows.", len(metadata_df))

    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")

    # ── run ───────────────────────────────────────────────────────────────
    total_done = total_skipped = total_errors = 0
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        d, s, e = run_split(split, X, y, metadata_df, processor, femi, args)
        total_done    += d
        total_skipped += s
        total_errors  += e

    log.info(
        "All done. done=%d  skipped=%d  errors=%d  features → %s",
        total_done, total_skipped, total_errors, FEMI_DST,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract FEMI (ViT-MAE) embeddings from embryo videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf_token",     default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--model_id",     default="ihlab/FEMI",
                        help="HuggingFace model ID")
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
