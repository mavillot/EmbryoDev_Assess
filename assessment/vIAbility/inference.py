"""
predict.py — single-video inference for the "ML approach" (classical
classifier on top of pre-extracted features: markers, FEMI or ResNet-34).

Pipeline
--------
1. Sample frames from the input video.
2. Extract features with the chosen method:
     - markers   : run the EmbryoVideo pipeline (same as extract_markers.py)
                   to get the morphokinetic evolution curves, then cluster
                   them with the saved TimeSeriesKMeans models and classify
                   with the saved TabPFN model — mirrors best_feat.py exactly.
     - femi      : ViT-MAE (FEMI) CLS-token embeddings per frame, masked-mean
                   pooled to one vector, then classified.
     - resnet34  : ResNet-34 penultimate-layer embeddings per frame,
                   masked-mean pooled to one vector, then classified.
3. Print the predicted class (and class probabilities, if the classifier
   supports predict_proba), plus how long frame sampling and
   feature-extraction-and-classification each took.

A fourth method, "multimodal", covers the DL approach instead: MultimodalClassifier
(R(2+1)D-18 video encoder + tabular clinical features), which is what both
train_multimodal.py and train3d_cnn.py train — the two scripts are
otherwise identical, so there's one architecture to support, not two. This
branch needs raw video frames (not a precomputed markers/femi/resnet
feature file) plus the same tabular clinical features the model was
trained on, since those can't be derived from the video alone.

Only the "markers" branch has a training script to mirror exactly
(best_feat.py). For the others, dataloader.py and model.py (the actual
frame preprocessing / MultimodalClassifier definition) weren't shared with
me, so:
  - femi/resnet34 assume a classifier trained on masked-mean-pooled
    per-video embeddings — adjust pool_embeddings/predict_embedding if
    yours was trained differently (e.g. on the full (T, D) sequence).
  - multimodal's frame tensor shape/normalization is my best-effort match
    to how run_epoch() calls model(frames, mask, tabular) — verify against
    dataloader.py/model.py before trusting its predictions.

Usage
-----
    python predict.py path/to/video.mp4 --method markers
    python predict.py path/to/video.mp4 --method resnet34 --device cuda
    python predict.py path/to/video.mp4 --method femi --classifier_path models/tabpfn_femi.joblib
    python predict.py path/to/video.mp4 --method multimodal \\
        --checkpoint models/assessment/vIAbility/multimodal/r2plus1d_18/15/iter0/best_multimodal_r2plus1d_18_15_iter0.pt \\
        --tabular 0.2,1,34,0,...   # comma-separated, tabular_dim values, same order/scaling as training
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# ── defaults (mirrors data_utils.DEFAULT_* used by the extraction scripts) ──
try:
    from data_utils import (
        DEFAULT_N_FRAMES, DEFAULT_START_TIME, DEFAULT_FREQ, DEFAULT_FRAME_SIZE,
        get_logger,
    )
    log = get_logger(__name__)
except ImportError:
    DEFAULT_N_FRAMES, DEFAULT_START_TIME, DEFAULT_FREQ, DEFAULT_FRAME_SIZE = 15, 0.0, 1.0, 224
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger(__name__)

CLASSMAP = {0: "no_FHB", 1: "FHB"}

_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_frames(video_path, start_time, freq, n_frames, resize):
    """
    Uniformly sample `n_frames` RGB frames from `video_path`, `freq` seconds
    apart, starting at `start_time`. Missing frames (video too short) are
    zero-padded and flagged in `mask`.

    This is a self-contained sampler — not tied to a metadata Excel lookup —
    so it works on any video path, not just ones already in the dataset. If
    your project's data_utils.extract_frames does something dataset-specific
    (e.g. resolving filenames through metadata), swap it in here instead;
    its source wasn't part of what I was given, so I couldn't replicate any
    dataset-specific behaviour it might have.

    Returns
    -------
    frames : list of (H, W, 3) uint8 RGB arrays
    mask   : list of int, 1 = valid frame, 0 = blank/padded
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frames, mask = [], []
    for i in range(n_frames):
        t = start_time + i * freq
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize))
            frames.append(frame)
            mask.append(1)
        else:
            frames.append(np.zeros((resize, resize, 3), dtype=np.uint8))
            mask.append(0)
    cap.release()

    n_valid = sum(mask)
    if n_valid == 0:
        raise ValueError(f"No valid frames could be read from {video_path}")
    log.info("Sampled %d/%d valid frames from '%s'", n_valid, n_frames, video_path)
    return frames, mask


# ─────────────────────────────────────────────────────────────────────────────
# Method: markers — mirrors best_feat.py's clustering + TabPFN pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_markers(frames, mask, args) -> dict:
    sys.path.insert(0, str(Path(args.project_root)))
    from assessment.embryo_utils.embryo_video import EmbryoVideo

    embryo = EmbryoVideo(frames=frames, mask=mask, cropped=True)
    embryo.get_evolution()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    embryo.save_evolution(filename=tmp_path)
    df = pd.read_csv(tmp_path).replace([np.inf, -np.inf], np.nan).fillna(0)
    Path(tmp_path).unlink(missing_ok=True)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.Categorical(df[col]).codes
    cols = df.columns[2:-1]   # same column slicing as best_feat.load_data

    model_dir = Path(args.model_dir)
    cluster_codes = []
    for c in cols:
        km_path = model_dir / f"ts_kmeans_{c}.pkl"
        if not km_path.exists():
            raise FileNotFoundError(f"Missing cluster model for column '{c}': {km_path}")
        km = joblib.load(km_path)
        ts = df[c].to_numpy().reshape(1, -1, 1)
        cluster_codes.append(int(km.predict(ts)[0]))

    X = np.array(cluster_codes).reshape(1, -1)
    clf_path = Path(args.classifier_path) if args.classifier_path else model_dir / "tabpfn_fhb.joblib"
    clf = joblib.load(clf_path)
    return classify(clf, X)


# ─────────────────────────────────────────────────────────────────────────────
# Methods: femi / resnet34 — per-frame embeddings, masked-mean pooled
# ─────────────────────────────────────────────────────────────────────────────

def embed_resnet34(frames, args) -> torch.Tensor:
    model = torch.load(args.backbone_path, weights_only=False, map_location=args.device)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(args.device).eval()

    batch = torch.stack([_PREPROCESS(f) for f in frames]).to(args.device)
    with torch.no_grad():
        feats = model(batch).squeeze(-1).squeeze(-1).cpu()   # (T, 512)
    return feats


def embed_femi(frames, args) -> torch.Tensor:
    from transformers import AutoImageProcessor, ViTMAEModel

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    femi = ViTMAEModel.from_pretrained(args.model_id).to(args.device)
    femi.config.mask_ratio = 0.0   # deterministic full-frame embeddings, see extract_femi.py
    femi.eval()

    pil_batch = [Image.fromarray(f) for f in frames]
    inputs = processor(images=pil_batch, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = femi(**inputs)
        feats = out.last_hidden_state[:, 0, :].cpu()   # (T, 768)
    return feats


def pool_embeddings(embeddings: torch.Tensor, mask: list[int]) -> np.ndarray:
    """Masked-mean pool a (T, D) embedding sequence into a single (D,) vector."""
    mask_t = torch.tensor(mask, dtype=torch.bool)
    pooled = embeddings[mask_t].mean(dim=0) if mask_t.any() else embeddings.mean(dim=0)
    return pooled.numpy().reshape(1, -1)


def predict_embedding(frames, mask, args, method: str) -> dict:
    embeddings = embed_resnet34(frames, args) if method == "resnet34" else embed_femi(frames, args)
    X = pool_embeddings(embeddings, mask)

    default_name = f"tabpfn_{method}.joblib"
    clf_path = Path(args.classifier_path) if args.classifier_path else Path(args.model_dir) / default_name
    clf = joblib.load(clf_path)
    return classify(clf, X)


# ─────────────────────────────────────────────────────────────────────────────
# Method: multimodal — MultimodalClassifier (R(2+1)D-18 + tabular), the DL
# approach trained by train_multimodal.py / train3d_cnn.py.
# ─────────────────────────────────────────────────────────────────────────────

def predict_multimodal(video_path, args) -> dict:
    sys.path.insert(0, str(Path(args.project_root)))
    from assessment.vIAbility.multimodal.model import MultimodalClassifier

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --method multimodal")
    if not args.tabular:
        raise ValueError("--tabular is required for --method multimodal (comma-separated clinical feature values)")

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    cfg = ckpt.get("config", {})

    # Sample frames the way the checkpoint was trained (n_frames/frame size
    # saved in its config), falling back to the CLI defaults if absent.
    n_frames = cfg.get("n_frames", args.n_frames)
    frame_size = cfg.get("frame_h", args.frame_size)   # assumes frame_h == frame_w, true in the shown configs
    frames, mask = sample_frames(video_path, args.start_time, args.frequency, n_frames, frame_size)

    tabular_values = [float(v) for v in args.tabular.split(",")]
    tabular_dim = cfg.get("tabular_dim", len(tabular_values))
    if len(tabular_values) != tabular_dim:
        raise ValueError(
            f"Expected {tabular_dim} tabular values (checkpoint was trained with tabular_dim={tabular_dim}), "
            f"got {len(tabular_values)}"
        )

    model = MultimodalClassifier(
        tabular_dim=tabular_dim,
        tabular_hidden=cfg.get("tabular_hidden_dim", 128),
        video_embed_dim=cfg.get("video_embed_dim", 512),
        num_classes=1,
        pretrained=False,
        dropout_head=cfg.get("dropout_head", 0.5),
        dropout_mid=cfg.get("dropout_mid", 0.3),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # (1, T, 3, H, W) — same per-frame ImageNet normalization used elsewhere
    # in this codebase. See the module docstring's caveat about this shape.
    frame_tensor   = torch.stack([_PREPROCESS(f) for f in frames]).unsqueeze(0).to(args.device)
    mask_tensor    = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(args.device)
    tabular_tensor = torch.tensor(tabular_values, dtype=torch.float32).unsqueeze(0).to(args.device)

    with torch.no_grad():
        prob = torch.sigmoid(model(frame_tensor, mask_tensor, tabular_tensor).view(-1)).item()
    pred = int(prob >= 0.5)

    return {
        "label": CLASSMAP.get(pred, str(pred)),
        "class_index": pred,
        "probabilities": {CLASSMAP[0]: round(1 - prob, 6), CLASSMAP[1]: round(prob, 6)},
        "checkpoint": str(args.checkpoint),
    }


# ─────────────────────────────────────────────────────────────────────────────
def classify(clf, X: np.ndarray) -> dict:
    pred = int(clf.predict(X)[0])
    result = {"label": CLASSMAP.get(pred, str(pred)), "class_index": pred}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        result["probabilities"] = {CLASSMAP.get(i, str(i)): float(p) for i, p in enumerate(proba)}
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predict FHB status for a single embryo video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--method", choices=["markers", "femi", "resnet34", "multimodal"], default="markers")
    parser.add_argument("--model_dir", default="models", help="Dir holding cluster models / classifier .joblib files")
    parser.add_argument("--classifier_path", default=None, help="Override the classifier .joblib path")
    parser.add_argument("--backbone_path", default="models/markers/stage_classif/resnet34/resnet34_we15_e100.pth",
                        help="Saved ResNet-34 feature extractor (only used for --method resnet34)")
    parser.add_argument("--model_id", default="ihlab/FEMI", help="HF model id for FEMI (only used for --method femi)")
    parser.add_argument("--project_root", default=str(Path.cwd()),
                        help="Repo root, so assessment.* modules are importable (markers/multimodal methods)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a MultimodalClassifier .pt checkpoint (required for --method multimodal)")
    parser.add_argument("--tabular", default=None,
                        help="Comma-separated tabular clinical features, same order/scaling as training "
                             "(required for --method multimodal)")
    parser.add_argument("--n_frames", type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument("--start_time", type=float, default=DEFAULT_START_TIME)
    parser.add_argument("--frequency", type=float, default=DEFAULT_FREQ)
    parser.add_argument("--frame_size", type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        parser.error(f"Video not found: {video_path}")

    t_start = time.perf_counter()

    if args.method == "multimodal":
        # Samples its own frames internally, using the checkpoint's config.
        result = predict_multimodal(video_path, args)
        t_done = time.perf_counter()
        result["timing_sec"] = {"total": round(t_done - t_start, 3)}
    else:
        frames, mask = sample_frames(video_path, args.start_time, args.frequency, args.n_frames, args.frame_size)
        t_sampled = time.perf_counter()

        if args.method == "markers":
            result = predict_markers(frames, mask, args)
        else:
            result = predict_embedding(frames, mask, args, args.method)
        t_done = time.perf_counter()

        result["timing_sec"] = {
            "frame_sampling": round(t_sampled - t_start, 3),
            "feature_extraction_and_classification": round(t_done - t_sampled, 3),
            "total": round(t_done - t_start, 3),
        }

    log.info("Prediction for '%s' (%s): %s", video_path.name, args.method, result["label"])
    if "probabilities" in result:
        for label, p in result["probabilities"].items():
            log.info("  P(%s) = %.4f", label, p)
    log.info("Timing (sec): %s", result["timing_sec"])

    print(result)


if __name__ == "__main__":
    main()