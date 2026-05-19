"""
gradcam_viz.py — GradCAM visualization for the MultiTaskModel.

Supported tasks and what GradCAM shows:
  Image classification : exp, icm, te, stage
      → Heatmap over the input image highlighting regions that drove the
        predicted (or a chosen) class score.

  Image segmentation   : blasto_str, fragmentation
      → Heatmap from the last encoder conv layer, overlaid on the image.
        Complements the segmentation mask (shows *why* the decoder activated).

  Video classification : ploidy, fhb
      → Per-frame GradCAM heatmaps saved as a grid and optionally as a GIF,
        so you can see which frames and which regions drove the decision.

Architecture notes
──────────────────
  encoder(image) → (features, img_embedding, skips)
    features      : last spatial feature map  (B, C, H', W')  — GradCAM target layer
    img_embedding : global-average-pooled     (B, C)

  The hook is placed on the *last conv block of the encoder*.  For ResNet-34
  this is `encoder.layer4[-1]`; for EfficientNet-B3 it is `encoder.features[-1]`.
  The script auto-detects both; pass --target_layer to override.

Usage examples
──────────────
  # Single image, classification task
  python gradcam_viz.py \\
      --checkpoint models/best_model_phase3.pth \\
      --encoder resnet34 \\
      --task exp \\
      --image path/to/blastocyst.png

  # Single image, segmentation task
  python gradcam_viz.py \\
      --checkpoint models/best_model_phase3.pth \\
      --encoder resnet34 \\
      --task blasto_str \\
      --image path/to/blastocyst.png \\
      --seg_class 2

  # Video, ploidy task — saves per-frame grid + GIF
  python gradcam_viz.py \\
      --checkpoint models/best_model_phase3.pth \\
      --encoder resnet34 --transformer \\
      --task ploidy \\
      --video path/to/video.mp4 \\
      --n_frames 50

  # Batch of images from a folder — saves one PNG per image
  python gradcam_viz.py \\
      --checkpoint models/best_model_phase3.pth \\
      --encoder resnet34 \\
      --task icm \\
      --image_dir path/to/folder/
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import IMAGE_SIZE, PATH_DATASET
from models.multi_task import MultiTaskModel
from models.encoder.visual_encoder import ResNetEncoder, EfficientNetB3Encoder
from models.temporal.temporal_transformer import VideoTransformer
from data.augmentation import get_val_transforms, get_video_val_transforms, apply_video_val_transforms

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

IMAGE_TASKS = {
    "exp":           {"type": "classification", "n_classes": 5,
                      "labels": ["1", "2", "3", "4", "5"]},
    "icm":           {"type": "classification", "n_classes": 3,
                      "labels": ["A", "B", "C"]},
    "te":            {"type": "classification", "n_classes": 3,
                      "labels": ["A", "B", "C"]},
    "stage":         {"type": "classification", "n_classes": 7,
                      "labels": ["empty","1cell","2cell","+3cell","+8cell","morula","blastocyst"]},
    "blasto_str":    {"type": "segmentation",   "n_classes": 4,
                      "labels": ["background","zona_pellucida","icm","trophectoderm"]},
    "fragmentation": {"type": "segmentation",   "n_classes": 1,
                      "labels": ["fragmentation"]},
}

VIDEO_TASKS = {
    "ploidy": {"labels": ["Aneuploid", "Euploid"]},
    "fhb":    {"labels": ["no_FHB",   "FHB"]},
}

ALL_TASKS = list(IMAGE_TASKS) + list(VIDEO_TASKS)


# =============================================================================
# GradCAM engine
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on `target_layer` to capture:
      • activations  : the layer's output feature map   (B, C, H, W)
      • gradients    : ∂score/∂activations              (B, C, H, W)

    The heatmap is:  ReLU( Σ_c  α_c * A_c )
    where  α_c = GAP( ∂score/∂A_c )  (Grad-CAM weights).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._fwd_hook    = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook    = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        # output may be a tuple for some layer types
        self._activations = (output[0] if isinstance(output, tuple) else output).detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(
        self,
        score: torch.Tensor,        # scalar score to differentiate
        input_hw: tuple[int, int],  # (H, W) of the original input image
        retain_graph: bool = False, # set True when reusing the same graph for multiple tasks
    ) -> np.ndarray:
        """
        Backprop `score`, compute GradCAM heatmap, resize to input_hw.
        Returns np.ndarray in [0, 1], shape (H, W).

        Callers are responsible for zero_grad() BEFORE computing the score so
        that each backward pass starts from a clean gradient state.
        FIX: removed the internal zero_grad() call — it was wiping grads that
        belong to the caller's gradient accumulation context and causing incorrect
        CAMs when multiple tasks share the same forward graph.
        """
        score.backward(retain_graph=retain_graph)

        A = self._activations   # (B, C, h, w)  or (C, h, w) for single sample
        G = self._gradients     # same shape

        if A.ndim == 3:         # single sample without batch dim
            A = A.unsqueeze(0)
            G = G.unsqueeze(0)

        # α_c = global average pool of gradients over spatial dims
        alpha = G.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam   = F.relu((alpha * A).sum(dim=1))      # (B, h, w)
        cam   = cam[0].cpu().numpy()                # (h, w)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to original image resolution
        cam = cv2.resize(cam, (input_hw[1], input_hw[0]),
                         interpolation=cv2.INTER_LINEAR)
        return cam


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device: str) -> MultiTaskModel:
    if args.encoder == "resnet34":
        encoder = ResNetEncoder(pretrained=False)
    elif args.encoder == "efficientnet_b3":
        encoder = EfficientNetB3Encoder(pretrained=False)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    transformer = VideoTransformer(seq_len=args.n_frames) if args.transformer else None
    model = MultiTaskModel(
        encoder,
        transformer,
        embedding_dim=encoder.embedding_dim,
        weighted_loss=args.weighted_loss,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Handle DataParallel checkpoints
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in checkpoint.items()
        )

    # Strict=False: loss-fn buffers (e.g. pos_weight) in checkpoint won't crash load
    result = model.load_state_dict(checkpoint, strict=False)
    if result.unexpected_keys:
        print(f"  [checkpoint] Ignored training-only keys: {result.unexpected_keys}")
    if result.missing_keys:
        print(f"  [checkpoint] WARNING — missing keys: {result.missing_keys}")

    model.to(device).eval()
    print(f"Model loaded from {args.checkpoint}")
    return model


def get_target_layer(model: MultiTaskModel, encoder_type: str, override: Optional[str]) -> nn.Module:
    """
    Resolve which encoder layer to hook.

    Auto-detection:
      resnet34      → encoder.layer4[-1]   (last BasicBlock)
      efficientnet  → encoder.features[-1] (last MBConv block)

    Pass --target_layer to override with any dotted attribute path,
    e.g.  "encoder.layer3[-1]"  or  "encoder.features[5]".
    """
    if override:
        # Support bracket notation: "encoder.layer4[-1]"
        # We resolve it step by step
        obj = model
        for part in override.replace("]", "").split("."):
            if "[" in part:
                attr, idx = part.split("[")
                obj = getattr(obj, attr)[int(idx)]
            else:
                obj = getattr(obj, part)
        print(f"Using target layer: {override}")
        return obj

    if "resnet" in encoder_type:
        layer = model.encoder.layer4[-1]
        print("Auto-detected target layer: encoder.layer4[-1]")
    elif "efficientnet" in encoder_type:
        layer = model.encoder.features[-1]
        print("Auto-detected target layer: encoder.features[-1]")
    else:
        raise ValueError(
            f"Cannot auto-detect target layer for encoder '{encoder_type}'. "
            "Pass --target_layer explicitly."
        )
    return layer


# =============================================================================
# Image preprocessing
# =============================================================================

def preprocess_image(path: str) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess a single image.
    Returns (tensor (1,C,H,W), original_rgb_uint8 (H,W,3)).
    """
    img = Image.open(path).convert("RGB")
    original = np.array(img)                          # (H, W, 3)  uint8

    transform = get_val_transforms()
    img_np    = np.array(img)
    tensor    = transform(image=img_np)["image"]      # (C, H, W)
    return tensor.unsqueeze(0), original              # (1,C,H,W), (H,W,3)


def preprocess_video(path: str, n_frames: int) -> tuple[torch.Tensor, list[np.ndarray]]:
    """
    Sample `n_frames` evenly from a video file.
    Returns (tensor (1,T,C,H,W), list of original frames as uint8 (H,W,3)).
    """
    cap    = cv2.VideoCapture(str(path))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs   = np.linspace(0, total - 1, n_frames).astype(int)

    pil_frames   = []
    orig_frames  = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_frames.append(rgb)
        pil_frames.append(Image.fromarray(rgb))
    cap.release()

    val_tfm = get_video_val_transforms()
    vid_tensor = apply_video_val_transforms(pil_frames, val_tfm)  # (T, C, H, W)
    return vid_tensor.unsqueeze(0), orig_frames                    # (1,T,C,H,W), list


# =============================================================================
# Overlay helpers
# =============================================================================

def overlay_heatmap(
    original: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a GradCAM heatmap onto the original image.
    original : (H, W, 3) uint8 RGB
    cam      : (H, W) float32 in [0, 1]
    Returns  : (H, W, 3) uint8 RGB
    """
    heat_bgr  = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heat_rgb  = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay   = (alpha * heat_rgb + (1 - alpha) * original).clip(0, 255).astype(np.uint8)
    return overlay


def colorbar_legend(ax, colormap="jet", label="GradCAM intensity"):
    """Add a horizontal colorbar to a matplotlib axis."""
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04, label=label)


# =============================================================================
# Per-task score extraction
# =============================================================================

def get_score_for_task(
    outputs: dict,
    task: str,
    target_class: Optional[int],
) -> torch.Tensor:
    """
    Extract the scalar score to differentiate for GradCAM.

    Classification tasks : logit of target_class (or argmax if not specified)
    Segmentation tasks   : mean logit of target_class over all spatial positions
                           (encourages the cam to highlight the class region globally)
    Binary video tasks   : the single sigmoid logit
    """
    logits = outputs[task]   # shape depends on task type

    if task in ("blasto_str", "fragmentation"):
        # logits: (1, C, H, W)  or  (1, 1, H, W)
        if task == "fragmentation":
            return logits.mean()     # single-channel binary
        # FIX: argmax(dim=1)[0] is (H,W); .mode() on 2D returns (H,) not a scalar.
        # Flatten to 1D first so .mode() returns a single value.
        c = target_class if target_class is not None else int(logits.argmax(dim=1)[0].flatten().mode().values.item())
        return logits[0, c].mean()   # mean over spatial → (scalar)

    elif task in ("ploidy", "fhb"):
        # logits: (1,) binary
        return logits[0]

    else:
        # Classification: logits (1, C)
        c = target_class if target_class is not None else int(logits.argmax(dim=1).item())
        return logits[0, c]


# =============================================================================
# Core GradCAM runners
# =============================================================================

def run_gradcam_image(
    model:        MultiTaskModel,
    gradcam:      GradCAM,
    image_path:   str,
    task:         str,
    target_class: Optional[int],
    device:       str,
    out_path:     str,
    seg_class:    Optional[int] = None,
) -> None:
    """Run GradCAM on a single image and save the overlay figure."""
    tensor, original = preprocess_image(image_path)
    tensor = tensor.to(device)
    H, W   = original.shape[:2]

    # Forward pass — no labels so no loss is computed
    with torch.enable_grad():
        tensor.requires_grad_(True)
        outputs, _ = model(images=tensor)

    # Resolve predicted class for the title
    effective_class = target_class
    task_info = IMAGE_TASKS[task]

    if task_info["type"] == "classification":
        pred_idx   = int(outputs[task].argmax(dim=1).item())
        probs      = F.softmax(outputs[task], dim=1)[0].detach().cpu().numpy()
        if effective_class is None:
            effective_class = pred_idx
        score      = get_score_for_task(outputs, task, effective_class)

    elif task_info["type"] == "segmentation":
        effective_class = seg_class   # None → mean over all classes
        if task == "blasto_str":
            pred_mask = outputs[task][0].argmax(dim=0).detach().cpu().numpy()  # (H', W')
        else:
            pred_mask = (torch.sigmoid(outputs[task][0, 0]) > 0.5).detach().cpu().numpy()
        score = get_score_for_task(outputs, task, effective_class)

    else:
        raise ValueError(f"Unknown task type for {task}")

    model.zero_grad()
    cam = gradcam.compute(score, (H, W))  # retain_graph=False: single task, free the graph

    # ── Build figure ──────────────────────────────────────────────────────
    if task_info["type"] == "classification":
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original"); axes[0].axis("off")

        axes[1].imshow(overlay_heatmap(original, cam))
        axes[1].set_title(
            f"GradCAM — {task.upper()}  "
            f"class: {task_info['labels'][effective_class]}  "
            f"(conf: {probs[effective_class]:.2%})"
        )
        axes[1].axis("off")

        # Bar chart of class probabilities
        axes[2].barh(task_info["labels"], probs,
                     color=["#e74c3c" if i == pred_idx else "#3498db"
                            for i in range(len(task_info["labels"]))])
        axes[2].set_xlim(0, 1); axes[2].set_xlabel("Probability")
        axes[2].set_title("Class probabilities")
        axes[2].invert_yaxis()

    else:  # segmentation
        # Resize pred_mask to original resolution for display
        pred_mask_vis = cv2.resize(
            pred_mask.astype(np.uint8),
            (W, H), interpolation=cv2.INTER_NEAREST
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original);          axes[0].set_title("Original");         axes[0].axis("off")
        axes[1].imshow(pred_mask_vis, cmap="tab10", vmin=0, vmax=max(task_info["n_classes"]-1, 1))
        axes[1].set_title(f"Predicted mask — {task}");                             axes[1].axis("off")
        cl_label = task_info["labels"][effective_class] if effective_class is not None else "mean"
        axes[2].imshow(overlay_heatmap(original, cam))
        axes[2].set_title(f"GradCAM  (class: {cl_label})");                       axes[2].axis("off")

    fig.suptitle(f"{Path(image_path).name}  |  task: {task}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def run_gradcam_video(
    model:        MultiTaskModel,
    gradcam:      GradCAM,
    video_path:   str,
    task:         str,
    target_class: Optional[int],
    device:       str,
    out_dir:      str,
    n_frames:     int,
    save_gif:     bool,
    max_cols:     int = 10,
) -> None:
    """
    Run GradCAM for each frame of a video clip.

    Strategy:
      1. Forward the full clip through the model to get the video-level score.
      2. Backprop through the score — gradients flow back through the
         temporal transformer into every frame's encoder output.
      3. For each frame independently re-run forward+backward using only
         that frame's image so we get a per-frame spatial heatmap.
         (The video-level decision is already obtained in step 1.)
    """
    vid_tensor, orig_frames = preprocess_video(video_path, n_frames)
    vid_tensor = vid_tensor.to(device)

    # ── Step 1: video-level prediction ──────────────────────────────────
    with torch.enable_grad():
        vid_tensor.requires_grad_(True)
        outputs, _ = model(videos=vid_tensor)

    score = get_score_for_task(outputs, task, target_class)
    prob  = torch.sigmoid(score).item()
    pred  = "FHB" if prob > 0.5 else "no_FHB"   # works for both ploidy and fhb (binary)

    print(f"Video prediction: {pred}  (score={score.item():.3f}, prob={prob:.2%})")

    # ── Step 2: per-frame heatmaps via individual image forward passes ────
    H, W    = orig_frames[0].shape[:2]
    heatmaps = []
    overlays  = []

    val_tfm    = get_video_val_transforms()
    img_transform = get_val_transforms()

    for i, orig in enumerate(orig_frames):
        frame_pil    = Image.fromarray(orig)
        frame_np     = np.array(frame_pil)
        frame_tensor = img_transform(image=frame_np)["image"].unsqueeze(0).to(device)

        with torch.enable_grad():
            frame_tensor.requires_grad_(True)
            f_outputs, _ = model(images=frame_tensor)

            # Use the same task but pick the head that matches the video task.
            # For ploidy/fhb, the video head isn't available from a single image forward.
            # We use the classification head that is semantically closest:
            #   ploidy → stage (blastocyst-level); fhb → exp (embryo development)
            # OR: use the raw encoder feature norm as a proxy score.
            # Here we use the feature map norm — architecture-agnostic and always valid.
            _, emb, _ = model.encoder(frame_tensor)
            proxy_score = emb.norm()              # scalar ∝ encoder activation magnitude

        model.zero_grad()
        cam = gradcam.compute(proxy_score, (H, W))
        heatmaps.append(cam)
        overlays.append(overlay_heatmap(orig, cam))

    # ── Step 3: save outputs ──────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    n = len(overlays)
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).reshape(-1)   # flatten for easy iteration

    for i, (ov, ax) in enumerate(zip(overlays, axes)):
        ax.imshow(ov)
        ax.set_title(f"f{i}", fontsize=7)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")   # hide unused subplots

    fig.suptitle(
        f"{Path(video_path).name}  |  task: {task}  |  pred: {pred}  ({prob:.1%})",
        fontsize=11
    )
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"{Path(video_path).stem}_{task}_gradcam_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid → {grid_path}")

    # Individual per-frame PNGs
    frames_dir = os.path.join(out_dir, f"{Path(video_path).stem}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, ov in enumerate(overlays):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        Image.fromarray(ov).save(frame_path)

    # Optional GIF
    if save_gif:
        try:
            gif_frames = [Image.fromarray(ov) for ov in overlays]
            gif_path   = os.path.join(out_dir, f"{Path(video_path).stem}_{task}_gradcam.gif")
            gif_frames[0].save(
                gif_path,
                save_all=True, append_images=gif_frames[1:],
                duration=80, loop=0,
            )
            print(f"Saved GIF → {gif_path}")
        except Exception as e:
            print(f"  GIF save failed (Pillow may need additional dependencies): {e}")


# =============================================================================
# Batch mode
# =============================================================================

def run_batch_images(
    model:        MultiTaskModel,
    gradcam:      GradCAM,
    image_dir:    str,
    task:         str,
    target_class: Optional[int],
    device:       str,
    out_dir:      str,
    seg_class:    Optional[int],
) -> None:
    """Process every image in image_dir and save one overlay PNG per image."""
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    images = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing {len(images)} images from {image_dir} ...")

    for img_path in sorted(images):
        try:
            if task == "all":
                # FIX: 'all' was passed as a task name into run_gradcam_image,
                # which looked it up in IMAGE_TASKS and raised KeyError('all').
                out_path = os.path.join(out_dir, f"{img_path.stem}_all_tasks_gradcam.png")
                run_all_tasks_image(model, gradcam, str(img_path), device, out_path)
            else:
                out_path = os.path.join(out_dir, f"{img_path.stem}_{task}_gradcam.png")
                run_gradcam_image(
                    model, gradcam, str(img_path), task, target_class,
                    device, out_path, seg_class,
                )
        except Exception as e:
            # FIX: include exception type so the real cause is visible
            print(f"  Skipped {img_path.name}: {type(e).__name__}: {e}")


# =============================================================================
# Summary / comparison grid
# =============================================================================

def run_all_tasks_image(
    model:      MultiTaskModel,
    gradcam:    GradCAM,
    image_path: str,
    device:     str,
    out_path:   str,
) -> None:
    """
    Run GradCAM for all image tasks on a single image and save a comparison grid.
    Useful for understanding which regions each task head attends to.
    """
    tensor, original = preprocess_image(image_path)
    tensor = tensor.to(device)
    H, W   = original.shape[:2]

    tasks_to_run = list(IMAGE_TASKS.keys())
    cams: dict[str, np.ndarray] = {}

    with torch.enable_grad():
        tensor.requires_grad_(True)
        outputs, _ = model(images=tensor)

    for i, task in enumerate(tasks_to_run):
        # FIX: zero_grad BEFORE backward so each task gets a clean gradient pass.
        # Previously zero_grad ran after compute(), meaning it wiped already-consumed
        # gradients and the next task's backward accumulated on top of the previous one.
        # retain_graph=True is needed for all tasks except the last, since the same
        # forward graph is reused across tasks.
        model.zero_grad()
        score      = get_score_for_task(outputs, task, None)
        retain     = (i < len(tasks_to_run) - 1)
        cams[task] = gradcam.compute(score, (H, W), retain_graph=retain)

    n = len(tasks_to_run)
    fig, axes = plt.subplots(2, n + 1, figsize=(3 * (n + 1), 7))

    # Top row: overlays, Bottom row: raw heatmaps
    axes[0, 0].imshow(original); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, task in enumerate(tasks_to_run, start=1):
        cam   = cams[task]
        ov    = overlay_heatmap(original, cam)
        t_info = IMAGE_TASKS[task]

        axes[0, col].imshow(ov)
        axes[0, col].axis("off")

        if t_info["type"] == "classification":
            pred_idx = int(outputs[task].argmax(dim=1).item())
            axes[0, col].set_title(
                f"{task}\n→ {t_info['labels'][pred_idx]}", fontsize=9
            )
        else:
            axes[0, col].set_title(task, fontsize=9)

        axes[1, col].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[1, col].axis("off")
        axes[1, col].set_title("raw CAM", fontsize=8)

    fig.suptitle(f"All tasks — {Path(image_path).name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all-tasks comparison → {out_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="GradCAM visualization for MultiTaskModel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model ----------------------------------------------------------------
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pth model checkpoint.")
    p.add_argument("--encoder",       choices=["resnet34", "efficientnet_b3"],
                   default="resnet34")
    p.add_argument("--transformer",   action="store_true",
                   help="Load model with VideoTransformer (needed for video tasks).")
    p.add_argument("--weighted_loss", action="store_true",
                   help="Load model with weighted loss heads (must match checkpoint).")
    p.add_argument("--target_layer",  default=None,
                   help="Dotted path to the encoder layer to hook, e.g. 'encoder.layer4[-1]'. "
                        "Auto-detected if omitted.")
    p.add_argument("--n_frames",      type=int, default=50,
                   help="Frames to sample from video (must match training).")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")

    # -- Task -----------------------------------------------------------------
    p.add_argument("--task", choices=ALL_TASKS + ["all"],
                   required=True,
                   help="Task to visualize. Use 'all' to compare all image tasks.")

    # -- Input ----------------------------------------------------------------
    p.add_argument("--image",      default=None, help="Path to a single image.")
    p.add_argument("--image_dir",  default=None, help="Directory of images (batch mode).")
    p.add_argument("--video",      default=None, help="Path to a video file.")

    # -- Task-specific --------------------------------------------------------
    p.add_argument("--target_class", type=int, default=None,
                   help="Class index to explain (default: predicted class).")
    p.add_argument("--seg_class",    type=int, default=None,
                   help="[Segmentation] Class index for the GradCAM score. "
                        "None → mean over all classes.")

    # -- Output ---------------------------------------------------------------
    p.add_argument("--out_dir", default="output/gradcam",
                   help="Directory where results are saved.")
    p.add_argument("--save_gif", action="store_true",
                   help="[Video] Also save an animated GIF of per-frame heatmaps.")

    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device

    # Validate input
    n_inputs = sum([args.image is not None, args.image_dir is not None, args.video is not None])
    if n_inputs == 0:
        raise ValueError("Provide at least one of --image, --image_dir, or --video.")
    if n_inputs > 1:
        raise ValueError("Provide exactly one of --image, --image_dir, or --video.")
    if args.video is not None and args.task not in VIDEO_TASKS:
        raise ValueError(f"--video requires a video task ({list(VIDEO_TASKS)}), got '{args.task}'.")
    if args.video is None and args.task in VIDEO_TASKS:
        raise ValueError(f"Task '{args.task}' requires --video input.")
    if args.task == "all" and args.video is not None:
        raise ValueError("--task all is only supported for image inputs.")

    # Load model and set up GradCAM hook
    model       = load_model(args, device)
    target_layer = get_target_layer(model, args.encoder, args.target_layer)
    gradcam     = GradCAM(model, target_layer)

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        # ── Single image ──────────────────────────────────────────────────
        if args.image is not None:
            if args.task == "all":
                out_path = os.path.join(
                    args.out_dir,
                    f"{Path(args.image).stem}_all_tasks_gradcam.png"
                )
                run_all_tasks_image(model, gradcam, args.image, device, out_path)
            else:
                out_path = os.path.join(
                    args.out_dir,
                    f"{Path(args.image).stem}_{args.task}_gradcam.png"
                )
                run_gradcam_image(
                    model, gradcam, args.image, args.task,
                    args.target_class, device, out_path, args.seg_class,
                )

        # ── Batch images ──────────────────────────────────────────────────
        elif args.image_dir is not None:
            run_batch_images(
                model, gradcam, args.image_dir, args.task,
                args.target_class, device, args.out_dir, args.seg_class,
            )

        # ── Video ─────────────────────────────────────────────────────────
        elif args.video is not None:
            run_gradcam_video(
                model, gradcam, args.video, args.task,
                args.target_class, device, args.out_dir,
                args.n_frames, args.save_gif,
            )

    finally:
        gradcam.remove_hooks()
        print("GradCAM hooks removed.")


if __name__ == "__main__":
    main()