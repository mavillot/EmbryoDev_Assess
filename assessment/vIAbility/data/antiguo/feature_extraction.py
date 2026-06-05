import os, shutil
import sys
import json
from pathlib import Path
import logging
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import argparse
import torch
from huggingface_hub import login
from transformers import AutoImageProcessor, ViTMAEModel
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.embryo_utils.embryo_video import EmbryoVideo
from assessment.vIAbility.data.dataset import parse_video_name, load_metadata, build_labels
from config.config import PATH_DATASET, SEED

PATH_source = os.path.join(PATH_DATASET, "Oslo_processed")
PATH_dst    = os.path.join(PATH_DATASET, "Oslo_viability")

PATH_source_videos = os.path.join(PATH_source, "videos")
PATH_excel = os.path.join(PATH_source, "Oslo_completo.xlsx")

classmap = {0 : "no_FHB", 1 : "FHB"}

FEMI_dst = os.path.join(PATH_dst, "femi")
MARKERS_dst = os.path.join(PATH_dst, "markers")

df = load_metadata(PATH_excel)

train = pd.read_csv(os.path.join(PATH_dst, "train.csv"))
test = pd.read_csv(os.path.join(PATH_dst, "test.csv"))

X_train, y_train = train['filename'].tolist(), train['label'].tolist()
X_test, y_test = test['filename'].tolist(), test['label'].tolist()


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),                    
    transforms.Normalize(
        mean =[0.485, 0.456, 0.406],                
        std  =[0.229, 0.224, 0.225]
)])


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_frames(video_path: str, tse:float, frame_hour: float, time_window: list[float], frame_size:  int,) -> tuple[list[np.ndarray], list[int]]:
    """
    Seek to each time-point and read one frame.
    Returns
    -------
    frames : list of (H, W, 3) uint8 RGB arrays — blank if unavailable
    mask   : list of int  1 = real frame,  0 = padded blank
    """
    frames, mask = [], []
    blank = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(video_path)

    try:
        for t in time_window:
            frame_idx = int((t - tse) * frame_hour)

            if frame_idx < 0 or not cap.isOpened():
                frames.append(blank.copy())
                mask.append(0)
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()

            if ret:
                img = cv2.resize(img, (frame_size, frame_size),
                                 interpolation=cv2.INTER_LINEAR)
                img = img[..., ::-1].copy()   # BGR → RGB
                frames.append(img)
                mask.append(1)
            else:
                frames.append(blank.copy())
                mask.append(0)
    finally:
        cap.release()

    return frames, mask

def extract_frames(video_path, df, start_time=0, freq=1, resize=224):
    time_window = [start_time + i*freq for i in range(0, 214)]
    id, well = parse_video_name(video_path)
    row = df[(df['ID'] == id) & (df['Well'] == well)]
    if row.empty:
        raise ValueError("Video not found in metadata")
    tse = row['Tse'].values[0]
    frame_hour = row['frame_hour'].values[0]
    frames, mask = load_frames(os.path.join(PATH_source_videos, video_path), tse, frame_hour, time_window, resize)
    log.info(f"Extracted {sum(mask)}/{len(mask)} valid frames from '{video_path}'")
    log.info(f'number of frames: {len(frames)}')
    return frames, mask

# ─────────────────────────────────────────────────────────────────────────────
# FEMI
# ─────────────────────────────────────────────────────────────────────────────

def embed_frames(
    frames:      list[np.ndarray],
    processor,
    model:       torch.nn.Module,
    device:      str,
    batch_size:  int,
) -> torch.Tensor:
    """
    Run FEMI on all frames in mini-batches and return CLS embeddings.
    Returns
    -------
    embeddings : FloatTensor  (T, 768)
    """
    all_embeddings = []

    for i in range(0, len(frames), batch_size):
        batch     = frames[i : i + batch_size]
        pil_batch = [Image.fromarray(f) for f in batch]
        tensor_batch = torch.stack([preprocess(img) for img in pil_batch])
        inputs    = processor(images=tensor_batch, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token: first position of last_hidden_state → (B, 768)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()

        all_embeddings.append(cls_emb)

    return torch.cat(all_embeddings, dim=0)   # (T, 768)

# ─────────────────────────────────────────────────────────────────────────────
# MARKERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_markers(frames, masks, y, filename, split):
    embryo = EmbryoVideo(frames=frames,mask=masks, cropped=True)
    embryo.get_evolution()
    filename = str(Path(PATH_dst) / split / classmap[y] /f"{filename}.csv")
    embryo.save_evolution(filename = filename)


def main(args) -> None:

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(hf_token)
        log.info("Logged in to HuggingFace.")
    else:
        log.info("No HF token provided — using cached credentials.")

    # ── Directories ───────────────────────────────────────────────────────

    os.makedirs(FEMI_dst, exist_ok=True)
    os.makedirs(MARKERS_dst, exist_ok=True)
    os.makedirs(os.path.join(FEMI_dst, 'train/FHB'), exist_ok=True)
    os.makedirs(os.path.join(MARKERS_dst, 'train/FHB'), exist_ok=True)
    os.makedirs(os.path.join(FEMI_dst, 'train/no_FHB'), exist_ok=True)
    os.makedirs(os.path.join(MARKERS_dst, 'train/no_FHB'), exist_ok=True)

    os.makedirs(os.path.join(FEMI_dst, 'test/FHB'), exist_ok=True)
    os.makedirs(os.path.join(MARKERS_dst, 'test/FHB'), exist_ok=True)
    os.makedirs(os.path.join(FEMI_dst, 'test/no_FHB'), exist_ok=True)
    os.makedirs(os.path.join(MARKERS_dst, 'test/no_FHB'), exist_ok=True)

    # Save extraction config for traceability
    config_path = os.path.join(FEMI_dst, "extraction_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    log.info("Extraction config saved → %s", config_path)

    # ── Load FEMI ─────────────────────────────────────────────────────────
    log.info("Loading FEMI from '%s' onto %s …", args.model_id, args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    femi      = ViTMAEModel.from_pretrained(args.model_id).to(args.device)
    femi.eval()
    log.info("FEMI loaded.")

    # ── Load metadata ─────────────────────────────────────────────────────
    df = load_metadata(PATH_excel)
    log.info("Metadata loaded: %d rows.", len(df))

    log.info("Starting feature extraction for %d training videos …", len(X_train))
    done, skipped, errors = 0, 0, 0
    for video_file, label in tqdm(zip(X_train, y_train), desc="Extracting", unit="video"):
        frames, mask = extract_frames(video_file, df, start_time=args.start_time, freq=args.frequency, resize=args.frame_size)
        femi_out_path = os.path.join(FEMI_dst, 'train', classmap[label],f"{video_file}.pt")
        markers_out_path = os.path.join(MARKERS_dst, 'train', classmap[label], f"{video_file}.csv")

        # Resume-friendly: skip if already done
        if os.path.exists(femi_out_path) and not args.overwrite:
            skipped += 1
            continue
        if os.path.exists(markers_out_path) and not args.overwrite:
            skipped += 1
            continue

        
        n_valid = sum(mask)
        if n_valid == 0:
            log.warning("All frames blank for '%s' — saving anyway.", video_file)

        # ── FEMI inference ────────────────────────────────────────────────
        try:
            embeddings = embed_frames(
                frames     = frames,
                processor  = processor,
                model      = femi,
                device     = args.device,
                batch_size = args.batch_frames,
            )
        except RuntimeError as exc:
            log.error("FEMI inference failed for '%s': %s", video_file, exc)
            errors += 1
            continue
        
        # ── Save ──────────────────────────────────────────────────────────
        torch.save(
            {
                "embeddings": embeddings,          # (T, 768)  float32
                "mask":       torch.tensor(mask),  # (T,)      int  1=valid
                "video_file": video_file,          # provenance
                "n_valid":    n_valid,
            },
            femi_out_path,
        )

        # ── Marker extraction ─────────────────────────────────────────────

        try:
            extract_markers(frames, mask, label, video_file, "train")
        except Exception as exc:
            log.error("Marker extraction failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        done += 1

    log.info("Starting feature extraction for %d test videos …", len(X_test))
    done, skipped, errors = 0, 0, 0
    for video_file, label in tqdm(zip(X_test, y_test), desc="Extracting", unit=" video"):
        frames, mask = extract_frames(video_file, df, start_time=args.start_time, freq=args.frequency, resize=args.frame_size)
        femi_out_path = os.path.join(FEMI_dst, 'test', classmap[label],f"{video_file}.pt")
        markers_out_path = os.path.join(MARKERS_dst, 'test', classmap[label], f"{video_file}.csv")

        # Resume-friendly: skip if already done
        if os.path.exists(femi_out_path) and not args.overwrite:
            skipped += 1
            continue
        if os.path.exists(markers_out_path) and not args.overwrite:
            skipped += 1
            continue

        
        n_valid = sum(mask)
        if n_valid == 0:
            log.warning("All frames blank for '%s' — saving anyway.", video_file)

        # ── FEMI inference ────────────────────────────────────────────────
        try:
            embeddings = embed_frames(
                frames     = frames,
                processor  = processor,
                model      = femi,
                device     = args.device,
                batch_size = args.batch_frames,
            )
        except RuntimeError as exc:
            log.error("FEMI inference failed for '%s': %s", video_file, exc)
            errors += 1
            continue
        
        # ── Save ──────────────────────────────────────────────────────────
        torch.save(
            {
                "embeddings": embeddings,          # (T, 768)  float32
                "mask":       torch.tensor(mask),  # (T,)      int  1=valid
                "video_file": video_file,          # provenance
                "n_valid":    n_valid,
            },
            femi_out_path,
        )

        # ── Marker extraction ─────────────────────────────────────────────

        try:
            extract_markers(frames, mask, label, video_file, "test")
        except Exception as exc:
            log.error("Marker extraction failed for '%s': %s", video_file, exc)
            errors += 1
            continue

        done += 1




    # ── Summary ───────────────────────────────────────────────────────────
    log.info(
        "Extraction complete. done=%d  skipped=%d  errors=%d  total=%d",
        done, skipped, errors, len(X_train),
    )
    log.info("Features saved to: %s", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract FEMI frame embeddings from embryo videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf_token", default=None, help="HuggingFace token for gated model access.If omitted, uses the HF_TOKEN env variable",)
    parser.add_argument("--model_id", default="ihlab/FEMI", help="HuggingFace model ID to load",)
    parser.add_argument("--n_frames", type=int, default=214, help="Number of frames to extract per video",)
    parser.add_argument("--start_time", type=float,default=0.0, help="Start time (hours) for frame sampling window",)
    parser.add_argument("--frequency",type=float,default=1.0,help="Time step (hours) between sampled frames",)
    parser.add_argument("--frame_size",type=int,default=224,help="Resize frames to (frame_size x frame_size) before passing to FEMI",)
    parser.add_argument("--batch_frames",type=int,default=16,help="Number of frames to pass through FEMI at once (reduce if OOM)",)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu",help="Device to run FEMI on (e.g. cpu, cuda, cuda:1)",)
    parser.add_argument("--overwrite",action="store_true",help="Re-extract features even if the .pt file already exists",)
    args = parser.parse_args()
    main(args)


