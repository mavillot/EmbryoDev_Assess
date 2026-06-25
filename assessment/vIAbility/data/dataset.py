import glob
import os, cv2, random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_metadata(excel_path: str) -> pd.DataFrame:
    """Load and validate the Excel metadata file."""
    df = pd.read_excel(excel_path)
    required = {"ID", "Well", "Fetal_Heart_Beat", "Tse", "frame_hour"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel file is missing columns: {missing}")
    return df

def parse_video_name(filename: str) -> tuple[str, int]:
    """Return (id, well) from a filename like  '<id>_Well_<well>.ext'."""
    try:
        id_part, well_part = Path(filename).stem.split("_Well_")
        well = int(well_part.split(".")[0])
        return id_part, well
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected filename format: '{filename}'") from exc
    
def build_labels(video_files: list[str], df: pd.DataFrame) -> list[int]:
    """Return a label for every video file in *video_files*."""
    labels = []
    for filename in video_files:
        vid_id, well = parse_video_name(filename)
        rows = df[(df["ID"] == vid_id) & (df["Well"] == well)]
        if rows.empty:
            raise KeyError(f"No metadata row found for video '{filename}'")
        labels.append(int(rows["Fetal_Heart_Beat"].iloc[0]))
    return labels
# Kinetics-400 normalization stats
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3, 1, 1, 1)

class VideoDataset(Dataset):
    START_TIMES = [0, 5, 7, 11, 15, 17, 24, 30,35, 37, 40, 42, 48,52, 55, 60, 65, 70,72, 80]
    FREQUENCIES = [0.2, 2, 5, 7, 15, 20, 24]


    # def __init__(self, video_files, labels, video_dir, df, augment=True):
    #     self.video_files = video_files
    #     self.labels      = labels
    #     self.video_dir   = video_dir
    #     self.df          = df
    #     self.augment     = augment

    def __init__(self, video_files, labels, video_dir, df, frame_w, frame_h, n_frames, augment=True, oversample=False, factor_clase_1=3, factor_clase_0=1.5, tabular_lookup=None):
        self.video_dir   = video_dir
        self.df          = df
        self.augment     = augment
        self.n_frames    = n_frames
        self.frame_w     = frame_w
        self.frame_h     = frame_h
        self.tabular_lookup = tabular_lookup
        if oversample:
            # Listas vacías para construir nuestro dataset balanceado y expandido
            final_video_files = []
            final_labels = []
            
            for video, label in zip(video_files, labels):
                if label == 1:
                    # Multiplicamos los videos de la clase 1 por el factor deseado (ej. 3 veces)
                    final_video_files.extend([video] * factor_clase_1)
                    final_labels.extend([label] * factor_clase_1)
                else:
                    # Multiplicamos o mantenemos la clase 0 según busques expandirla (ej. un 50% más mediante factor 1.5)
                    # Para factores decimales, usamos probabilidad:
                    veces = int(factor_clase_0) + (1 if random.random() < (factor_clase_0 % 1) else 0)
                    final_video_files.extend([video] * veces)
                    final_labels.extend([label] * veces)
                    
            self.video_files = final_video_files
            self.labels      = final_labels
        else:
            self.video_files = video_files
            self.labels      = labels

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        filename = self.video_files[idx]
        label    = self.labels[idx]

        # Temporal sampling augmentation: only for positive class (label==1)

        if self.augment and label == 1:
            start_time = random.choice(self.START_TIMES)
            frequency  = random.choice(self.FREQUENCIES)
        else:
            start_time, frequency = 0, 1

        frames, mask = self._load_frames(filename, start_time, frequency)

        # (T, H, W, C) → (C, T, H, W)
        frames_t = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0

        if self.augment:
            frames_t = self._augment(frames_t)

        frames_t = (frames_t - MEAN) / STD

        mask_t  = torch.tensor(mask,  dtype=torch.bool)
        label_t = torch.tensor(label, dtype=torch.long)

        if self.tabular_lookup is not None:
            tab = self.tabular_lookup[filename]
            tab = torch.tensor(tab, dtype=torch.float32)
            return frames_t, mask_t, tab, label_t
        return frames_t, mask_t, label_t

    def _augment(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Temporally-consistent augmentations on (C, T, H, W).
        The same spatial transform is applied to every frame to preserve motion.
        """
        C, T, H, W = frames.shape

        # ── 1. Horizontal flip ───────────────────────────────────────────
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])

        # ── 2. Vertical flip ─────────────────────────────────────────────
        # Safe for embryo videos which have no canonical up/down orientation
        if random.random() < 0.3:
            frames = torch.flip(frames, dims=[-2])

        # ── 3. Random 90° rotation ───────────────────────────────────────
        # k ∈ {1,2,3} → 90°, 180°, 270°.  Again safe for embryo data.
        if random.random() < 0.3:
            k = random.randint(1, 3)
            # torch.rot90 operates on the last two dims (H, W)
            frames = torch.rot90(frames, k=k, dims=[-2, -1])

        # ── 4. Random crop and resize ────────────────────────────────────
        # Simulates zoom / slight misframing. Crop to 90–100 % of the frame,
        # then resize back to (H, W) so the tensor shape is unchanged.
        if random.random() < 0.4:
            scale   = random.uniform(0.85, 1.0)
            crop_h  = int(H * scale)
            crop_w  = int(W * scale)
            top     = random.randint(0, H - crop_h)
            left    = random.randint(0, W - crop_w)
            # Crop: frames is (C, T, H, W) → slice last two dims
            frames  = frames[:, :, top:top + crop_h, left:left + crop_w]
            # Resize back: interpolate expects (N, C, H, W); treat T as batch
            frames  = frames.permute(1, 0, 2, 3)          # (T, C, crop_h, crop_w)
            frames  = torch.nn.functional.interpolate(
                frames, size=(H, W), mode="bilinear", align_corners=False
            )
            frames  = frames.permute(1, 0, 2, 3)          # (C, T, H, W)

        # ── 5. Brightness / contrast jitter ─────────────────────────────
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            frames     = torch.clamp(frames * contrast + (brightness - 1.0), 0.0, 1.0)

        # ── 6. Gaussian noise ────────────────────────────────────────────
        # Simulates sensor noise in low-light embryo microscopy
        if random.random() < 0.3:
            noise  = torch.randn_like(frames) * 0.02
            frames = torch.clamp(frames + noise, 0.0, 1.0)

        # ── 7. Random channel shuffle ────────────────────────────────────
        # Embryo videos are effectively grayscale rendered as RGB; shuffling
        # channels prevents the model from relying on any single channel.
        if random.random() < 0.2:
            perm   = torch.randperm(C)
            frames = frames[perm]

        # ── 8. Temporal dropout ──────────────────────────────────────────
        # Zero out up to 2 frames to simulate dropped frames / motion blur
        n_drop = random.randint(0, 2)
        if n_drop:
            drop_idx = random.sample(range(T), min(n_drop, T))
            for t in drop_idx:
                frames[:, t] = 0.0

        return frames

    def _load_frames(self, filename, start_time, frequency):
        vid_id, well = parse_video_name(filename)
        rows = self.df[(self.df["ID"] == vid_id) & (self.df["Well"] == well)]
        tse, frame_hour = rows["Tse"].iloc[0], rows["frame_hour"].iloc[0]

        time_window = [start_time + i * frequency for i in range(self.n_frames)]
        frames, mask = [], []

        cap = cv2.VideoCapture(filename)
        try:
            for t in time_window:
                frame_idx = int((t - tse) * frame_hour)
                if frame_idx < 0 or not cap.isOpened():
                    frames.append(self._blank()); mask.append(1)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, img = cap.read()
                if ret:
                    img = cv2.resize(img, (self.frame_w, self.frame_h),
                                     interpolation=cv2.INTER_LINEAR)
                    img = img[..., ::-1].copy()
                    frames.append(img); mask.append(0)
                else:
                    frames.append(self._blank()); mask.append(1)
        finally:
            cap.release()
        return frames, mask

    def _blank(self):
        return np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
    
def mixup_collate(batch, alpha=0.4):
    """
    Drop-in replacement for the default DataLoader collate_fn.
    Applies mixup across the batch with Beta(alpha, alpha) mixing coefficient.

    Each item in `batch` is (frames, mask, label) from VideoDataset.__getitem__. or (frames, mask, tabular, label) if tabular features are used.

    Returns
    -------
    frames_mixed : (B, C, T, H, W)  float32
    mask_mixed   : (B, T)           bool   — union of both masks (conservative)
    tabular      : (B, D)           float32 — unchanged tabular features
    labels_mixed : (B,)             float32 — soft labels for BCEWithLogitsLoss
    """
    if len(batch[0]) == 3:
        tab = False
        frames, masks, labels = zip(*batch)
    else:
        tab = True
        frames, masks, tabular, labels = zip(*batch)
        tabular = torch.stack(tabular)  # (B, D)

    frames = torch.stack(frames)          # (B, C, T, H, W)
    masks  = torch.stack(masks)           # (B, T)
    labels = torch.stack(labels).float()  # (B,)

    # Draw one λ per batch (scalar mixup — simpler and works just as well)
    lam = float(np.random.beta(alpha, alpha))

    # Roll the batch by 1 to get the "partner" for each sample
    idx_b = torch.roll(torch.arange(len(frames)), shifts=1)

    frames_mixed = lam * frames + (1 - lam) * frames[idx_b]

    # Conservative mask: a frame is padding if EITHER source clip padded it
    masks_mixed  = masks | masks[idx_b]

    # Soft label: interpolate between the two hard labels
    labels_mixed = lam * labels + (1 - lam) * labels[idx_b]

    if tab:
        return frames_mixed, masks_mixed, tabular, labels_mixed
    else:
        return frames_mixed, masks_mixed, labels_mixed


def get_dataloaders(video_dir, excel_path, frame_w=112, frame_h=112, n_frames=30, augment=True, oversample=False, factor1=2, factor0=1.5, batch_size=8, num_workers=4,  device="cuda"):
    df         = pd.read_excel(excel_path)

    X_train = glob.glob(os.path.join(video_dir, "train/*/*.avi"))
    X_test  = glob.glob(os.path.join(video_dir, "test/*/*.avi"))
    y_train = build_labels(X_train, df)
    y_test  = build_labels(X_test, df)

    train_ds = VideoDataset(X_train, y_train, video_dir, df, frame_w=frame_w, frame_h=frame_h, n_frames=n_frames, augment=augment, oversample=oversample, factor_clase_1=factor1, factor_clase_0=factor0)
    test_ds  = VideoDataset(X_test,  y_test,  video_dir, df, frame_w=frame_w, frame_h=frame_h, n_frames=n_frames, augment=False, oversample=False)

    pin = (device == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=pin, drop_last=True, collate_fn=mixup_collate, )
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=pin, collate_fn=mixup_collate,)

    return train_loader, test_loader