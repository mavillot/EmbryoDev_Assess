import os, cv2, sys
import tqdm
import numpy as np
from pathlib import Path
from glob import glob
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET, SEED

PATH_PROCESSED  = os.path.join(PATH_DATASET, "Oslo_processed")
VIDEO_DIR       = os.path.join(PATH_PROCESSED, "videos")
EXCEL_PATH      = os.path.join(PATH_PROCESSED, "Oslo_completo.xlsx")
 
FRAME_H, FRAME_W = 250, 250          # expected frame size (H x W)
TEST_SIZE        = 0.1


# ─────────── Paths & constants ───────────────────────────────
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
        id_part, well_part = filename.split("_Well_")
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

def split_dataset( video_dir: str, df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED,) -> tuple[list, list, list, list]:
    """Return stratified train / test splits of (filenames, labels)."""
    all_files = sorted(os.listdir(video_dir))          # sort for reproducibility
    all_labels = build_labels(all_files, df)
 
    X_train, X_test, y_train, y_test = train_test_split(
        all_files, all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test

# ─────────── Frame extraction ───────────────────────────────

def _blank_frame() -> np.ndarray:
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

def extract_frames_for_split(video_files: list[str], df: pd.DataFrame, video_dir: str, start_time: int = 0, frequency: int = 1) -> tuple[list[list[np.ndarray]], list[list[int]]]:
    """
    For each video in *video_files* sample one frame per TIME_WINDOW step.
 
    Returns
    -------
    video_dataset : list of lists — shape [N_videos][N_timesteps] of np.ndarray
    mask_dataset  : list of lists — shape [N_videos][N_timesteps]
                    0 = real frame, 1 = padded blank (before recording start / read error)
    """
    video_dataset: list[list[np.ndarray]] = []
    mask_dataset:  list[list[int]]        = []
    TIME_WINDOW      = [start_time + i * frequency for i in range(30)]  

    for video_file in video_files:
        frames: list[np.ndarray] = []
        masks:  list[int]        = []
 
        vid_id, well = parse_video_name(video_file)
        rows = df[(df["ID"] == vid_id) & (df["Well"] == well)]
 
        if rows.empty:
            print("No metadata for '%s' — skipping.", video_file)
            continue
 
        tse         = rows["Tse"].iloc[0]
        frame_hour  = rows["frame_hour"].iloc[0]
        video_fp    = os.path.join(video_dir, video_file)
 
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            print("Cannot open video '%s' — filling with blanks.", video_fp)
            video_dataset.append([_blank_frame()] * len(TIME_WINDOW))
            mask_dataset.append([1] * len(TIME_WINDOW))
            continue
 
        try:
            for time in TIME_WINDOW:
                frame_idx = int((time - tse) * frame_hour)
 
                if frame_idx < 0:                   # before recording started
                    frames.append(_blank_frame())
                    masks.append(1)
                    continue
 
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, img = cap.read()
 
                if ret:
                    frames.append(img)
                    masks.append(0)
                else:
                    frames.append(_blank_frame())
                    masks.append(1)
        finally:
            cap.release()           # always release, even on exception
 
        video_dataset.append(frames)
        mask_dataset.append(masks)
 
    return video_dataset, mask_dataset



def build_dataset(video_dir: str = VIDEO_DIR,excel_path: str = EXCEL_PATH,) -> dict:
    """
    Build train and test video datasets.
 
    Returns a dict with keys:
        'X_train', 'mask_train', 'y_train',
        'X_test',  'mask_test',  'y_test'
    """
    df = load_metadata(excel_path)
    print(f"Loaded metadata for {len(df)} videos.")
    X_train, X_test, y_train, y_test = split_dataset(video_dir, df)
    print(f"Train/test split: {len(X_train)} train videos, {len(X_test)} test videos.")
 
    print("Extracting training frames …")
    X_DATA, MASK_DATA, Y_DATA = [], [], []
    for x in [0,11,24,30,47,52,72,80]:
        print(f"Start time: {x}h")
        for i in [0.2,2,7,15,24]:
            X_train_data, mask_train = extract_frames_for_split(X_train, df, video_dir, start_time=x, frequency=i)
            X_DATA.extend(X_train_data)
            MASK_DATA.extend(mask_train)
            Y_DATA.extend(y_train)
 
    print("Extracting test frames …")
    X_test_data, mask_test = extract_frames_for_split(X_test, df, video_dir, start_time=0, frequency=1)
 
    return {
        "X_train":    X_DATA,
        "mask_train": MASK_DATA,
        "y_train":    Y_DATA,
        'X_train_files': X_train,  # for reference
        "X_test":     X_test_data,
        "mask_test":  mask_test,
        "y_test":     y_test,
        'X_test_files': X_test,    # for reference
    }

def load_dataset(dataset_path: str, map_location: str = 'cpu') -> dict:
    """Load a preprocessed dataset from disk."""
    return torch.load(dataset_path, map_location=map_location, weights_only=False)

 
if __name__ == "__main__":
    dataset = build_dataset()
    torch.save(dataset, f'../Datasets/Processed/Oslo_processed/dataset_different_frequencies.pt')
    print(f"Train samples : {len(dataset['y_train'])}")
    print(f"Test  samples : {len(dataset['y_test'])}")
    print(f"Frames/video  : {len(dataset['X_train'][0]) if dataset['X_train'] else 0}")
