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

