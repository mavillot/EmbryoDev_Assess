import os
import sys
import json
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.data.dataset import build_labels, load_metadata, parse_video_name
from assessment.vIAbility.data.anitguo.extract_FEMI_features_antiguo import load_frames
from sklearn.model_selection import train_test_split
from config.config import SEED, PATH_DATASET

PATH_DATASET = Path(PATH_DATASET)/'Oslo_processed'
path_videos  = PATH_DATASET/'videos'
excel_path = PATH_DATASET/'Oslo_completo.xlsx'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16  # tune down to 8 if OOM persists

model = torch.load('models/markers/stage_classif/resnet34/resnet34_we15_e100.pth', weights_only=False)
model = torch.nn.Sequential(*list(model.children())[:-1])
model = model.to(DEVICE)  # move model to GPU once
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),                    
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                
        std=[0.229, 0.224, 0.225]
)])

def extract_frames(video_path, df, start_time=0, freq=1, resize=224):
    time_window = [start_time + i*freq for i in range(0, 214)]
    id, well = parse_video_name(video_path)
    row = df[(df['ID'] == id) & (df['Well'] == well)]
    if row.empty:
        raise ValueError("Video not found in metadata")
    tse = row['Tse'].values[0]
    frame_hour = row['frame_hour'].values[0]
    frames, mask = load_frames(path_videos / video_path, tse, frame_hour, time_window, resize)
    return frames, mask


def extract_features_batched(frames, model, device, batch_size=BATCH_SIZE):
    """Run inference in mini-batches to avoid OOM."""
    transformed = [preprocess(frame) for frame in frames]
    all_features = []

    with torch.no_grad():  # no gradient tracking needed for inference
        for i in range(0, len(transformed), batch_size):
            batch = torch.stack(transformed[i:i + batch_size]).to(device)
            feats = model(batch).squeeze(-1).squeeze(-1)  # (B, 512)
            all_features.append(feats.cpu())              # move back to CPU immediately
            del batch, feats
            torch.cuda.empty_cache()

    return torch.cat(all_features, dim=0)  # (T, 512)


class ResNet34Dataset:
    def __init__(self, files, model, df):
        self.files = files
        self.model = model
        self.df = df
        self.labels = build_labels(files, df)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        video_path = self.files[idx]
        frames, mask = extract_frames(video_path, self.df, start_time=0, freq=1, resize=224)
        features = extract_features_batched(frames, self.model, DEVICE)
        return video_path, frames, np.array(mask), features, self.labels[idx]
    
    def save_features(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(len(self)):
            print(f"Processing {self.files[idx]} ({idx+1}/{len(self)})")
            try:
                video_path, frames, mask, features, _ = self[idx]
                features_np = features.numpy()  # already on CPU
                if len(features_np.shape) > 2:
                    features_np = features_np.reshape(features_np.shape[0], -1)
                torch.save(
                    {
                        "embeddings": features_np,
                        "mask":       torch.tensor(mask),
                        "video_file": video_path,
                        'label':      self.labels[idx]
                    },
                    output_dir / f"{video_path}_features.pt",
                )
            except Exception as e:
                print(f"  ERROR on {self.files[idx]}: {e}")
                torch.cuda.empty_cache()  # recover and continue


files = [str(p.name) for p in path_videos.glob('*')]
df = load_metadata(excel_path)

labels = build_labels(files, df)
train_files, val_files, y_train, y_val = train_test_split(
    files, labels, stratify=labels, test_size=0.2, random_state=SEED
)
train_dataset = ResNet34Dataset(train_files, model, df)
val_dataset   = ResNet34Dataset(val_files,   model, df)

train_dataset.save_features(PATH_DATASET/'features_resnet34/train')
val_dataset.save_features(PATH_DATASET/'features_resnet34/test')