import os, sys
from pathlib import Path
import glob
import cv2
import torch
import torchvision
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import PATH_DATASET, PATH_MODELS
from assessment.vIAbility.tabular.dataset import parse_video_name

DIR_PATH = os.path.join(PATH_DATASET, "Oslo_viability/videos")
LABELS_PATH = os.path.join(PATH_DATASET, "Oslo_viability/original.xlsx")
OUT_DIR = os.path.join("", "timing_results_stage")
model_path = os.path.join(PATH_MODELS, "markers/stage_classif/resnet34/resnet34_we15_e100.pth")

CLASSES      = ["empty", "1cell", "2cell", "+3cell", "+8cell","morula" ,"blastocyst"] 
IMG_SIZE     = 224          # ResNet canonical input
MEAN         = [0.485, 0.456, 0.406]   # ImageNet stats
STD          = [0.229, 0.224, 0.225]

os.makedirs(OUT_DIR, exist_ok=True)

CELL_COUNT_EVENTS = {
    "t2": 2, "t3": 3, "t8": 4, "tM": 5, "tB": 6,
}

def first_index(values, target):
    """Index of first occurrence of `target` in `values`, or 0 if absent."""
    try:
        return values.index(target)
    except ValueError:
        return 0

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def enforce_monotonic(preds, min_confirm=4):
    """
    Cleans a sequence of cell-count predictions so it's non-decreasing,
    treating isolated spikes as noise.

    A new (higher) value only gets "confirmed" and becomes the current
    value once it appears `min_confirm` times in a row. Until confirmed,
    the last confirmed value is kept.

    Parameters
    ----------
    preds : list[int]
        Raw frame-by-frame predictions.
    min_confirm : int
        Number of consecutive repeats needed to confirm a change.

    Returns
    -------
    list[int]
    """
    if not preds:
        return []

    cleaned = []
    current = preds[0]          # currently confirmed value
    candidate = None            # possible next value
    candidate_count = 0

    for v in preds:
        if v == current:
            candidate = None
            candidate_count = 0
        elif v > current:
            if v == candidate:
                candidate_count += 1
            else:
                candidate = v
                candidate_count = 1

            if candidate_count >= min_confirm:
                current = candidate
                candidate = None
                candidate_count = 0
        # v < current -> treated as noise (count shouldn't drop)

        cleaned.append(current)

    return cleaned




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    true_df = pd.read_excel(LABELS_PATH, sheet_name="Morula_Blast_nov")
    files = glob.glob(os.path.join(DIR_PATH, "*/*/*.avi"))
    records = {name: [] for name in list(CELL_COUNT_EVENTS)}

    for f in tqdm(files):
        cap = cv2.VideoCapture(f)
        ret, frame = cap.read()
        evolution = []
        while ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = get_transforms()(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(frame)
                _, predicted = torch.max(outputs.data, 1)
                evolution.append(predicted.item())
            ret, frame = cap.read()

        vid_id, well = parse_video_name(f)

        true_row = true_df[(true_df["ID"] == vid_id) & (true_df["Well"] == well)]
        if true_row.empty:
            continue
        true_row = true_row.iloc[0]
        tse, frame_hour = true_row["Tse"], true_row["frame_hour"]

        n_cells = enforce_monotonic(evolution)

        for event, target in CELL_COUNT_EVENTS.items():
            gt = true_row[event]
            if pd.isna(gt):
                continue
            pred = first_index(n_cells, target)
            pred = tse + pred/frame_hour
            records[event].append({
                "ID": vid_id,
                "Well": well,
                "prediction": pred,
                "ground_truth": gt,
                "error": abs(pred - gt),
            })


    # Save one CSV per event (t2, t3, t8, tM, tB) and plot error histograms
    for event, rows in records.items():
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, f"{event}.csv"), index=False)

        if not df.empty:
            plt.figure()
            plt.hist(df["error"])
            plt.title(f"{event} error distribution")
            plt.xlabel("Absolute error")
            plt.ylabel("Count")
            plt.savefig(os.path.join(OUT_DIR, f"{event}_hist.png"))
            plt.close()


if __name__ == "__main__":
    main()