import os, sys
from pathlib import Path
import glob

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.config import PATH_DATASET
from assessment.vIAbility.tabular.dataset import parse_video_name

DIR_PATH = os.path.join(PATH_DATASET, "Oslo_viability/markers")
LABELS_PATH = os.path.join(PATH_DATASET, "Oslo_viability/original.xlsx")
OUT_DIR = os.path.join("", "timing_results")
os.makedirs(OUT_DIR, exist_ok=True)

# Events based on the first frame where `n_cells` reaches a given value
CELL_COUNT_EVENTS = {
    "t2": 2, "t3": 3, "t4": 4, "t5": 5,
    "t6": 6, "t7": 7, "t8": 8,
}
# Events based on the first frame where `cleavage` reaches a given value
CLEAVAGE_EVENTS = {
    "tM": 1,
    "tB": 2,
}

# Fragmentation is predicted as a value in [0, 1]; it must be rescaled to
# 0-100 and snapped to the established clinical grading scale.
FRAGMENTATION_LEVELS = [
    (60, 75),
    (40, 50),
    (20, 35),
    (10, 15),
    (1, 5),
]


def first_index(values, target):
    """Index of first occurrence of `target` in `values`, or 0 if absent."""
    try:
        return values.index(target)
    except ValueError:
        return 0


def map_fragmentation(pred):
    """Rescale a [0, 1] fragmentation prediction to 0-100 and snap it to
    the nearest established grading level (5, 15, 35, 50, 75)."""
    pct = pred * 100
    for threshold, level in FRAGMENTATION_LEVELS:
        if pct > threshold:
            return level
    return 0


def main():
    files = glob.glob(os.path.join(DIR_PATH, "*/*/*.csv"))
    true_df = pd.read_excel(LABELS_PATH, sheet_name="Morula_Blast_nov")

    event_names = list(CELL_COUNT_EVENTS) + list(CLEAVAGE_EVENTS) + ["Fragmentation"]
    # records[event] -> list of dicts {ID, Well, prediction, ground_truth, error}
    records = {name: [] for name in event_names}

    for f in tqdm(files):
        name = Path(f).stem
        predicted_df = pd.read_csv(f)
        vid_id, well = parse_video_name(name)

        true_row = true_df[(true_df["ID"] == vid_id) & (true_df["Well"] == well)]
        if true_row.empty:
            continue
        true_row = true_row.iloc[0]

        n_cells = predicted_df["n_cells"].tolist()
        cleavage = predicted_df["cleavage"].tolist()
        fragmentation_idx = predicted_df["fragmentation_idx"].tolist()

        for event, target in {**CELL_COUNT_EVENTS, **CLEAVAGE_EVENTS}.items():
            gt = true_row[event]
            if pd.isna(gt):
                continue
            values = n_cells if event in CELL_COUNT_EVENTS else cleavage
            pred = first_index(values, target)
            records[event].append({
                "ID": vid_id,
                "Well": well,
                "prediction": pred,
                "ground_truth": gt,
                "error": abs(pred - gt),
            })

        frag_value = true_row["Fragmentation - Value"]
        frag_time = true_row["Fragmentation - Time"]
        if not pd.isna(frag_value) and not pd.isna(frag_time):
            frag_hour = round(frag_time)
            raw_frag = (
                fragmentation_idx[frag_hour]
                if 0 <= frag_hour < len(fragmentation_idx)
                else 0
            )
            pred_frag = map_fragmentation(raw_frag)
            records["Fragmentation"].append({
                "ID": vid_id,
                "Well": well,
                "prediction": pred_frag,
                "ground_truth": frag_value,
                "error": abs(pred_frag - frag_value),
            })

    # Save one CSV per event (t2, t3, ..., tM, tB, Fragmentation) and plot error histograms
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