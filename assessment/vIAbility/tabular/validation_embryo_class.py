import os, sys
import tqdm
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config.config import PATH_DATASET
from assessment.vIAbility.tabular.dataset import parse_video_name

dir_path = os.path.join(PATH_DATASET, "Oslo_viability/markers")
labels_path = os.path.join(PATH_DATASET, "Oslo_viability/original.xlsx")
files = glob.glob(os.path.join(dir_path, "*/*/*.csv"))
true_df      = pd.read_excel(labels_path, sheet_name="Morula_Blast_nov")

errors = {'t2': [], 't4': [],  't8': [], 'tM': [], 'tB': [], 'Fragmentation': []}
for f in tqdm.tqdm(files):
    name = Path(f).stem
    predicted_df = pd.read_csv(f)
    
    id, well = parse_video_name(name)

    true_row = true_df[(true_df["Slide ID"] == id) & (true_df["Well"] == well)]
    n_cells = list(predicted_df['n_cells'])
    if not pd.isna(true_row['t2'].iloc[0]):
        try:
            t2 = n_cells.index(2)
        except:
            t2 = 0
        errors['t2'].append(abs(t2 - true_row['t2'].iloc[0]))
    if not pd.isna(true_row['t4'].iloc[0]):
        try:
            t4 = n_cells.index(4)
        except:
            t4 = 0
        errors['t4'].append(abs(t4 - true_row['t4'].iloc[0]))
    if not pd.isna(true_row['t8'].iloc[0]):
        try:
            t8 = n_cells.index(8)
        except:
            t8 = 0
        errors['t8'].append(abs(t8 - true_row['t8'].iloc[0]))
    cleavage = list(predicted_df['cleavage'])
    if not pd.isna(true_row['tM'].iloc[0]):
        try:
            tM = cleavage.index(1)
        except:
            tM = 0
        errors['tM'].append(abs(tM - true_row['tM'].iloc[0]))
    if not pd.isna(true_row['tB'].iloc[0]):
        try:
            tB = cleavage.index(2)
        except:
            tB = 0
        tB = 0 if not tB else tB
        errors['tB'].append(abs(tB - true_row['tB'].iloc[0]))

    fragmentation_idx = list(predicted_df['fragmentation_idx'])
    if not pd.isna(true_row['Fragmentation - Value'].iloc[0]) and not pd.isna(true_row['Fragmentation - Time'].iloc[0]):
        frag_hour = round(true_row['Fragmentation - Time'].iloc[0])
        try:   
            frag = fragmentation_idx[frag_hour]
        except:
            frag = 0
        errors['Fragmentation'].append(abs(frag - true_row['Fragmentation - Value'].iloc[0]))

   

for key, value in errors.items():
    plt.figure()
    plt.hist(value, bins=20)
    plt.title(f"Error distribution for {key}")
    plt.xlabel("Absolute error (frames)")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig(f"error_distribution_{key}.png")

print("Errors:")
for key, value in errors.items():
    print(f"{key}: {len(value)} samples")


print("Mean absolute errors:")
for key, value in errors.items():
    print(key)
    mean_error = sum(value) / len(value) if value else 0
    length = len(value)
    median_error = sorted(value)[length // 2] if length > 0 else 0
    print(f"Mean: {mean_error:.2f} hours")
    print(f"Median: {median_error:.2f} hours")
    print(f"Min: {min(value):.2f} hours")
    print(f"Max: {max(value):.2f} hours")
    print('#############################################')