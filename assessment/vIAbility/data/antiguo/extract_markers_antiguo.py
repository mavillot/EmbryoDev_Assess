import glob,sys,os
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import build_dataset
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET, SEED
from assessment.embryo_utils.embryo_video import EmbryoVideo

PATH_PROCESSED  = os.path.join(PATH_DATASET, "Oslo_processed")
VIDEO_DIR       = os.path.join(PATH_PROCESSED, "videos")
EXCEL_PATH      = os.path.join(PATH_PROCESSED, "Oslo_completo.xlsx")
SAVE_DIR        = os.path.join(PATH_DATASET, "Viability_markers")


if __name__ == "__main__":
    dataset = build_dataset(VIDEO_DIR, EXCEL_PATH)
    lbl_dict = {0: "no_HB", 1: "HB"}
    for i in tqdm(range(1, len(dataset['y_train'])), desc="Processing training videos"):
        embryo = EmbryoVideo(frames=dataset['X_train'][i],mask=dataset['mask_train'][i], cropped=True)
        embryo.get_evolution()
        label = lbl_dict[dataset['y_train'][i]]
        name = dataset['X_train_files'][i]
        filename = str(Path(SAVE_DIR) / label /f"{name}.csv")
        embryo.save_evolution(filename = filename)

