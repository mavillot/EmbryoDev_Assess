import os, shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.data.dataset import parse_video_name, load_metadata, build_labels
from config.config import PATH_DATASET, SEED

PATH_source = os.path.join(PATH_DATASET, "Oslo_processed")
PATH_dst    = os.path.join(PATH_DATASET, "Oslo_viability")

PATH_source_videos = os.path.join(PATH_source, "videos")
PATH_excel = os.path.join(PATH_source, "Oslo_completo.xlsx")

classmap = {0 : "no_FHB", 1 : "FHB"}

files = os.listdir(PATH_source_videos)
df = load_metadata(PATH_excel)
y = build_labels(files, df)

X_train, X_test, y_train, y_test = train_test_split(files, y, test_size=0.2, random_state=SEED)

df=pd.DataFrame({"filename": X_train, "label": y_train})
df.to_csv(os.path.join(PATH_dst, "train.csv"), index=False)
df=pd.DataFrame({"filename": X_test, "label": y_test})
df.to_csv(os.path.join(PATH_dst, "test.csv"), index=False)


