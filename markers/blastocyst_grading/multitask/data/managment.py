from importlib.resources import path

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

DATA = {
    'exp': {'train': 'EXP_silver', 'test': 'EXP_gold', 'classes': [0, 1, 2, 3, 4]},
    'icm': {'train': 'ICM_silver', 'test': 'ICM_gold', 'classes': [0, 1, 2]},
    'te':  {'train': 'TE_silver',  'test': 'TE_gold',  'classes': [0, 1, 2]},
}

def dataset_grading(path, struct, idx, split='train'):
    data = []
    if split == 'train':
        name = "Gardner_train_silver.csv"
        csv_file = pd.read_csv(path / name,  encoding="latin-1", sep=";")
        value = 3
    if split == 'test':
        name = "Gardner_test_gold_onlyGardnerScores.csv"
        csv_file = pd.read_csv(path / name,  sep=";")
        value = 'ND'
    # DATASET GRADING
    image_root = path / "Images"
    
    for img_name in csv_file['Image']:
        img_path = image_root / img_name
        img = Image.open(img_path).convert("RGB") 
        str_value = csv_file.loc[csv_file["Image"] == img_name,DATA[struct][split]].values[0]
        values = {'image': img,'exp': None, 'icm': None, 'te': None, 'dataset_id':idx}
        if not pd.isna(str_value) and ((struct !='te' and struct != 'icm') or  (str_value!= value)):
            values[struct] = int(str_value)
            data.append(values)
    return data


def dataset_seg_blastocyst(path, idx, split='train'):
    data = []
    class_map = {'A':0, 'B':1, 'C':2}
    # DATASET SEGMENTATION BLASTOCYST
    image_root = path /split
    #check if file exists
    file = False
    if (path / "labels.csv").exists():
        csv_grades = pd.read_csv(path / "labels.csv", sep=";")
        file = True
    for img_name in image_root.iterdir():
        img_path = image_root / img_name.name

        if file:
            exp = int(csv_grades.loc[csv_grades["File Name"] == img_name.stem, "Expansion"].values[0])
            icm = int(class_map[csv_grades.loc[csv_grades["File Name"] == img_name.stem, "ICM"].values[0]])
            te = int(class_map[csv_grades.loc[csv_grades["File Name"] == img_name.stem, "TE"].values[0]])
            if te == 2:
                te = None

        img = cv2.imread(str(img_path))
        # mask = None
        data.append({'image': img, 'exp': exp, 'icm': icm, 'te': te, 'dataset_id':idx})
    
    return data

def dataset_oslo_blastocyst(path, idx):
    data = []
    class_map = {'A':0, 'B':1, 'C':2}
    image_root = path / 'images'
    #check if file exists
    file = False
    if (path / "extracted_blasto_data.csv").exists():
        csv_grades = pd.read_csv(path / "extracted_blasto_data.csv", sep=",")
        file = True
    for img_name in image_root.iterdir():
        img_path = image_root / img_name.name
        img= 'images/' + img_name.name
        exp = None
        if file:
            try:
                icm = int(class_map[csv_grades.loc[csv_grades["path"] == img, "icm"].values[0]])
            except (IndexError, ValueError,KeyError):
                icm = None
            try:
                te = int(class_map[csv_grades.loc[csv_grades["path"] == img, "te"].values[0]])
            except (IndexError, ValueError, KeyError):
                te = None
        img = cv2.imread(str(img_path))
        data.append({'image': img, 'exp': exp, 'icm': icm, 'te': te, 'dataset_id':idx})
    
    return data


def prepare_datasets(path, split = 'train',task=None, frames=50):
    """
    Prepare datasets for multi-task or single-task training.
    
    Args:
        task: one of 'exp', 'icm', 'te', 'seg', 'frag', 'ploidy', or None for all tasks.
    """
    print("Preparing datasets...")

    # --- Load all image datasets ---
    data_grading_exp = dataset_grading(path / "Blastocyst_Dataset", 'exp', 0, split=split)
    data_grading_te  = dataset_grading(path / "Blastocyst_Dataset", 'te',  1, split=split)
    data_grading_icm = dataset_grading(path / "Blastocyst_Dataset", 'icm', 2, split=split)
    print(f"Grading samples — EXP: {len(data_grading_exp)}, TE: {len(data_grading_te)}, ICM: {len(data_grading_icm)}")

    data_seg_blasto       = dataset_seg_blastocyst(path / "PCMR_blastocyts",   3, split=split)
    print(f"PCMR grading samples: {len(data_seg_blasto)}")
    data_oslo_blasto       = dataset_oslo_blastocyst(path / "Oslo_viability",   4) 
    print(f"Oslo grading samples: {len(data_oslo_blasto)}")

    all_img_samples = (data_grading_exp + data_grading_te + data_grading_icm + data_seg_blasto + data_oslo_blasto)


    # --- Filter and clean for single-task mode ---
    # Maps each task to: the field that must be present, and fields to null out
    TASK_CONFIG = {
        'exp':  {'keep': 'exp',       'null': ['icm', 'te']},
        'icm':  {'keep': 'icm',       'null': ['exp', 'te']},
        'te':   {'keep': 'te',        'null': ['exp', 'icm']},
    }

    if task is not None:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(TASK_CONFIG.keys()) + ['ploidy']}")
        
        config = TASK_CONFIG[task]
        img_samples = []
        for sample in all_img_samples:
            if sample[config['keep']] is not None:
                sample = sample.copy()
                for field in config['null']:
                    sample[field] = None
                img_samples.append(sample)
        print(f"Single-task '{task}' samples after filtering: {len(img_samples)}")
    else:
        img_samples = all_img_samples


    # --- Print class distributions ---
    for grade in ('exp', 'icm', 'te'):
        values = [s[grade] for s in img_samples if s[grade] is not None]
        if values:
            print(f"{grade.upper()} class distribution: {np.unique(values, return_counts=True)}")

    print(f"\nFinal — image samples: {len(img_samples)}")
    return img_samples
