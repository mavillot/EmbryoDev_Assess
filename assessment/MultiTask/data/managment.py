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
        values = {'image': img, 'mask_str': None, 'mask_frag': None, 'exp': None, 'icm': None, 'te': None, 'dataset_id':idx}
        if not pd.isna(str_value) and ((struct !='te' and struct != 'icm') or  (str_value!= value)):
            values[struct] = int(str_value)
            data.append(values)
    return data

def dataset_ploidy(path, idx, split='train', n_frames=50):
    data = []
    # DATASET PLOIDY
    video_root = path  / split
    class_map ={'Aneuploides':0, 'Euploides':1}
    for class_name in class_map.keys():
        class_dir = video_root / class_name
        for video_name in tqdm(class_dir.iterdir()):
            cap = cv2.VideoCapture(str(video_name))
            N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idxs = np.linspace(0, N - 1, n_frames).astype(int)
            frames = []
            for idx in frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            
            cap.release()
            print(len(frames))
            ploidy = class_map[class_name]
            data.append({'video': frames, 'ploidy': ploidy, 'dataset_id': idx})
    data
    return data

def dataset_seg_blastocyst(path, idx, split='train'):
    data = []
    class_map = {'A':0, 'B':1, 'C':2}
    # DATASET SEGMENTATION BLASTOCYST
    image_root = path /split
    mask_root = path /"GT_Mask"
    #check if file exists
    file = False
    if (path / "labels.csv").exists():
        csv_grades = pd.read_csv(path / "labels.csv", sep=";")
        file = True
    for img_name in image_root.iterdir():
        img_path = image_root / img_name.name
        mask_path = mask_root / img_name.name

        if file:
            exp = int(csv_grades.loc[csv_grades["File Name"] == img_name.stem, "Expansion"].values[0])
            icm = int(class_map[csv_grades.loc[csv_grades["File Name"] == img_name.stem, "ICM"].values[0]])
            te = int(class_map[csv_grades.loc[csv_grades["File Name"] == img_name.stem, "TE"].values[0]])
            if te == 2:
                te = None
            # exp = None
            # te = None
            # icm = None
        else:
            exp = None
            icm = None
            te = None

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # assuming single channel mask

        mask[mask == 75] = 1  # Zona Pellucida
        mask[mask == 150] = 2  # Inner Cell Mass
        mask[mask == 255] = 3  # Trophectoderm
        # mask = None
        data.append({'image': img, 'mask_str': mask, 'mask_frag': None, 'exp': exp, 'icm': icm, 'te': te, 'dataset_id':idx})
    
    return data

def dataset_seg_fragmentation(path,idx, split='train'):
    data = []
    # DATASET SEGMENTATION FRAGMENTATION
    image_root = path  / "images"/split
    mask_root = path  / "masks"/split
    for img_name in image_root.iterdir():
        img_path = image_root / img_name.name
        mask_path = mask_root / (img_name.stem + ".png")

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)  # assuming single channel mask

        mask[mask > 0] = 1  # Fragmented areas as class 1

        data.append({'image': img, 'mask_str': None, 'mask_frag': mask, 'exp': None, 'icm': None, 'te': None, 'dataset_id':idx})
    
    return data

def prepare_datasets(path, transformer, split = 'train',task=None):
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
    data_seg_blasto_quiron = dataset_seg_blastocyst(path / "Blastocyst_quiron", 4, split=split)
    print(f"Blastocyst seg samples: {len(data_seg_blasto) + len(data_seg_blasto_quiron)}")

    data_seg_frag = dataset_seg_fragmentation(path / "CleavageEmbryo", 5, split=split)
    print(f"Fragmentation seg samples: {len(data_seg_frag)}")

    all_img_samples = (data_grading_exp + data_grading_te + data_grading_icm +
                       data_seg_blasto + data_seg_blasto_quiron + data_seg_frag)

    # --- Load video dataset ---
    if transformer:
        data_ploidy = dataset_ploidy(path / "Quiron_ploidy", 0,split=split, n_frames=50)
        print(f"Ploidy samples: {len(data_ploidy)}")
    else:
        data_ploidy = []
        print("Skipping ploidy dataset (transformer not enabled)")

    # --- Filter and clean for single-task mode ---
    # Maps each task to: the field that must be present, and fields to null out
    TASK_CONFIG = {
        'exp':  {'keep': 'exp',       'null': ['icm', 'te', 'mask_str', 'mask_frag']},
        'icm':  {'keep': 'icm',       'null': ['exp', 'te', 'mask_str', 'mask_frag']},
        'te':   {'keep': 'te',        'null': ['exp', 'icm', 'mask_str', 'mask_frag']},
        'seg':  {'keep': 'mask_str',  'null': ['exp', 'icm', 'te', 'mask_frag']},
        'frag': {'keep': 'mask_frag', 'null': ['exp', 'icm', 'te', 'mask_str']},
    }

    if task is not None and task != 'ploidy':
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
    elif task == 'ploidy':
        img_samples = []
    else:
        img_samples = all_img_samples

    video_samples = data_ploidy if (task is None or task == 'ploidy') else []

    # --- Print class distributions ---
    for grade in ('exp', 'icm', 'te'):
        values = [s[grade] for s in img_samples if s[grade] is not None]
        if values:
            print(f"{grade.upper()} class distribution: {np.unique(values, return_counts=True)}")

    print(f"\nFinal — image samples: {len(img_samples)}, video samples: {len(video_samples)}")
    return img_samples, video_samples

def prepare_datasets_antiguo(path, transformer, task = None):
    """Prepare all datasets for multi-task training."""
    print("Preparing datasets...")
    
    # Grading datasets
    data_grading_exp = dataset_grading(path / "Blastocyst_Dataset", 'exp', 0)
    data_grading_te = dataset_grading(path / "Blastocyst_Dataset", 'te', 1)
    data_grading_icm = dataset_grading(path / "Blastocyst_Dataset", 'icm', 2)
    print(f"Grading samples: {len(data_grading_exp) + len(data_grading_te) + len(data_grading_icm)}")

    print(f"EXP class distribution: {np.unique([d['exp'] for d in data_grading_exp], return_counts=True)}")
    print(f"ICM class distribution: {np.unique([d['icm'] for d in data_grading_icm], return_counts=True)}")
    print(f"TE class distribution: {np.unique([d['te'] for d in data_grading_te], return_counts=True)}")
    
    # Segmentation datasets
    data_seg_blasto = dataset_seg_blastocyst(path / "PCMR_blastocyts", 3)
    data_seg_blasto_quiron = dataset_seg_blastocyst(path / "Blastocyst_quiron", 4)
    print(f"Blastocyst segmentation samples: {len(data_seg_blasto) + len(data_seg_blasto_quiron)}")
    print(f"EXP class distribution: {np.unique([d['exp'] for d in data_seg_blasto], return_counts=True)}")
    print(f"ICM class distribution: {np.unique([d['icm'] for d in data_seg_blasto], return_counts=True)}")
    print(f"TE class distribution: {np.unique([d['te'] for d in data_seg_blasto], return_counts=True)}")
    
    data_seg_fragmentation = dataset_seg_fragmentation(path / "CleavageEmbryo", 5)
    print(f"Fragmentation segmentation samples: {len(data_seg_fragmentation)}")

    # Ploidy dataset
    data_ploidy = dataset_ploidy(path / "Quiron_ploidy", 0, n_frames=50) if transformer else None
    print(f"Ploidy samples: {len(data_ploidy)}") if transformer else print("Skipping ploidy dataset (transformer not enabled)")   
    print(f"Ploidy class distribution: {np.unique([d['ploidy'] for d in data_ploidy], return_counts=True)}") if transformer else None
    
    
    # Combine into multi-task dataset
    
    img_samples = data_grading_exp + data_grading_te + data_grading_icm + data_seg_blasto + data_seg_blasto_quiron + data_seg_fragmentation
    video_samples = data_ploidy if transformer else []
    final_img = img_samples
    
    if task:
        final_img, = []
        if task == 'exp':
            for sample in img_samples:
                if sample['exp']:
                    sample['te'] = None
                    sample['icm'] = None
                    sample['mask_str'] = None
                    final_img.append(sample)
        elif task == 'te':
            for sample in img_samples:
                if sample['te']:
                    sample['exp'] = None
                    sample['icm'] = None
                    sample['mask_str'] = None
                    final_img.append(sample)
        elif task == 'icm':
            for sample in img_samples:
                if sample['icm']:
                    sample['exp'] = None
                    sample['te'] = None
                    sample['mask_str'] = None
                    final_img.append(sample)
        elif task == 'seg':
            for sample in img_samples:
                if sample['mask_str'] is not None:
                    sample['exp'] = None
                    sample['te'] = None
                    sample['icm'] = None
                    final_img.append(sample)
        elif task == 'frag':
            for sample in img_samples:
                if not (sample['mask_str'] is None):
                    final_img.append(sample)
       
    return final_img, video_samples