" python assessment/MultiTask/train.py --tasks exp te icm seg stage --weighted_loss --data_augmentation --multi_gpu"

import glob
import cv2,sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assessment.vIAbility.data.data_utils import (load_split, extract_frames)
from assessment.vIAbility.data.dataset import (load_metadata)

DATA = {
    'exp': {'train': 'EXP_silver', 'test': 'EXP_gold', 'classes': [0, 1, 2, 3, 4]},
    'icm': {'train': 'ICM_silver', 'test': 'ICM_gold', 'classes': [0, 1, 2]},
    'te':  {'train': 'TE_silver',  'test': 'TE_gold',  'classes': [0, 1, 2]},
}
GRADES = {'A':0, 'B':1, 'C':2}

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
        values = {'image': img, 'mask_str': None, 'mask_frag': None, 'exp': None, 'icm': None, 'te': None, 'stage': None, 'dataset_id':idx}
        if not pd.isna(str_value) and ((struct !='te' and struct != 'icm') or  (str_value!= value)):
            values[struct] = int(str_value)
            data.append(values)
    return data

def dataset_grading_Akriti(path, idx, split='train'):
    data = []
    name = "extracted_blasto_data.csv"
    image_root = path / "images"
    files = glob.glob(str(image_root / "*.jpg"))
    files = sorted(files)
    X_train, X_test = train_test_split(files, test_size=0.2, random_state=42)
    csv_file = pd.read_csv(path / name)
    print(len(csv_file))
    if split == 'train':
        files = X_train
    else:
        files = X_test
    for file in files:
        name = file.split("/")[-1]
        img_path = "images/"+name
        img = Image.open(file).convert("RGB") 

        icm = csv_file.loc[csv_file["path"] == img_path,'icm'].values[0] 
        te = csv_file.loc[csv_file["path"] == img_path,'te'].values[0]
        te = GRADES.get(te, None)
        icm = GRADES.get(icm, None)
        values = {'image': img, 'mask_str': None, 'mask_frag': None, 'exp': None, 'icm': icm, 'te': te, 'stage': None, 'dataset_id':idx}
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
            N = min(N, 1000)  
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
            data.append({'video': frames, 'ploidy': ploidy, 'fhb':None, 'dataset_id': idx})
    data
    return data

def dataset_fhb(path, idx, split='train', n_frames=50, start_time=0, freq=4):
    data = []
    X, y = load_split(split)
    for video_file, label in tqdm(zip(X, y), total=len(X)):
        df = load_metadata(path / "Oslo_completo.xlsx")
        frames, _ = extract_frames(video_file, df,n_frames=n_frames, start_time=start_time, freq=freq, resize=256)

        data.append({'video': frames,'ploidy': None, 'fhb': label, 'dataset_id': idx})
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
        data.append({'image': img, 'mask_str': mask, 'mask_frag': None, 'exp': exp, 'icm': icm, 'te': te,'stage': None, 'dataset_id':idx})
    
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

        data.append({'image': img, 'mask_str': None, 'mask_frag': mask, 'exp': None, 'icm': None, 'te': None,'stage': None, 'dataset_id':idx})
    
    return data

def dataset_stage_classif(path,idx, split='train'):
    data = []
    # DATASET SEGMENTATION FRAGMENTATION
    image_root = path  / split
    classmap = {'empty':0, '1cell':1, '2cell':2, '+3cell':3, '+8cell':4, 'morula':5, 'blastocyst':6}
    files = glob.glob(str(image_root / "*/*"))
    for img_name in files:
        img = cv2.imread(str(img_name))
        stage = classmap[Path(img_name).parent.stem]
        data.append({'image': img, 'mask_str': None, 'mask_frag': None, 'exp': None, 'icm': None, 'te': None, 'stage': stage, 'dataset_id':idx})
    
    return data

def prepare_datasets(path, transformer, split='train', tasks=None, frames=50):
    """
    Prepare datasets for multi-task or single-task training.

    Args:
        path:        root dataset path (Path object)
        transformer: bool — whether the video transformer is enabled
        split:       'train' or 'test'
        tasks:       list of tasks to include, e.g. ['exp', 'icm', 'seg', 'ploidy'].
                     Pass None (or omit) to include ALL tasks.
                     Valid values: 'exp', 'icm', 'te', 'seg', 'frag', 'stage', 'ploidy', 'fhb'
        frames:      number of frames to sample from each video
    """
    ALL_TASKS = {'exp', 'icm', 'te', 'seg', 'frag', 'stage', 'ploidy', 'fhb'}

    # Normalise: None means all tasks; validate any explicit list
    if tasks is None:
        tasks = list(ALL_TASKS)
    else:
        tasks = list(tasks)   # accept any iterable
        unknown = set(tasks) - ALL_TASKS
        if unknown:
            raise ValueError(f"Unknown task(s): {unknown}. Choose from: {ALL_TASKS}")

    tasks_set = set(tasks)
    print(f"Preparing datasets for tasks: {sorted(tasks_set)} ...")

    # ── Video datasets ────────────────────────────────────────────────────────
    all_vid_samples = []
    if transformer:
        if 'fhb' in tasks_set:
            data_fhb = dataset_fhb(path / 'Oslo_viability', 7, split=split, n_frames=frames)
            print(f"FHB samples: {len(data_fhb)}")
            all_vid_samples.extend(data_fhb)
        if 'ploidy' in tasks_set:
            data_ploidy = dataset_ploidy(path / "Quiron_ploidy/videos", 6, split=split, n_frames=frames)
            data_ploidy1 = dataset_ploidy(path / "EasyFiv/videos", 10, split=split, n_frames=frames)
            print(f"Ploidy samples: {len(data_ploidy)}")
            all_vid_samples.extend(data_ploidy)
            all_vid_samples.extend(data_ploidy1)
    else:
        print("Skipping ploidy and fhb datasets (transformer not enabled)")

    # ── Image datasets ────────────────────────────────────────────────────────
    all_img_samples = []
    Akriti = False
    if 'exp' in tasks_set:
        data_grading_exp = dataset_grading(path / "Blastocyst_Dataset", 'exp', 0, split=split)
        all_img_samples.extend(data_grading_exp)
        print(f"Grading samples — EXP: {len(data_grading_exp)}")
    if 'te' in tasks_set:
        data_grading_te = dataset_grading(path / "Blastocyst_Dataset", 'te', 1, split=split)
        data_grading_te_icm = dataset_grading_Akriti(path / "Oslo_viability", 9, split=split)
        Akriti = True
        all_img_samples.extend(data_grading_te)
        all_img_samples.extend(data_grading_te_icm)
        print(f"Grading samples — TE: {len(data_grading_te)}")
    if 'icm' in tasks_set:
        data_grading_icm = dataset_grading(path / "Blastocyst_Dataset", 'icm', 2, split=split)
        if not Akriti:
            data_grading_icm = dataset_grading_Akriti(path / "Oslo_viability", 9, split=split)
        all_img_samples.extend(data_grading_icm)
        print(f"Grading samples — ICM: {len(data_grading_icm)}")
    if 'stage' in tasks_set:
        data_stage_classif = dataset_stage_classif(path / "stage_classif/Images", 3, split=split)
        all_img_samples.extend(data_stage_classif)
        print(f"Stage classification samples: {len(data_stage_classif)}")
    # Blastocyst segmentation is loaded when 'seg' is requested OR when any grading
    # task is present (those datasets carry both grade labels and segmentation masks)
    if tasks_set & {'seg', 'exp', 'te', 'icm'}:
        data_seg_blasto = dataset_seg_blastocyst(path / "PCMR_blastocyts", 4, split=split)
        all_img_samples.extend(data_seg_blasto)
        print(f"Blastocyst seg samples (PCMR): {len(data_seg_blasto)}")
    if 'seg' in tasks_set:
        data_seg_blasto_quiron = dataset_seg_blastocyst(path / "Blastocyst_quiron", 5, split=split)
        all_img_samples.extend(data_seg_blasto_quiron)
        print(f"Blastocyst seg samples (Quiron): {len(data_seg_blasto_quiron)}")
    if 'frag' in tasks_set:
        data_seg_frag = dataset_seg_fragmentation(path / "CleavageEmbryo", 8, split=split)
        all_img_samples.extend(data_seg_frag)
        print(f"Fragmentation seg samples: {len(data_seg_frag)}")

    # ── Per-task field config ─────────────────────────────────────────────────
    # 'keep'  : the label field that must be non-None for a sample to be included
    # 'field' : the sample dict key that carries this task's label
    TASK_CONFIG = {
        'exp':    {'field': 'exp',      'video': False},
        'icm':    {'field': 'icm',      'video': False},
        'te':     {'field': 'te',       'video': False},
        'stage':  {'field': 'stage',    'video': False},
        'seg':    {'field': 'mask_str', 'video': False},
        'frag':   {'field': 'mask_frag','video': False},
        'ploidy': {'field': 'ploidy',   'video': True},
        'fhb':    {'field': 'fhb',      'video': True},
    }

    # Determine which label fields to KEEP (active tasks) and which to NULL OUT
    img_tasks   = [t for t in tasks_set if not TASK_CONFIG[t]['video']]
    video_tasks = [t for t in tasks_set if     TASK_CONFIG[t]['video']]

    keep_img_fields   = {TASK_CONFIG[t]['field'] for t in img_tasks}
    keep_video_fields = {TASK_CONFIG[t]['field'] for t in video_tasks}

    all_img_fields   = {TASK_CONFIG[t]['field'] for t in TASK_CONFIG if not TASK_CONFIG[t]['video']}
    all_video_fields = {TASK_CONFIG[t]['field'] for t in TASK_CONFIG if     TASK_CONFIG[t]['video']}

    null_img_fields   = all_img_fields   - keep_img_fields
    null_video_fields = all_video_fields - keep_video_fields

    # ── Filter image samples ──────────────────────────────────────────────────
    # Keep a sample if at least one of its active-task fields is non-None.
    # Null out all fields belonging to tasks that were NOT requested.
    img_samples = []
    for sample in all_img_samples:
        if any(sample.get(f) is not None for f in keep_img_fields):
            sample = sample.copy()
            for f in null_img_fields:
                sample[f] = None
            img_samples.append(sample)

    # ── Filter video samples ──────────────────────────────────────────────────
    video_samples = []
    for sample in all_vid_samples:
        if any(sample.get(f) is not None for f in keep_video_fields):
            sample = sample.copy()
            for f in null_video_fields:
                sample[f] = None
            video_samples.append(sample)

    # ── Class distribution report ─────────────────────────────────────────────
    for grade in ('exp', 'icm', 'te', 'stage'):
        if grade in tasks_set:
            values = [s[grade] for s in img_samples if s.get(grade) is not None]
            if values:
                print(f"{grade.upper()} class distribution: {np.unique(values, return_counts=True)}")

    print(f"\nFinal — image samples: {len(img_samples)}, video samples: {len(video_samples)}")
    return img_samples, video_samples