import os, sys
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from assessment.vIAbility.tabular.tabular_study import clean_diagnose, one_hot_encode
from assessment.vIAbility.data.dataset import VideoDataset, mixup_collate

class Multimodal_Data():
    def __init__(self, video_path, excel_path):
        self.video_path = video_path
        self.excel_path = excel_path
        self.labels=[]
        self.videos={}
        self.df = self._load_excel_data()
        self.columns =  [ 'Age', 'Age_mann', 'smoke', 'snus','IVF/ICSI', 'number of oocytes', 'ovulatory_factor', 'tubal_factor', 'uterine_factor', 'endometriosis_factor',
                    'male_factor', 'age_factor', 'unexplained_factor']
        self.n_data = len(self.columns)

    def _load_excel_data(self):
        df = pd.read_csv(self.excel_path)
        df = df.drop_duplicates(subset=["ID", "Well"])
        return df

    def preprocess(self):
        def fill_mean(df, column):
            mean_value = df[column].mean()
            if str(df[column].dtype).startswith('Int') or str(df[column].dtype).startswith('int'):
                mean_value = round(mean_value)
            df[column] = df[column].fillna(mean_value)
            return df

        self.df.dropna(subset=["HCG Test"], inplace=True)
        self.df['Fetal Heart Beat']= self.df['Fetal Heart Beat'].fillna(0)
        self.df.loc[self.df['Fetal Heart Beat'] == 'mi', 'Fetal Heart Beat'] = None
        self.df.dropna(subset=["Fetal Heart Beat"], inplace=True)
        self.df['smoke'] = self.df['smoke'].fillna(0)
        self.df['snus'] = self.df['snus'].fillna(0)
        for column in ['Age', 'Age_mann', 'number of oocytes']:
            self.df = fill_mean(self.df, column)
        return self.df
        

    def get_labels(self,class_name):
        if class_name=="0":
            return 0
        else:
            return 1
    
    def extract_tabular_features(self, video_pth):
        id_part, well_part = Path(video_pth).stem.split("_Well_")
        well = int(well_part.split(".")[0])
        row = self.df[(self.df["ID"] == id_part) & (self.df["Well"] == well)]
        return row[self.columns].fillna(0).values.flatten() 
    
    def load_data(self):
        videos = glob.glob(self.video_path + '*/*.avi')
        for video_pth in videos:
            class_name = Path(video_pth).parent.stem
            self.labels.append(self.get_labels(class_name))
            tabular_feat = self.extract_tabular_features(video_pth)
            self.videos[video_pth] = tabular_feat
        return self.videos, self.labels


def get_dataloaders(video_dir, excel_path, frame_w=112, frame_h=112, n_frames=30, augment=True, oversample=False, 
                    factor1=2, factor0=1.5, mixup=True, batch_size=8, num_workers=4,  device="cuda", log=None):
    if log is None:
        import logging
        log = logging.getLogger(__name__)

    X_train = glob.glob(os.path.join(video_dir, "train/*/*.avi"))
    X_test  = glob.glob(os.path.join(video_dir, "test/*/*.avi"))

    log.info(f"Number of training videos: {len(X_train)}")
    log.info(f"Number of test videos: {len(X_test)}")

    mm_data = Multimodal_Data(video_dir, excel_path)
    mm_data.preprocess()

    train_tab = {p: mm_data.extract_tabular_features(p) for p in X_train if mm_data.extract_tabular_features(p).shape[0] == mm_data.n_data}
    test_tab  = {p: mm_data.extract_tabular_features(p) for p in X_test if mm_data.extract_tabular_features(p).shape[0] == mm_data.n_data}
    

    X_train = [p for p in X_train if p in train_tab]
    X_test  = [p for p in X_test if p in test_tab]

    log.info(f"Final number of training videos: {len(X_train)}")
    log.info(f"Final number of test videos: {len(X_test)}")

    y_train = [mm_data.get_labels(Path(p).parent.stem) for p in X_train]
    y_test  = [mm_data.get_labels(Path(p).parent.stem) for p in X_test]

    log.info(f"Number of training samples: {len(y_train)} ")
    log.info(f"Number of training samples: {y_train[0] + y_train.count(1)} (0: {y_train.count(0)}, 1: {y_train.count(1)})")
    log.info(f"Number of test samples: {y_test[0] + y_test.count(1)} (0: {y_test.count(0)}, 1: {y_test.count(1)})")

    train_ds = VideoDataset(X_train, y_train, video_dir, mm_data.df,frame_w=frame_w, frame_h=frame_h, n_frames=n_frames,
                            augment=augment, oversample=oversample,factor_clase_1=factor1, factor_clase_0=factor0, 
                            tabular_lookup=train_tab)
    test_ds = VideoDataset(X_test, y_test, video_dir, mm_data.df,frame_w=frame_w, frame_h=frame_h, n_frames=n_frames,
                           augment=False, oversample=False,tabular_lookup=test_tab)

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=pin,drop_last=True, collate_fn=mixup_collate if mixup else None,)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,num_workers=num_workers, pin_memory=pin,collate_fn=mixup_collate if mixup else None,)
    
    return train_loader, test_loader