import os, sys
from pathlib import Path
import glob
import pandas as pd

from assessment.vIAbility.tabular.tabular_study import clean_diagnose, one_hot_encode

class Multimodal_Data():
    def __init__(self, video_path, excel_path):
        self.video_path = video_path
        self.excel_path = excel_path
        self.labels=[]
        self.videos={}
        self.df = self._load_excel_data()

    def _load_excel_data(self):
        return pd.read_excel(self.excel_path, sheet_name="Morula_Blast_nov")

    def preprocess(self):
        def fill_mean(df, column):
            mean_value = df[column].mean()
            if str(df[column].dtype).startswith('Int') or str(df[column].dtype).startswith('int'):
                mean_value = round(mean_value)
            df[column] = df[column].fillna(mean_value)
            return df
        implantation = {'mi': None,'hcg testpositiv': 1,'pos': 1, 'positiv': 1,'biokjemisk': 1, 
                        'neg': 0,'negativ': 0,'negative': 0, 'negativ (står positiv i lf uten mer informasjon, men neg. i sys.x.)': 0}
        smoke_map = {'0': 0,'nei': 0, '3': 1, '5': 1,'0-5/d': 1, '1-5': 1,'1 (0-5/d)': 1, '1 (ikke daglig)': 1, '1-10/d': 2, 
                     '10/d': 2, '1 (6-15/d)': 2,'1 (1-10/d)': 2, 'mi': None}
        snus_map = {'0': 0,'nei': 0, '1': 1, '2': 1, '3': 1, '5': 1, '4-5pk/d': 1, '1 (en sjelden gang)': 1,  '1 (0-1d/v)': 1, 
                    '1-2 d/v': 1, '1(slutter)': 1, 'mi': None}
        
        self.df['HCG Test'] = self.df['HCG Test'].map(implantation)
        self.df.dropna(subset=["HCG Test"], inplace=True)
        self.df['Fetal Heart Beat']= self.df['Fetal Heart Beat'].fillna(0)
        self.df.loc[self.df['Fetal Heart Beat'] == 'mi', 'Fetal Heart Beat'] = None
        self.df.dropna(subset=["Fetal Heart Beat"], inplace=True)
        self.df['smoke'] = self.df['smoke'].map(smoke_map).fillna(0)
        self.df['snus'] = self.df['snus'].map(snus_map).fillna(0)
        self.df['Fragmentation - Value'] = self.df['Fragmentation - Value'].fillna(0)
        for column in ['Age', 'Age_mann']:
            self.df = fill_mean(self.df, column)
        diagnose = [str(d).lower() for d in self.df["Diagnose"]]
        list_diseases = clean_diagnose(diagnose)
        self.df = one_hot_encode(self.df, list_diseases)
        self.df.drop(columns=["Diagnose"], inplace=True)

        return self.df
        

    def get_labels(self,class_name):
        if class_name=="no_FHB":
            return 0
        else:
            return 1
    
    def extract_tabular_features(self, video_pth):

        columns = [ 'Age', 'Age_mann', 'smoke', 'snus', 'Fragmentation - Value', 'ovulatory_factor', 'tubal_factor', 'uterine_factor', 'endometriosis_factor',
                    'male_factor', 'age_factor', 'unexplained_factor']
        id_part, well_part = Path(video_pth).stem.split("_Well_")
        well = int(well_part.split(".")[0])
        row = self.df[(self.df["ID"] == id_part) & (self.df["Well"] == well)]
        return row.loc[:, columns].fillna(0).values.flatten() 
    
    def load_data(self):
        videos = glob.glob(self.video_path + '*/*.avi')
        for video_pth in videos:
            class_name = Path(video_pth).parent.stem
            self.labels.append(self.get_labels(class_name))
            tabular_feat = self.extract_tabular_features(video_pth)
            self.videos[video_pth] = tabular_feat
        return self.videos, self.labels
