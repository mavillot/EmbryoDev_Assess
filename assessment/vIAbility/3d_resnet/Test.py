import pandas as pd
from glob import glob


# To verify distribution of Independent dataset
"""
excel_file='Excel/ClassLabels.xlsx'
video_files='Data/Set1_IndependentDataset/*.avi'
videos=glob(video_files)

dataDF = pd.read_excel(excel_file, sheet_name='Set1_IndependentDataset', header=0, index_col=None)
i=1
for video in videos:
    videoName = video.split('/')[-1].split('.av')[0]
    row_id = dataDF.index[dataDF['Video_Id'] == videoName]
    print(i,dataDF['Outcome'][row_id].item())
    i += 1
"""