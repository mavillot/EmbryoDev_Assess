import os, sys
import pandas as pd
from pathlib import Path
import glob
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import PATH_DATASET
from assessment.embryo_utils.embryo_video import EmbryoVideo


path = Path(PATH_DATASET) / 'videos_limpios/videos'
videos = glob.glob(str(path / '*.mp4'))

preds = {'video':[], 'time':[]}

for f in videos[1:]: 
    print(f"Evaluating {f}")
    video = EmbryoVideo(f, n_MAX = 2000, n_total = 2000, cropped = True)
    values = video.get_evolution()
    try:
        blasto = values['stage_classif'].index('blastocyst')
        # blasto = values['cleavage'].index(2)
    except:
        blasto = len(video)
    fps = video.get_fps()
    time = blasto / fps
    segs = time % 60
    mins = time  // 60
    f = Path(f)
    preds['video'].append(f.stem)
    preds['time'].append(f"{int(mins)}.{int(segs)}")

pd.DataFrame(preds).to_csv('blasto_mom_stage.csv', index=False)
    
