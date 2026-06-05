import glob,sys,os
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from assessment.embryo_utils.embryo_video import EmbryoVideo



if __name__ == "__main__":
    path_videos = '../Datasets/Processed/Quiron_ploidy/videos/'
    path_save = Path('../Datasets/Processed/Quiron_ploidy/markers/')
    videos = glob.glob(path_videos + '*/*/*.mp4')
    for file in tqdm(videos,desc="Processing videos"):
        print(f"Processing {file}")
        embryo = EmbryoVideo(file, cropped=True)
        file = Path(file)
        split_class = Path(file.parent.parent.stem) / file.parent.stem
        embryo.get_evolution()
        #filename = str(Path(path_save) / split_class /f"{file.stem}.csv")
        #print(filename)
        #embryo.save_evolution(filename = filename)