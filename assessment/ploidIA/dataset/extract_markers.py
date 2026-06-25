import glob,sys,os
from tqdm import tqdm
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from assessment.embryo_utils.embryo_video import EmbryoVideo
from config.config import PATH_DATASET

PATH_PROCESSED = Path(PATH_DATASET) / 'EasyFiv'


if __name__ == "__main__":
    path_videos = PATH_PROCESSED / 'videos'
    path_save = PATH_PROCESSED / 'markers'
    videos = glob.glob(str(path_videos) + '*/*/*/*.mp4')
    print(f"Found {len(videos)} videos")
    for file in tqdm(videos,desc="Processing videos"):
        print(f"Processing {file}")
        embryo = EmbryoVideo(file, cropped=False, n_MAX = 200, n_total = 1300)
        embryo.get_evolution()
        file = Path(file)
        split_class = Path(file.parent.parent.stem) / file.parent.stem
        filename = str(Path(path_save) / split_class /f"{file.stem}.csv")
        print(f"Saving markers to {filename}")
        embryo.save_evolution(filename = filename)