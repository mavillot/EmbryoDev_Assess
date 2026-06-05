import os, cv2, random
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET

PATH_PROCESSED = os.path.join(PATH_DATASET, "stage_classif/EVALUATE")
PATH_DATASET = os.path.join(PATH_DATASET, "Oslo_processed")


def load_metadata(excel_path: str) -> pd.DataFrame:
    """Load and validate the Excel metadata file."""
    df = pd.read_excel(excel_path)
    return df

def parse_video_name(filename: str) -> tuple[str, int]:
    """Return (id, well) from a filename like  '<id>_Well_<well>.ext'."""
    try:
        id_part, well_part = filename.split("_Well_")
        well = int(well_part.split(".")[0])
        return id_part, well
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected filename format: '{filename}'") from exc
    

def extract_frames(video_fp, tse, start_time, end_time, frame_hour):
    cap = cv2.VideoCapture(video_fp)
    frames = []
    start_frame = int((pd.to_numeric(start_time) - pd.to_numeric(tse)) * frame_hour)
    end_frame = int((pd.to_numeric(end_time) - pd.to_numeric(tse))  * frame_hour)
    try:
        for frame_idx in range(start_frame, end_frame):
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, img = cap.read()
                if ret:
                    img = img[..., ::-1].copy()  # BGR→RGB
                    frames.append(img)   
    finally:
        cap.release()
    return frames

def save_frames(frames, filename, save_dir=os.path.join(PATH_PROCESSED, "Images")):
    os.makedirs(save_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(save_dir, f"{filename}_frame_{i}.jpg"), frame)

def save_frames_stage(video_fp, df, save_dir = PATH_PROCESSED):
    vid_id, well = parse_video_name(os.path.basename(video_fp))
    rows = df[(df["ID"] == vid_id) & (df["Well"] == well)]
    tse, frame_hour = rows["Tse"].iloc[0], rows["frame_hour"].iloc[0]
    try:
        # t2 = df['t2'].iloc[0]
        # t3 = df['t3'].iloc[0]
        # t4 = df['t4'].iloc[0]
        # t5 = df['t5'].iloc[0] if not None else t4 + 5
        # t8 = df['t8'].iloc[0] if not None else t5 + 5
        # t9 = df['t9+'].iloc[0] if not None else t8 + 5
        tm = df['tM'].iloc[0]
        tsb = df['tSB'].iloc[0]
        tb = df['tB'].iloc[0]

        #frames = extract_frames(video_fp, tse, start_time=tse, end_time=tse+8, frame_hour=frame_hour)
        #save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "empty"))
        #frames = extract_frames(video_fp, tse, start_time=t2, end_time=t3, frame_hour=frame_hour)
        #save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "2cell"))
        #frames = extract_frames(video_fp, tse, start_time=t3, end_time=t4, frame_hour=frame_hour)
        #save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "3cell"))
        #frames = extract_frames(video_fp, tse, start_time=t4, end_time=t5, frame_hour=frame_hour)
        #save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "4cell"))
        #frames = extract_frames(video_fp, tse, start_time=t5, end_time=t8, frame_hour=frame_hour)
        save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "+4cell"))
        frames = extract_frames(video_fp, tse, start_time=t8, end_time=tm, frame_hour=frame_hour)
        save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "+8cell"))
        if tsb:
            frames = extract_frames(video_fp, tse, start_time=tm, end_time=tsb, frame_hour=frame_hour)
            save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "morula"))
            if tb:
                frames = extract_frames(video_fp, tse, start_time=tsb, end_time=tb, frame_hour=frame_hour)
                save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "blastocyst"))
            else:
                frames = extract_frames(video_fp, tse, start_time=tsb, end_time=tsb+2, frame_hour=frame_hour)
                save_frames(frames, filename=os.path.basename(video_fp), save_dir=os.path.join(save_dir, "blastocyst"))
    except Exception as exc:
        print(f"Error processing {video_fp}: {exc}")


def main():
    df = load_metadata(os.path.join(PATH_DATASET, "Oslo_completo.xlsx"))
    video_fps = [os.path.join(PATH_DATASET, "videos", f) for f in os.listdir(os.path.join(PATH_DATASET, "videos")) if f.endswith((".mp4", ".avi"))]
    for video_fp in video_fps[:15]:
        save_frames_stage(video_fp, df)

if __name__ == "__main__":
    main()