import random
from pathlib import Path
import numpy as np
import pandas as pd
from glob import glob
import cv2
import tensorflow as tf

class Video_Data():
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels=[]
        self.videos=[]

    def get_labels(self,class_name):
        if class_name=="no_FHB":
            return 0
        else:
            return 1

    def load_data(self):
        videos = glob(self.data_path + '*/*.avi')
        #print(len(videos))
        #flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")
        for video_pth in videos:
            temp_even=[]
            temp_odd=[]
            class_name = Path(video_pth).parent.stem
            cap = cv2.VideoCapture(video_pth)
            i=0
            while (cap.isOpened()):
                ret, frame = cap.read()
                try:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    if i % 2 ==0:
                        temp_even.append(frame)
                    else:
                        temp_odd.append(frame)
                    i += 1
                except:
                    break
            cap.release()
            temp_even = np.array(temp_even)
            temp_odd = np.array(temp_odd)

            self.videos.append(temp_even)
            self.videos.append(temp_odd)


            for i in range(2):
                self.labels.append(self.get_labels(class_name))

        dataset=np.array(self.videos)
        temp = list(zip(dataset, self.labels))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        res1, res2 = list(res1), list(res2)
        train_x, train_y = np.array(res1), np.array(res2)
        return train_x, train_y


