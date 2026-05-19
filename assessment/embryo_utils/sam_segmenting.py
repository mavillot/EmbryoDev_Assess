# INSTALACIÓN
#!pip install git+https://github.com/facebookresearch/segment-anything.git
#!pip install opencv-python pycocotools matplotlib onnxruntime onnx
#pip install SemTorch
#!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ../models/sam_vit_h_4b8939.pth


import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
torch.cuda.set_device(0)
import PIL
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull

from fastai.basics import *
from fastai.vision import models
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *
from fastai.callback import *

# SemTorch
from semtorch import get_segmentation_learner
from pathlib import Path
import random
import torchvision.transforms as transforms


# SAM MODEL
sam_checkpoint = "models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

#HRNET
path='models/hrnet.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(path)
model = model.cpu()

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



def conectivity(region, polar):
    dil=cv2.dilate(region,np.ones((8,8)))
    return np.sum(polar[dil==255].flatten())>0

def counting_polar_corpuscles(image, zp_mask):
    masks = mask_generator.generate(image)
    region=ConvexHull(zp_mask)
    region=cv2.dilate(region,np.ones((15,15)))
    list_elements=[]
    for m in masks:
        if np.sum(region[m['segmentation']].flatten())>0 and np.sum(m['segmentation'][region==False].flatten())==0:
            list_elements.append(m)
        if list_elements==[]:
            return 0, []
      
    areas=[el['area'] for el in list_elements]
    embryo=list_elements[areas.index(max(areas))]
    candidates=[el['segmentation'] for el in list_elements if el['area']!= embryo['area'] and np.sum(embryo['segmentation'][el['segmentation']].flatten())==0]
    final_candidates=[]
    for c in candidates:
        if conectivity(255*embryo['segmentation'].astype('uint8'), c):
            final_candidates.append(c)
    return len(final_candidates), final_candidates