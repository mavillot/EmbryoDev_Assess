import os, sys, argparse, random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fastai.vision.all import *
from fastai.callback.all import *

from albumentations import (
    Compose, HorizontalFlip, Rotate,
    ElasticTransform, GridDistortion,
    Transpose, ShiftScaleRotate
)

from semtorch import get_segmentation_learner
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET

number_of_the_seed = 42

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed, reproducible=False)


def get_y_fn(x):
    return Path(str(x).replace("images","masks").replace("jpg","png"))

class AlbumentationsSeg(ItemTransform):
    def __init__(self, aug): 
        self.aug = aug
        
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])

class MaskCreator(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        #Convert to array
        mask = np.array(mask)
        mask[mask==0]=0
        mask[mask==255]=1  
        
        # Back to PILMask
        return img, PILMask.create(mask)
    
def _pred_target(input, target):
    target = target.squeeze(1)
    preds = input.argmax(dim=1)
    return preds, target

def accuracy(input, target):
    preds, target = _pred_target(input, target)
    return (preds == target).float().mean()

def background_accuracy(input, target):
    preds, target = _pred_target(input, target)
    mask = target == 0
    return (preds[mask] == target[mask]).float().mean()

def fragmentation_accuracy(input, target):
    preds, target = _pred_target(input, target)
    mask = target == 1
    return (preds[mask] == target[mask]).float().mean()

def build_callbacks(fname):
    return [
        EarlyStoppingCallback(monitor="valid_loss", patience=2),
        SaveModelCallback(monitor="valid_loss", fname=fname)
    ]


def main(args):
    path = os.path.join(PATH_DATASET, args.dataset)
    path_images = os.path.join(path, "images")
    path_images_train = os.path.join(path_images, "train")
    transforms=Compose([HorizontalFlip(p=0.5),Rotate(p=0.3,limit=10), 
                        ElasticTransform(p=0.3), GridDistortion(p=0.3),
                        Transpose(p=0.3), ShiftScaleRotate(p=0.3)],p=1)

    codes = ['Background','Fragmentation']

    trainDB = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                    get_items=get_image_files,
                    get_y=get_y_fn,
                    splitter=RandomSplitter(valid_pct=0.2, seed=number_of_the_seed),
                    item_tfms=[Resize(args.size, method=ResizeMethod.Pad, pad_mode='zeros'), MaskCreator(), AlbumentationsSeg(transforms)],
                    batch_tfms=Normalize.from_stats(*imagenet_stats)
                    )

    batch_size = args.batch_size
    path_images=str(path_images_train)
    trainDLS = trainDB.dataloaders(path_images,
                                   bs=batch_size)
    trainDLS.show_batch(max_n=16, figsize=(12, 10))
    plt.savefig('output/fragmentation/train_batch_augmentation.png')

    learn = get_segmentation_learner(
        dls=trainDLS,
        number_classes=len(codes),
        segmentation_type="Semantic Segmentation",
        architecture_name=args.model_name,
        backbone_name=args.backbone_name,
        metrics=[accuracy, fragmentation_accuracy, background_accuracy],
        wd=args.weight_decay,
        pretrained=True,
        normalize=True).to_fp16()

    cbs_before = build_callbacks("fragmentation/hrnet_before_unfreeze")
    cbs_after  = build_callbacks("fragmentation/hrnet_after_unfreeze")

    learn.fit_one_cycle(
        args.epochs_before_unfreeze, 
        lr_max=args.lr,
        cbs=cbs_before)

    learn.unfreeze()

    learn.fit_one_cycle(
        args.epochs_after_unfreeze,
        lr_max=slice(args.lr/10, args.lr),
        cbs=cbs_after)


    aux=learn.model
    aux=aux.cpu()
    traced_cell=torch.jit.script(aux)
    traced_cell.save("models/markers/fragmentation/fragmentation_hrnet.pth")

if __name__ == "__main__":
    seed_everything(number_of_the_seed)
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Embryo fragmentation training script")

    parser.add_argument("--dataset", type=str, default="CleavageEmbryo",
                        help="Name of the directory containing the dataset")
    parser.add_argument("--size", type=int, default=480,
                        help="Size of the images (they will be resized to size x size)")
    parser.add_argument("--model_name", type=str, default="hrnet",
                        help="Model architecture to use")
    parser.add_argument("--backbone_name", type=str, default="hrnet_w30",
                        help="Backbone architecture to use")
    parser.add_argument("--epochs_before_unfreeze", type=int, default=1)
    parser.add_argument("--epochs_after_unfreeze", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)