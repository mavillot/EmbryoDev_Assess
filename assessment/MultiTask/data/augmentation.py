import cv2,sys
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import IMAGE_SIZE


def get_train_transforms(image_size=IMAGE_SIZE):
    h, w = image_size
    return A.Compose([
        # Spatial transforms first
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        A.ElasticTransform(
            alpha=120, sigma=120 * 0.05,
            border_mode=cv2.BORDER_REFLECT, p=0.3
        ),
        A.GridDistortion(
            num_steps=5, distort_limit=0.3,
            border_mode=cv2.BORDER_REFLECT, p=0.3
        ),

        
        A.Resize(h, w),

        # Appearance transforms (image only)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Normalize + tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask_str': 'mask', 'mask_frag': 'mask'})


def get_val_transforms(image_size=IMAGE_SIZE):
    h, w = image_size
    return A.Compose([
        A.Resize(h, w), 
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'mask_str': 'mask', 'mask_frag': 'mask'})