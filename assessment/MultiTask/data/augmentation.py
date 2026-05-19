import cv2, sys
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import IMAGE_SIZE


def get_train_transforms(image_size=IMAGE_SIZE):
    h, w = image_size
    return A.Compose([
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        A.ElasticTransform(
            alpha=120, sigma=6,                          
            border_mode=cv2.BORDER_REFLECT, p=0.3
        ),
        A.GridDistortion(
            num_steps=5, distort_limit=0.3,
            border_mode=cv2.BORDER_REFLECT, p=0.3
        ),

        A.Resize(h, w),

        # Appearance transforms (image only)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussNoise(p=0.3),                             
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


def get_video_train_transforms(image_size=IMAGE_SIZE):
    """
    Augmentation pipeline for video frames.

    Spatial transforms (flip, rotate, crop) are applied consistently across ALL
    frames in a clip via a shared ReplayCompose — call `.replay(params, image=frame)`
    for frames 2..T after calling the transform on frame 1.

    Appearance transforms (color jitter, blur, noise) are applied independently
    per frame so the model learns temporal robustness.
    """
    h, w = image_size

    spatial = A.ReplayCompose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT, p=0.4
        ),
        A.Resize(h, w),
    ])

    appearance = A.Compose([
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.4),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return spatial, appearance


def get_video_val_transforms(image_size=IMAGE_SIZE):
    """Returns a single deterministic pipeline for validation video frames."""
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def apply_video_train_transforms(frames, spatial_tfm, appearance_tfm):
    """
    Apply augmentation to a list of PIL/numpy frames.

    Spatial transforms are replayed from the first frame so all frames
    receive the same geometric perturbation. Appearance transforms are
    applied independently.

    Args:
        frames:         list of PIL Images or np.ndarrays  (T frames)
        spatial_tfm:    ReplayCompose returned by get_video_train_transforms()[0]
        appearance_tfm: Compose returned by get_video_train_transforms()[1]

    Returns:
        torch.Tensor of shape (T, C, H, W)
    """
    import torch
    from PIL import Image

    aug_frames = []
    replay_params = None

    for i, frame in enumerate(frames):
        if isinstance(frame, Image.Image):
            frame = np.array(frame.convert('RGB'))
        elif frame.ndim == 2:                   # grayscale → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Apply shared spatial transform
        if i == 0:
            result = spatial_tfm(image=frame)
            replay_params = result['replay']
        else:
            result = A.ReplayCompose.replay(replay_params, image=frame)

        frame_spatial = result['image']

        # Apply per-frame appearance transform
        frame_tensor = appearance_tfm(image=frame_spatial)['image']
        aug_frames.append(frame_tensor)

    return torch.stack(aug_frames, dim=0)   # (T, C, H, W)


def apply_video_val_transforms(frames, val_tfm):
    """
    Apply deterministic val transforms to a list of PIL/numpy frames.

    Returns:
        torch.Tensor of shape (T, C, H, W)
    """
    import torch
    from PIL import Image

    aug_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            frame = np.array(frame.convert('RGB'))
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        aug_frames.append(val_tfm(image=frame)['image'])

    return torch.stack(aug_frames, dim=0)   # (T, C, H, W)