import torch, sys, os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import random
from PIL import Image
from transformers import AutoImageProcessor

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import IMAGE_SIZE
from data.augmentation import (
    apply_video_train_transforms,
    apply_video_val_transforms,
)

processor = AutoImageProcessor.from_pretrained("ihlab/FEMI")
# ---------------------------------------------------------------------------
# Fallback transforms (used when no albumentations pipeline is provided)
# ---------------------------------------------------------------------------

def transform_image(img, size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img)
    return img

def transform_femi(img):
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = Image.fromarray(img.astype(np.uint8))
    inputs = processor(images=img, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)

def transform_mask(mask, size):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    mask = torch.from_numpy(mask).long()
    mask = transforms.Resize(
        size,
        interpolation=transforms.InterpolationMode.NEAREST
    )(mask.unsqueeze(0)).squeeze(0)
    return mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """
    Handles heterogeneous datasets for multi-task learning:
    - Images with grading and segmentation labels
    - Videos with ploidy / fhb labels

    Args:
        image_samples:      list of dicts with keys:
                            'image', 'mask_str', 'mask_frag',
                            'exp', 'icm', 'te', 'stage', 'dataset_id'
        video_samples:      list of dicts with keys:
                            'video' (list[PIL]), 'ploidy', 'fhb', 'dataset_id'
        transform:          albumentations Compose for images  (or None)
        video_transform:    tuple (spatial_ReplayCompose, appearance_Compose)
                            OR a plain Compose for val  (or None)
        is_train:           if True and video_transform is provided, spatial+
                            appearance augmentation is applied to videos;
                            otherwise only the val deterministic pipeline runs.
        use_femi_processor: if True, apply the FEMI image processor to every
                            image and video frame after augmentation.
    """

    def __init__(
        self,
        image_samples=None,
        video_samples=None,
        img_size = None,
        transform=None,
        video_transform=None,
        is_train=False,
        use_femi_processor=False,
    ):
        self.image_samples  = image_samples  if image_samples  is not None else []
        self.video_samples  = video_samples  if video_samples  is not None else []
        self.img_size       = img_size       if img_size       is not None else IMAGE_SIZE
        self.transform      = transform
        self.video_transform = video_transform
        self.is_train       = is_train
        self.use_femi_processor = use_femi_processor

        self.samples = (
            [('image', s) for s in self.image_samples] +
            [('video', s) for s in self.video_samples]
        )
        self.dataset_ids = [s.get('dataset_id', 'default') for _, s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_type, sample = self.samples[idx]

        # ------------------------------------------------------------------ #
        #  IMAGE sample                                                        #
        # ------------------------------------------------------------------ #
        if sample_type == 'image':
            img      = sample['image']
            mask_str  = sample.get('mask_str',  None)
            mask_frag = sample.get('mask_frag', None)
            exp   = sample.get('exp',   None)
            icm   = sample.get('icm',   None)
            te    = sample.get('te',    None)
            stage = sample.get('stage', None)

            if self.transform is not None:
                if isinstance(img, Image.Image):
                    img = np.array(img.convert('RGB'))
                if isinstance(mask_str,  Image.Image): mask_str  = np.array(mask_str)
                if isinstance(mask_frag, Image.Image): mask_frag = np.array(mask_frag)

                aug_input = {'image': img}
                if mask_str  is not None: aug_input['mask_str']  = mask_str
                if mask_frag is not None: aug_input['mask_frag'] = mask_frag

                augmented = self.transform(**aug_input)
                img       = augmented['image']
                mask_str  = torch.as_tensor(augmented['mask_str']).long()  if mask_str  is not None else None
                mask_frag = torch.as_tensor(augmented['mask_frag']).long() if mask_frag is not None else None
            else:
                img = transform_femi(img) if self.use_femi_processor else transform_image(img, size=self.img_size)
                if mask_str  is not None: mask_str  = transform_mask(mask_str, size=self.img_size)
                if mask_frag is not None: mask_frag = transform_mask(mask_frag, size=self.img_size)

            if self.use_femi_processor and self.transform is not None:
                img = transform_femi(img)

            return {
                'type':       'image',
                'image':      img,
                'mask_str':   mask_str,
                'mask_frag':  mask_frag,
                'exp':        exp,
                'icm':        icm,
                'te':         te,
                'stage':      stage,
                'dataset_id': sample.get('dataset_id', None),
            }

        # ------------------------------------------------------------------ #
        #  VIDEO sample                                                        #
        # ------------------------------------------------------------------ #
        else:
            vid    = sample['video']   # list of PIL Images
            ploidy = sample.get('ploidy', None)
            fhb    = sample.get('fhb',    None)

            if self.video_transform is not None:
                if self.is_train and isinstance(self.video_transform, tuple):
                    # Tuple means (spatial_ReplayCompose, appearance_Compose)
                    spatial_tfm, appearance_tfm = self.video_transform
                    vid = apply_video_train_transforms(vid, spatial_tfm, appearance_tfm)
                else:
                    # Single Compose → deterministic val pipeline
                    vid = apply_video_val_transforms(vid, self.video_transform)
            else:
                # Fallback: simple per-frame resize + normalize (no augmentation)
                frame_transform = transform_femi if self.use_femi_processor else transform_image
                if self.use_femi_processor:
                    vid = torch.stack([frame_transform(f) for f in vid], dim=0)
                else:
                    vid = torch.stack([frame_transform(f, size=self.img_size) for f in vid], dim=0)

            return {
                'type':       'video',
                'video':      vid,
                'ploidy':     ploidy,
                'fhb':        fhb,
                'dataset_id': sample.get('dataset_id', None),
            }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def multitask_collate(batch):
    batch_out = {}

    images = [b for b in batch if b['type'] == 'image']
    videos = [b for b in batch if b['type'] == 'video']

    if images:
        batch_out['images'] = torch.stack([b['image'] for b in images])

        # FIX: only stack masks when ALL samples in the batch have them;
        #      mixed batches (some None, some tensor) would crash torch.stack.
        for key in ['mask_str', 'mask_frag']:
            vals = [b[key] for b in images]
            if all(v is None for v in vals):
                batch_out[key] = None
            elif any(v is None for v in vals):
                # Partial mask availability — keep as None to avoid silent errors.
                # Callers should ensure dataset_id grouping avoids mixing.
                batch_out[key] = None
            else:
                batch_out[key] = torch.stack(vals)

        for key in ['exp', 'icm', 'te', 'stage']:
            vals = [b[key] for b in images]
            if all(v is None for v in vals):
                batch_out[key] = None
            else:
                # Replace None entries with -1 (ignored by cross-entropy by default)
                safe_vals = [v if v is not None else -1 for v in vals]
                batch_out[key] = torch.tensor(safe_vals, dtype=torch.long)

    if videos:
        batch_out['videos'] = torch.stack([b['video'] for b in videos])

        for key in ['ploidy', 'fhb']:
            vals = [b[key] for b in videos]
            if all(v is None for v in vals):
                batch_out[key] = None
            else:
                safe_vals = [v if v is not None else float('nan') for v in vals]
                batch_out[key] = torch.tensor(safe_vals, dtype=torch.float32)

    return batch_out


# ---------------------------------------------------------------------------
# Class-balancing utilities
# ---------------------------------------------------------------------------

# Which label fields are considered for balancing, in priority order.
# The first non-None label found on a sample is used to assign its weight.
# Priority reflects severity of imbalance observed in practice:
#   icm and te are most skewed, then exp, stage; masks get a neutral weight.
_BALANCE_PRIORITY = ['icm', 'te', 'exp', 'stage', 'ploidy', 'fhb', 'mask_str', 'mask_frag']


def compute_sample_weights(
    dataset: "MultiTaskDataset",
    task_weights: dict | None = None,
) -> np.ndarray:
    """
    Compute a per-sample sampling weight for class-balanced oversampling.

    Strategy
    --------
    For each sample, find its *primary label* — the first non-None label in
    _BALANCE_PRIORITY order.  Compute inverse-frequency weights within each
    (dataset_id, task, class) bucket so that every class is drawn equally
    often.  Samples with only mask labels (no discrete class) get weight 1.0.

    Args:
        dataset:      a MultiTaskDataset instance
        task_weights: optional dict mapping task name → float multiplier,
                      e.g. {'icm': 2.0, 'te': 2.0} to up-weight already-
                      boosted tasks even further.  Defaults to 1.0 per task.

    Returns:
        np.ndarray of shape (len(dataset),) with positive float weights.
    """
    task_weights = task_weights or {}
    n = len(dataset.samples)
    weights = np.ones(n, dtype=np.float64)

    # Build (dataset_id, task, class) → list[sample_idx] mapping
    from collections import defaultdict
    bucket: dict[tuple, list[int]] = defaultdict(list)

    for idx, (_, sample) in enumerate(dataset.samples):
        ds_id = sample.get('dataset_id', 'default')
        assigned = False
        for field in _BALANCE_PRIORITY:
            val = sample.get(field)
            if val is not None and not isinstance(val, (np.ndarray, list)):
                bucket[(ds_id, field, int(val))].append(idx)
                assigned = True
                break
        # Samples with only mask labels stay at weight 1.0 (already bucketless)
        if not assigned:
            pass

    # Inverse-frequency: weight = 1 / count  within each bucket,
    # then normalise so the mean weight per dataset_id group stays ~1.
    for (ds_id, task, cls), indices in bucket.items():
        w = 1.0 / len(indices)
        tw = task_weights.get(task, 1.0)
        for i in indices:
            weights[i] = w * tw

    # Normalise per dataset_id so groups don't dominate each other
    group_indices: dict = {}
    for idx, (_, sample) in enumerate(dataset.samples):
        ds_id = sample.get('dataset_id', 'default')
        group_indices.setdefault(ds_id, []).append(idx)

    for ds_id, indices in group_indices.items():
        group_w = weights[indices]
        mean_w  = group_w.mean()
        if mean_w > 0:
            weights[indices] = group_w / mean_w   # mean weight per group = 1.0

    return weights


# ---------------------------------------------------------------------------
# Batch sampler — with optional class-balanced oversampling
# ---------------------------------------------------------------------------

class MultiTaskBatchSampler(Sampler):
    """
    Yields batches where all samples come from the same dataset_id.

    When sample_weights is provided (from compute_sample_weights), minority
    classes are oversampled so each batch sees a more balanced class mix.
    Without weights the behaviour is identical to the original sampler.

    Args:
        dataset:        MultiTaskDataset instance
        batch_size:     samples per batch
        shuffle:        shuffle batches each epoch
        sample_weights: per-sample weights from compute_sample_weights(),
                        or None to use uniform sampling (no oversampling)
        oversample_factor: how many times larger the oversampled epoch is
                        relative to the original dataset size.
                        1.0 = same number of batches as without oversampling;
                        2.0 = twice as many (each minority sample seen ~2×).
                        Only used when sample_weights is not None.
    """

    def __init__(
        self,
        dataset,
        batch_size:       int,
        shuffle:          bool = True,
        sample_weights:   np.ndarray | None = None,
        oversample_factor: float = 1.5,
    ):
        self.dataset           = dataset
        self.batch_size        = batch_size
        self.shuffle           = shuffle
        self.sample_weights    = sample_weights
        self.oversample_factor = oversample_factor

        # Group dataset indices by dataset_id
        self.groups: dict[str, list[int]] = {}
        for idx, ds_id in enumerate(dataset.dataset_ids):
            self.groups.setdefault(ds_id, []).append(idx)

    def _sample_indices(self, indices: list[int]) -> list[int]:
        """
        Draw indices for one group, with or without oversampling.

        Without weights: simple shuffle of the original indices.
        With weights:    weighted-random draw of
                         round(len * oversample_factor) indices with replacement.
        """
        if self.sample_weights is None:
            if self.shuffle:
                random.shuffle(indices)
            return list(indices)

        # Weighted draw with replacement
        group_weights = self.sample_weights[indices]
        total         = round(len(indices) * self.oversample_factor)
        probs         = group_weights / group_weights.sum()
        drawn         = np.random.choice(indices, size=total, replace=True, p=probs)
        if self.shuffle:
            np.random.shuffle(drawn)
        return drawn.tolist()

    def __iter__(self):
        all_batches = []

        for indices in self.groups.values():
            sampled = self._sample_indices(indices)

            for i in range(0, len(sampled), self.batch_size):
                batch = sampled[i : i + self.batch_size]
                # Pad last (potentially short) batch by re-drawing from the group
                if len(batch) < self.batch_size:
                    pad = np.random.choice(
                        indices,
                        size=self.batch_size - len(batch),
                        replace=True,
                    ).tolist()
                    batch = batch + pad
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        if self.sample_weights is None:
            return sum(
                (len(v) + self.batch_size - 1) // self.batch_size
                for v in self.groups.values()
            )
        # With oversampling, each group draws oversample_factor × more samples
        return sum(
            max(1, round(len(v) * self.oversample_factor) // self.batch_size)
            for v in self.groups.values()
        )