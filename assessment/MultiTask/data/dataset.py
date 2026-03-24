import torch,sys,os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import random
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import IMAGE_SIZE

def transform_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR)(img)
    img = transforms.ToTensor()(img)  # converts to (C, H, W) float in [0, 1]
    img = transforms.Normalize(       # ImageNet stats for pretrained ResNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img)
    return img

def transform_mask(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    mask = torch.from_numpy(mask).long()
    mask = transforms.Resize(
        IMAGE_SIZE,
        interpolation=transforms.InterpolationMode.NEAREST
    )(mask.unsqueeze(0)).squeeze(0)
    return mask



class MultiTaskDataset(Dataset):
    """
    Handles heterogeneous datasets for multitask:
    - Images with grading and segmentation labels
    - Videos with ploidy labels
    Missing labels are set to None
    """
    def __init__(self, image_samples=None, video_samples=None, transform=None):
        """
        image_samples: list of dicts with keys: 'image', 'mask_str', 'mask_frag', 'exp', 'icm', 'te', 'dataset_id'
        video_samples: list of dicts with keys: 'video', 'ploidy', 'dataset_id'
        """
        self.image_samples = image_samples if image_samples is not None else []
        self.video_samples = video_samples if video_samples is not None else []
        self.transform = transform
    
        # Combine both datasets with type tags
        self.samples = [('image', s) for s in self.image_samples] + \
                       [('video', s) for s in self.video_samples]
        self.dataset_ids = [s.get('dataset_id', 'default') for _, s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_type, sample = self.samples[idx]

        if sample_type == 'image':
            img = sample['image']          # PIL Image
            mask_str = sample.get('mask_str', None)  # numpy or None
            mask_frag = sample.get('mask_frag', None)

            exp = sample.get('exp', None)
            icm = sample.get('icm', None)
            te = sample.get('te', None)

            if self.transform is not None:
                if isinstance(img, Image.Image):
                    img = np.array(img.convert('RGB'))
                if isinstance(mask_str, Image.Image):
                    mask_str = np.array(mask_str)
                if isinstance(mask_frag, Image.Image):
                    mask_frag = np.array(mask_frag)
                aug_input = {'image': img}
                if mask_str  is not None: aug_input['mask_str']  = mask_str
                if mask_frag is not None: aug_input['mask_frag'] = mask_frag
                
                augmented = self.transform(**aug_input)
                img       = augmented['image']
                mask_str  = augmented['mask_str'].long()  if mask_str  is not None else None
                mask_frag = augmented['mask_frag'].long() if mask_frag is not None else None
            else:
                # fallback to old behavior if no transform passed
                img = transform_image(img)
                if mask_str  is not None: mask_str  = transform_mask(mask_str)
                if mask_frag is not None: mask_frag = transform_mask(mask_frag)

            return {
                'type': 'image',
                'image': img,
                'mask_str': mask_str,
                'mask_frag': mask_frag,
                'exp': exp,
                'icm': icm,
                'te': te,
                'dataset_id': sample.get('dataset_id', None)
            }

        else:  # video
            vid = sample['video']  # list of PIL images
            ploidy = sample.get('ploidy', None)

            vid = torch.stack([transform_image(f) for f in vid], dim=0)

            return {
                'type': 'video',
                'video': vid,
                'ploidy': ploidy,
                'dataset_id': sample.get('dataset_id', None)
            }



def multitask_collate(batch):
    batch_out = {}

    images = [b for b in batch if b['type'] == 'image']
    videos = [b for b in batch if b['type'] == 'video']

    if images:
        batch_out['images'] = torch.stack([b['image'] for b in images])
    
        for key in ['mask_str', 'mask_frag']:
            vals = [b[key] for b in images]
            if all(v is None for v in vals):
                batch_out[key] = None
            else:
                batch_out[key] = torch.stack(vals)

        # ---- CLASSIFICATION LABELS ----
        for key in ['exp', 'icm', 'te']:
            vals = [b[key] for b in images]
            if all(v is None for v in vals):
                batch_out[key] = None
            else:
                batch_out[key] = torch.tensor(vals, dtype=torch.long)

    if videos:
        batch_out['videos'] = torch.stack([b['video'] for b in videos])

        vals = [b['ploidy'] for b in videos]
        if all(v is None for v in vals):
            batch_out['ploidy'] = None
        else:
            batch_out['ploidy'] = torch.tensor(vals, dtype=torch.float32)

    return batch_out


class MultiTaskBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by dataset_id
        self.groups = {}
        for idx, ds_id in enumerate(dataset.dataset_ids):
            self.groups.setdefault(ds_id, []).append(idx)

    def __iter__(self):
        all_batches = []

        for _, indices in self.groups.items():
            if self.shuffle:
                random.shuffle(indices)

            # Create batches within each dataset
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                #if the batch is smaller than batch_size, we can fill it with random samples from the same group
                while len(batch) < self.batch_size:
                    batch.append(random.choice(indices))
                # there is an other option to discard smaller batches.    
                # if len(batch) == self.batch_size:
                all_batches.append(batch)
                

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum((len(v) + self.batch_size - 1) // self.batch_size for v in self.groups.values())
