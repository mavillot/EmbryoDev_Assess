import torch,sys,os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import random
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
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



class MultiTaskDataset(Dataset):
    """
    Handles heterogeneous datasets for multitask:
    - Images with grading and segmentation labels
    Missing labels are set to None
    """
    def __init__(self, transform=None, image_samples=None):
        """
        image_samples: list of dicts with keys: 'image', 'mask_str', 'mask_frag', 'exp', 'icm', 'te', 'dataset_id'
        """
        self.image_samples = image_samples if image_samples is not None else []
        self.transform = transform
        
        # Combine both datasets with type tags
        self.samples = [('image', s) for s in self.image_samples]
        self.dataset_ids = [s.get('dataset_id', 'default') for _, s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_type, sample = self.samples[idx]

        if sample_type == 'image':
            img = sample['image']          

            exp = sample.get('exp', None)
            icm = sample.get('icm', None)
            te = sample.get('te', None)

            if self.transform is not None:
                if isinstance(img, Image.Image):
                    img = np.array(img.convert('RGB'))
                aug_input = {'image': img}
                
                
                augmented = self.transform(**aug_input)
                img       = augmented['image']
            else:
                # fallback to old behavior if no transform passed
                img = transform_image(img)

            return {
                'type': 'image',
                'image': img,
                'exp': exp,
                'icm': icm,
                'te': te,
                'dataset_id': sample.get('dataset_id', None)
            }



def multitask_collate(batch):
    batch_out = {}

    images = [b for b in batch if b['type'] == 'image']

    if images:
        batch_out['images'] = torch.stack([b['image'] for b in images])
        # ---- CLASSIFICATION LABELS ----
        for key in ['exp', 'icm', 'te']:
            vals = [b[key] for b in images]
            if all(v is None for v in vals):
                batch_out[key] = None
            else:
                batch_out[key] = torch.tensor(vals, dtype=torch.long)
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
