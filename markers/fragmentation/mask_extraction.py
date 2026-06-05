import os,sys
import json
import numpy as np
from pathlib import Path
import cv2
from pycocotools import mask as maskUtils

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET
# Paths
path_data = os.path.join(PATH_DATASET, "CleavageEmbryo")
json_path = os.path.join(path_data, "annotations/instances_train.json")
images_dir = os.path.join(path_data, "images")  # directorio con las imágenes
output_dir = os.path.join(path_data, "masks_blastomere/train")   # donde guardar las máscaras
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
with open(json_path, "r") as f:
    coco = json.load(f)

# Index annotations by image_id
annotations_by_image = {}
for ann in coco["annotations"]:
    if ann["category_id"] == 1:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# Process each image
for img in coco["images"]:
    img_id = img["id"]
    h, w = img["height"], img["width"]

    # empty mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if img_id in annotations_by_image:
        for ann in annotations_by_image[img_id]:
            seg = ann["segmentation"]

            # Polygons
            if isinstance(seg, list):
                rles = maskUtils.frPyObjects(seg, h, w)
                rle = maskUtils.merge(rles)
            # RLE
            else:
                rle = seg

            m = maskUtils.decode(rle)
            mask = np.maximum(mask, m)

    # Guardar máscara
    out_name = os.path.splitext(img["file_name"])[0] + ".png"
    cv2.imwrite(os.path.join(output_dir, out_name), mask * 255)

print("Masks extracted correctly.")