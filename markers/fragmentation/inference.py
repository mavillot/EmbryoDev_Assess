import cv2, glob,os, sys
import tqdm
import PIL
import torch
torch.cuda.set_device(0)
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

torch._C._jit_set_profiling_mode(False)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET

  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_image(image):
    to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )])
    return to_tensor_norm(image).unsqueeze(0).to(device)

def predict_fragmentation(model, image_path, shape):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    image = transforms.Resize(shape)(img)
    input_tensor = transform_image(image)
    
    with torch.no_grad():
        outputs = model(input_tensor)

    outputs = torch.argmax(outputs, 1)
    mask = outputs.squeeze(0).cpu().numpy()
    mask[mask == 1] = 255
    return np.array(transforms.Resize((orig_h, orig_w))(Image.fromarray(mask.astype('uint8'))))

def color_inference(image, mask):
    # Convert mask to 3 channels
    mask_3c = np.stack([mask]*3, axis=-1)
    # Create color overlay
    color_mask = np.zeros_like(mask_3c)
    color_mask[mask > 0] = [255, 0, 0]  # Red color for fragmentation areas
    # Blend original image and color mask
    blended = cv2.addWeighted(np.array(image), 0.7, color_mask, 0.3, 0)
    return Image.fromarray(blended)


def main(args):
    path = args.images
    list_images = sorted(glob.glob(os.path.join(path, '*')))
    if len(list_images) == 0:
        list_images = [path]

    print(f"Number of images: {len(list_images)}")
    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    torch._C._jit_set_profiling_mode(False)
    model = torch.jit.load(args.checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    save_path = os.path.join(args.dir_predictions)
    save_color_path = os.path.join(args.dir_predictions+'_color')
    os.makedirs(save_path, exist_ok=True)
    if args.color:
        os.makedirs(save_color_path, exist_ok=True)

    print("Predicting images...")
    for path_image in tqdm.tqdm(list_images):
        mask_pred = predict_fragmentation(model, path_image, args.size)
        fname = os.path.basename(path_image).replace('.jpg','.png')
        Image.fromarray(mask_pred.astype('uint8')).save(os.path.join(save_path, fname))
        if args.color:
            img = Image.open(path_image).convert("RGB")
            color_img = color_inference(img, mask_pred)
            color_img.save(os.path.join(save_color_path, fname))
    print(f"Predicted masks saved in {save_path}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fragmentation marker evaluation")
    
    parser.add_argument("--images", type=str, help="Name of the directory containing the dataset")
    parser.add_argument("--checkpoint", type=str, default="models/markers/fragmentation/fragmentation_hrnet.pth",
                        help="Path to the trained model")
    parser.add_argument("--size", type=int, default=(480, 480),
                        help="Size of the images (they will be resized to size x size)")
    parser.add_argument("--dir_predictions", type=str, default="output/markers/fragmentation/inference",
                        help="Whether to save the predicted masks")
    parser.add_argument("--color", type=bool, default=True, help="Whether to save the fusion of mask and image in color")

    args = parser.parse_args()
    main(args)

