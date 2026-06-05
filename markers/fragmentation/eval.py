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
get_y_fn = lambda x: Path(str(x).replace("images","masks").replace("jpg","png"))

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


def accuracy(target, prediction):
    true_detec = np.logical_not(np.logical_xor(target, prediction))
    return np.sum(true_detec)/(np.sum(np.ones_like(target))+1)

def precision(target, prediction):
    intersection = np.logical_and(target, prediction)
    return np.sum(intersection)/(np.sum(prediction)+1)

def recall(target, prediction):
    intersection = np.logical_and(target, prediction)
    return np.sum(intersection)/(np.sum(target)+1)

def jaccard(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    return np.sum(intersection) / (np.sum(union)+1)

def dice(target, prediction):
    intersection = np.logical_and(target, prediction)
    return 2*np.sum(intersection) / (np.sum(target) + np.sum(prediction)+1)

def metrics(target, prediction):
    return {'accuracy': accuracy(target, prediction),
            'precision': precision(target, prediction),
            'recall': recall(target, prediction),
            'specificity': recall(1-target,1- prediction),
            'jaccard':jaccard(target, prediction),
            'dice': dice(target, prediction)}

def summary_metrics(y_test,predictions):
    a,p,r,s,j,d=0.,0.,0.,0.,0.,0.
    n=len(predictions)
    for i in range(n):
        preds= predictions[i]==255
        gt= y_test[i]==255
        metricas=metrics(gt,preds)
        a+=metricas['accuracy']
        p+=metricas['precision']
        r+=metricas['recall']
        s+=metricas['specificity']
        j+=metricas['jaccard']
        d+=metricas['dice']
    return {'accuracy': a/n,
            'precision': p/n,
            'recall': r/n,
            'specificity': s/n,
            'jaccard':j/n,
            'dice': d/n}

def main(args):
    path = os.path.join(PATH_DATASET, args.dataset)
    path_images = os.path.join(path, "images")
    path_images_test = os.path.join(path_images, "val")
    list_images = sorted(glob.glob(os.path.join(path_images_test, '*')))
    list_masks = [get_y_fn(x) for x in list_images]
    print(f"Number of test images: {len(list_images)}")
    #load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    torch._C._jit_set_profiling_mode(False)
    model = torch.jit.load(args.checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    if args.save_predictions:
        save_path = os.path.join("output/markers/fragmentation/predictions/")
        os.makedirs(save_path, exist_ok=True)
    preds = []
    print("Predicting test images...")
    for path_image in tqdm.tqdm(list_images):
        mask_pred = predict_fragmentation(model, path_image, args.size)
        preds.append(mask_pred)
        if args.save_predictions:
            fname = os.path.basename(path_image).replace('.jpg','.png')
            Image.fromarray(mask_pred.astype('uint8')).save(os.path.join(save_path, fname))
    
    masks = [np.array(Image.open(x)) for x in list_masks]
    print("Evaluation metrics:")
    metricas = summary_metrics(masks, preds)
    #write in a file the results
    with open(os.path.join("output/markers/fragmentation/evaluation_metrics.txt"), "w") as f:
        for k,v in metricas.items():
            print(f"{k}: {v:.4f}")
            f.write(f"{k}: {v:.4f}\n")
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fragmentation marker evaluation")
    
    parser.add_argument("--dataset", type=str, default="CleavageEmbryo",
                        help="Name of the directory containing the dataset")
    parser.add_argument("--checkpoint", type=str, default="models/markers/fragmentation/fragmentation_hrnet.pth",
                        help="Path to the trained model")
    parser.add_argument("--size", type=int, default=(480, 480),
                        help="Size of the images (they will be resized to size x size)")
    parser.add_argument("--save_predictions", type=bool, default=True,
                        help="Whether to save the predicted masks")

    args = parser.parse_args()
    main(args)

