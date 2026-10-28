import os
import csv
import argparse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from glob import glob

DATA = {'EXP': {'classes': [0, 1, 2, 3, 4], 'class_map':{0:1,1:2,2:3,3:4,4:5} },
        'ICM': {'classes': [0,1,2], 'class_map':{0:'A',1:'B',2:'C'} },
        'TE':  {'classes': [0,1,2], 'class_map':{0:'A',1:'B',2:'C'} },
        'STAGE':  {'classes': [i for i in range(7)], 'class_map':{0: "empty", 1:"1cell", 2:"2cell",
                                                                  3:"+3cell", 4:"+8cell", 5:"morula",
                                                                  6:"blastocyst" } }}


def get_transforms():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_image_paths(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.tif',  '*.bmp', '*.BMP')
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(path, ext)))
        return files
    else:
        raise ValueError(f"Invalid path: {path}")


def load_model(checkpoint_path, num_classes, device, model_name='inception_v3'):
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(loaded, nn.Module):
        # checkpoint stored the whole model, not a state_dict
        model = loaded
    else:
        model = torchvision.models.__dict__[model_name](init_weights=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
        model.AuxLogits = None
        model.load_state_dict(loaded)

    model.to(device)
    model.eval()
    return model


def run_inference(model, image_paths, device):
    results = []
    transform = get_transforms()

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        results.append([
            os.path.basename(img_path),
            DATA[args.task]['class_map'][pred],
            *probs.squeeze().cpu().tolist()
        ])

    return results

def run(image, checkpoint, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, num_classes=len(DATA[task]['classes']),device=device)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    transform = get_transforms()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return DATA[task]['class_map'][pred]


def save_csv(results, output_path):
    with open(output_path, 'w', newline='') as f:
        columns = ['filename', 'prediction'] + [f'prob_{cls}' for cls in DATA[args.task]['class_map'].values()]
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(results)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_classes=len(DATA[args.task]['classes']),
        device=device
    )

    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} image(s) to process")
    results = run_inference(model, image_paths, device)
    save_csv(results, args.output)

    print("Inference completed.")

    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task",type=str,
                        help="Task to be graded. Select between EXP, ICM, TE")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to image or directory")
    parser.add_argument("--model_name", type=str, default="inception_v3",
                        help="Model architecture to use")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model.pth")
    parser.add_argument("--output", type=str, default="output/markers/blastocyst_grading/predictions.csv",
                        help="Output CSV file")

    args = parser.parse_args()
    main(args)





