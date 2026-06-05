import cv2
import glob
import os
import sys
import sklearn
import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
from collections import defaultdict, OrderedDict
import torchvision.transforms as transforms
import torchvision

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET


class SegmentationMetrics:
    """
    Standardized metrics for comparing different methods.
    All metrics computed the same way across different model architectures.
    """
    
    @staticmethod
    def accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
        """Pixel-wise accuracy."""
        true_detec = np.logical_not(np.logical_xor(target, prediction))
        return np.sum(true_detec) / (np.sum(np.ones_like(target)) + 1e-10)
    
    @staticmethod
    def precision(target: np.ndarray, prediction: np.ndarray) -> float:
        """Precision (positive predictive value)."""
        intersection = np.logical_and(target, prediction)
        return np.sum(intersection) / (np.sum(prediction) + 1e-10)
    
    @staticmethod
    def recall(target: np.ndarray, prediction: np.ndarray) -> float:
        """Recall (sensitivity, true positive rate)."""
        intersection = np.logical_and(target, prediction)
        return np.sum(intersection) / (np.sum(target) + 1e-10)
    
    @staticmethod
    def specificity(target: np.ndarray, prediction: np.ndarray) -> float:
        """Specificity (true negative rate)."""
        return SegmentationMetrics.recall(1 - target, 1 - prediction)
    
    @staticmethod
    def jaccard(target: np.ndarray, prediction: np.ndarray) -> float:
        """IoU (Jaccard index)."""
        intersection = np.logical_and(target, prediction)
        union = np.logical_or(target, prediction)
        return np.sum(intersection) / (np.sum(union) + 1e-10)
    
    @staticmethod
    def dice(target: np.ndarray, prediction: np.ndarray) -> float:
        """Dice coefficient (F1 score for segmentation)."""
        intersection = np.logical_and(target, prediction)
        return 2 * np.sum(intersection) / (np.sum(target) + np.sum(prediction) + 1e-10)
    
    @staticmethod
    def compute_all(target: np.ndarray, prediction: np.ndarray) -> dict:
        """Compute all metrics for a single image."""
        return {
            'accuracy': SegmentationMetrics.accuracy(target, prediction),
            'precision': SegmentationMetrics.precision(target, prediction),
            'recall': SegmentationMetrics.recall(target, prediction),
            'specificity': SegmentationMetrics.specificity(target, prediction),
            'jaccard': SegmentationMetrics.jaccard(target, prediction),
            'dice': SegmentationMetrics.dice(target, prediction)
        }
    
    @staticmethod
    def aggregate_metrics(metrics_list: list) -> dict:
        """Aggregate metrics across multiple images."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return aggregated

class MultiClassMetrics:
    def accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
        """Overall accuracy."""
        return accuracy_score(target, prediction)
    
    
    def precision(target: np.ndarray, prediction: np.ndarray) -> float:
        """Macro-averaged precision."""
        return precision_score(target, prediction, average='macro', zero_division=0)
    
    def recall(target: np.ndarray, prediction: np.ndarray) -> float:
        """Macro-averaged recall."""
        return recall_score(target, prediction, average='macro', zero_division=0)
    
    def f1(target: np.ndarray, prediction: np.ndarray) -> float:
        """Macro-averaged F1 score."""
        return f1_score(target, prediction, average='macro', zero_division=0)
    
    def confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """Confusion matrix."""
        return confusion_matrix(target, prediction)
    
    def compute_all(target: np.ndarray, prediction: np.ndarray) -> dict:
        """Compute all metrics for a single set of predictions."""
        return {
            'accuracy': MultiClassMetrics.accuracy(target, prediction),
            'precision': MultiClassMetrics.precision(target, prediction),
            'recall': MultiClassMetrics.recall(target, prediction),
            'f1': MultiClassMetrics.f1(target, prediction),
            'confusion_matrix': MultiClassMetrics.confusion_matrix(target, prediction)
        }
    

class FragmentationEvaluator:
    """Evaluator for fragmentation segmentation (binary segmentation)."""
    
    def __init__(self, dataset_path: str, checkpoint_path: str, device: str = 'cuda'):
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        
        # Image transformations
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the model checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        # Handle both TorchScript and regular PyTorch models
        try:
            # Try loading as TorchScript
            torch._C._jit_set_profiling_mode(False)
            self.model = torch.jit.load(self.checkpoint_path)
            self.model = self.model.to(self.device)
            print("Loaded TorchScript model")
        except:
            # Try loading as regular PyTorch checkpoint
            self.model = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(self.model, dict) and 'state_dict' in self.model:
                # Need to initialize model architecture first
                raise ValueError("Regular checkpoint detected. Please provide model architecture.")
            print("Loaded PyTorch model")
        
        self.model.eval()
    
    def predict_mask(self, image_path: Path, target_size: tuple = (480, 480)) -> np.ndarray:
        """Predict fragmentation mask for a single image."""
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Resize for model
        image = transforms.Resize(target_size)(img)
        input_tensor = self.to_tensor_norm(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Post-process
        outputs = torch.argmax(outputs, 1)
        mask = outputs.squeeze(0).cpu().numpy()
        mask[mask == 1] = 255
        
        # Resize back to original size
        mask_resized = np.array(
            transforms.Resize((orig_h, orig_w))(Image.fromarray(mask.astype('uint8')))
        )
        
        return mask_resized
    
    def evaluate(self, split: str = 'test', save_predictions: bool = True) -> dict:
        """Evaluate on dataset split."""
        # Get image and mask paths
        images_dir = self.dataset_path / 'images' / split
        masks_dir = self.dataset_path / 'masks' / split
        
        image_paths = sorted(images_dir.glob('*.jpg')) + sorted(images_dir.glob('*.png'))
        
        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Found {len(image_paths)} images in {split} split")
        
        # Load model
        self.load_model()
        
        # Create output directory
        if save_predictions:
            pred_dir = Path("output/comparison/fragmentation/predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate
        metrics_list = []
        predictions = []
        
        print("Running inference...")
        for img_path in tqdm.tqdm(image_paths):
            # Get corresponding mask path
            mask_path = masks_dir / img_path.name.replace('.jpg', '.png')
            if not mask_path.exists():
                mask_path = masks_dir / img_path.name
            
            if not mask_path.exists():
                print(f"Warning: No mask found for {img_path.name}, skipping")
                continue
            
            # Predict
            pred_mask = self.predict_mask(img_path)
            predictions.append(pred_mask)
            
            # Load ground truth
            gt_mask = np.array(Image.open(mask_path))
            
            # Compute metrics (binary masks)
            pred_binary = (pred_mask == 255).astype(bool)
            gt_binary = (gt_mask == 255).astype(bool)
            
            metrics = SegmentationMetrics.compute_all(gt_binary, pred_binary)
            metrics_list.append(metrics)
            
            # Save prediction
            if save_predictions:
                pred_path = pred_dir / img_path.name.replace('.jpg', '.png')
                Image.fromarray(pred_mask.astype('uint8')).save(pred_path)
        
        # Aggregate metrics
        aggregated = SegmentationMetrics.aggregate_metrics(metrics_list)
        
        # Save results
        results_dir = Path("output/comparison/fragmentation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed metrics
        with open(results_dir / "evaluation_metrics.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("FRAGMENTATION SEGMENTATION EVALUATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Split: {split}\n")
            f.write(f"Number of images: {len(metrics_list)}\n\n")
            
            for metric_name, values in aggregated.items():
                f.write(f"{metric_name.capitalize()}:\n")
                f.write(f"  Mean: {values['mean']:.4f}\n")
                f.write(f"  Std:  {values['std']:.4f}\n")
                f.write(f"  Min:  {values['min']:.4f}\n")
                f.write(f"  Max:  {values['max']:.4f}\n\n")
        
        # Save CSV
        df = pd.DataFrame([{
            f"{k}_mean": v['mean'],
            f"{k}_std": v['std']
        } for k, v in aggregated.items()][0], index=[0])
        df.to_csv(results_dir / "metrics_summary.csv", index=False)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for metric_name, values in aggregated.items():
            print(f"{metric_name.capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")
        print("="*60)
        
        return aggregated


class GradingEvaluator:
    """Evaluator for embryo grading (classification)."""
    
    def __init__(self, dataset_path: str, checkpoint_path: str, task: str = 'exp', device: str = 'cuda', target_size: tuple = (299, 299), model_name: str = 'inception_v3'):
        """
        Args:
            dataset_path: Path to dataset
            checkpoint_path: Path to model checkpoint
            task: Grading task ('exp', 'icm', or 'te')
            device: Device for inference
            target_size: Target size for image resizing
        """

        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = checkpoint_path
        self.task = task
        self.device = device
        self.target_size = target_size
        self.model_name = model_name
        self.model = None
        self.DATA = {'exp': {'train': 'EXP_silver', 'test': 'EXP_gold', 'classes': [0, 1, 2, 3, 4]},
                     'icm': {'train': 'ICM_silver', 'test': 'ICM_gold', 'classes': [0, 1, 2]},
                     'te':  {'train': 'TE_silver',  'test': 'TE_gold',  'classes': [0, 1, 2]},
                     }
        
        # Number of classes per task
        self.num_classes = {'exp': 5, 'icm': 3, 'te': 3}[task]
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _build_model(self) -> torch.nn.Module:
        model = torchvision.models.__dict__[self.model_name](init_weights=False)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
        if hasattr(model, "aux_logits"):
            model.aux_logits = False
            model.AuxLogits = None
        return model

    def load_model(self):
        """Load the model checkpoint."""
        # Handle both TorchScript and regular PyTorch models
        print(f"Loading model from {self.checkpoint_path}")
        try:
            # Try loading as TorchScript
            torch._C._jit_set_profiling_mode(False)
            self.model = torch.jit.load(self.checkpoint_path)
            self.model = self.model.to(self.device)
            print("Loaded TorchScript model")
        except Exception:
            # Try loading as regular PyTorch checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint
                print("Loaded PyTorch model")
            else:
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                if not isinstance(state_dict, (dict, OrderedDict)):
                    raise ValueError("Unsupported checkpoint format for grading model.")

                if any(k.startswith("module.") for k in state_dict.keys()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_state_dict[k.replace("module.", "", 1)] = v
                    state_dict = new_state_dict

                self.model = self._build_model()
                self.model.load_state_dict(state_dict)
                print("Loaded PyTorch state_dict")

            self.model = self.model.to(self.device)

        self.model.eval()
    
    def predict_lbl(self, image_path: Path) -> int:
        """Predict grading label for a single image."""
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        image = transforms.Resize(self.target_size)(img)
        input_tensor = self.to_tensor_norm(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        pred_label = torch.argmax(outputs, 1).item()
        return pred_label
    
    def evaluate(self, split: str = 'test', save_predictions: bool = True) -> dict:
        """Evaluate grading performance."""

        images_dir = self.dataset_path / 'Images' 
        lbl_path = self.dataset_path / "Gardner_test_gold_onlyGardnerScores.csv"
        csv_file = pd.read_csv(lbl_path,  sep=";")
        image_paths = []
        for img_name in csv_file['Image']:
            image_paths.append(images_dir / img_name)  

        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Found {len(image_paths)} images in {split} split")
        
        # Load model
        self.load_model()
        
        # Create output directory
        if save_predictions:
            pred_dir = Path("output/comparison/grading/predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate
        images = []
        labels = []
        predictions = []
        
        print("Running inference...")
        for img_path in tqdm.tqdm(image_paths):
            # Get corresponding mask path
            
            str_value = csv_file.loc[csv_file["Image"] == img_path.name,self.DATA[self.task][split]].values[0]
            if  str_value!= 'ND' and not pd.isna(str_value):
                images.append(img_path.name)
                labels.append(int(str_value))
                # Predict
                pred_label = self.predict_lbl(img_path)
                predictions.append(pred_label)
                
            else:
                print(f"Warning: No label found for {img_path.name}, skipping")
                continue
           
        # Compute metrics (multi-class classification)
        metrics = MultiClassMetrics.compute_all(labels, predictions)
            
            # Save prediction
        if save_predictions:
            pred_path = pred_dir / f'predictions_{self.task}.csv'
            df = pd.DataFrame({'image': images, 'label': labels, 'prediction': predictions})
            df.to_csv(pred_path, index=False)
        
        results_dir = Path("output/comparison/grading")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed metrics
        with open(results_dir / f"evaluation_metrics_{self.task}.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write(f"GRADING {self.task} EVALUATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Split: {split}\n")
            f.write(f"Number of images: {len(images)}\n\n")
            
            for metric_name, value in metrics.items():
                f.write(f"{metric_name.capitalize()}:{value}\n")
        f.close()



class SegmentationEvaluator:
    """Evaluator for blastocyst segmentation (multi-class segmentation)."""
    
    def __init__(self, dataset_path: str, checkpoint_path: str, device: str = 'cuda'):
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.DATA = {'ZP':75, 'ICM': 150, 'TE': 255}
        
        # Image transformations
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the model checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        # Handle both TorchScript and regular PyTorch models
        try:
            # Try loading as TorchScript
            torch._C._jit_set_profiling_mode(False)
            self.model = torch.jit.load(self.checkpoint_path)
            self.model = self.model.to(self.device)
            print("Loaded TorchScript model")
        except:
            # Try loading as regular PyTorch checkpoint
            self.model = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(self.model, dict) and 'state_dict' in self.model:
                # Need to initialize model architecture first
                raise ValueError("Regular checkpoint detected. Please provide model architecture.")
            print("Loaded PyTorch model")
        
        self.model.eval()
    
    def predict_mask(self, image_path: Path, target_size: tuple = (480, 480)) -> np.ndarray:
        """Predict segmentation mask for a single image."""
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Resize for model
        image = transforms.Resize(target_size)(img)
        input_tensor = self.to_tensor_norm(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Post-process
        outputs = torch.argmax(outputs, 1)
        mask = outputs.squeeze(0).cpu().numpy()

        mask[mask == 1] = 75
        mask[mask == 2] = 255
        mask[mask == 3] = 150
        
        # Resize back to original size
        mask_resized = np.array(
            transforms.Resize((orig_h, orig_w))(Image.fromarray(mask.astype('uint8')))
        )
        return mask_resized
    
    def evaluate(self, split: str = 'test', save_predictions: bool = True) -> dict:
        """Evaluate on dataset split."""
        # Get image and mask paths
        images_dir = self.dataset_path / split
        masks_dir = self.dataset_path / 'GT_Mask' 

        image_paths = sorted(images_dir.glob('*.BMP')) 
        if not image_paths:
            raise ValueError(f"No images found in {images_dir}")
        print(f"Found {len(image_paths)} images in {split} split")
        # Load model
        self.load_model()
        # Create output directory
        results_dir = Path("output/comparison/segmentation")
        results_dir.mkdir(parents=True, exist_ok=True)  

        if save_predictions:
            pred_dir = Path("output/comparison/segmentation/predictions")
            pred_dir.mkdir(parents=True, exist_ok=True)
        # Evaluate
        metrics_list = {'ZP': [], 'TE': [], 'ICM': []}
        predictions = []
        print("Running inference...")
        
        for img_path in tqdm.tqdm(image_paths):
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                print(f"Warning: No mask found for {img_path.name}, skipping")
                continue
            pred_mask = self.predict_mask(img_path)
            predictions.append(pred_mask)

            # Load ground truth
            gt_mask = np.array(Image.open(mask_path))
            
            # Compute metrics (binary masks)
            for struct in ['ZP', 'TE', 'ICM']:
                pred_binary = (pred_mask == self.DATA[struct]).astype(bool)
                gt_binary = (gt_mask == self.DATA[struct]).astype(bool)
                metrics = SegmentationMetrics.compute_all(gt_binary, pred_binary)
                metrics_list[struct].append(metrics)
        
            # Save prediction
            if save_predictions:   
                pred_path = pred_dir / img_path.name.replace('.BMP', '.png')
                Image.fromarray(pred_mask.astype('uint8')).save(pred_path)

        # Aggregate metrics
        # Aggregate metrics for each structure
        aggregated = {}
        for struct in ['ZP', 'TE', 'ICM']:
            aggregated[struct] = SegmentationMetrics.aggregate_metrics(metrics_list[struct])
        
        # Save results
        results_dir = Path("output/comparison/segmentation")
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "evaluation_metrics.txt", "w") as f:
            f.write("="*60 + "\n")
            f.write("BLASTOCYST SEGMENTATION EVALUATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Dataset: {self.dataset_path}\n")
            f.write(f"Split: {split}\n")
            f.write(f"Number of images: {len(image_paths)}\n\n")
            
            for struct in aggregated.keys():
                f.write(f"Structure: {struct}\n")
                for metric_name, values in aggregated[struct].items():
                    f.write(f"{metric_name.capitalize()}:\n")
                    f.write(f"  Mean: {values['mean']:.4f}\n")
                    f.write(f"  Std:  {values['std']:.4f}\n")
                    f.write(f"  Min:  {values['min']:.4f}\n")
                    f.write(f"  Max:  {values['max']:.4f}\n\n")
        
        # Save CSV
        df = pd.DataFrame([{
            f"{struct}_{k}_mean": v['mean'],
            f"{struct}_{k}_std": v['std']
        } for struct, metrics in aggregated.items() for k, v in metrics.items()][0], index=[0])
        df.to_csv(results_dir / "metrics_summary.csv", index=False)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for struct, metrics in aggregated.items():
            print(f"{struct.capitalize()}:")
            for metric_name, values in metrics.items():
                print(f"  {metric_name.capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")
        print("="*60)
        
        return aggregated
            


def main(args):
    """Main comparison evaluation routine."""
    
    if args.task == 'fragmentation':
        evaluator = FragmentationEvaluator(
            dataset_path=os.path.join(PATH_DATASET, args.dataset),
            checkpoint_path=args.checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        results = evaluator.evaluate(
            split=args.split,
            save_predictions=args.save_predictions
        )
    
    elif args.task == 'grading':
        # Grading evaluation
        evaluator = GradingEvaluator(
            dataset_path=os.path.join(PATH_DATASET, args.dataset),
            checkpoint_path=args.checkpoint,
            task=args.grading_task,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model_name=args.grading_model_name
        )
        results = evaluator.evaluate(
            split=args.split,
            save_predictions=args.save_predictions
        )
    
    elif args.task == 'segmentation':
        evaluator = SegmentationEvaluator(
            dataset_path=os.path.join(PATH_DATASET, args.dataset),
            checkpoint_path=args.checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        results = evaluator.evaluate(
            split=args.split,
            save_predictions=args.save_predictions
        )
    
    else:   
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified evaluation for method comparison")
    
    # Task selection
    parser.add_argument("--task",type=str, choices=['fragmentation', 'grading', 'segmentation'],required=True,help="Task to evaluate")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, help="Name of the directory containing the dataset")
    parser.add_argument("--split", type=str, default='test', choices=['train', 'val', 'test'], help="Dataset split to evaluate")
    
    # Model
    parser.add_argument("--checkpoint",type=str,required=True, help="Path to the trained model checkpoint")
    
    # Task-specific arguments
    parser.add_argument("--grading_task",type=str,choices=['exp', 'icm', 'te'],help="Grading task (required if task=grading)")
    parser.add_argument("--grading_model_name", type=str, default="inception_v3",
                        help="Grading model architecture (default: inception_v3)")
    
    # Output
    parser.add_argument("--save_predictions",action="store_true",help="Whether to save predicted masks/labels")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task == 'grading' and args.grading_task is None:
        parser.error("--grading_task is required when --task=grading")
    
    main(args)