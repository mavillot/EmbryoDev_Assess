import torch
import sys
import cv2
import tqdm
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from models.multi_task import MultiTaskModel
from models.encoder.visual_encoder import ResNetEncoder
from models.temporal.temporal_transformer import VideoTransformer
from data.dataset import MultiTaskDataset, multitask_collate, MultiTaskBatchSampler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET
from assessment.MultiTask.data.managment import (
    dataset_grading,
    dataset_ploidy,
    dataset_seg_blastocyst,
    dataset_seg_fragmentation
)


class MetricsTracker:
    """Track and compute various metrics during evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        # Classification metrics
        self.exp_labels = []
        self.exp_preds = []
        self.icm_labels = []
        self.icm_preds = []
        self.te_labels = []
        self.te_preds = []
        self.ploidy_labels = []
        self.ploidy_preds = []
        
        # Segmentation metrics
        self.frag_iou = []
        self.frag_dice = []
        self.frag_accuracy = []
        self.frag_precision = []
        self.frag_recall = []
        
        self.seg_metrics = {
            'dice': defaultdict(list),
            'iou': defaultdict(list),
            'accuracy': defaultdict(list),
            'precision': defaultdict(list),
            'recall': defaultdict(list)
        }
        
        # Counters
        self.total_images = 0
        self.total_videos = 0
    
    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
        """
        Compute Dice score for segmentation.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            eps: Epsilon for numerical stability
            
        Returns:
            Dice score
        """
        pred = pred.float()
        target = target.float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return ((2 * intersection + eps) / (union + eps)).mean().item()
    
    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
        """
        Compute IoU score for segmentation.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            eps: Epsilon for numerical stability
            
        Returns:
            IoU score
        """
        intersection = (pred & target).sum(dim=(1, 2, 3))
        union = (pred | target).sum(dim=(1, 2, 3))
        return ((intersection + eps) / (union + eps)).mean().item()
    
    @staticmethod
    def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute pixel-wise accuracy for segmentation.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            
        Returns:
            Pixel accuracy
        """
        pred = pred.bool()
        target = target.bool()
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def precision_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
        """
        Compute precision for segmentation.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            eps: Epsilon for numerical stability
            
        Returns:
            Precision score
        """
        pred = pred.bool()
        target = target.bool()
        true_positive = (pred & target).sum(dim=(1, 2, 3))
        predicted_positive = pred.sum(dim=(1, 2, 3))
        return ((true_positive + eps) / (predicted_positive + eps)).mean().item()
    
    @staticmethod
    def recall_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
        """
        Compute recall (sensitivity) for segmentation.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            eps: Epsilon for numerical stability
            
        Returns:
            Recall score
        """
        pred = pred.bool()
        target = target.bool()
        true_positive = (pred & target).sum(dim=(1, 2, 3))
        actual_positive = target.sum(dim=(1, 2, 3))
        return ((true_positive + eps) / (actual_positive + eps)).mean().item()
    
    def update_classification(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ):
        """Update classification metrics from batch."""
        # EXP classification
        if batch.get('exp') is not None and 'exp' in outputs:
            pred = outputs['exp'].argmax(dim=1)
            self.exp_labels.extend(batch['exp'].cpu().tolist())
            self.exp_preds.extend(pred.cpu().tolist())
        
        # ICM classification
        if batch.get('icm') is not None and 'icm' in outputs:
            pred = outputs['icm'].argmax(dim=1)
            self.icm_labels.extend(batch['icm'].cpu().tolist())
            self.icm_preds.extend(pred.cpu().tolist())
        
        # TE classification
        if batch.get('te') is not None and 'te' in outputs:
            pred = outputs['te'].argmax(dim=1)
            self.te_labels.extend(batch['te'].cpu().tolist())
            self.te_preds.extend(pred.cpu().tolist())
        
        # Ploidy classification
        if batch.get('ploidy') is not None and 'ploidy' in outputs:
            logits = outputs['ploidy']
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().view(-1)
            self.ploidy_labels.extend(batch['ploidy'].cpu().tolist())
            self.ploidy_preds.extend(preds.cpu().tolist())
    
    def update_segmentation(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ):
        """Update segmentation metrics from batch."""
        # Fragmentation segmentation (binary)
        if batch.get('mask_frag') is not None and 'fragmentation' in outputs:
            logits = outputs['fragmentation']
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5)
            target = batch['mask_frag'].unsqueeze(1)
            
            self.frag_iou.append(self.iou_score(preds, target))
            self.frag_dice.append(self.dice_score(preds, target))
            self.frag_accuracy.append(self.pixel_accuracy(preds, target))
            self.frag_precision.append(self.precision_score(preds, target))
            self.frag_recall.append(self.recall_score(preds, target))
        
        # Blastocyst structure segmentation (multiclass)
        if batch.get('mask_str') is not None and 'blasto_str' in outputs:
            logits = outputs['blasto_str']
            preds = logits.argmax(dim=1)
            target = batch['mask_str']
            
            # Compute metrics for each class
            for c in range(logits.shape[1]):
                pred_c = (preds == c).unsqueeze(1)
                target_c = (target == c).unsqueeze(1)
                
                self.seg_metrics['dice'][c].append(self.dice_score(pred_c, target_c))
                self.seg_metrics['iou'][c].append(self.iou_score(pred_c, target_c))
                self.seg_metrics['accuracy'][c].append(self.pixel_accuracy(pred_c, target_c))
                self.seg_metrics['precision'][c].append(self.precision_score(pred_c, target_c))
                self.seg_metrics['recall'][c].append(self.recall_score(pred_c, target_c))
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {
            'total_images': self.total_images,
            'total_videos': self.total_videos
        }
        
        # Helper function to compute per-class metrics
        def compute_metrics(labels, preds, num_classes):
            metrics = {}
            for c in range(num_classes):
                # Binary classification for each class (one-vs-rest)
                y_true = np.array([1 if label == c else 0 for label in labels])
                y_pred = np.array([1 if pred == c else 0 for pred in preds])
                
                # True Positives, False Positives, True Negatives, False Negatives
                tp = np.sum((y_true == 1) & (y_pred == 1))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                
                # Precision = TP / (TP + FP)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                
                # Recall (Sensitivity) = TP / (TP + FN)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # Specificity = TN / (TN + FP)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                metrics[f'class_{c}'] = {
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity
                }
            
            # Compute macro-averaged metrics
            precisions = [metrics[f'class_{c}']['precision'] for c in range(num_classes)]
            recalls = [metrics[f'class_{c}']['recall'] for c in range(num_classes)]
            specificities = [metrics[f'class_{c}']['specificity'] for c in range(num_classes)]
            
            metrics['macro_avg'] = {
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'specificity': np.mean(specificities)
            }
            
            return metrics
        
        # Classification summaries with detailed metrics
        if self.exp_labels:
            summary['exp_accuracy'] = accuracy_score(self.exp_labels, self.exp_preds)
            summary['exp_f1'] = f1_score(self.exp_labels, self.exp_preds, average='weighted', zero_division=0)
            exp_metrics = compute_metrics(self.exp_labels, self.exp_preds, num_classes=5)
            summary['exp_metrics'] = exp_metrics
        
        if self.icm_labels:
            summary['icm_accuracy'] = accuracy_score(self.icm_labels, self.icm_preds)
            summary['icm_f1'] = f1_score(self.icm_labels, self.icm_preds, average='weighted', zero_division=0)
            icm_metrics = compute_metrics(self.icm_labels, self.icm_preds, num_classes=3)
            summary['icm_metrics'] = icm_metrics
        
        if self.te_labels:
            summary['te_accuracy'] = accuracy_score(self.te_labels, self.te_preds)
            summary['te_f1'] = f1_score(self.te_labels, self.te_preds, average='weighted', zero_division=0)
            te_metrics = compute_metrics(self.te_labels, self.te_preds, num_classes=3)
            summary['te_metrics'] = te_metrics
        
        if self.ploidy_labels:
            summary['ploidy_accuracy'] = accuracy_score(self.ploidy_labels, self.ploidy_preds)
            summary['ploidy_f1'] = f1_score(self.ploidy_labels, self.ploidy_preds, average='binary', zero_division=0)
            ploidy_metrics = compute_metrics(self.ploidy_labels, self.ploidy_preds, num_classes=2)
            summary['ploidy_metrics'] = ploidy_metrics
        
        # Segmentation summaries
        if self.frag_dice:
            summary['frag_dice'] = np.mean(self.frag_dice)
            summary['frag_iou'] = np.mean(self.frag_iou)
            summary['frag_accuracy'] = np.mean(self.frag_accuracy)
            summary['frag_precision'] = np.mean(self.frag_precision)
            summary['frag_recall'] = np.mean(self.frag_recall)
        
        if self.seg_metrics['dice']:
            for c in self.seg_metrics['dice'].keys():
                if self.seg_metrics['dice'][c]:
                    summary[f'seg_class_{c}_dice'] = np.mean(self.seg_metrics['dice'][c])
                    summary[f'seg_class_{c}_iou'] = np.mean(self.seg_metrics['iou'][c])
                    summary[f'seg_class_{c}_accuracy'] = np.mean(self.seg_metrics['accuracy'][c])
                    summary[f'seg_class_{c}_precision'] = np.mean(self.seg_metrics['precision'][c])
                    summary[f'seg_class_{c}_recall'] = np.mean(self.seg_metrics['recall'][c])
        
        return summary


class MultiTaskEvaluator:
    """
    Evaluator for multi-task model supporting both single and multi-GPU.
    """
    
    def __init__(
        self,
        model: nn.Module,
        checkpoint_path: str,
        device: str = 'cuda',
        multi_gpu: bool = False,
        use_transformer: bool = True,
        output_dir: str = "output/assessment/MultiTask/evaluation"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model architecture (uninitialized)
            checkpoint_path: Path to model checkpoint
            device: Device to run evaluation on
            multi_gpu: Whether to use multiple GPUs
            use_transformer: Whether to enable transformer for video processing
            output_dir: Directory to save evaluation results
        """
        self.base_model = model
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        self.use_transformer = use_transformer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = MetricsTracker()
        
        # Load model
        self._load_model()
        
        print(f"Evaluator initialized:")
        print(f"  Device: {self.device}")
        print(f"  Multi-GPU: {self.multi_gpu}")
        if self.multi_gpu:
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Transformer enabled: {self.use_transformer}")
        print(f"  Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle DataParallel checkpoint (remove 'module.' prefix)
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            checkpoint = new_state_dict
        
        self.base_model.load_state_dict(checkpoint)
        self.base_model.to(self.device)
        
        # Wrap with DataParallel if using multi-GPU
        if self.multi_gpu:
            self.model = nn.DataParallel(self.base_model)
            print(f"Model wrapped with DataParallel")
        else:
            self.model = self.base_model
        
        self.model.eval()
        print(f"Model loaded from {self.checkpoint_path}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        return batch
    
    @staticmethod
    def save_confusion_matrix(
        cm: np.ndarray,
        filepath: Path,
        title: str = 'Confusion Matrix',
        labels: Optional[List[str]] = None
    ):
        """Save confusion matrix as image."""
        plt.figure(figsize=(10, 8))
        
        if labels:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {filepath}")
    
    @torch.no_grad()
    def evaluate_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Model outputs
        """
        # Move to device
        batch = self._move_batch_to_device(batch)
        
        # Forward pass
        outputs, _ = self.model(
            images=batch.get('images'),
            videos=batch.get('videos'),
            exp_labels=None,
            icm_labels=None,
            te_labels=None,
            blasto_str_labels=None,
            fragmentation_labels=None,
            ploidy_labels=None
        )
        
        return outputs
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Run full evaluation on dataset.
        
        Args:
            dataloader: DataLoader for evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nStarting evaluation...")
        self.metrics.reset()
        
        # Evaluate all batches
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            # Count samples
            if batch.get('images') is not None:
                self.metrics.total_images += batch['images'].shape[0]
            if batch.get('videos') is not None:
                self.metrics.total_videos += batch['videos'].shape[0]
            
            # Get predictions
            outputs = self.evaluate_batch(batch)
            
            # Update metrics
            self.metrics.update_classification(outputs, batch)
            self.metrics.update_segmentation(outputs, batch)
        
        # Generate reports
        self._generate_reports()
        
        # Get summary
        summary = self.metrics.get_summary()
        
        print("\nEvaluation complete!")
        print(f"Total images: {summary['total_images']}")
        print(f"Total videos: {summary['total_videos']}")
        
        return summary
    
    def _generate_reports(self):
        """Generate detailed evaluation reports."""
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MULTI-TASK MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total images evaluated: {self.metrics.total_images}\n")
            f.write(f"Total videos evaluated: {self.metrics.total_videos}\n\n")
            
            # EXP Classification
            if self.metrics.exp_labels:
                f.write("="*60 + "\n")
                f.write("EXP CLASSIFICATION\n")
                f.write("="*60 + "\n")
                
                labels = sorted(list(set(self.metrics.exp_labels)))
                cm = confusion_matrix(self.metrics.exp_labels, self.metrics.exp_preds, labels=labels)
                self.save_confusion_matrix(
                    cm, 
                    self.output_dir / 'confusion_matrix_EXP.png',
                    title='EXP Confusion Matrix',
                    labels=[f'Class {i}' for i in labels]
                )
                
                report = classification_report(
                    self.metrics.exp_labels, 
                    self.metrics.exp_preds, 
                    labels=labels,
                    zero_division=0
                )
                f.write(report)
                f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'exp_metrics' in summary:
                    f.write("\nPer-Class Metrics:\n")
                    f.write("-" * 60 + "\n")
                    for i in range(5):
                        if f'class_{i}' in summary['exp_metrics']:
                            metrics = summary['exp_metrics'][f'class_{i}']
                            f.write(f"Class {i}:\n")
                            f.write(f"  Precision:   {metrics['precision']:.4f}\n")
                            f.write(f"  Recall:      {metrics['recall']:.4f}\n")
                            f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                    
                    macro = summary['exp_metrics']['macro_avg']
                    f.write(f"\nMacro Average:\n")
                    f.write(f"  Precision:   {macro['precision']:.4f}\n")
                    f.write(f"  Recall:      {macro['recall']:.4f}\n")
                    f.write(f"  Specificity: {macro['specificity']:.4f}\n")
                f.write("\n\n")
            
            # ICM Classification
            if self.metrics.icm_labels:
                f.write("="*60 + "\n")
                f.write("ICM CLASSIFICATION\n")
                f.write("="*60 + "\n")
                
                labels = sorted(list(set(self.metrics.icm_labels)))
                cm = confusion_matrix(self.metrics.icm_labels, self.metrics.icm_preds, labels=labels)
                self.save_confusion_matrix(
                    cm,
                    self.output_dir / 'confusion_matrix_ICM.png',
                    title='ICM Confusion Matrix',
                    labels=[f'Class {i}' for i in labels]
                )
                
                report = classification_report(
                    self.metrics.icm_labels,
                    self.metrics.icm_preds,
                    labels=labels,
                    zero_division=0
                )
                f.write(report)
                f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'icm_metrics' in summary:
                    f.write("\nPer-Class Metrics:\n")
                    f.write("-" * 60 + "\n")
                    for i in range(3):
                        if f'class_{i}' in summary['icm_metrics']:
                            metrics = summary['icm_metrics'][f'class_{i}']
                            f.write(f"Class {i}:\n")
                            f.write(f"  Precision:   {metrics['precision']:.4f}\n")
                            f.write(f"  Recall:      {metrics['recall']:.4f}\n")
                            f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                    
                    macro = summary['icm_metrics']['macro_avg']
                    f.write(f"\nMacro Average:\n")
                    f.write(f"  Precision:   {macro['precision']:.4f}\n")
                    f.write(f"  Recall:      {macro['recall']:.4f}\n")
                    f.write(f"  Specificity: {macro['specificity']:.4f}\n")
                f.write("\n\n")
            
            # TE Classification
            if self.metrics.te_labels:
                f.write("="*60 + "\n")
                f.write("TE CLASSIFICATION\n")
                f.write("="*60 + "\n")
                
                labels = sorted(list(set(self.metrics.te_labels)))
                cm = confusion_matrix(self.metrics.te_labels, self.metrics.te_preds, labels=labels)
                self.save_confusion_matrix(
                    cm,
                    self.output_dir / 'confusion_matrix_TE.png',
                    title='TE Confusion Matrix',
                    labels=[f'Class {i}' for i in labels]
                )
                
                report = classification_report(
                    self.metrics.te_labels,
                    self.metrics.te_preds,
                    labels=labels,
                    zero_division=0
                )
                f.write(report)
                f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'te_metrics' in summary:
                    f.write("\nPer-Class Metrics:\n")
                    f.write("-" * 60 + "\n")
                    for i in range(3):
                        if f'class_{i}' in summary['te_metrics']:
                            metrics = summary['te_metrics'][f'class_{i}']
                            f.write(f"Class {i}:\n")
                            f.write(f"  Precision:   {metrics['precision']:.4f}\n")
                            f.write(f"  Recall:      {metrics['recall']:.4f}\n")
                            f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                    
                    macro = summary['te_metrics']['macro_avg']
                    f.write(f"\nMacro Average:\n")
                    f.write(f"  Precision:   {macro['precision']:.4f}\n")
                    f.write(f"  Recall:      {macro['recall']:.4f}\n")
                    f.write(f"  Specificity: {macro['specificity']:.4f}\n")
                f.write("\n\n")
            
            # Ploidy Classification
            if self.metrics.ploidy_labels:
                f.write("="*60 + "\n")
                f.write("PLOIDY CLASSIFICATION\n")
                f.write("="*60 + "\n")
                
                labels = [0, 1]
                cm = confusion_matrix(self.metrics.ploidy_labels, self.metrics.ploidy_preds, labels=labels)
                self.save_confusion_matrix(
                    cm,
                    self.output_dir / 'confusion_matrix_Ploidy.png',
                    title='Ploidy Confusion Matrix',
                    labels=['Euploid', 'Aneuploid']
                )
                
                report = classification_report(
                    self.metrics.ploidy_labels,
                    self.metrics.ploidy_preds,
                    labels=labels,
                    target_names=['Euploid', 'Aneuploid'],
                    zero_division=0
                )
                f.write(report)
                f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'ploidy_metrics' in summary:
                    f.write("\nPer-Class Metrics:\n")
                    f.write("-" * 60 + "\n")
                    class_names = ['Euploid', 'Aneuploid']
                    for i in range(2):
                        if f'class_{i}' in summary['ploidy_metrics']:
                            metrics = summary['ploidy_metrics'][f'class_{i}']
                            f.write(f"{class_names[i]}:\n")
                            f.write(f"  Precision:   {metrics['precision']:.4f}\n")
                            f.write(f"  Recall:      {metrics['recall']:.4f}\n")
                            f.write(f"  Specificity: {metrics['specificity']:.4f}\n")
                    
                    macro = summary['ploidy_metrics']['macro_avg']
                    f.write(f"\nMacro Average:\n")
                    f.write(f"  Precision:   {macro['precision']:.4f}\n")
                    f.write(f"  Recall:      {macro['recall']:.4f}\n")
                    f.write(f"  Specificity: {macro['specificity']:.4f}\n")
                f.write("\n\n")
            
            # Fragmentation Segmentation
            if self.metrics.frag_iou:
                f.write("="*60 + "\n")
                f.write("FRAGMENTATION SEGMENTATION\n")
                f.write("="*60 + "\n")
                f.write(f"Dice Score:      {np.mean(self.metrics.frag_dice):.4f} ± {np.std(self.metrics.frag_dice):.4f}\n")
                f.write(f"IoU Score:       {np.mean(self.metrics.frag_iou):.4f} ± {np.std(self.metrics.frag_iou):.4f}\n")
                f.write(f"Pixel Accuracy:  {np.mean(self.metrics.frag_accuracy):.4f} ± {np.std(self.metrics.frag_accuracy):.4f}\n")
                f.write(f"Precision:       {np.mean(self.metrics.frag_precision):.4f} ± {np.std(self.metrics.frag_precision):.4f}\n")
                f.write(f"Recall:          {np.mean(self.metrics.frag_recall):.4f} ± {np.std(self.metrics.frag_recall):.4f}\n")
                f.write("\n\n")
            
            # Blastocyst Structure Segmentation
            if self.metrics.seg_metrics['dice']:
                f.write("="*60 + "\n")
                f.write("BLASTOCYST STRUCTURE SEGMENTATION\n")
                f.write("="*60 + "\n")
                
                class_names = {0: 'Background', 1: 'Zona Pellucida (ZP)', 2: 'Inner Cell Mass (ICM)', 3: 'Trophectoderm (TE)'}
                
                for c in sorted(self.metrics.seg_metrics['dice'].keys()):
                    if self.metrics.seg_metrics['dice'][c]:
                        dice_scores = self.metrics.seg_metrics['dice'][c]
                        iou_scores = self.metrics.seg_metrics['iou'][c]
                        accuracy_scores = self.metrics.seg_metrics['accuracy'][c]
                        precision_scores = self.metrics.seg_metrics['precision'][c]
                        recall_scores = self.metrics.seg_metrics['recall'][c]
                        
                        f.write(f"{class_names.get(c, f'Class {c}')}:\n")
                        f.write(f"  Dice Score:     {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}\n")
                        f.write(f"  IoU Score:      {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}\n")
                        f.write(f"  Pixel Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}\n")
                        f.write(f"  Precision:      {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}\n")
                        f.write(f"  Recall:         {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}\n\n")
        
        print(f"Detailed report saved to {report_path}")
        
        # Save metrics summary as CSV
        summary = self.metrics.get_summary()
        summary_df = pd.DataFrame([summary])
        summary_path = self.output_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Metrics summary saved to {summary_path}")
        
        # Export standardized comparison metrics (same format as single-task methods)
        self._export_comparison_metrics()
    
    def _export_comparison_metrics(self):
        """Export metrics in standardized format for method comparison."""
        comparison_dir = Path("output/comparison/multitask")
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Fragmentation comparison
        if self.metrics.frag_dice:
            frag_metrics = {
                'accuracy_mean': np.mean(self.metrics.frag_accuracy),
                'accuracy_std': np.std(self.metrics.frag_accuracy),
                'precision_mean': np.mean(self.metrics.frag_precision),
                'precision_std': np.std(self.metrics.frag_precision),
                'recall_mean': np.mean(self.metrics.frag_recall),
                'recall_std': np.std(self.metrics.frag_recall),
                'specificity_mean': np.mean([1 - r for r in self.metrics.frag_recall]),  # Approximation
                'specificity_std': np.std([1 - r for r in self.metrics.frag_recall]),
                'jaccard_mean': np.mean(self.metrics.frag_iou),
                'jaccard_std': np.std(self.metrics.frag_iou),
                'dice_mean': np.mean(self.metrics.frag_dice),
                'dice_std': np.std(self.metrics.frag_dice)
            }
            
            frag_df = pd.DataFrame([frag_metrics])
            frag_df.to_csv(comparison_dir / "fragmentation_metrics.csv", index=False)
            
            with open(comparison_dir / "fragmentation_metrics.txt", "w") as f:
                f.write("="*60 + "\n")
                f.write("MULTI-TASK: FRAGMENTATION SEGMENTATION\n")
                f.write("="*60 + "\n\n")
                for key, value in frag_metrics.items():
                    metric_name = key.replace('_mean', '').replace('_std', '')
                    if '_mean' in key:
                        std_value = frag_metrics.get(f"{metric_name}_std", 0)
                        f.write(f"{metric_name.capitalize()}: {value:.4f} ± {std_value:.4f}\n")
        
        # Classification comparison (EXP, ICM, TE)
        summary = self.metrics.get_summary()
        
        for task in ['exp', 'icm', 'te']:
            if f'{task}_metrics' in summary:
                task_metrics = summary[f'{task}_metrics']
                
                # Export per-class and macro metrics
                comparison_data = {
                    'accuracy': summary.get(f'{task}_accuracy', 0),
                    'f1_score': summary.get(f'{task}_f1', 0),
                    'macro_precision': task_metrics['macro_avg']['precision'],
                    'macro_recall': task_metrics['macro_avg']['recall'],
                    'macro_specificity': task_metrics['macro_avg']['specificity']
                }
                
                # Add per-class metrics
                num_classes = 5 if task == 'exp' else 3
                for i in range(num_classes):
                    if f'class_{i}' in task_metrics:
                        comparison_data[f'class_{i}_precision'] = task_metrics[f'class_{i}']['precision']
                        comparison_data[f'class_{i}_recall'] = task_metrics[f'class_{i}']['recall']
                        comparison_data[f'class_{i}_specificity'] = task_metrics[f'class_{i}']['specificity']
                
                task_df = pd.DataFrame([comparison_data])
                task_df.to_csv(comparison_dir / f"{task}_classification_metrics.csv", index=False)
        
        # Ploidy comparison
        if 'ploidy_metrics' in summary:
            ploidy_metrics = summary['ploidy_metrics']
            
            comparison_data = {
                'accuracy': summary.get('ploidy_accuracy', 0),
                'f1_score': summary.get('ploidy_f1', 0),
                'macro_precision': ploidy_metrics['macro_avg']['precision'],
                'macro_recall': ploidy_metrics['macro_avg']['recall'],
                'macro_specificity': ploidy_metrics['macro_avg']['specificity'],
                'euploid_precision': ploidy_metrics['class_0']['precision'],
                'euploid_recall': ploidy_metrics['class_0']['recall'],
                'euploid_specificity': ploidy_metrics['class_0']['specificity'],
                'aneuploid_precision': ploidy_metrics['class_1']['precision'],
                'aneuploid_recall': ploidy_metrics['class_1']['recall'],
                'aneuploid_specificity': ploidy_metrics['class_1']['specificity']
            }
            
            ploidy_df = pd.DataFrame([comparison_data])
            ploidy_df.to_csv(comparison_dir / "ploidy_classification_metrics.csv", index=False)
        
        print(f"Comparison metrics exported to {comparison_dir}")


def prepare_test_datasets(path: Path) -> MultiTaskDataset:
    """Prepare test datasets for evaluation."""
    print("Preparing test datasets...")
    
    # Grading datasets
    data_grading_exp = dataset_grading(path / "Blastocyst_Dataset", 'exp', 0, split='test')
    data_grading_te = dataset_grading(path / "Blastocyst_Dataset", 'te', 1, split='test')
    data_grading_icm = dataset_grading(path / "Blastocyst_Dataset", 'icm', 2, split='test')
    print(f"Grading samples: {len(data_grading_exp) + len(data_grading_te) + len(data_grading_icm)}")
    
    # Segmentation datasets
    data_seg_blasto = dataset_seg_blastocyst(path / "PCMR_blastocyts", 3, split='test')
    data_seg_blasto_quiron = dataset_seg_blastocyst(path / "Blastocyst_quiron", 4, split='test')
    print(f"Blastocyst segmentation samples: {len(data_seg_blasto) + len(data_seg_blasto_quiron)}")
    
    data_seg_fragmentation = dataset_seg_fragmentation(path / "CleavageEmbryo", 5, split='test')
    print(f"Fragmentation segmentation samples: {len(data_seg_fragmentation)}")
    
    # Ploidy dataset
    data_ploidy = dataset_ploidy(path / "Quiron_ploidy", 0, n_frames=50, split='test')
    print(f"Ploidy samples: {len(data_ploidy)}")
    
    # Combine into multi-task dataset
    test_dataset = MultiTaskDataset(
        image_samples=(
            data_grading_exp +
            data_grading_te +
            data_grading_icm +
            data_seg_blasto +
            data_seg_blasto_quiron +
            data_seg_fragmentation
        ),
        video_samples=data_ploidy,
    )
    
    print(f"Total test samples: {len(test_dataset)}\n")
    return test_dataset


def main(args):
    """Main evaluation routine."""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate transformer requirement
    if not args.transformer:
        print("WARNING: Transformer is disabled - video evaluation will be skipped")
    
    # Prepare dataset
    path = Path(args.path)
    test_dataset = prepare_test_datasets(path)
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=MultiTaskBatchSampler(test_dataset, batch_size=args.batch_size),
        collate_fn=multitask_collate,
        num_workers=args.num_workers
    )
    
    # Initialize model
    encoder = ResNetEncoder(pretrained=False)
    
    # Conditionally initialize transformer
    if args.transformer:
        transformer = VideoTransformer()
        print("Transformer enabled for video evaluation")
    else:
        transformer = None
        print("Transformer disabled - video samples will be skipped")
    
    model = MultiTaskModel(encoder, transformer, embedding_dim=512)
    
    # Initialize evaluator
    evaluator = MultiTaskEvaluator(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        multi_gpu=args.multi_gpu,
        use_transformer=args.transformer,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    summary = evaluator.evaluate(test_loader)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif not isinstance(value, dict):  # Skip nested metric dicts
            print(f"{key}: {value}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiTask Model Evaluation with Multi-GPU Support"
    )
    
    # Data paths
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/assessment/MultiTask/evaluation",
        help="Directory to save evaluation results"
    )
    
    # Processing parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    
    # GPU settings
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Use multiple GPUs if available"
    )
    
    # Model settings
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Enable transformer for video evaluation (required for video samples)"
    )
    
    args = parser.parse_args()
    main(args)