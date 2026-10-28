"""Evaluation entry point for MultiTask checkpoints."""

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
from models.temporal.temporal_transformer import VideoTransformer
from data.dataset import MultiTaskDataset, multitask_collate, MultiTaskBatchSampler
from data.managment import prepare_datasets

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET, OUTPUT_PATH

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
        self.stage_labels = []
        self.stage_preds = []
        self.te_labels = []
        self.te_preds = []
        self.ploidy_labels = []
        self.ploidy_preds = []
        self.fhb_labels = []
        self.fhb_preds = []
        
        # Segmentation metrics
        self.frag_iou = []
        self.frag_dice = []
        self.frag_accuracy = []
        self.frag_precision = []
        self.frag_recall = []
        self.frag_specificity = []
        
        self.seg_metrics = {
            'dice': defaultdict(list),
            'iou': defaultdict(list),
            'accuracy': defaultdict(list),
            'precision': defaultdict(list),
            'recall': defaultdict(list),
            'specificity': defaultdict(list)
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
    
    def specificity_score(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
        """
        Compute specificity for segmentation.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            eps: Epsilon for numerical stability
        """
        pred = pred.bool()
        target = target.bool()
        true_negative = ((~pred) & (~target)).sum(dim=(1, 2, 3))
        actual_negative = (~target).sum(dim=(1, 2, 3))
        return ((true_negative + eps) / (actual_negative + eps)).mean().item()
    
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
        
        # STAGE classification
        if batch.get('stage') is not None and 'stage' in outputs:
            pred = outputs['stage'].argmax(dim=1)
            self.stage_labels.extend(batch['stage'].cpu().tolist())
            self.stage_preds.extend(pred.cpu().tolist())
        
        # Ploidy classification
        if batch.get('ploidy') is not None and 'ploidy' in outputs:
            logits = outputs['ploidy']
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().view(-1)
            self.ploidy_labels.extend(batch['ploidy'].cpu().tolist())
            self.ploidy_preds.extend(preds.cpu().tolist())
        
        if batch.get('fhb') is not None and 'fhb' in outputs:
            logits = outputs['fhb']
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().view(-1)
            self.fhb_labels.extend(batch['fhb'].cpu().tolist())
            self.fhb_preds.extend(preds.cpu().tolist())
    
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
            self.frag_specificity.append(self.specificity_score(preds, target))
        
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
                self.seg_metrics['specificity'][c].append(self.specificity_score(pred_c, target_c))
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {
            'total_images': self.total_images,
            'total_videos': self.total_videos
        }
        
        # Helper function to compute per-class metrics
        def compute_metrics(labels, preds, num_classes):
            labels = np.asarray(labels)
            preds  = np.asarray(preds)

            assert len(labels) == len(preds) > 0, "labels and preds must be non-empty and same length"

            metrics = {}

            for c in range(num_classes):
                if num_classes == 1:
                    # preds are already 0/1
                    y_true = labels
                    y_pred = preds
                else:
                    # one-vs-rest
                    y_true = (labels == c).astype(int)
                    y_pred = (preds  == c).astype(int)

                tp = int(np.sum((y_true == 1) & (y_pred == 1)))
                fp = int(np.sum((y_true == 0) & (y_pred == 1)))
                tn = int(np.sum((y_true == 0) & (y_pred == 0)))
                fn = int(np.sum((y_true == 1) & (y_pred == 0)))

                precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                f1          = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

                metrics[f'class_{c}'] = {
                    'precision':   precision,
                    'recall':      recall,
                    'specificity': specificity,
                    'f1_score':    f1,
                    'accuracy':    accuracy,
                    'support':     int(np.sum(y_true)),  # number of true positives in ground truth
                }

            # macro average over all classes (works for both binary and multi-class)
            keys = ['precision', 'recall', 'specificity', 'f1_score', 'accuracy']
            metrics['macro_avg'] = {
                k: float(np.mean([metrics[f'class_{c}'][k] for c in range(num_classes)]))
                for k in keys
            }

            return metrics
        
        # Classification summaries with detailed metrics
        if self.exp_labels:
            summary['exp_metrics'] = {}
            exp_metrics = compute_metrics(self.exp_labels, self.exp_preds, num_classes=5)
            summary['exp_metrics']['accuracy'] = accuracy_score(self.exp_labels, self.exp_preds)
            summary['exp_metrics']['precision'] = exp_metrics['macro_avg']['precision']
            summary['exp_metrics']['recall'] = exp_metrics['macro_avg']['recall']
            summary['exp_metrics']['specificity'] = exp_metrics['macro_avg']['specificity']
            summary['exp_metrics']['f1_score'] = exp_metrics['macro_avg']['f1_score']
        
        if self.icm_labels:
            summary['icm_metrics'] = {}
            icm_metrics = compute_metrics(self.icm_labels, self.icm_preds, num_classes=3)
            summary['icm_metrics']['accuracy'] = accuracy_score(self.icm_labels, self.icm_preds)
            summary['icm_metrics']['precision'] = icm_metrics['macro_avg']['precision']
            summary['icm_metrics']['recall'] = icm_metrics['macro_avg']['recall']
            summary['icm_metrics']['specificity'] = icm_metrics['macro_avg']['specificity']
            summary['icm_metrics']['f1_score'] = icm_metrics['macro_avg']['f1_score']
        
        if self.te_labels:
            summary['te_metrics'] = {}
            summary['te_metrics']['accuracy'] = accuracy_score(self.te_labels, self.te_preds)
            te_metrics = compute_metrics(self.te_labels, self.te_preds, num_classes=3)
            summary['te_metrics']['precision'] = te_metrics['macro_avg']['precision']
            summary['te_metrics']['recall'] = te_metrics['macro_avg']['recall']
            summary['te_metrics']['specificity'] = te_metrics['macro_avg']['specificity']
            summary['te_metrics']['f1_score'] = te_metrics['macro_avg']['f1_score']
        
        if self.stage_labels:
            summary['stage_metrics'] = {}
            summary['stage_metrics']['accuracy'] = accuracy_score(self.stage_labels, self.stage_preds)
            stage_metrics = compute_metrics(self.stage_labels, self.stage_preds, num_classes=5)
            summary['stage_metrics']['precision'] = stage_metrics['macro_avg']['precision']
            summary['stage_metrics']['recall'] = stage_metrics['macro_avg']['recall']
            summary['stage_metrics']['specificity'] = stage_metrics['macro_avg']['specificity']
            summary['stage_metrics']['f1_score'] = stage_metrics['macro_avg']['f1_score']
        
        if self.ploidy_labels:
            summary['ploidy_metrics'] = {}
            summary['ploidy_metrics']['accuracy'] = accuracy_score(self.ploidy_labels, self.ploidy_preds)
            ploidy_metrics = compute_metrics(self.ploidy_labels, self.ploidy_preds, num_classes=1)
            summary['ploidy_metrics']['precision'] = ploidy_metrics['macro_avg']['precision']
            summary['ploidy_metrics']['recall'] = ploidy_metrics['macro_avg']['recall']
            summary['ploidy_metrics']['specificity'] = ploidy_metrics['macro_avg']['specificity']
            summary['ploidy_metrics']['f1_score'] = ploidy_metrics['macro_avg']['f1_score']
        
        if self.fhb_labels:
            summary['fhb_metrics'] = {}
            summary['fhb_metrics']['accuracy'] = accuracy_score(self.fhb_labels, self.fhb_preds)
            fhb_metrics = compute_metrics(self.fhb_labels, self.fhb_preds, num_classes=1)
            summary['fhb_metrics']['precision'] = fhb_metrics['macro_avg']['precision']
            summary['fhb_metrics']['recall'] = fhb_metrics['macro_avg']['recall']
            summary['fhb_metrics']['specificity'] = fhb_metrics['macro_avg']['specificity']
            summary['fhb_metrics']['f1_score'] = fhb_metrics['macro_avg']['f1_score']
        
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
        output_dir: str = "output/assessment/MultiTask/evaluation/"
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
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle DataParallel checkpoint (remove 'module.' prefix)
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            checkpoint = new_state_dict
        
        # Use strict=False so that loss-function buffers saved during training
        # (e.g. fragmentation_loss_fn.pos_weight, blasto_str_loss_fn params)
        # do not cause a crash when the eval model is built without weighted_loss=True.
        # Those buffers are only needed during training; we log them as a heads-up.
        result = self.base_model.load_state_dict(checkpoint, strict=False)
        if result.unexpected_keys:
            print(f"  [checkpoint] Ignored training-only keys: {result.unexpected_keys}")
        if result.missing_keys:
            print(f"  [checkpoint] WARNING — missing keys (architecture mismatch?): {result.missing_keys}")
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
            stage_labels=None,
            blasto_str_labels=None,
            fragmentation_labels=None,
            ploidy_labels=None,
            fhb_labels=None
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
                    labels=range(1,6)
                )
                
                report = classification_report(
                    self.metrics.exp_labels, 
                    self.metrics.exp_preds, 
                    labels=labels,
                    digits=4,
                    zero_division=0
                )
                # f.write(report)
                # f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'exp_metrics' in summary:                 
                    macro = summary['exp_metrics']
                    f.write(f"Accuracy :   {100* macro['accuracy']:.2f}\n")
                    f.write(f"Precision:   {100*macro['precision']:.2f}\n")
                    f.write(f"Recall:      {100*macro['recall']:.2f}\n")
                    f.write(f"Specificity: {100*macro['specificity']:.2f}\n")
                    f.write(f"F1 Score:    {100*macro['f1_score']:.2f}\n")
                f.write("\n\n")
            
            # ICM Classification
            if self.metrics.icm_labels:
                f.write("="*60 + "\n")
                f.write("ICM CLASSIFICATION\n")
                f.write("="*60 + "\n")
                class_map = {'A':0, 'B':1, 'C':2}
                labels = sorted(list(set(self.metrics.icm_labels)))
                cm = confusion_matrix(self.metrics.icm_labels, self.metrics.icm_preds, labels=labels)
                self.save_confusion_matrix(
                    cm,
                    self.output_dir / 'confusion_matrix_ICM.png',
                    title='ICM Confusion Matrix',
                    labels=list(class_map.keys())
                )
                
                report = classification_report(
                    self.metrics.icm_labels,
                    self.metrics.icm_preds,
                    labels=labels,
                    digits=4,
                    zero_division=0
                )
                # f.write(report)
                # f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'icm_metrics' in summary:              
                    macro = summary['icm_metrics']
                    f.write(f"Accuracy :   {100*macro['accuracy']:.2f}\n")
                    f.write(f"Precision:   {100*macro['precision']:.2f}\n")
                    f.write(f"Recall:      {100*macro['recall']:.2f}\n")
                    f.write(f"Specificity: {100*macro['specificity']:.2f}\n")
                    f.write(f"F1 Score:    {100*macro['f1_score']:.2f}\n")
                f.write("\n\n")
            
            # TE Classification
            if self.metrics.te_labels:
                f.write("="*60 + "\n")
                f.write("TE CLASSIFICATION\n")
                f.write("="*60 + "\n")
                class_map = {'A':0, 'B':1, 'C':2}
                labels = sorted(list(set(self.metrics.te_labels)))
                cm = confusion_matrix(self.metrics.te_labels, self.metrics.te_preds, labels=labels)
                self.save_confusion_matrix(
                    cm,
                    self.output_dir / 'confusion_matrix_TE.png',
                    title='TE Confusion Matrix',
                    labels=list(class_map.keys())
                )
                
                report = classification_report(
                    self.metrics.te_labels,
                    self.metrics.te_preds,
                    labels=labels,
                    digits=4,
                    zero_division=0
                )
                # f.write(report)
                # f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'te_metrics' in summary:                   
                    macro = summary['te_metrics']
                    f.write(f"Accuracy :   {100*macro['accuracy']:.2f}\n")
                    f.write(f"Precision:   {100*macro['precision']:.2f}\n")
                    f.write(f"Recall:      {100*macro['recall']:.2f}\n")
                    f.write(f"Specificity: {100*macro['specificity']:.2f}\n")
                    f.write(f"F1 Score:    {100*macro['f1_score']:.2f}\n")
                f.write("\n\n")
            
            # STAGE Classification
            if self.metrics.stage_labels:
                f.write("="*60 + "\n")
                f.write("STAGE CLASSIFICATION\n")
                f.write("="*60 + "\n")
                classmap = {'empty':0, '1cell':1, '2cell':2, '+3cell':3, '+8cell':4, 'morula':5, 'blastocyst':6}
                labels = sorted(list(set(self.metrics.stage_labels)))
                cm = confusion_matrix(self.metrics.stage_labels, self.metrics.stage_preds, labels=labels)
                self.save_confusion_matrix(
                    cm,
                    self.output_dir / 'confusion_matrix_STAGE.png',
                    title='STAGE Confusion Matrix',
                    labels=list(classmap.keys())
                )
                
                report = classification_report(
                    self.metrics.stage_labels,
                    self.metrics.stage_preds,
                    labels=labels,
                    digits=4,
                    zero_division=0
                )
                # f.write(report)
                # f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'stage_metrics' in summary:                   
                    macro = summary['stage_metrics']
                    f.write(f"Accuracy :   {100*macro['accuracy']:.2f}\n")
                    f.write(f"Precision:   {100*macro['precision']:.2f}\n")
                    f.write(f"Recall:      {100*macro['recall']:.2f}\n")
                    f.write(f"Specificity: {100*macro['specificity']:.2f}\n")
                    f.write(f"F1 Score:    {100*macro['f1_score']:.2f}\n")
                f.write("\n\n")
            
            # Ploidy Classification
            if self.metrics.ploidy_labels:
                f.write("="*60 + "\n")
                f.write("PLOIDY CLASSIFICATION\n")
                f.write("="*60 + "\n")
                
                # Use numeric labels [0, 1] for sklearn functions
                numeric_labels = [0, 1]
                label_names = ['Aneuploid', 'Euploid']
                
                # Only compute confusion matrix if we have actual labels
                unique_labels = sorted(list(set(self.metrics.ploidy_labels)))
                if unique_labels:
                    cm = confusion_matrix(
                        self.metrics.ploidy_labels, 
                        self.metrics.ploidy_preds, 
                        labels=numeric_labels
                    )
                    self.save_confusion_matrix(
                        cm,
                        self.output_dir / 'confusion_matrix_Ploidy.png',
                        title='Ploidy Confusion Matrix',
                        labels=label_names,
                    )
                
                report = classification_report(
                    self.metrics.ploidy_labels,
                    self.metrics.ploidy_preds,
                    labels=numeric_labels,
                    target_names=label_names,
                    digits=4,
                    zero_division=0
                )
                # f.write(report)
                # f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'ploidy_metrics' in summary:
                    class_names = ['Aneuploid', 'Euploid']
                    macro = summary['ploidy_metrics']
                    f.write(f"Accuracy :   {100*macro['accuracy']:.2f}\n")
                    f.write(f"Precision:   {100*macro['precision']:.2f}\n")
                    f.write(f"Recall:      {100*macro['recall']:.2f}\n")
                    f.write(f"Specificity: {100*macro['specificity']:.2f}\n")
                    f.write(f"F1 Score:    {100*macro['f1_score']:.2f}\n")
                f.write("\n\n")

            # fhb Classification
            if self.metrics.fhb_labels:
                f.write("="*60 + "\n")
                f.write("FHB CLASSIFICATION\n")
                f.write("="*60 + "\n")
                
                # Use numeric labels [0, 1] for sklearn functions
                numeric_labels = [0, 1]
                label_names = ['no_FHB', 'FHB']
                
                # Only compute confusion matrix if we have actual labels
                unique_labels = sorted(list(set(self.metrics.fhb_labels)))
                if unique_labels:
                    cm = confusion_matrix(
                        self.metrics.fhb_labels, 
                        self.metrics.fhb_preds, 
                        labels=numeric_labels
                    )
                    self.save_confusion_matrix(
                        cm,
                        self.output_dir / 'confusion_matrix_FHB.png',
                        title='FHB Confusion Matrix',
                        labels=label_names,
                    )
                
                report = classification_report(
                    self.metrics.fhb_labels,
                    self.metrics.fhb_preds,
                    labels=numeric_labels,
                    target_names=label_names,
                    digits=4,
                    zero_division=0
                )
                # f.write(report)
                # f.write("\n")
                
                # Add per-class precision, recall, specificity
                summary = self.metrics.get_summary()
                if 'fhb_metrics' in summary:
                    class_names = ['no_FHB', 'FHB']
                    macro = summary['fhb_metrics']
                    f.write(f"Accuracy :   {100*macro['accuracy']:.2f}\n")
                    f.write(f"Precision:   {100*macro['precision']:.2f}\n")
                    f.write(f"Recall:      {100*macro['recall']:.2f}\n")
                    f.write(f"Specificity: {100*macro['specificity']:.2f}\n")
                    f.write(f"F1 Score:    {100*macro['f1_score']:.2f}\n")
                f.write("\n\n")
            
            # Fragmentation Segmentation
            if self.metrics.frag_iou:
                f.write("="*60 + "\n")
                f.write("FRAGMENTATION SEGMENTATION\n")
                f.write("="*60 + "\n")
                f.write(f"Accuracy:        {100*np.mean(self.metrics.frag_accuracy):.2f} ± {100*np.std(self.metrics.frag_accuracy):.2f}\n")
                f.write(f"Precision:       {100*np.mean(self.metrics.frag_precision):.2f} ± {100*np.std(self.metrics.frag_precision):.2f}\n")
                f.write(f"Recall:          {100*np.mean(self.metrics.frag_recall):.2f} ± {100*np.std(self.metrics.frag_recall):.2f}\n")
                f.write(f"Specificity:     {100*np.mean(self.metrics.frag_specificity):.2f} ± {100*np.std(self.metrics.frag_specificity):.2f}\n")
                f.write(f"Dice Score:      {100*np.mean(self.metrics.frag_dice):.2f} ± {100*np.std(self.metrics.frag_dice):.2f}\n")
                f.write(f"IoU Score:       {100*np.mean(self.metrics.frag_iou):.2f} ± {100*np.std(self.metrics.frag_iou):.2f}\n\n")
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
                        f.write(f"  Accuracy:       {100*np.mean(accuracy_scores):.2f} ± {100*np.std(accuracy_scores):.2f}\n")
                        f.write(f"  Precision:      {100*np.mean(precision_scores):.2f} ± {100*np.std(precision_scores):.2f}\n")
                        f.write(f"  Recall:         {100*np.mean(recall_scores):.2f} ± {100*np.std(recall_scores):.2f}\n")
                        f.write(f"  Specificity:    {100*np.mean(self.metrics.seg_metrics['specificity'][c]):.2f} ± {100*np.std(self.metrics.seg_metrics['specificity'][c]):.2f}\n")
                        f.write(f"  Dice Score:     {100*np.mean(dice_scores):.2f} ± {100*np.std(dice_scores):.2f}\n")
                        f.write(f"  IoU Score:      {100*np.mean(iou_scores):.2f} ± {100*np.std(iou_scores):.2f}\n\n")
        
        print(f"Detailed report saved to {report_path}")
        
        # Save metrics summary as CSV
        summary = self.metrics.get_summary()
        summary_df = pd.DataFrame([summary])
        summary_path = self.output_dir / "metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Metrics summary saved to {summary_path}")
    


def main(args):
    """Main evaluation routine."""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Validate transformer requirement
    if not args.transformer:
        print("WARNING: Transformer is disabled - video evaluation will be skipped")
    
    # Prepare dataset
    path = Path(args.path)
    image_samples, video_samples = prepare_datasets(path, args.transformer, split="test", tasks=args.tasks)
    test_dataset = MultiTaskDataset(
        image_samples=image_samples,
        video_samples=video_samples,
        img_size=(224, 224),
        use_femi_processor=args.encoder == "femi",
    )
    
    print(f"Total test samples: {len(test_dataset)}\n")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=MultiTaskBatchSampler(test_dataset, batch_size=args.batch_size),
        collate_fn=multitask_collate,
        num_workers=args.num_workers
    )
    
    # Initialize model
    if args.encoder == "resnet34":
        from models.encoder.visual_encoder import ResNetEncoder
        encoder = ResNetEncoder(pretrained=True)
    elif args.encoder == "efficientnet_b3":
        from models.encoder.visual_encoder import EfficientNetB3Encoder
        encoder = EfficientNetB3Encoder(pretrained=True)
    elif args.encoder == "femi":
        from models.encoder.femi_encoder import FEMIEncoder
        encoder = FEMIEncoder(freeze=False, embedding_dim=512)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
    
    # Conditionally initialize transformer
    if args.transformer:
        from models.temporal.temporal_transformer import VideoTransformer
        transformer = VideoTransformer(args.frames) 
        print("Transformer enabled for video evaluation")
    else:
        transformer = None
        print("Transformer disabled - video samples will be skipped")

    model = MultiTaskModel(encoder, transformer, embedding_dim=encoder.embedding_dim, weighted_loss=args.weighted_loss)
    
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
    parser.add_argument("--path",type=str,required=False,help="Path to dataset directory", default=PATH_DATASET)
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder architecture (default: resnet34)")
    parser.add_argument("--checkpoint",type=str,required=True,help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, help="Use baseline model", default=None)
    
    # Output
    parser.add_argument("--output_dir",type=str,default=OUTPUT_PATH + "/assessment/MultiTask/femi/",help="Directory to save evaluation results")
    
    # Processing parameters
    parser.add_argument("--batch_size",type=int,default=1,help="Batch size for evaluation")
    parser.add_argument("--num_workers",type=int,default=4,help="Number of dataloader workers")
    
    # GPU settings
    parser.add_argument("--multi_gpu",action="store_true",help="Use multiple GPUs if available")
    
    # Model settings
    parser.add_argument("--transformer",action="store_true",help="Enable transformer for video evaluation (required for video samples)")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames used during training (must match checkpoint)")
    parser.add_argument("--weighted_loss", action="store_true", help="Build model with weighted loss heads (must match training checkpoint)")
    args = parser.parse_args()
    main(args)