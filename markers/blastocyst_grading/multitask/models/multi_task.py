import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.heads.classification import ClassificationHead

WEIGHTS_LOSS = {'exp': torch.tensor([1.3889, 1.4104, 0.5584, 0.2319, 1.4104]),  
                'icm': torch.tensor([0.1000, 0.1224, 2.8443]),                  
                'te': torch.tensor([0.1224, 0.2445, 2.6331]), }

def weighted_focal_loss(logits, targets, weight=None, gamma=2.0):
    """
    Multiclass focal loss (exp, icm, te).
    logits:  (B, C) raw logits
    targets: (B,)   class indices (0..C-1)
    weight:  (C,)   per-class weights or None
    """
    log_probs = F.log_softmax(logits, dim=1)                          # (B, C)
    probs     = torch.exp(log_probs)                                   # (B, C)
    pt        = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (B,)
    ce_loss   = F.nll_loss(log_probs, targets, weight=weight, reduction='none')  # (B,)
    focal_loss = (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def binary_focal_loss(logits, targets, pos_weight=None, gamma=2.0):
    """
    Binary focal loss (ploidy).
    logits:     (B,) raw logits — single scalar per sample
    targets:    (B,) float binary labels (0.0 or 1.0)
    pos_weight: scalar tensor to upweight positive class, or None
    """
    probs  = torch.sigmoid(logits)                                     # (B,)
    pt     = torch.where(targets == 1, probs, 1 - probs)              # (B,) prob of correct class
    bce    = F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=pos_weight, reduction='none'
             )                                                          # (B,)
    focal_loss = (1 - pt) ** gamma * bce
    return focal_loss.mean()

# ---------- Multitask model skeleton ----------
class MultiTaskModel(nn.Module):
    def __init__(self, encoder, embedding_dim=512, weighted_loss=False):
        """
        encoder: shared encoder (ResNet34)
        embedding_dim: size of encoder embedding
        """
        super().__init__()
        self.encoder = encoder
        self.weighted_loss = weighted_loss

        # ----- Image grading heads -----
        self.exp_head = ClassificationHead(embedding_dim, 5)   # EXP 1-5
        self.icm_head = ClassificationHead(embedding_dim, 3)   # ICM
        self.te_head  = ClassificationHead(embedding_dim, 3)   # TE 1-3

        
    def _get_weight(self, key, device):
        """Move loss weight to the correct device dynamically."""
        w = WEIGHTS_LOSS.get(key)
        return w.to(device) if w is not None else None

    def forward(
        self,
        images=None,
        exp_labels=None,
        icm_labels=None,
        te_labels=None,
        ploidy_labels=None,
        HB_labels=None,
        blasto_str_labels=None,
        fragmentation_labels=None,
    ):
        outputs = {}
        losses = {}

        # -------- IMAGE TASKS --------
        if images is not None:
            device = images.device 
            features, img_embedding, skips = self.encoder(images)

            # GRADING TASKS
            # Compute logits for each grading task
            exp_logits = self.exp_head(img_embedding)
            icm_logits = self.icm_head(img_embedding)
            te_logits  = self.te_head(img_embedding)

            outputs['exp'] = exp_logits
            outputs['icm'] = icm_logits
            outputs['te']  = te_logits

            # Compute losses if labels exist
            if exp_labels is not None:
                if self.weighted_loss:
                    losses['exp'] = weighted_focal_loss(exp_logits, exp_labels, weight=self._get_weight('exp', device))
                else:
                    losses['exp'] = F.cross_entropy(exp_logits, exp_labels)
            if icm_labels is not None:
                if self.weighted_loss:
                    losses['icm'] = weighted_focal_loss(icm_logits, icm_labels, weight=self._get_weight('icm', device))
                else:
                    losses['icm'] = F.cross_entropy(icm_logits, icm_labels)
            if te_labels is not None:
                if self.weighted_loss:
                    losses['te'] = weighted_focal_loss(te_logits, te_labels, weight=self._get_weight('te', device))
                else:
                    losses['te'] = F.cross_entropy(te_logits, te_labels)
            
            

        return outputs, losses
