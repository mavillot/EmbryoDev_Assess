import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class TabularEncoder(nn.Module):
    """
    Small MLP that projects the tabular feature vector to a fixed-size embedding.

        tabular_dim  →  hidden_dim*2  →  out_dim
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalClassifier(nn.Module):
    """
    R(2+1)D-18 video backbone  +  tabular MLP encoder  →  fusion head.

    Parameters
    ----------
    tabular_dim      : number of tabular input features (default 12)
    tabular_hidden   : hidden dim for tabular encoder   (default 128)
    video_embed_dim  : penultimate feature size of R(2+1)D-18 (512)
    num_classes      : 1 for binary BCE
    pretrained       : load ImageNet/Kinetics weights for the backbone
    dropout_head     : dropout before fusion head first linear
    dropout_mid      : dropout between fusion head linears
    """

    def __init__(self, tabular_dim: int, tabular_hidden: int, video_embed_dim: int, num_classes: int, pretrained: bool, dropout_head: float, dropout_mid: float,):
        super().__init__()

    
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        _backbone = r2plus1d_18(weights=weights)
        self.video_encoder = nn.Sequential(*list(_backbone.children())[:-1])
        # output: [B, 512, 1, 1, 1]  → flatten to [B, 512]

        self.tabular_encoder = TabularEncoder(in_dim=tabular_dim,hidden_dim=tabular_hidden,out_dim=tabular_hidden,dropout=dropout_mid,)

        fusion_dim = video_embed_dim + tabular_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout_head),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_mid),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Linear(fusion_dim // 4, num_classes),
        )

    def freeze_backbone(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = True

    def backbone_parameters(self):
        return self.video_encoder.parameters()

    def head_parameters(self):
        """All non-backbone parameters (tabular encoder + fusion head)."""
        return list(self.tabular_encoder.parameters()) + list(self.head.parameters())

    def forward(
        self,
        frames: torch.Tensor,           # [B, C, T, H, W]
        mask: torch.Tensor,             # [B, T]  (padded-frame mask, passed through)
        tabular: torch.Tensor,          # [B, tabular_dim]
    ) -> torch.Tensor:                  # [B, 1]

        # Video branch
        v = self.video_encoder(frames)  # [B, 512, 1, 1, 1]
        v = v.flatten(1)                # [B, 512]

        # Tabular branch
        t = self.tabular_encoder(tabular.float())  # [B, tabular_hidden]

        # Fusion
        fused = torch.cat([v, t], dim=1)  # [B, fusion_dim]
        return self.head(fused)           # [B, 1]

