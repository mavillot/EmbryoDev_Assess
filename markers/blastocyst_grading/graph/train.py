#!/usr/bin/env python3

"""
Multitask blastocyst grading — CNN + KG Attention (PyTorch).

Three classification heads:
  - Expansion : 5 classes  (0-4, already integer indices in CSV)
  - TE        : 3 classes  (0-2, already integer indices in CSV)
                            3 = not gradable → masked out of TE/ICM loss
  - ICM       : 3 classes  (0-2, already integer indices in CSV)
                            3 = not gradable → masked out of TE/ICM loss

Key design decisions:
  - TE/ICM loss is masked: value 3 samples do not contribute gradient
  - Class weights computed automatically from non-masked samples
  - Upsampling done independently per head, then merged
  - At inference: if Expansion ∈ {0,1} → TE and ICM are forced to 3

Expected CSV columns: Image, Expansion, TE, ICM  (all integer)

Usage:
    python train_best.py
    python train_best.py --epochs 50 --w_exp 1.0 --w_te 1.0 --w_icm 1.0
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE  = 224
CKPT_DIR  = Path("models/markers/blastocyst_grading")

# CSV already contains integer indices — no mapping needed
# Expansion : 0-4  (5 classes)
# TE / ICM  : 0-2  (3 classes)  |  3 = not gradable (mask sentinel)

N_EXPANSION      = 5
N_TE             = 3
N_ICM            = 3
MASK_LABEL       = 3    # TE/ICM value meaning "not gradable" → excluded from loss

# Expansion indices for which TE/ICM are not gradable
# (adjust if the convention in your dataset differs)
NOT_GRADABLE_EXP = {0, 1}   # predicted expansion → force TE/ICM = MASK_LABEL

DATA_MAP = {
    'exp': {'train': 'EXP_silver', 'test': 'EXP_gold', 'classes': [0, 1, 2, 3, 4]},
    'icm': {'train': 'ICM_silver', 'test': 'ICM_gold', 'classes': [0, 1, 2]},
    'te':  {'train': 'TE_silver',  'test': 'TE_gold',  'classes': [0, 1, 2]},
}

KG_TRIPLES = [
    "embryo, has_quality, blastocyst_expansion",
    "embryo, has_quality, cell_symmetry",
    "embryo, has_quality, inner_cell_mass",
    "embryo, has_quality, trophectoderm_quality",
]
KG_EMBED_DIM = 64
KG_MAX_LEN   = 10


# ---------------------------------------------------------------------------
# KG tokenisation
# ---------------------------------------------------------------------------

def build_kg_token_ids(
        triples: list[str],
        max_len: int = KG_MAX_LEN,
) -> tuple[torch.LongTensor, int]:
    vocab: dict[str, int] = {}
    token_ids = []
    for triple in triples:
        tokens = triple.replace(",", "").split()
        ids = []
        for t in tokens[:max_len]:
            if t not in vocab:
                vocab[t] = len(vocab) + 1
            ids.append(vocab[t])
        ids += [0] * (max_len - len(ids))
        token_ids.append(ids)
    return torch.tensor(token_ids, dtype=torch.long), len(vocab) + 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BlastoGradingDataset(Dataset):
    """
    Returns (image, label_exp, label_te, label_icm).
    label_te / label_icm = -1 when the sample is D (not gradable).
    The loss function masks these out.
    """
    def __init__(self, df: pd.DataFrame, image_root: Path, transform=None):
        self.df         = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(self.image_root / row["Image"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_exp = int(row[DATA_MAP['exp']['train']])
        label_te  = int(row[DATA_MAP['te']['train']])
        label_icm = int(row[DATA_MAP['icm']['train']])

        # Remap MASK_LABEL (3) to -1 so masked_ce_loss can use >= 0 check
        label_te  = -1 if label_te  == MASK_LABEL else label_te
        label_icm = -1 if label_icm == MASK_LABEL else label_icm

        return image, label_exp, label_te, label_icm


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Per-head independent upsampling
# ---------------------------------------------------------------------------

def _upsample_column(df: pd.DataFrame, col: str, mask_val: int = MASK_LABEL) -> pd.DataFrame:
    """
    Oversample minority classes in `col`, ignoring rows where col == mask_val.
    """
    gradable  = df[df[col] != mask_val]
    max_count = gradable[col].value_counts().max()
    parts = []
    for cls in gradable[col].unique():
        subset = gradable[gradable[col] == cls]
        parts.append(resample(subset, replace=True,
                               n_samples=max_count, random_state=42))
    return pd.concat(parts)


def build_balanced_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      1. Upsample Expansion independently (all 6 classes).
      2. Upsample TE independently (A/B/C only, D rows kept as-is).
      3. Upsample ICM independently (A/B/C only, D rows kept as-is).
      4. Union of all three + original df, deduplicated on index,
         then shuffle.

    This ensures every rare class gets enough representation
    for each head without conflating the imbalance across heads.
    """
    up_exp = _upsample_column(df, DATA_MAP['exp']['train'])
    up_te  = _upsample_column(df, DATA_MAP['te']['train'])
    up_icm = _upsample_column(df, DATA_MAP['icm']['train'])

    merged = pd.concat([df, up_exp, up_te, up_icm]) \
               .sample(frac=1, random_state=42) \
               .reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Class weights  (computed from non-D samples only for TE/ICM)
# ---------------------------------------------------------------------------

def compute_class_weights(
        df: pd.DataFrame,
        col: str,
        n_classes: int,
        device: torch.device,
        mask_val: int = MASK_LABEL,
) -> torch.Tensor:
    """
    Inverse-frequency weights for CrossEntropyLoss.
    Rows where col == mask_val are excluded from the count.
    """
    gradable = df[df[col] != mask_val]
    counts   = np.array([
        (gradable[col] == c).sum() for c in range(n_classes)
    ], dtype=np.float32)
    counts  = np.where(counts == 0, 1, counts)   # avoid div-by-zero
    weights = counts.sum() / (n_classes * counts)
    print(f"  {col}: counts={counts.astype(int).tolist()}  "
          f"weights={[round(w, 3) for w in weights.tolist()]}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    dataset_path = Path(PATH_DATASET) / "Blastocyst_Dataset"
    image_root   = dataset_path / "Images"
    csv          = pd.read_csv(dataset_path / "Gardner_train_silver.csv",  encoding="latin-1", sep=";")

    # Stratified split on Expansion (most granular, always present)
    train_df, val_df = train_test_split(
        csv,
        test_size=args.val_fraction,
        stratify=csv[DATA_MAP['exp']['train']],
        random_state=42,
    )

    # Upsample only the training fold — no leakage into val
    train_df = build_balanced_train(train_df)

    print(f"\nTrain: {len(train_df)} samples (after upsample) | Val: {len(val_df)}")
    print("Train Expansion:", train_df[DATA_MAP['exp']['train']].value_counts().sort_index().to_dict())
    print("Train TE :",       train_df[DATA_MAP['te']['train']].value_counts().sort_index().to_dict())
    print("Train ICM:",       train_df[DATA_MAP['icm']['train']].value_counts().sort_index().to_dict())

    train_ds = BlastoGradingDataset(train_df, image_root, get_transforms(True))
    val_ds   = BlastoGradingDataset(val_df,   image_root, get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    return train_loader, val_loader, train_df


# ---------------------------------------------------------------------------
# Masked cross-entropy loss
# ---------------------------------------------------------------------------

def masked_ce_loss(logits: torch.Tensor,
                   targets: torch.Tensor,
                   weight: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss that ignores samples where target == -1 (D grade).
    Returns scalar 0 (no gradient) if all samples in batch are masked.
    """
    mask = targets >= 0                          # True where label is valid
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return F.cross_entropy(logits[mask], targets[mask], weight=weight)


def compute_loss(logits_exp, logits_te, logits_icm,
                 y_exp, y_te, y_icm,
                 w_exp, w_te, w_icm,
                 ce_weight_exp, ce_weight_te, ce_weight_icm):
    l_exp = F.cross_entropy(logits_exp, y_exp,  weight=ce_weight_exp)
    l_te  = masked_ce_loss(logits_te,   y_te,   ce_weight_te)
    l_icm = masked_ce_loss(logits_icm,  y_icm,  ce_weight_icm)
    total = w_exp * l_exp + w_te * l_te + w_icm * l_icm
    return total, l_exp.item(), l_te.item(), l_icm.item()


# ---------------------------------------------------------------------------
# KG Attention Layer
# ---------------------------------------------------------------------------

class KGAttentionLayer(nn.Module):
    def __init__(self, cnn_channels: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool      = nn.AdaptiveAvgPool2d(1)

        self.triple_queries = nn.Linear(embed_dim, embed_dim)
        self.cnn_keys       = nn.Linear(cnn_channels, embed_dim)
        self.cnn_values     = nn.Linear(cnn_channels, embed_dim)

        self.input_queries  = nn.Linear(cnn_channels, embed_dim)
        self.triple_keys    = nn.Linear(embed_dim, embed_dim)
        self.triple_values  = nn.Linear(embed_dim, embed_dim)

    def forward(self, cnn_features, triple_embeds):
        B     = cnn_features.size(0)
        scale = self.embed_dim ** 0.5

        cnn_flat    = self.pool(cnn_features).view(B, -1)
        triple_flat = triple_embeds.mean(dim=1).unsqueeze(0).expand(B, -1, -1)

        # Outward: KG queries × CNN keys/values
        tq      = self.triple_queries(triple_flat)
        ck      = self.cnn_keys(cnn_flat).unsqueeze(1)
        cv      = self.cnn_values(cnn_flat).unsqueeze(1)
        out_w   = F.softmax(torch.bmm(tq, ck.transpose(1, 2)) / scale, dim=1)
        out_agg = (out_w * cv).mean(dim=1)

        # Inward: CNN queries × KG keys/values
        iq      = self.input_queries(cnn_flat).unsqueeze(1)
        tk      = self.triple_keys(triple_flat)
        tv      = self.triple_values(triple_flat)
        in_w    = F.softmax(torch.bmm(iq, tk.transpose(1, 2)) / scale, dim=2)
        in_agg  = torch.bmm(in_w, tv).squeeze(1)

        return torch.cat([out_agg, in_agg], dim=1)   # (B, 2*d)


# ---------------------------------------------------------------------------
# CNN Encoder  (mirrors notebook architecture)
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.block3 = nn.Sequential(        # branching point for KGA
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(5, stride=2)
        self.flat  = nn.Sequential(nn.Flatten(), nn.Dropout(0.4))

    def forward(self, x):
        x            = self.block1(x)
        x            = self.block2(x)
        cnn_features = self.block3(x)       # (B, 64, H, W)
        x            = self.relu3(cnn_features)
        x            = self.pool3(x)
        x_flat       = self.flat(x)
        return cnn_features, x_flat


def _flat_dim() -> int:
    enc = CNNEncoder().eval()
    with torch.no_grad():
        _, flat = enc(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    return flat.shape[1]


# ---------------------------------------------------------------------------
# Grading head
# ---------------------------------------------------------------------------

class GradingHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Full multitask model
# ---------------------------------------------------------------------------

class MultitaskBlastoModel(nn.Module):
    def __init__(self, vocab_size: int, token_ids: torch.LongTensor,
                 embed_dim: int = KG_EMBED_DIM):
        super().__init__()
        self.kg_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.register_buffer("token_ids", token_ids)

        self.encoder  = CNNEncoder()
        fused_dim     = _flat_dim() + 2 * embed_dim

        self.kga = KGAttentionLayer(cnn_channels=64, embed_dim=embed_dim)

        self.head_expansion = GradingHead(fused_dim, N_EXPANSION)
        self.head_te        = GradingHead(fused_dim, N_TE)
        self.head_icm       = GradingHead(fused_dim, N_ICM)

    def forward(self, x):
        triple_embeds        = self.kg_embedding(self.token_ids)
        cnn_features, x_flat = self.encoder(x)
        kga_out              = self.kga(cnn_features, triple_embeds)
        fused                = torch.cat([x_flat, kga_out], dim=1)
        return (
            self.head_expansion(fused),
            self.head_te(fused),
            self.head_icm(fused),
        )


# ---------------------------------------------------------------------------
# Evaluation  (masked accuracy for TE and ICM)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, ce_w_exp, ce_w_te, ce_w_icm,
             w_exp, w_te, w_icm):
    model.eval()
    totals = dict(loss=0, exp=0, te=0, icm=0)
    correct = dict(exp=0, te=0, icm=0)
    counts  = dict(exp=0, te=0, icm=0)   # te/icm count only non-D samples

    for images, y_exp, y_te, y_icm in loader:
        images = images.to(device)
        y_exp  = y_exp.to(device)
        y_te   = y_te.to(device)
        y_icm  = y_icm.to(device)

        lo_exp, lo_te, lo_icm = model(images)
        loss, l_exp, l_te, l_icm = compute_loss(
            lo_exp, lo_te, lo_icm, y_exp, y_te, y_icm,
            w_exp, w_te, w_icm, ce_w_exp, ce_w_te, ce_w_icm)

        bs = images.size(0)
        totals["loss"] += loss.item() * bs
        totals["exp"]  += l_exp       * bs
        totals["te"]   += l_te        * bs
        totals["icm"]  += l_icm       * bs

        # Expansion: all samples
        correct["exp"] += (lo_exp.argmax(1) == y_exp).sum().item()
        counts["exp"]  += bs

        # TE / ICM: only non-D samples
        for key, logits, y in [("te", lo_te, y_te), ("icm", lo_icm, y_icm)]:
            mask = y >= 0
            if mask.sum() > 0:
                correct[key] += (logits.argmax(1)[mask] == y[mask]).sum().item()
                counts[key]  += mask.sum().item()

    n = counts["exp"]
    acc_exp = correct["exp"] / n
    acc_te  = correct["te"]  / max(counts["te"],  1)
    acc_icm = correct["icm"] / max(counts["icm"], 1)
    mean_acc = (acc_exp + acc_te + acc_icm) / 3

    return {
        "loss":     totals["loss"] / n,
        "loss_exp": totals["exp"]  / n,
        "loss_te":  totals["te"]   / n,
        "loss_icm": totals["icm"]  / n,
        "acc_exp":  acc_exp,
        "acc_te":   acc_te,
        "acc_icm":  acc_icm,
        "mean_acc": mean_acc,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader,
          optimizer, scheduler, args, device,
          ce_w_exp, ce_w_te, ce_w_icm):

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mean_acc = -1.0
    best_ckpt     = None

    hdr = (f"{'Ep':>4} | {'TrLoss':>8} | "
           f"{'VLoss':>8} {'LExp':>7} {'LTE':>7} {'LICM':>7} | "
           f"{'AccExp':>7} {'AccTE':>7} {'AccICM':>7} {'Mean':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n = 0

        for images, y_exp, y_te, y_icm in train_loader:
            images = images.to(device)
            y_exp  = y_exp.to(device)
            y_te   = y_te.to(device)
            y_icm  = y_icm.to(device)

            optimizer.zero_grad()
            lo_exp, lo_te, lo_icm = model(images)
            loss, *_ = compute_loss(
                lo_exp, lo_te, lo_icm, y_exp, y_te, y_icm,
                args.w_exp, args.w_te, args.w_icm,
                ce_w_exp, ce_w_te, ce_w_icm)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n += images.size(0)

        scheduler.step()

        m = evaluate(model, val_loader, device,
                     ce_w_exp, ce_w_te, ce_w_icm,
                     args.w_exp, args.w_te, args.w_icm)

        print(f"{epoch:>4} | {running_loss/n:>8.4f} | "
              f"{m['loss']:>8.4f} {m['loss_exp']:>7.4f} "
              f"{m['loss_te']:>7.4f} {m['loss_icm']:>7.4f} | "
              f"{m['acc_exp']:>7.4f} {m['acc_te']:>7.4f} "
              f"{m['acc_icm']:>7.4f} {m['mean_acc']:>7.4f}")

        if m["mean_acc"] > best_mean_acc:
            best_mean_acc = m["mean_acc"]
            if best_ckpt and best_ckpt.exists():
                best_ckpt.unlink()
            best_ckpt = CKPT_DIR / (
                f"best_mean{best_mean_acc:.4f}"
                f"_exp{m['acc_exp']:.4f}"
                f"_te{m['acc_te']:.4f}"
                f"_icm{m['acc_icm']:.4f}"
                f"_ep{epoch:02d}.pt"
            )
            torch.save(model.state_dict(), best_ckpt)
            print(f"  → saved: {best_ckpt.name}")

    print(f"\nBest mean val accuracy: {best_mean_acc:.4f}")
    print(f"Checkpoint: {best_ckpt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, train_df = build_dataloaders(args)

    print("\nClass weights (inverse-frequency, masked samples excluded for TE/ICM):")
    ce_w_exp = compute_class_weights(train_df, DATA_MAP['exp']['train'], N_EXPANSION, device)
    ce_w_te  = compute_class_weights(train_df, DATA_MAP['te']['train'], N_TE, device)
    ce_w_icm = compute_class_weights(train_df, DATA_MAP['icm']['train'], N_ICM, device)

    token_ids, vocab_size = build_kg_token_ids(KG_TRIPLES, KG_MAX_LEN)
    model = MultitaskBlastoModel(
        vocab_size=vocab_size,
        token_ids=token_ids,
        embed_dim=KG_EMBED_DIM,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    train(model, train_loader, val_loader,
          optimizer, scheduler, args, device,
          ce_w_exp, ce_w_te, ce_w_icm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multitask blastocyst grading — CNN + KG Attention (PyTorch)")

    parser.add_argument("--epochs",       type=int,   default=25)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--w_exp",        type=float, default=1.0,
                        help="Loss weight for Expansion head")
    parser.add_argument("--w_te",         type=float, default=1.0,
                        help="Loss weight for TE head")
    parser.add_argument("--w_icm",        type=float, default=1.0,
                        help="Loss weight for ICM head")

    args = parser.parse_args()
    main(args)