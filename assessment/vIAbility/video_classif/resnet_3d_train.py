"""
resnet3d_train.py
────────────────────────────────────────────────────────────────────────────────
Keras 3 / TensorFlow 2.16+ training script for the ResNet-3D model.

Uses the same data pipeline, task, and evaluation logic as cnn_3d.py:
  Task    : Binary classification  (Fetal_Heart_Beat  0 / 1)
  Data    : Oslo_viability videos + Oslo_completo.xlsx labels
  Extras  : Label smoothing · Early stopping · F-beta checkpointing
            Two-stage training (head-only → full fine-tune)
            Mixed precision · Training curves · Confusion matrix
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score,
)

# ── Keras 3 — import keras before tensorflow to avoid backend conflicts ───────
import keras
from keras import layers as L, ops

import tensorflow as tf

# ── Project imports (same as cnn_3d.py) ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from assessment.vIAbility.data.dataset import get_dataloaders
from config.config import PATH_DATASET, SEED

# ── Resnet building blocks ────────────────────────────────────────────────────
from resnet_3d import conv_blocks, add_residual_block, ResizeVideo

# ── Reproducibility ───────────────────────────────────────────────────────────
keras.utils.set_random_seed(SEED)

VIDEO_DIR  = os.path.join(PATH_DATASET, "Oslo_viability/videos")
EXCEL_PATH = os.path.join(PATH_DATASET, "Oslo_viability/Oslo_completo.xlsx")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # data
    frame_h: int        = 224
    frame_w: int        = 224
    n_frames: int       = 25
    num_workers: int    = 4

    # loss
    weight_loss: float  = 1.48      # positive-class weight in BCE

    # two-stage fine-tuning
    freeze_epochs: int  = 30        # stage-1 epochs (head only)
    finetune_epochs:int = 31        # stage-2 epochs (full model)
    lr_head: float      = 3e-4
    lr_backbone: float  = 1e-5
    warmup_epochs: int  = 3
    weight_decay: float = 1e-4
    beta: float         = 0.5       # F-beta exponent
    save_by: str        = "val_fbeta"

    # regularisation
    label_smoothing: float = 0.2
    dropout_head: float    = 0.5

    # training
    batch_size: int     = 8

    # early stopping
    patience: int       = 10
    min_delta: float    = 1e-4

    # augmentation / sampling
    augmentation: bool  = True
    oversample: bool    = False
    factor1: float      = 2.0
    factor0: float      = 1.5

    # paths
    ckpt_dir: str       = "models/assessment/vIAbility/resnet3d"
    out_dir: str        = "output/assessment/vIAbility/resnet3d"
    ckpt_name: str      = field(init=False)

    # mixed precision
    amp: bool           = True

    def __post_init__(self):
        self.ckpt_name = f"best_resnet3d_{self.n_frames}.keras"


CFG = TrainConfig()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Mixed precision
# ─────────────────────────────────────────────────────────────────────────────
if CFG.amp:
    keras.mixed_precision.set_global_policy("mixed_float16")
    log.info("Mixed precision: float16 compute / float32 variables")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Model  — binary head replacing the original 3-class softmax head
# ─────────────────────────────────────────────────────────────────────────────
def build_binary_model(
    width: int = 224, height: int = 224,
    depth: int = 3,   n_frames: int = 25,
) -> keras.Model:

    inputs = L.Input(shape=(n_frames, height, width, depth))
    x = inputs

    # ── Backbone (identical to resnet_3d.get_model, minus classification head)
    x = conv_blocks(filters=16, kernel_size=(3, 7, 7), padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    x = ResizeVideo(height // 2, width // 2)(x)

    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(height // 4, width // 4)(x)

    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(height // 8, width // 8)(x)

    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(height // 16, width // 16)(x)

    x = add_residual_block(x, 128, (3, 3, 3))
    x = L.GlobalAveragePooling3D()(x)
    x = L.Flatten()(x)

    # ── Binary head ──────────────────────────────────────────────────────────
    x = L.Dropout(CFG.dropout_head)(x)
    x = L.Dense(
        128, activation="relu",
        kernel_regularizer=keras.regularizers.L2(CFG.weight_decay),
    )(x)
    x = L.Dropout(CFG.dropout_head)(x)
    # Force float32 output regardless of AMP policy
    outputs = L.Dense(1, activation="sigmoid", dtype="float32")(x)

    return keras.Model(inputs, outputs, name="ResNet3D_Binary")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Data adapter  — PyTorch DataLoader → tf.data.Dataset
# ─────────────────────────────────────────────────────────────────────────────
def torch_loader_to_tf_dataset(
    loader, n_frames: int, height: int, width: int
) -> tf.data.Dataset:
    """
    PyTorch loader yields  (frames, mask, labels)
      frames : (B, C, T, H, W)  float32  [0, 1]
      labels : (B,)             int64

    tf.data.Dataset yields  (x, y)
      x : (T, H, W, C)  float32
      y : (1,)          float32
    """
    def generator():
        for frames, _mask, labels in loader:
            # (B, C, T, H, W) → (B, T, H, W, C)
            x_batch = frames.numpy().transpose(0, 2, 3, 4, 1).astype(np.float32)
            y_batch = labels.numpy().astype(np.float32).reshape(-1, 1)
            for xi, yi in zip(x_batch, y_batch):
                yield xi, yi

    sig = (
        tf.TensorSpec(shape=(n_frames, height, width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,),                          dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=sig)
    ds = ds.batch(loader.batch_size or CFG.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Loss — label-smoothed, positive-class-weighted BCE
# ─────────────────────────────────────────────────────────────────────────────
class SmoothedWeightedBCE(keras.losses.Loss):

    def __init__(
        self,
        smoothing: float  = CFG.label_smoothing,
        pos_weight: float = CFG.weight_loss,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smoothing  = smoothing
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.cast(y_pred, tf.float32)
        eps     = self.smoothing
        y_smooth = y_true * (1.0 - eps) + 0.5 * eps
        bce     = tf.keras.backend.binary_crossentropy(y_smooth, y_pred)
        weight  = y_true * (self.pos_weight - 1.0) + 1.0
        return tf.reduce_mean(weight * bce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(smoothing=self.smoothing, pos_weight=self.pos_weight)
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Metric — F-beta  (Keras 3 compatible)
# ─────────────────────────────────────────────────────────────────────────────
class FBetaMetric(keras.metrics.Metric):
    """Accumulates TP / FP / FN over the epoch, returns F-beta on result()."""

    def __init__(self, beta: float = CFG.beta, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.beta      = float(beta)
        self.threshold = float(threshold)
        # Keras 3: add_weight requires explicit name= and shape= kwargs
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        preds  = tf.cast(y_pred >= self.threshold, tf.float32)
        self.tp.assign_add(tf.reduce_sum(preds * y_true))
        self.fp.assign_add(tf.reduce_sum(preds * (1.0 - y_true)))
        self.fn.assign_add(tf.reduce_sum((1.0 - preds) * y_true))

    def result(self):
        b2  = self.beta ** 2
        num = (1.0 + b2) * self.tp
        den = num + b2 * self.fn + self.fp + keras.backend.epsilon()
        return num / den

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(beta=self.beta, threshold=self.threshold)
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Callback — linear LR warm-up
# ─────────────────────────────────────────────────────────────────────────────
class LinearWarmupCallback(keras.callbacks.Callback):
    """Ramps LR from ~0 → base_lr linearly over warmup_epochs."""

    def __init__(self, warmup_epochs: int, base_lr: float):
        super().__init__()
        self.warmup_epochs = max(1, warmup_epochs)
        self.base_lr       = base_lr
        self._local_epoch  = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._local_epoch += 1
        factor = min(1.0, self._local_epoch / self.warmup_epochs)
        new_lr = float(self.base_lr * factor)
        # Keras 3: optimizer.learning_rate is a plain Python float after compile;
        # use the public assign() API to update it safely.
        self.model.optimizer.learning_rate = new_lr
        log.info("Warmup LR → %.2e  (local epoch %d)", new_lr, self._local_epoch)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Freeze / unfreeze helpers
# ─────────────────────────────────────────────────────────────────────────────
_HEAD_KEYWORDS = {"flatten", "dropout", "dense", "activation"}

def _is_head(layer) -> bool:
    return any(k in layer.name.lower() for k in _HEAD_KEYWORDS)

def freeze_backbone(model: keras.Model):
    for layer in model.layers:
        if not _is_head(layer):
            layer.trainable = False

def unfreeze_all(model: keras.Model):
    for layer in model.layers:
        layer.trainable = True


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Plots & evaluation
# ─────────────────────────────────────────────────────────────────────────────
def save_training_plots(hist1, hist2, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def cat(key):
        a = hist1.history.get(key, [])
        b = hist2.history.get(key, []) if hist2 else []
        return a + b

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(cat("loss"),            label="Train")
    axes[0].plot(cat("val_loss"),        label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(cat("binary_accuracy"), label="Train")
    axes[1].plot(cat("val_binary_accuracy"), label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path); plt.close()
    log.info("Training curves → %s", path)


def evaluate(model: keras.Model, ds: tf.data.Dataset, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    all_labels, all_preds = [], []

    for x_batch, y_batch in ds:
        probs = model(x_batch, training=False).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.numpy().flatten().astype(int).tolist())

    y_true = np.array(all_labels, dtype=int)
    y_pred = np.array(all_preds,  dtype=int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No HB (0)", "HB (1)"],
        yticklabels=["No HB (0)", "HB (1)"],
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path); plt.close()
    log.info("Confusion matrix → %s", cm_path)

    # Classification report
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    report = classification_report(
        y_true, y_pred,
        target_names=["No Heartbeat (0)", "Heartbeat (1)"],
        zero_division=0,
    )
    log.info("\n%s", report)

    rpt_path = os.path.join(out_dir, "classification_report.txt")
    with open(rpt_path, "w") as fh:
        fh.write("Validation — ResNet-3D binary classifier\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(report)
        fh.write(f"\nOverall accuracy : {acc:.4f}\n\n")
        for i, name in enumerate(["No Heartbeat (0)", "Heartbeat (1)"]):
            fh.write(
                f"  {name:<22}  precision={prec[i]:.4f}  "
                f"recall={rec[i]:.4f}  f1={f1[i]:.4f}  "
                f"support={int(sup[i])}\n"
            )
    log.info("Classification report → %s", rpt_path)


def save_config(cfg: TrainConfig, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "training_config.json")
    with open(path, "w") as fh:
        json.dump(vars(cfg), fh, indent=4)
    log.info("Config → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Training
# ─────────────────────────────────────────────────────────────────────────────
def make_callbacks(ckpt_path: str, monitor: str, mode: str, patience: int):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor, mode=mode,
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode,
            patience=patience, min_delta=CFG.min_delta,
            restore_best_weights=True, verbose=1,
        ),
    ]


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=CFG.weight_decay,
        ),
        loss=SmoothedWeightedBCE(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            FBetaMetric(beta=CFG.beta, name="fbeta_metric"),
        ],
    )


def train():
    log.info("Keras %s  |  TF %s  |  GPUs: %s",
             keras.__version__, tf.__version__,
             tf.config.list_physical_devices("GPU"))

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading data …")
    train_loader, val_loader = get_dataloaders(
        video_dir   = VIDEO_DIR,
        excel_path  = EXCEL_PATH,
        frame_h     = CFG.frame_h,
        frame_w     = CFG.frame_w,
        n_frames    = CFG.n_frames,
        augment     = CFG.augmentation,
        oversample  = CFG.oversample,
        factor1     = CFG.factor1,
        factor0     = CFG.factor0,
        batch_size  = CFG.batch_size,
        num_workers = CFG.num_workers,
        device      = "cpu",
    )
    log.info("Train: %d  |  Val: %d",
             len(train_loader.dataset), len(val_loader.dataset))

    tf_train = torch_loader_to_tf_dataset(
        train_loader, CFG.n_frames, CFG.frame_h, CFG.frame_w)
    tf_val   = torch_loader_to_tf_dataset(
        val_loader,   CFG.n_frames, CFG.frame_h, CFG.frame_w)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_binary_model(
        width=CFG.frame_w, height=CFG.frame_h,
        depth=3, n_frames=CFG.n_frames,
    )
    model.summary(print_fn=log.info)

    # ── Checkpoint / monitor setup ────────────────────────────────────────────
    ckpt_dir  = os.path.join(CFG.ckpt_dir, str(CFG.n_frames))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, CFG.ckpt_name)
    out_dir   = os.path.join(CFG.out_dir,  str(CFG.n_frames))

    monitor = {
        "val_fbeta":  "val_fbeta_metric",
        "val_acc":    "val_binary_accuracy",
        "val_loss":   "val_loss",
    }.get(CFG.save_by, "val_fbeta_metric")
    mode = "min" if CFG.save_by == "val_loss" else "max"

    # ── Stage 1: head only ────────────────────────────────────────────────────
    freeze_backbone(model)
    n_trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
    n_total     = sum(np.prod(w.shape) for w in model.weights)
    log.info("Stage 1 — backbone frozen | trainable %.2fM / %.2fM params",
             n_trainable / 1e6, n_total / 1e6)

    compile_model(model, CFG.lr_head)
    warmup1 = LinearWarmupCallback(CFG.warmup_epochs, CFG.lr_head)
    cbs1    = make_callbacks(ckpt_path, monitor, mode, CFG.patience) + [warmup1]

    log.info("Stage 1: up to %d epochs", CFG.freeze_epochs)
    hist1 = model.fit(
        tf_train,
        validation_data=tf_val,
        epochs=CFG.freeze_epochs,
        callbacks=cbs1,
        verbose=1,
    )

    # ── Stage 2: full fine-tune ───────────────────────────────────────────────
    unfreeze_all(model)
    log.info("Stage 2 — all layers trainable | lr = %.0e", CFG.lr_backbone)

    compile_model(model, CFG.lr_backbone)
    warmup2 = LinearWarmupCallback(CFG.warmup_epochs, CFG.lr_backbone)
    cbs2    = make_callbacks(ckpt_path, monitor, mode, CFG.patience) + [warmup2]

    log.info("Stage 2: up to %d epochs", CFG.finetune_epochs)
    hist2 = model.fit(
        tf_train,
        validation_data=tf_val,
        epochs=CFG.finetune_epochs,
        callbacks=cbs2,
        verbose=1,
    )

    log.info("Training complete. Best checkpoint: %s", ckpt_path)

    # ── Outputs ───────────────────────────────────────────────────────────────
    save_training_plots(hist1, hist2, out_dir)
    save_config(CFG, out_dir)

    # Load best weights and evaluate
    best_model = keras.models.load_model(
        ckpt_path,
        custom_objects={
            "SmoothedWeightedBCE": SmoothedWeightedBCE,
            "FBetaMetric":         FBetaMetric,
        },
    )
    evaluate(best_model, tf_val, out_dir)


if __name__ == "__main__":
    train()