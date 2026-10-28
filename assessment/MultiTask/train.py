import torch
import sys, tqdm
import argparse
import json, random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch.optim as optim
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import DataLoader
from models.multi_task import MultiTaskModel
from data.dataset import MultiTaskDataset, multitask_collate, MultiTaskBatchSampler, compute_sample_weights
from data.managment import prepare_datasets
from data.augmentation import get_train_transforms, get_val_transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import PATH_DATASET, PATH_MODELS


WEIGHTS_LOSS = {
    'blasto_str':    2.0,   # upweight segmentation
    'fragmentation': 2.0,
    'exp':           1.0,
    'icm':           1.0,
    'te':            1.0,
    'stage':         1.0,
    'ploidy':        1.0,
    'fhb':           1.0
}

class Trainer:
    """
    Flexible trainer for multi-task model supporting both single and multi-GPU training.
    """
    
    def __init__(
        self,
        model,
        dataset,
        val_dataset=None,
        batch_size=2,
        lr=1e-3,
        device='cuda',
        multi_gpu=True,
        freeze_encoder=False,
        best_model_path="best_model.pth",
        oversample=False,
        oversample_factor=1.5,
        task_weights=None,
    ):
        self.base_model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        self.freeze_encoder = freeze_encoder
        self.best_model_path = best_model_path
        self.history           = {'train_loss': [], 'val_loss': [], 'task_losses': {}}
        self.oversample        = oversample
        self.oversample_factor = oversample_factor
        self.task_weights      = task_weights
        
        # Setup model and optimizer
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_dataloader()
        
    def _setup_model(self):
        """Initialize model on device(s)."""
        self.base_model.to(self.device)
        
        if self.multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.base_model)
        else:
            print(f"Using single device: {self.device}")
            self.model = self.base_model
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            # Trainer._setup_model
            encoder = self.model.module.encoder if self.multi_gpu else self.model.encoder
            backbone = encoder.encoder if hasattr(encoder, 'proj_main') else encoder  # FEMI: ViT only
            for param in backbone.parameters():
                param.requires_grad = False

            # Trainer.unfreeze_encoder, new branch
        elif hasattr(encoder, 'proj_main'):  # FEMIEncoder: ViT blocks, deepest first
            layer_groups = list(encoder.encoder.encoder.layer)[::-1]
            print("Encoder frozen")
    
    def _setup_optimizer(self):
        """Setup optimizer with only trainable parameters."""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=self.lr)
    
    def _setup_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)      
        
    def _setup_dataloader(self):
        """Setup dataloader with custom batch sampler.
        When self.oversample=True, computes inverse-frequency weights and passes
        them to MultiTaskBatchSampler for minority-class oversampling.
        Val loader always uses uniform sampling for unbiased metrics.
        """
        if self.oversample:
            sample_weights = compute_sample_weights(
                self.dataset, task_weights=self.task_weights
            )
            self._log_weight_summary(sample_weights)
        else:
            sample_weights = None

        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=MultiTaskBatchSampler(
                self.dataset, self.batch_size,
                sample_weights=sample_weights,
                oversample_factor=self.oversample_factor,
            ),
            collate_fn=multitask_collate,
        )
        if len(self.dataloader) == 0:
            raise ValueError(
                "Dataloader has 0 batches. Check dataset size, batch_size, and dataset configuration."
            )
        # Val loader always uniform — unbiased validation metrics
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_sampler=MultiTaskBatchSampler(self.val_dataset, self.batch_size),
                collate_fn=multitask_collate,
            )
        else:
            self.val_dataloader = None
    def _log_weight_summary(self, sample_weights):
        """Print per-task class weight statistics so imbalance correction is visible."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for idx, (_, sample) in enumerate(self.dataset.samples):
            for field in ['icm', 'te', 'exp', 'stage', 'ploidy', 'fhb']:
                val = sample.get(field)
                if val is not None and not isinstance(val, list):
                    buckets[f"{field}_cls{int(val)}"].append(sample_weights[idx])
                    break
        print("  Oversampling weight summary (mean weight per class):")
        for key in sorted(buckets):
            vals = buckets[key]
            print(f"    {key:20s}: mean={sum(vals)/len(vals):.3f}  n={len(vals)}")


    def _move_batch_to_device(self, batch):
        """Move batch tensors to device."""
        images = batch.get('images', None)
        videos = batch.get('videos', None)
        
        if images is not None:
            batch['images'] = images.to(self.device)
            for key in ['mask_str', 'mask_frag', 'exp', 'icm', 'te', 'stage']:
                if batch.get(key) is not None:
                    batch[key] = batch[key].to(self.device)
        
        if videos is not None:
            batch['videos'] = videos.to(self.device)
            for key in ['ploidy', 'fhb']:
                if batch.get(key) is not None:
                    batch[key] = batch[key].to(self.device)
                
        return batch
    
    def _compute_loss(self, batch):
        """Forward pass and loss computation."""
        outputs, losses = self.model(
            images=batch.get('images'),
            videos=batch.get('videos'),
            blasto_str_labels=batch.get('mask_str'),
            fragmentation_labels=batch.get('mask_frag'),
            exp_labels=batch.get('exp'),
            icm_labels=batch.get('icm'),
            te_labels=batch.get('te'),
            stage_labels=batch.get('stage'),
            ploidy_labels=batch.get('ploidy'),
            fhb_labels=batch.get('fhb')
        )
        
        # Sum all task losses
        combined_loss =  sum(WEIGHTS_LOSS.get(k, 1.0) * v for k, v in losses.items())
        
        # For DataParallel, losses are already per-GPU; average them
        if self.multi_gpu:
            loss = combined_loss.mean()
        else:
            loss = combined_loss
            
        return loss, outputs, losses
    
    def unfreeze_encoder(self, layers='all', new_lr=1e-4):
        """
        Unfreeze encoder layers and add them to optimizer.
        
        Args:
            layers: 'all' to unfreeze everything, or an int to unfreeze last N blocks
            new_lr: learning rate for the unfrozen encoder params
        """
        encoder = self.base_model.encoder

        if layers == 'all':
            candidates = list(encoder.parameters())
        else:
            # Deepest -> shallowest layer groups, architecture-specific.
            if hasattr(encoder, 'layer4'):  # ResNetEncoder
                layer_groups = [encoder.layer4, encoder.layer3, encoder.layer2, encoder.layer1]
            elif hasattr(encoder, 'block5'):  # EfficientNetB3Encoder
                layer_groups = [encoder.block5, encoder.block4, encoder.block3, encoder.block2, encoder.block1]
            else:
                raise ValueError(
                    f"unfreeze_encoder: don't know how to unfreeze layer groups for "
                    f"encoder type {type(encoder).__name__}"
                )
            candidates = [p for group in layer_groups[:layers] for p in group.parameters()]

        # Only unfreeze params that are still frozen to avoid duplicates
        params_to_unfreeze = [p for p in candidates if not p.requires_grad]

        for p in params_to_unfreeze:
            p.requires_grad = True

        if params_to_unfreeze:
            self.optimizer.add_param_group({'params': params_to_unfreeze, 'lr': new_lr})
            print(f"Added {len(params_to_unfreeze)} new param tensors to optimizer at lr={new_lr}")
        else:
            print("No new params to unfreeze.")
    
    def train_epoch(self, epoch, total_epochs):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        task_loss_totals = {}
        num_batches = len(self.dataloader)
        if num_batches == 0:
            raise RuntimeError("Dataloader has 0 batches; cannot compute average loss.")
        
        pbar = tqdm.tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Compute loss
            loss, outputs, task_losses = self._compute_loss(batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            for k, v in task_losses.items():
                val = v.mean().item() if self.multi_gpu else v.item()
                task_loss_totals[k] = task_loss_totals.get(k, 0) + val
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print("  Per-task losses:")
        for k, v in task_loss_totals.items():
            print(f"    {k}: {v / num_batches:.4f}")
        avg_task_losses = {k: v / num_batches for k, v in task_loss_totals.items()}
        return avg_loss, avg_task_losses   
    

    def validate_epoch(self):
        """Validation loop — no gradients, no optimizer step."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_batch_to_device(batch)
                loss, _, _ = self._compute_loss(batch)
                total_loss += loss.item()
        return total_loss / len(self.val_dataloader)
    
    def save_checkpoint(self, path):
        """Save model checkpoint (unwrap DataParallel if needed)."""
        state_dict = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint (handle DataParallel prefix)."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Remove 'module.' prefix if present
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            checkpoint = new_state_dict
        
        # Load into base model (not wrapped version)
        self.base_model.load_state_dict(checkpoint)
        print(f"Model loaded from {path}")
    
    def train(
        self, 
        epochs=10, 
        early_stopping=True, 
        patience=5, 
        min_delta=0.0
    ):
        """
        Main training loop with early stopping.
        
        Args:
            epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change in loss to count as improvement
            
        Returns:
            best_loss: Best validation loss achieved
        """
        print(f"\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Multi-GPU: {self.multi_gpu}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Encoder frozen: {self.freeze_encoder}")
        print(f"Validation set: {'Yes' if self.val_dataloader else 'No'}\n")
        
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            train_loss, avg_task_losses = self.train_epoch(epoch, epochs)
            # Record train loss
            self.history['train_loss'].append(train_loss)
            # Record per-task train losses
            for task, loss in avg_task_losses.items():
                if task not in self.history['task_losses']:
                    self.history['task_losses'][task] = []
                self.history['task_losses'][task].append(loss)

            if self.val_dataloader is not None:
                val_loss = self.validate_epoch()
                self.history['val_loss'].append(val_loss)
                monitor_loss = val_loss
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            else:
                monitor_loss = train_loss
                print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}")

            self.scheduler.step(monitor_loss)
            # Early stopping now tracks monitor_loss (val if available, else train)
            if monitor_loss < (best_loss - min_delta):
                best_loss = monitor_loss
                epochs_no_improve = 0
                self.save_checkpoint(self.best_model_path)
            else:
                epochs_no_improve += 1
                if early_stopping and epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

            torch.cuda.empty_cache()
        
        # Load best model
        print(f"\nLoading best model (loss: {best_loss:.4f})")
        self.load_checkpoint(self.best_model_path)
        
        return best_loss
    
    def save_history(self, path=None):
        """Save loss history to a JSON file."""
        
        path = path or self.best_model_path.parent / "loss_history.json"
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Loss history saved to {path}")

    def plot_history(self, path=None, phase_epochs=None):
        """
        Plot and save train/val loss curves + per-task losses.
        
        Args:
            path: where to save the plot (default: next to best model)
            phase_epochs: list of epoch indices where phases changed
                        e.g. [10, 20] to draw vertical lines at epoch 10 and 20
        """
        

        path = path or self.best_model_path.parent / "loss_curves.png"
        
        n_tasks = len(self.history['task_losses'])
        n_plots = 1 + (1 if n_tasks > 0 else 0)   # combined loss + per-task losses
        
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        epochs_range = range(1, len(self.history['train_loss']) + 1)

        # --- Plot 1: Combined train/val loss ---
        ax = axes[0]
        ax.plot(epochs_range, self.history['train_loss'], label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax.plot(epochs_range, self.history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')

        # Draw vertical lines at phase boundaries
        if phase_epochs:
            for pe in phase_epochs:
                ax.axvline(x=pe, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
                ax.text(pe + 0.2, ax.get_ylim()[1] * 0.95, f'Phase\nchange', 
                        color='red', fontsize=8, va='top')

        ax.set_title('Combined Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Per-task losses ---
        if n_tasks > 0:
            ax2 = axes[1]
            for task_name, losses in self.history['task_losses'].items():
                task_epochs = range(1, len(losses) + 1)
                ax2.plot(task_epochs, losses, label=task_name, linewidth=2)

            if phase_epochs:
                for pe in phase_epochs:
                    ax2.axvline(x=pe, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

            ax2.set_title('Per-Task Losses')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to {path}")


def main(args):
    """Main training script."""
    # Setup
    """Main training script."""
    path = Path(PATH_DATASET)
    image_samples, video_samples = prepare_datasets(path, args.transformer, 'train', args.tasks, args.frames)

    val_ratio    = 0.15
    all_samples  = image_samples  # your list of dicts

    random.shuffle(all_samples)
    split        = int(len(all_samples) * (1 - val_ratio))
    train_images = all_samples[:split]
    val_images   = all_samples[split:]

    # Split video samples
    random.shuffle(video_samples)
    vid_split    = int(len(video_samples) * (1 - val_ratio))
    train_videos = video_samples[:vid_split]
    val_videos   = video_samples[vid_split:]

    size =(224,224)

    train_dataset = MultiTaskDataset(
        image_samples=train_images,
        video_samples=train_videos,
        img_size=size,
        transform=get_train_transforms(size, normalize=args.encoder != "femi") if args.data_augmentation else None,
        use_femi_processor=args.encoder == "femi",
    )
    val_dataset = MultiTaskDataset(
        image_samples=val_images,
        video_samples=val_videos,    
        img_size=size,
        transform=get_val_transforms(normalize=args.encoder != "femi"),
        use_femi_processor=args.encoder == "femi",
    )

    print(f"Train samples: {len(train_dataset)} (img: {len(train_images)}, vid: {len(train_videos)})")
    print(f"Val samples:   {len(val_dataset)}   (img: {len(val_images)},   vid: {len(val_videos)})")

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

    if args.transformer:
        from models.temporal.temporal_transformer import VideoTransformer
        transformer = VideoTransformer(args.frames) 
    else:
        transformer = None
    model = MultiTaskModel(encoder, transformer, embedding_dim=encoder.embedding_dim, weighted_loss=args.weighted_loss)

    tasks_tag = "_".join(sorted(args.tasks)) if args.tasks else "all"
    model_path = Path(args.path_model) / args.encoder / ("transformer" if args.transformer else "no_transformer") / tasks_tag

    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"Created model directory: {model_path}")

    phase_boundary_epochs = []
    current_epoch = 0
    # --- Phase 1: Freeze encoder, train heads only ---
    print("=" * 60)
    print("PHASE 1: Training with frozen encoder")
    print("=" * 60)
    import json as _json
    _task_weights = _json.loads(args.task_weights) if args.task_weights else None

    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=1e-3,
        device=args.device,
        multi_gpu=False,
        freeze_encoder=True,
        best_model_path=model_path / "best_model_phase1.pth",
        oversample=args.oversample,
        oversample_factor=args.oversample_factor,
        task_weights=_task_weights,
    )
    best_loss_phase1 = trainer.train(epochs=20, patience=5)
    current_epoch += len(trainer.history['train_loss'])
    phase_boundary_epochs.append(current_epoch)

    if args.phase2: 
        # --- Phase 2: Unfreeze last 2 encoder blocks ---
        print("\n" + "=" * 60)
        print("PHASE 2: Training last 2 encoder blocks")
        print("=" * 60)

        trainer.best_model_path = model_path / "best_model_phase2.pth"   
        trainer.unfreeze_encoder(layers=1, new_lr=1e-4)
        best_loss_phase2 = trainer.train(epochs=5, patience=5)
        current_epoch += len(trainer.history['train_loss']) - phase_boundary_epochs[-1]  
        phase_boundary_epochs.append(current_epoch)

    if args.phase3:
        # --- Phase 3: Unfreeze full encoder ---
        print("\n" + "=" * 60)
        print("PHASE 3: Unfreeze the full encoder")
        print("=" * 60)

        trainer.best_model_path = model_path / "best_model_phase3.pth"   # FIXED: was PATH_MODELS
        trainer.unfreeze_encoder(layers=2, new_lr=1e-5)
        best_loss_phase3 = trainer.train(epochs=5, patience=7)

    # Save and plot at the very end
    trainer.save_history(model_path / "loss_history.json")
    trainer.plot_history(
        path=model_path / "loss_curves.png",
        phase_epochs=phase_boundary_epochs
    )
    # Save final model
    final_model_path = model_path / f"multitask_model_final_{best_loss_phase3:.4f}.pth"
    trainer.save_checkpoint(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    print(f"Phase 1 best loss: {best_loss_phase1:.4f}")
    if args.phase2:
        print(f"Phase 2 best loss: {best_loss_phase2:.4f}")
    if args.phase3:
        print(f"Phase 3 best loss: {best_loss_phase3:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiTask Model Training")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder architecture (default: resnet34)")
    parser.add_argument("--tasks", nargs="+",default=None,help=(
            "Tasks to include in training. Pass one or more of: "
            "exp icm te seg frag stage ploidy fhb. "
            "Omit to include all tasks (default behaviour)."
        ),
    )
    parser.add_argument("--transformer", action="store_true", help="Use transformer model")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames for video input (default: 16)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--multi_gpu", action="store_true", help="Use DataParallel multi-GPU training")
    parser.add_argument('--weighted_loss', action='store_true', help="use weight loss for classification")
    parser.add_argument('--data_augmentation', action= 'store_true', help='whether data augmentation is applied')
    parser.add_argument('--path_model', type = str, default= PATH_MODELS + '/assessment/Multitask/')
    parser.add_argument('--oversample', action='store_true', help='Oversample minority classes using inverse-frequency weights.')
    parser.add_argument('--oversample_factor', type=float, default=1.5,help='Epoch size multiplier when oversampling (default 1.5).')
    parser.add_argument('--task_weights',type=str, default=None,help='JSON dict of task multipliers, e.g. \'{"icm":2.0,"te":2.0}\'')
    parser.add_argument('--phase2', action='store_true', help='Whether to run phase 2 (unfreeze last encoder block)')
    parser.add_argument('--phase3', action='store_true', help='Whether to run phase 3 (unfreeze full encoder)')
    args = parser.parse_args()
    main(args)