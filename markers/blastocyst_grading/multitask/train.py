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
from models.encoder.visual_encoder import ResNetEncoder, EfficientNetB3Encoder
from data.dataset import MultiTaskDataset, multitask_collate, MultiTaskBatchSampler
from data.managment import prepare_datasets
from data.augmentation import get_train_transforms, get_val_transforms

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import PATH_DATASET


WEIGHTS_LOSS = {
    'exp':           1.0,
    'icm':           1.0,
    'te':            1.0,

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
        best_model_path="best_model.pth"
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
        self.history = {'train_loss': [],'val_loss': [],'task_losses': {}}
        
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
            encoder = self.model.module.encoder if self.multi_gpu else self.model.encoder
            for param in encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")
    
    def _setup_optimizer(self):
        """Setup optimizer with only trainable parameters."""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=self.lr)
    
    def _setup_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)      
        
    def _setup_dataloader(self):
        """Setup dataloader with custom batch sampler."""
        self.dataloader = DataLoader(
            self.dataset, 
            batch_sampler=MultiTaskBatchSampler(self.dataset, self.batch_size),
            collate_fn=multitask_collate
        )
        if len(self.dataloader) == 0:
            raise ValueError(
                "Dataloader has 0 batches. Check dataset size, batch_size, and dataset configuration."
            )
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_sampler=MultiTaskBatchSampler(self.val_dataset, self.batch_size),
                collate_fn=multitask_collate
            )
        else:
            self.val_dataloader = None
    
    def _move_batch_to_device(self, batch):
        """Move batch tensors to device."""
        images = batch.get('images', None)
        
        if images is not None:
            batch['images'] = images.to(self.device)
            for key in ['exp', 'icm', 'te']:
                if batch.get(key) is not None:
                    batch[key] = batch[key].to(self.device)
        

                
        return batch
    
    def _compute_loss(self, batch):
        """Forward pass and loss computation."""
        outputs, losses = self.model(
            images=batch.get('images'),
            exp_labels=batch.get('exp'),
            icm_labels=batch.get('icm'),
            te_labels=batch.get('te')
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
            layer_groups = [encoder.layer4, encoder.layer3, encoder.layer2, encoder.layer1]
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
    image_samples= prepare_datasets(path, 'train', args.baseline)

    val_ratio    = 0.15
    all_samples  = image_samples  # your list of dicts

    random.shuffle(all_samples)
    split        = int(len(all_samples) * (1 - val_ratio))
    train_images = all_samples[:split]
    val_images   = all_samples[split:]


    train_dataset = MultiTaskDataset(
        image_samples=train_images,
        transform=get_train_transforms() if args.data_augmentation else None
    )
    val_dataset = MultiTaskDataset(
        image_samples=val_images,  
        transform=get_val_transforms()
    )

    print(f"Train samples: {len(train_dataset)} (img: {len(train_images)}")
    print(f"Val samples:   {len(val_dataset)}   (img: {len(val_images)}")

    # Initialize model
    if args.encoder == "resnet34":
        encoder = ResNetEncoder(pretrained=True)
    elif args.encoder == "efficientnet_b3":
        encoder = EfficientNetB3Encoder(pretrained=True)
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")


    model = MultiTaskModel(encoder, embedding_dim=encoder.embedding_dim, weighted_loss=args.weighted_loss)

    model_path = Path(args.path_model) / args.encoder 
    if args.baseline:
        model_path = model_path / args.baseline

    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
        print(f"Created model directory: {model_path}")

    phase_boundary_epochs = []
    current_epoch = 0
    # --- Phase 1: Freeze encoder, train heads only ---
    print("=" * 60)
    print("PHASE 1: Training with frozen encoder")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        dataset=train_dataset,          
        val_dataset=val_dataset,        
        batch_size=args.batch_size,     
        lr=1e-3,
        device=args.device,             
        multi_gpu=args.multi_gpu,       
        freeze_encoder=True,
        best_model_path=model_path / "best_model_phase1.pth"
    )
    best_loss_phase1 = trainer.train(epochs=10, patience=5)
    current_epoch += len(trainer.history['train_loss'])
    phase_boundary_epochs.append(current_epoch)

    # --- Phase 2: Unfreeze last 2 encoder blocks ---
    print("\n" + "=" * 60)
    print("PHASE 2: Training last 2 encoder blocks")
    print("=" * 60)

    trainer.best_model_path = model_path / "best_model_phase2.pth"   
    trainer.unfreeze_encoder(layers=2, new_lr=1e-4)
    best_loss_phase2 = trainer.train(epochs=10, patience=5)
    current_epoch += len(trainer.history['train_loss']) - phase_boundary_epochs[-1]  
    phase_boundary_epochs.append(current_epoch)

    # --- Phase 3: Unfreeze full encoder ---
    print("\n" + "=" * 60)
    print("PHASE 3: Unfreeze the full encoder")
    print("=" * 60)

    trainer.best_model_path = model_path / "best_model_phase3.pth"   # FIXED: was PATH_MODELS
    trainer.unfreeze_encoder(layers='all', new_lr=1e-5)
    best_loss_phase3 = trainer.train(epochs=20, patience=7)

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
    print(f"Phase 2 best loss: {best_loss_phase2:.4f}")
    print(f"Phase 3 best loss: {best_loss_phase3:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiTask Model Training")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder architecture (default: resnet34)")
    parser.add_argument("--baseline", type=str, default=None, help="Use baseline model")    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--multi_gpu", action="store_true", help="Use DataParallel multi-GPU training")
    parser.add_argument('--weighted_loss', action='store_true', help="use weight loss for classification")
    parser.add_argument('--data_augmentation', action= 'store_true', help='whether data augmentation is applied')
    parser.add_argument('--path_model', type = str, default= 'models/markers/blastocyst_grading/multitask/')
    args = parser.parse_args()
    main(args)