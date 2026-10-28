import pandas as pd 
import torch
import sys
import cv2
import argparse
from pathlib import Path
import numpy as np
from collections import OrderedDict
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from PIL import Image
import time

from models.multi_task import MultiTaskModel
from models.encoder.visual_encoder import ResNetEncoder, EfficientNetB3Encoder
from models.temporal.temporal_transformer import VideoTransformer
from data.dataset import transform_image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import OUTPUT_PATH


def color_inference(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Create colored overlay of mask on image.
    
    Args:
        image: Original image (H, W, 3) in RGB or BGR format
        mask: Binary or multi-class mask (H, W)
        color: RGB color for the mask overlay (default: red)
        
    Returns:
        Blended image with colored mask overlay
    """
    # Ensure image is 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert mask to 3 channels
    mask_3c = np.stack([mask] * 3, axis=-1)
    
    # Create color overlay
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[mask > 0] = color
    
    # Blend original image and color mask
    blended = cv2.addWeighted(image, 0.8, color_mask, 0.5, 0)
    
    return blended


class MultiTaskInference:
    """
    Inference engine for multi-task model supporting both single and multi-GPU.
    """
    
    def __init__(
        self,
        encoder: str,
        checkpoint_path: str,
        device: str = 'cuda',
        multi_gpu: bool = False,
        batch_size: int = 8,
        use_transformer: bool = True,
        tasks: Optional[List[str]] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            encoder: Encoder architecture name (e.g., 'resnet34')
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            multi_gpu: Whether to use multiple GPUs
            batch_size: Batch size for inference
            use_transformer: Whether to enable transformer for video processing
            tasks: Heads to run (e.g. ['icm']); None runs all heads
        """
        self.tasks = tasks
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.multi_gpu = multi_gpu and torch.cuda.device_count() > 1
        self.batch_size = batch_size
        self.use_transformer = use_transformer
        
        # Timings (seconds)
        self.timings = {
            'model_loading': 0.0,
            'data_loading': 0.0,
            'inference': 0.0,
            'n_samples': 0,
        }
        
        # Initialize model
        t0 = time.perf_counter()
        self.model = self._load_model()
        self._sync()
        self.timings['model_loading'] = time.perf_counter() - t0
        
        print(f"Inference engine initialized:")
        print(f"  Device: {self.device}")
        print(f"  Multi-GPU: {self.multi_gpu}")
        if self.multi_gpu:
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Transformer enabled: {self.use_transformer}")
        print(f"  Tasks: {self.tasks if self.tasks else 'all'}")

    def _sync(self):
        """Wait for pending GPU work so timings are accurate."""
        if str(self.device).startswith('cuda'):
            torch.cuda.synchronize()
    
    def print_timings(self):
        """Print a summary of the collected timings."""
        t = self.timings
        n = max(t['n_samples'], 1)
        print("\n" + "="*80)
        print("TIMING SUMMARY")
        print("="*80)
        print(f"  Model loading : {t['model_loading']:.3f} s")
        print(f"  Data loading  : {t['data_loading']:.3f} s ({t['data_loading'] / n * 1000:.1f} ms/sample)")
        print(f"  Inference     : {t['inference']:.3f} s ({t['inference'] / n * 1000:.1f} ms/sample)")
        print(f"  Samples       : {t['n_samples']}")
        print("="*80)
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        start_time = time.time()
        # Initialize model architecture
        if args.encoder == "resnet34":
            encoder = ResNetEncoder(pretrained=False)
        elif args.encoder == "efficientnet_b3":
            encoder = EfficientNetB3Encoder(pretrained=False)
        else:
            raise ValueError(f"Unknown encoder: {args.encoder}")
        
        # Conditionally initialize transformer
        if self.use_transformer:
            transformer = VideoTransformer(args.n_frames)
            print("Transformer enabled for video processing")
        else:
            transformer = None
            print("Transformer disabled - video processing unavailable")
        
        model = MultiTaskModel(encoder, transformer, embedding_dim=512)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle DataParallel checkpoint (remove 'module.' prefix)
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            checkpoint = new_state_dict
        
        checkpoint = {k: v for k, v in checkpoint.items() if not k.endswith("loss_fn.pos_weight")}

        model.load_state_dict(checkpoint)
        model.to(self.device)
        
        # Wrap with DataParallel if using multi-GPU
        if self.multi_gpu:
            model = nn.DataParallel(model)
            print(f"Model wrapped with DataParallel")
        
        model.eval()
        print(f"Model loaded from {self.checkpoint_path}")
        end_time = time.time()
        print(f"Checkpoint loaded in {end_time - start_time:.6f} seconds")
        
        return model
    
    @staticmethod
    def load_image(image_path: Path, size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            size: Target size (height, width)
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        img = cv2.imread(str(image_path))
        img = transform_image(img, size)
        return img
    
    @staticmethod
    def load_video(video_path: Path, n_frames: int = 50) -> torch.Tensor:
        """
        Load and preprocess a single video.
        
        Args:
            video_path: Path to video file
            n_frames: Number of frames to sample
            
        Returns:
            Preprocessed video tensor [T, C, H, W]
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, total_frames - 1, n_frames).astype(int)
        
        frames = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        cap.release()
        video_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
        return video_tensor
    
    def load_images_batch(
        self, 
        image_paths: List[Path], 
        size: Tuple[int, int] = (512, 512)
    ) -> Tuple[torch.Tensor, List[Path]]:
        """
        Load multiple images into a batch.
        
        Args:
            image_paths: List of paths to image files
            size: Target size (height, width)
            
        Returns:
            Batch of images [B, C, H, W] and corresponding paths
        """
        images = []
        valid_paths = []
        
        for img_path in tqdm(image_paths, desc="Loading images"):
            try:
                img = self.load_image(img_path, size)
                images.append(img)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        if not images:
            return None, []
        
        batch = torch.stack(images, dim=0)
        return batch, valid_paths
    
    def load_videos_batch(
        self, 
        video_paths: List[Path], 
        n_frames: int = 50
    ) -> Tuple[torch.Tensor, List[Path]]:
        """
        Load multiple videos into a batch.
        
        Args:
            video_paths: List of paths to video files
            n_frames: Number of frames to sample per video
            
        Returns:
            Batch of videos [B, T, C, H, W] and corresponding paths
        """
        videos = []
        valid_paths = []
        
        for vid_path in tqdm(video_paths, desc="Loading videos"):
            try:
                vid = self.load_video(vid_path, n_frames)
                videos.append(vid)
                valid_paths.append(vid_path)
            except Exception as e:
                print(f"Error loading {vid_path}: {e}")
        
        if not videos:
            return None, []
        
        batch = torch.stack(videos, dim=0)
        return batch, valid_paths
    
    @torch.no_grad()
    def predict_batch(
        self, 
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on a batch of images and/or videos.
        
        Args:
            images: Batch of images [B, C, H, W] or None
            videos: Batch of videos [B, T, C, H, W] or None
            
        Returns:
            Dictionary of predictions
        """
        # Move to device
        if images is not None:
            images = images.to(self.device)
        if videos is not None:
            videos = videos.to(self.device)
        
        # Run inference
        outputs, _ = self.model(images=images, videos=videos, tasks=self.tasks)

        return outputs
    
    def profile_model(
        self,
        output_dir: Path,
        image_size: Tuple[int, int] = (512, 512),
        n_frames: int = 16
    ) -> List[Dict]:
        """
        Measure parameters, FLOPs and memory for the multitask model and for
        each task on its own (shared encoder + that task's head only).

        Measured with batch size 1 and random input of the given size.
        - params_M:       parameters of the modules the configuration uses
        - flops_G:        FLOPs of one forward pass, as counted by PyTorch
                          (2 x multiply-accumulates; only conv/matmul/attention ops)
        - weights_MB:     size of the used parameters and buffers
        - activation_MB:  extra peak GPU memory during the forward pass (CUDA only)
        - peak_MB:        weights_MB + activation_MB (CUDA only)
        """
        from torch.utils.flop_counter import FlopCounterMode

        base = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        on_cuda = str(self.device).startswith('cuda')

        image_heads = {
            'exp': base.exp_head, 'icm': base.icm_head, 'te': base.te_head,
            'stage': base.stage_head, 'blasto_str': base.blasto_str_head,
            'fragmentation': base.fragmentation_head,
        }
        video_heads = {'ploidy': base.ploidy_head, 'fhb': base.fhb_head}
        video_shared = [base.encoder]
        if base.video_transformer is not None:
            video_shared.append(base.video_transformer)

        # (name, input type, tasks passed to forward, modules the configuration uses)
        configs = [
            ('multitask_images', 'image', None, [base.encoder, *image_heads.values()]),
            ('multitask_videos', 'video', None, video_shared + list(video_heads.values())),
        ]
        configs += [(t, 'image', [t], [base.encoder, h]) for t, h in image_heads.items()]
        configs += [(t, 'video', [t], video_shared + [h]) for t, h in video_heads.items()]

        def n_params(modules):
            return sum(p.numel() for m in modules for p in m.parameters())

        def n_bytes(modules):
            return sum(t.numel() * t.element_size()
                       for m in modules for t in list(m.parameters()) + list(m.buffers()))

        rows = []
        for name, kind, tasks, modules in configs:
            if kind == 'image':
                kwargs = {'images': torch.randn(1, 3, *image_size, device=self.device)}
            else:
                kwargs = {'videos': torch.randn(1, n_frames, 3, *image_size, device=self.device)}

            with torch.no_grad():
                base(tasks=tasks, **kwargs)  # warm-up
                self._sync()

                with FlopCounterMode(display=False) as flop_counter:
                    base(tasks=tasks, **kwargs)
                flops = flop_counter.get_total_flops()

                activation_mb = peak_mb = None
                weights_mb = n_bytes(modules) / 2**20
                if on_cuda:
                    self._sync()
                    mem_before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    base(tasks=tasks, **kwargs)
                    self._sync()
                    activation_mb = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
                    peak_mb = weights_mb + activation_mb

            rows.append({
                'config': name,
                'input': kind,
                'params_M': n_params(modules) / 1e6,
                'flops_G': flops / 1e9,
                'weights_MB': weights_mb,
                'activation_MB': activation_mb,
                'peak_MB': peak_mb,
            })

        total_params = sum(p.numel() for p in base.parameters()) / 1e6
        print("\n" + "="*80)
        print(f"MODEL PROFILE (batch 1, input {image_size[0]}x{image_size[1]}, {n_frames} frames for videos)")
        print("="*80)
        print(f"  Total parameters in checkpoint: {total_params:.2f} M")
        for r in rows:
            mem = f"{r['peak_MB']:.1f} MB" if r['peak_MB'] is not None else "n/a (CPU)"
            print(f"  {r['config']:<18} params {r['params_M']:.2f} M | FLOPs {r['flops_G']:.2f} G | peak memory {mem}")
        print("="*80)

        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "profile.csv", index=False)
        return rows

    def diagnose_bn_gap(
        self,
        image_paths: List[Path],
        output_dir: Path,
        n: int = 4,
        size: Tuple[int, int] = (512, 512)
    ):
        """
        Compare predictions in train vs eval mode on the same batch.
        Helps diagnose BatchNorm train/eval gap issues.

        Args:
            image_paths: List of image paths to sample from
            output_dir: Directory to save the diagnostic plot
            n: Number of images to visualize
            size: Image size
        """
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample n images
        sample_paths = image_paths[:n]
        batch_images, valid_paths = self.load_images_batch(sample_paths, size)
        if batch_images is None:
            print("No images loaded for diagnosis.")
            return

        batch_images = batch_images.to(self.device)

        # Get base model (unwrap DataParallel if needed)
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        with torch.no_grad():
            # --- Train mode prediction ---
            base_model.train()
            outputs_train, _ = base_model(images=batch_images)
            pred_train_str  = torch.argmax(outputs_train['blasto_str'], dim=1).cpu()
            pred_train_frag = (torch.sigmoid(outputs_train['fragmentation'][:, 0]) > 0.5).cpu().numpy().astype('uint8')

            # --- Eval mode prediction ---
            base_model.eval()
            outputs_eval, _ = base_model(images=batch_images)
            pred_eval_str  = torch.argmax(outputs_eval['blasto_str'], dim=1).cpu()
            pred_eval_frag = (torch.sigmoid(outputs_eval['fragmentation'][:, 0]) > 0.5).cpu().numpy().astype('uint8')

        # --- Plot ---
        n_actual = len(valid_paths)
        fig, axes = plt.subplots(n_actual, 5, figsize=(18, 4 * n_actual))
        if n_actual == 1:
            axes = [axes]

        col_titles = ['Image', 'blasto_str\n(train mode)', 'blasto_str\n(eval mode)',
                    'fragmentation\n(train mode)', 'fragmentation\n(eval mode)']
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=10)

        for i in range(n_actual):
            img_display = batch_images[i].cpu().permute(1, 2, 0).numpy()
            # Denormalize from ImageNet stats for display
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_display = (img_display * std + mean).clip(0, 1)

            axes[i][0].imshow(img_display)
            axes[i][1].imshow(pred_train_str[i],  cmap='jet', vmin=0, vmax=3)
            axes[i][2].imshow(pred_eval_str[i],   cmap='jet', vmin=0, vmax=3)
            axes[i][3].imshow(pred_train_frag[i], cmap='gray', vmin=0, vmax=1)
            axes[i][4].imshow(pred_eval_frag[i],  cmap='gray', vmin=0, vmax=1)

            axes[i][0].set_ylabel(valid_paths[i].name, fontsize=7)
            for ax in axes[i]:
                ax.axis('off')

        # Summary text
        str_match  = (pred_train_str == pred_eval_str).float().mean().item()
        frag_match = (pred_train_frag == pred_eval_frag).mean()
        fig.suptitle(
            f"BN Gap Diagnosis | blasto_str pixel agreement: {str_match:.1%} | "
            f"fragmentation pixel agreement: {frag_match:.1%}\n"
            f"(100% agreement = no BN gap, low % = BatchNorm issue)",
            fontsize=11, y=1.01
        )

        plt.tight_layout()
        save_path = output_dir / "bn_gap_diagnosis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nBN Gap Diagnosis:")
        print(f"  blasto_str  pixel agreement (train vs eval): {str_match:.1%}")
        print(f"  fragmentation pixel agreement (train vs eval): {frag_match:.1%}")
        print(f"  Plot saved to: {save_path}")

        if str_match < 0.95 or frag_match < 0.95:
            print("\n  ⚠️  WARNING: Significant train/eval gap detected!")
            print("     → Apply SyncBatchNorm or GroupNorm fix in your model.")
        else:
            print("\n  ✅  No significant BN gap detected.")

        # Restore eval mode
        base_model.eval()

    def process_images(
        self, 
        image_paths: List[Path],
        output_dir: Path,
        size: Tuple[int, int] = (512, 512)
    ) -> List[Dict]:
        """
        Process multiple images with batching.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            size: Target size for images
            
        Returns:
            List of result dictionaries
        """
        # Create subdirectories for outputs
        predictions_dir = output_dir / "predictions"
        color_dir = output_dir / "color"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        color_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        # Define colors for different mask types (BGR format for cv2)
        colors = {
            'fragmentation': (0, 0, 255),      # Red in BGR
            'blasto_str': {
                0: (0, 0, 0),                   # Background - Black
                1: (255, 0, 0),                 # ZP - Blue
                2: (0, 255, 0),                 # ICM - Green
                3: (0, 0, 255)                  # TE - Red
            }
        }
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Load batch
            t0 = time.perf_counter()
            batch_images, valid_paths = self.load_images_batch(batch_paths, size)
            self.timings['data_loading'] += time.perf_counter() - t0
            if batch_images is None:
                continue
            
            # Run inference (includes host-to-device transfer)
            t0 = time.perf_counter()
            outputs = self.predict_batch(images=batch_images)
            self._sync()
            self.timings['inference'] += time.perf_counter() - t0
            self.timings['n_samples'] += len(valid_paths)
            
            # Process each image in batch
            for j, img_path in enumerate(valid_paths):
                result = {
                    'path': img_path.name,
                    'type': 'image'
                }
                
                # Load original image
                original_img = cv2.imread(str(img_path))
                H, W = original_img.shape[:2]
                
                # Extract predictions
                if 'exp' in outputs:
                    result['exp'] = outputs['exp'][j].argmax(dim=0).cpu().item()
                if 'icm' in outputs:
                    result['icm'] = outputs['icm'][j].argmax(dim=0).cpu().item()
                if 'te' in outputs:
                    result['te'] = outputs['te'][j].argmax(dim=0).cpu().item()
                
                result['ploidy'] = None
                
                # Process fragmentation segmentation
                if 'fragmentation' in outputs:
                    # Get binary mask
                    mask = (torch.sigmoid(outputs['fragmentation'][j][0]) > 0.5).cpu().numpy().astype('uint8') * 255
                    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    # Save prediction mask
                    mask_path = predictions_dir / f"fragmentation_{img_path.stem}.png"
                    cv2.imwrite(str(mask_path), mask_resized)
                    result['fragmentation_mask'] = mask_path.name
                    
                    # Create and save colored overlay
                    colored = color_inference(original_img, mask_resized, colors['fragmentation'])
                    color_path = color_dir / f"fragmentation_{img_path.stem}.png"
                    cv2.imwrite(str(color_path), colored)
                    result['fragmentation_colored'] = color_path.name
                
                # Process blastocyst structure segmentation
                if 'blasto_str' in outputs:
                    # Get multiclass mask
                    mask = outputs['blasto_str'][j].argmax(dim=0).cpu().numpy().astype('uint8')
                    mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                    
                    # Save prediction mask (with class values scaled for visibility)
                    mask_vis = (mask_resized * 85).astype('uint8')  # Scale to 0, 85, 170, 255
                    mask_path = predictions_dir / f"blasto_str_{img_path.stem}.png"
                    cv2.imwrite(str(mask_path), mask_vis)
                    result['blasto_str_mask'] = mask_path.name
                    
                    # Create colored overlay with different colors for each class
                    colored = np.copy(original_img)
                    for class_id, color in colors['blasto_str'].items():
                        if class_id == 0:  # Skip background
                            continue
                        class_mask = (mask_resized == class_id).astype('uint8')
                        colored = color_inference(colored, class_mask, color)
                    
                    color_path = color_dir / f"blasto_str_{img_path.stem}.png"
                    cv2.imwrite(str(color_path), colored)
                    result['blasto_str_colored'] = color_path.name
                
                results.append(result)
                
                # Print individual result
                print(f"\nResults for {img_path.name}:")
                print(f"  EXP: {result.get('exp', 'N/A')}")
                print(f"  ICM: {result.get('icm', 'N/A')}")
                print(f"  TE: {result.get('te', 'N/A')}")
                if 'fragmentation_mask' in result:
                    print(f"  Fragmentation mask saved: {result['fragmentation_mask']}")
                if 'blasto_str_mask' in result:
                    print(f"  Blastocyst structure mask saved: {result['blasto_str_mask']}")
        
        return results
    
    def process_videos(
        self, 
        video_paths: List[Path],
        n_frames: int = 50
    ) -> List[Dict]:
        """
        Process multiple videos with batching.
        
        Args:
            video_paths: List of video paths
            n_frames: Number of frames to sample per video
            
        Returns:
            List of result dictionaries
        """
        # Check if transformer is enabled
        if not self.use_transformer:
            print("ERROR: Cannot process videos - transformer is disabled")
            print("Please enable transformer with --transformer flag")
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(video_paths), self.batch_size):
            batch_paths = video_paths[i:i + self.batch_size]
            
            # Load batch
            t0 = time.perf_counter()
            batch_videos, valid_paths = self.load_videos_batch(batch_paths, n_frames)
            self.timings['data_loading'] += time.perf_counter() - t0
            if batch_videos is None:
                continue
            
            # Run inference (includes host-to-device transfer)
            t0 = time.perf_counter()
            outputs = self.predict_batch(videos=batch_videos)
            self._sync()
            self.timings['inference'] += time.perf_counter() - t0
            self.timings['n_samples'] += len(valid_paths)
            
            # Process each video in batch
            for j, vid_path in enumerate(valid_paths):
                result = {
                    'path': vid_path.name,
                    'type': 'video'
                }
                
                # Extract ploidy prediction
                if 'ploidy' in outputs:
                    ploidy_pred = (torch.sigmoid(outputs['ploidy'][j]) > 0.5).cpu().item()
                    result['ploidy'] = int(ploidy_pred)
                else:
                    result['ploidy'] = None
                
                result['exp'] = None
                result['icm'] = None
                result['te'] = None
                
                results.append(result)
                
                # Print individual result
                print(f"\nResults for {vid_path.name}:")
                print(f"  Ploidy: {result['ploidy']}")
        
        return results


def get_file_paths(input_path: str, extensions: List[str]) -> List[Path]:
    """
    Get list of file paths from input (file or directory).
    
    Args:
        input_path: Path to file or directory
        extensions: List of valid file extensions
        
    Returns:
        List of file paths
    """
    path = Path(input_path)
    
    if path.is_dir():
        files = [f for f in path.iterdir() if f.suffix.lower() in extensions]
    elif path.is_file():
        files = [path] if path.suffix.lower() in extensions else []
    else:
        files = []
    
    return sorted(files)


def main(args):
    """Main inference routine."""
    # Validate inputs
    if args.input_videos is not None and not args.transformer:
        print("WARNING: Video input provided but transformer is disabled")
        print("Videos will not be processed. Enable with --transformer flag")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    inferencer = MultiTaskInference(
        encoder=args.encoder,
        checkpoint_path=args.checkpoint,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        multi_gpu=args.multi_gpu,
        batch_size=args.batch_size,
        use_transformer=args.transformer,
        tasks=args.tasks
    )
    
    if args.profile:
        inferencer.profile_model(
            output_dir,
            image_size=(args.image_size, args.image_size),
            n_frames=args.n_frames
        )

    all_results = []
    if args.diagnose:
        if args.input_images is None:
            print("ERROR: --diagnose requires --input_images to be provided")
        else:
            print("\n" + "="*80)
            print("RUNNING BATCHNORM GAP DIAGNOSIS")
            print("="*80)
            image_paths = get_file_paths(args.input_images, ['.png', '.jpg', '.jpeg', '.bmp'])
            if image_paths:
                inferencer.diagnose_bn_gap(
                    image_paths=image_paths,
                    output_dir=output_dir / "diagnosis",
                    n=args.diagnose_n,
                    size=(args.image_size, args.image_size)
                )
            else:
                print("No valid images found for diagnosis.")
            print("="*80 + "\n")

    

    
    # Process images
    if args.input_images is not None:
        print("\n" + "="*80)
        print("PROCESSING IMAGES")
        print("="*80)
        
        image_paths = get_file_paths(
            args.input_images, 
            ['.png', '.jpg', '.jpeg', '.bmp']
        )
        
        if image_paths:
            print(f"Found {len(image_paths)} images")
            image_results = inferencer.process_images(
                image_paths, 
                output_dir,
                size=(args.image_size, args.image_size)
            )
            all_results.extend(image_results)
        else:
            print("No valid images found")
    
    # Process videos
    if args.input_videos is not None:
        print("\n" + "="*80)
        print("PROCESSING VIDEOS")
        print("="*80)
        
        video_paths = get_file_paths(
            args.input_videos, 
            ['.mp4', '.avi', '.mov']
        )
        
        if video_paths:
            print(f"Found {len(video_paths)} videos")
            video_results = inferencer.process_videos(
                video_paths,
                n_frames=args.n_frames
            )
            all_results.extend(video_results)
        else:
            print("No valid videos found")
    
    # Save results to CSV
    if all_results:
        results_file = output_dir / "results_inference.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(results_file, index=False)
        print("\n" + "="*80)
        print(f"Results saved to {results_file}")
        print("="*80)
        print(f"\nProcessed {len(all_results)} files total")
    else:
        print("\nNo results to save")
    
    # Timing summary (saved so it can be compared against the baseline)
    if inferencer.timings['n_samples'] > 0:
        inferencer.print_timings()
        pd.DataFrame([inferencer.timings]).to_csv(output_dir / "timings.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiTask Model Inference with Multi-GPU Support")

    # Input/Output
    parser.add_argument("--input_images", type=str, default=None, help="Path to input images or directory containing images")
    parser.add_argument("--input_videos", type=str, default=None, help="Path to input videos or directory containing videos")
    parser.add_argument("--encoder", type=str, default="resnet34", help="Encoder architecture (default: resnet34)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_PATH + "/assessment/MultiTask/inference/", help="Directory to save inference results")

    # Processing parameters
    parser.add_argument("--n_frames", type=int, default=16, help="Number of frames to sample from each video")
    parser.add_argument("--image_size", type=int, default=512, help="Target size for images (width and height)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")

    # GPU settings
    parser.add_argument("--multi_gpu", action="store_true", help="Use multiple GPUs if available")

    # Model settings
    parser.add_argument("--transformer", action="store_true", help="Enable transformer for video processing (required for video inference)")
    parser.add_argument("--tasks", nargs="+", default=None, choices=["exp", "icm", "te", "stage", "blasto_str", "fragmentation", "ploidy", "fhb"], help="Run only these heads (skipped heads cost no compute). Default: all heads")
    parser.add_argument("--profile", action="store_true", help="Measure parameters, FLOPs and memory for the multitask model and each task alone")
    parser.add_argument("--diagnose", action="store_true", help="Run BatchNorm train/eval gap diagnosis before inference")
    parser.add_argument("--diagnose_n", type=int, default=4, help="Number of images to use for BN diagnosis (default: 4)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.input_images is None and args.input_videos is None and not args.profile:
        parser.error("At least one of --input_images, --input_videos or --profile must be provided")
    
    main(args)