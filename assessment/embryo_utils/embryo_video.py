"""
Embryo Video Analysis Class

This module provides functionality for analyzing embryo development videos,
including temporal tracking of morphological features, division timing,
and blastocyst formation detection.
"""
"""
Embryo Video Analysis Class

This module provides functionality for analyzing embryo development videos,
including temporal tracking of morphological features, division timing,
and blastocyst formation detection.
"""

import cv2,os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from assessment.embryo_utils.crop import coord_embyo
from assessment.embryo_utils.seg_model import predict_mask
from assessment.embryo_utils.functions import intersection, counting_pronuclei, n_cells_cleaning
#from sam_segmenting import counting_polar_corpuscles
from assessment.embryo_utils.embryo_img import EmbryoImage


class EmbryoVideo:
    """
    A class for analyzing embryo development videos with temporal tracking.
    
    This class processes video files to track embryo development over time,
    extracting morphological features at each frame and identifying key
    developmental events (cell divisions, blastocyst formation).
    
    Attributes:
        path (str): Path to the video file
        fps (float): Frames per second of the video
        n_MAX (int): Maximum number of frames to process
        video_frames (List[np.ndarray]): List of video frames
        cropped_video_frames (List[np.ndarray]): List of cropped frames
        segmented_frames (List[np.ndarray]): List of segmentation masks
        values (Dict[str, List]): Time-series data of morphological features
        
    Example:
        >>> video = EmbryoVideo(path="embryo_dev.mp4", n_MAX=1000)
        >>> evolution = video.get_evolution(crop_video=True)
        >>> blasto_time = video.get_blasto_formation(real_segs=3000)
        >>> video.save_evolution(evolution, "results.csv")
    """
    
    # Supported video formats
    SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Default features to track
    DEFAULT_FEATURES = [
        'diameter', 'area', 'ZP_R', 'ZP_r', 'ZP_thickness', 'ZP_area', 'ZP_symmetry',
        'TE_area', 'TE_area_ratio', 'TE_fractal_d', 'TE_mean_thickness',
        'ICM_area', 'ICM_area_ratio', 'ICM_eccentricity',
        'BC_area', 'BC_area_ratio', 'n_cells', 'cleavage', 'fragmentation_idx', 
        'expansion', 'te_grading', 'icm_grading', 'stage_classif'
    ]
    root_dir = ''
    PATH_SEGMENTATION_MODEL = root_dir +'models/markers/blastocyst_segmentation/segmentation_hrnet.pth'
    
    def __init__(
        self,
        path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
        mask: Optional[List[np.ndarray]] = None,
        n_MAX: int = 214,
        n_total:int = 214,
        cropped: bool = False,
        values: Optional[List[str]] = None
    ):
        """
        Initialize the EmbryoVideo object.
        
        Args:
            path: Path to the video file
            n_MAX: Maximum number of frames to process (default: 214)
            cropped: Whether the video is already cropped (default: False)
            values: List of feature names to track (default: all standard features)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If video file does not exist
        """
        if path: 
            self.path = path
            
            # Validate file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")
            
            # Validate and load video
            if not path.lower().endswith(self.SUPPORTED_FORMATS):
                raise ValueError(
                    f"Unsupported file format. Supported formats: {self.SUPPORTED_FORMATS}"
                )
            self.video = cv2.VideoCapture(path)
            if not self.video.isOpened():
                raise ValueError(f"Failed to open video file: {path}")
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            # Frame storage
            self.video_frames: List[np.ndarray] = []
            self.mask = [1] * n_MAX  # Default mask (1 = real frame)
        else: 
            if frames is None:
                raise ValueError("Either 'path' or 'frames' must be provided")  
            else:
                self.video_frames = frames[:n_MAX]
                length = len(self.video_frames)
                self.mask = mask[:n_MAX] if mask else None
                self.video = None
        
        
        self.is_cropped = cropped
        self.cropped_video_frames: List[np.ndarray] = []
        self.segmented_frames: List[np.ndarray] = []
        self.n_MAX = min(n_MAX, length)
        self.n_total= min(n_total, self.n_MAX)
        self.step = max(1, self.n_total // self.n_MAX)
        
        # Embryo tracking
        self.xxyy: Optional[Tuple[int, int, int, int]] = None  # Embryo coordinates
        
        # Developmental events
        self.blasto_frame: Optional[int] = None  # Frame where blastocyst forms
        self.blasto_formation: Optional[float] = None  # Blastocyst formation time (hours)
        self.division_times: Dict[int, float] = {}  # Division events
        self.cleavage_states: Dict[int, int] = {}  # Cleavage stage at each frame
        
        # Cell features
        self.polar_corpuscles: Optional[int] = None
        self.n_pronuclei: Optional[int] = None
        
        # Time-series data
        if values is None:
            values = self.DEFAULT_FEATURES
        self.values: Dict[str, List] = {key: [] for key in values}
    
    def __iter__(self):
        """
        Iterate over video frames.
        
        Yields:
            np.ndarray: Video frame
        """
        if not self.video_frames:
            self.get_video_frames()
        for frame in self.video_frames:
            yield frame
    
    def __len__(self) -> int:
        """
        Get the number of frames loaded.
        
        Returns:
            Number of frames currently loaded
        """
        return len(self.get_video_frames()) if self.video_frames else 0
    
    def __repr__(self) -> str:
        """String representation of the EmbryoVideo object."""
        n_frames = len(self.video_frames)
        return (
            f"EmbryoVideo(path={self.path}, "
            f"fps={self.fps:.2f}, "
            f"frames_loaded={n_frames}/{self.n_MAX})"
        )
    
    # ==================== Video Properties ====================
    
    def get_fps(self) -> float:
        """
        Get the frames per second of the video.
        
        Returns:
            FPS value
        """
        return self.fps
    
    def get_total_frames(self) -> int:
        """
        Get the total number of frames in the video.
        
        Returns:
            Total frame count
        """
        return len(self.get_video_frames()) if self.video_frames else 0
    
    
    # ==================== Frame Extraction Methods ====================
    
    def get_video_frames(self, simplify = True) -> List[np.ndarray]:
        """
        Extract all frames from the video up to n_MAX.
        
        Returns:
            List of video frames
        """
        if self.video_frames:
            return self.video_frames
        
        if simplify:
            
            with tqdm(total=min(self.n_MAX, self.get_total_frames()),desc="Extracting frames") as pbar:
                for i in range(self.n_MAX):
                    # Seek to the target frame for this sample, then read it.
                    # (Previously this read first and seeked afterward, which
                    # caused frame 0 to be read twice: once via the initial
                    # seek-to-0 before the loop, and again on iteration 1
                    # since iteration 0's seek target was also 0.)
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, i*self.step)
                    ret, frame = self.video.read() 
                    self.video_frames.append(frame)
                    pbar.update(1)
            return self.video_frames
        
        else:
            n_frame = 0
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video.read()
            
            # Get initial embryo coordinates
            if ret:
                self.xxyy = coord_embyo(frame)
            
            # Extract frames
            with tqdm(total=min(self.n_MAX, self.get_total_frames()), 
                    desc="Extracting frames") as pbar:
                while ret and n_frame < self.n_MAX:
                    self.video_frames.append(frame)
                    ret, frame = self.video.read()
                    n_frame += 1
                    pbar.update(1)
            return self.video_frames
    
    def remove_video_frames(self) -> None:
        """
        Clear video frames from memory to free up space.
        """
        self.video_frames = []
    
    def get_cropped_frames(self, tam: int = 500) -> List[np.ndarray]:
        """
        Get cropped frames focused on the embryo region.
        
        This method adaptively updates the crop region when significant
        changes in brightness are detected (e.g., focus adjustments).
        
        Args:
            tam: Target size for cropping (not currently used)
            
        Returns:
            List of cropped frames
        """
        if self.cropped_video_frames:
            return self.cropped_video_frames
            
        # If already cropped, just return the full frames
        if self.is_cropped:
            self.cropped_video_frames = self.get_video_frames()
            return self.cropped_video_frames
        
        # Ensure frames are actually loaded before we try to crop them.
        # get_video_frames(simplify=True) (the default path) never sets
        # self.xxyy, so we can't rely on it being populated here -- the
        # loop below computes its own crop coords from frame 0 anyway.
        if not self.video_frames:
            self.get_video_frames()
        
        # Adaptive cropping with brightness-based recalibration
        for i, frame in enumerate(tqdm(self.video_frames, desc="Cropping frames")):
            if i == 0:
                # Initialize on first frame
                xmin, xmax, ymin, ymax = coord_embyo(frame)
                _, thres = cv2.threshold(frame[ymin:ymax, xmin:xmax, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                baseline_brightness = np.mean(thres)
            else:
                # Check for brightness changes (e.g., focus adjustments)
                _, thres = cv2.threshold(frame[ymin:ymax, xmin:xmax, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                current_brightness = np.mean(thres)
                
                # Recalibrate if significant brightness change detected
                if current_brightness > baseline_brightness + 10:
                    xmin, xmax, ymin, ymax = coord_embyo(frame)
                    baseline_brightness = current_brightness
            
            self.cropped_video_frames.append(frame[ymin:ymax, xmin:xmax, :])
        
        return self.cropped_video_frames
    
    def save_cropped_frames(self, filename: str = 'output/cropped_video.mp4') -> None:
        """
        Save the cropped video frames to a file.
        
        Args:
            filename: Output video file path
        """
        if not self.cropped_video_frames:
            self.get_cropped_frames()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get frame dimensions
        height, width = self.cropped_video_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in tqdm(self.cropped_video_frames, desc="Saving video"):
            out.write(frame)
        
        out.release()
        print(f"Cropped video saved to {filename}")
    
    def remove_cropped_frames(self) -> None:
        """
        Clear cropped frames from memory to free up space.
        """
        self.cropped_video_frames = []
    
    # ==================== Segmentation Methods ====================
    
    def get_segmented_frames(self) -> List[np.ndarray]:
        """
        Get segmentation masks for all cropped frames.
        
        Returns:
            List of segmentation masks
        """
        if self.segmented_frames:
            return self.segmented_frames
        
        if not self.cropped_video_frames:
            self.get_cropped_frames()
        
        for frame in tqdm(self.get_cropped_frames(), desc="Segmenting frames"):
            mask = predict_mask(frame, self.PATH_SEGMENTATION_MODEL, task='Segmentation')
            self.segmented_frames.append(mask)
        
        return self.segmented_frames
    
    def save_segmented_video(self, filename: str = 'output/segmented_video.mp4') -> None:
        """
        Save the segmented video to a file.
        
        Args:
            filename: Output video file path
        """
        if not self.segmented_frames:
            self.get_segmented_frames()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get frame dimensions
        height, width = self.segmented_frames[0].shape[:2]
        
        # Create video writer (slower fps for segmentation video)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 1, (width, height))
        
        # Write frames
        for frame in tqdm(self.segmented_frames, desc="Saving segmented video"):
            # Convert grayscale to BGR if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        
        out.release()
        print(f"Segmented video saved to {filename}")
    
    # ==================== Feature Extraction Methods ====================
    
    def remove_values(self) -> None:
        """
        Clear all extracted feature values.
        """
        self.values = {key: [] for key in self.values.keys()}
    
    def get_evolution(self) -> Dict[str, List]:
        """
        Extract temporal evolution of morphological features.
        
        This method processes each frame to extract all tracked features,
        creating a time-series dataset of embryo development.
        
        Args:
            crop_video: If True, process cropped frames; if False, process full frames
            
        Returns:
            Dictionary mapping feature names to lists of values over time
        """
        # Determine if we need to compute values
        compute = any(not values for values in self.values.values())
        
        if not compute:
            return self.values
        
        # Clear existing values
        self.remove_values()
        
        if not self.cropped_video_frames:
            self.get_cropped_frames()
        # Process each frame
        for i, frame in enumerate(tqdm(self.cropped_video_frames, desc="Extracting features")):
            # Create embryo image object
            if self.mask[i] == 0:
                frame_values = {key: None for key in self.values.keys()}
            else:
                embryo = EmbryoImage(img=frame, cropped_image=True, compute=True)
                # Extract all values
                frame_values = embryo.get_all_values()
            
            # Store values
            for key in self.values.keys():
                self.values[key].append(frame_values.get(key, None))
        
        # Clean up cell count data (remove noise/outliers)
        if 'n_cells' in self.values:
            self.values['n_cells'] = n_cells_cleaning(self.values['n_cells'])
        
        if 'cleavage' in self.values:
            self.values['cleavage'] = n_cells_cleaning(self.values['cleavage'])
        
        return self.values
    
    def save_evolution(
        self, 
        values: Optional[Dict[str, List]] = None,
        filename: str = 'output/assessment/PloidIA/evolution.csv'
    ) -> None:
        """
        Save the evolution data to a CSV file.
        
        Args:
            values: Feature values to save (uses self.values if None)
            filename: Output CSV file path
        """
        if values is None:
            values = self.values
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create DataFrame and save
        df = pd.DataFrame(values)
        
        # Add time columns
        df.insert(0, 'frame', range(len(df)))
        df.insert(1, 'time_hours', [i*self.step/12 for i in range(self.n_MAX)])
        
        df.to_csv(filename, index=False)
        print(f"Evolution data saved to {filename}")
    
    # ==================== Developmental Event Detection ====================
    
    def get_blasto_frame(self, N: int = 40) -> int:
        """
        Detect the frame where blastocyst formation occurs.
        
        Blastocyst formation is detected by the intersection of TE and ICM areas.
        
        Args:
            N: Window size for intersection detection
            
        Returns:
            Frame number where blastocyst formation is detected
        """
        if self.blasto_frame is not None:
            return self.blasto_frame
        
        # Ensure we have evolution data
        if not self.values.get('TE_area') or not self.values.get('ICM_area'):
            raise ValueError(
                "Evolution data not available. Run get_evolution() first."
            )
        
        self.blasto_frame = intersection(
            self.values['TE_area'], 
            self.values['ICM_area'], 
            N
        )
        
        return self.blasto_frame
    
    def get_blasto_formation(self, real_segs: float = 3000) -> float:
        """
        Calculate the real time when blastocyst formation occurs.
        
        Args:
            real_segs: Real time duration represented by the video (in seconds)
            
        Returns:
            Blastocyst formation time in hours
        """
        if self.blasto_formation is not None:
            return self.blasto_formation
        
        blasto_frame = self.get_blasto_frame()
        
        # Calculate real time in hours
        self.blasto_formation = blasto_frame * real_segs / (self.fps * 3600)
        
        return self.blasto_formation
    
    def detect_divisions(self, threshold: int = 1) -> Dict[int, int]:
        """
        Detect cell division events in the video.
        
        Args:
            threshold: Minimum cell count increase to be considered a division
            
        Returns:
            Dictionary mapping frame numbers to cell counts after division
        """
        if 'n_cells' not in self.values or not self.values['n_cells']:
            raise ValueError(
                "Cell count data not available. Run get_evolution() first."
            )
        
        divisions = {}
        prev_count = 0
        
        for frame, count in enumerate(self.values['n_cells']):
            if count is not None and count > prev_count + threshold:
                divisions[frame] = count
                prev_count = count
        
        return divisions
    
    def get_division_times(self, real_segs: float = 3000) -> Dict[int, float]:
        """
        Get the real times (in hours) when cell divisions occur.
        
        Args:
            real_segs: Real time duration represented by the video (in seconds)
            
        Returns:
            Dictionary mapping cell counts to division times in hours
        """
        divisions = self.detect_divisions()
        
        division_times = {}
        for frame, cell_count in divisions.items():
            time_hours = frame * real_segs / (self.fps * 3600)
            division_times[cell_count] = time_hours
        
        return division_times
    
    # ==================== Visualization Methods ====================
    
    def plot_evolution(
        self, 
        features: Optional[List[str]] = None,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot the temporal evolution of selected features.
        
        Args:
            features: List of features to plot (default: all)
            filename: Output file path (if None, displays plot)
            figsize: Figure size as (width, height)
        """
        if not self.values:
            raise ValueError(
                "No evolution data available. Run get_evolution() first."
            )
        
        if features is None:
            features = list(self.values.keys())
        
        # Filter out empty or all-None features
        features = [
            f for f in features 
            if f in self.values and any(v is not None for v in self.values[f])
        ]
        
        if not features:
            raise ValueError("No valid features to plot")
        
        # Calculate grid size
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        # Time axis (in hours)
        time_hours = np.arange(len(self.values[features[0]])) / (self.fps * 3600)
        
        # Plot each feature
        for idx, feature in enumerate(features):
            ax = axes[idx]
            values = self.values[feature]
            
            ax.plot(time_hours, values, linewidth=2)
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_title(f'{feature.replace("_", " ").title()} Over Time')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Evolution plot saved to {filename}")
        else:
            plt.show()
    
    def plot_feature(
        self,
        feature: str,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        mark_events: bool = True
    ) -> None:
        """
        Plot a single feature with optional event markers.
        
        Args:
            feature: Feature name to plot
            filename: Output file path (if None, displays plot)
            figsize: Figure size as (width, height)
            mark_events: Whether to mark developmental events
        """
        if feature not in self.values:
            raise ValueError(f"Feature '{feature}' not found in values")
        
        plt.figure(figsize=figsize)
        
        # Time axis
        time_hours = np.arange(len(self.values[feature])) / (self.fps * 3600)
        
        # Plot feature
        plt.plot(time_hours, self.values[feature], linewidth=2, label=feature)
        
        # Mark blastocyst formation if available and requested
        if mark_events and self.blasto_frame is not None:
            blasto_time = self.blasto_frame / (self.fps * 3600)
            plt.axvline(
                blasto_time, 
                color='red', 
                linestyle='--', 
                label='Blastocyst Formation'
            )
        
        plt.xlabel('Time (hours)')
        plt.ylabel(feature.replace('_', ' ').title())
        plt.title(f'{feature.replace("_", " ").title()} Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Feature plot saved to {filename}")
        else:
            plt.show()
    
    # ==================== Summary Methods ====================
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the video analysis.
        
        Returns:
            Dictionary containing key metrics and statistics
        """
        summary = {
            'video_path': self.path,
            'fps': self.fps,
            'total_frames': len(self.video_frames),
            'duration_hours': len(self.video_frames) / (self.fps * 3600),
        }
        
        # Add blastocyst formation info if available
        if self.blasto_frame is not None:
            summary['blasto_frame'] = self.blasto_frame
            summary['blasto_formation_hours'] = self.get_blasto_formation()
        
        # Add feature statistics if available
        if self.values:
            summary['features_tracked'] = list(self.values.keys())
            summary['feature_stats'] = {}
            
            for feature, values in self.values.items():
                # Filter out None values
                valid_values = [v for v in values if v is not None]
                
                if valid_values:
                    summary['feature_stats'][feature] = {
                        'mean': np.mean(valid_values),
                        'std': np.std(valid_values),
                        'min': np.min(valid_values),
                        'max': np.max(valid_values),
                    }
        
        return summary
    
    def print_summary(self) -> None:
        """
        Print a formatted summary of the video analysis.
        """
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("EMBRYO VIDEO ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nVideo: {summary['video_path']}")
        print(f"FPS: {summary['fps']:.2f}")
        print(f"Frames: {summary['total_frames']}")
        print(f"Duration: {summary['duration_hours']:.2f} hours")
        
        if 'blasto_formation_hours' in summary:
            print(f"\nBlastocyst Formation:")
            print(f"  Frame: {summary['blasto_frame']}")
            print(f"  Time: {summary['blasto_formation_hours']:.2f} hours")
        
        if 'features_tracked' in summary:
            print(f"\nFeatures Tracked: {len(summary['features_tracked'])}")
            print(f"  {', '.join(summary['features_tracked'][:5])}")
            if len(summary['features_tracked']) > 5:
                print(f"  ... and {len(summary['features_tracked']) - 5} more")
        
        print("\n" + "="*60 + "\n")
    
    # ==================== Cleanup Methods ====================
    
    def release(self) -> None:
        """
        Release video capture and free resources.
        """
        if self.video is not None:
            self.video.release()
    
    def __del__(self):
        """Destructor to ensure video is released."""
        self.release()
    
    # ==================== Legacy/Commented Methods ====================
    
    # Keeping these as comments for potential future implementation
    
    # def get_n_pronuclei(self, frame: np.ndarray) -> int:
    #     """
    #     Count the number of pronuclei in a frame.
    #     
    #     Args:
    #         frame: Input frame
    #         
    #     Returns:
    #         Number of pronuclei detected
    #     """
    #     return counting_pronuclei(frame)
    
    # def get_n_polar_corpuscles(self, frame: np.ndarray) -> Tuple[int, Any]:
    #     """
    #     Count the number of polar corpuscles in a frame.
    #     
    #     Args:
    #         frame: Input frame
    #         
    #     Returns:
    #         Tuple of (count, elements)
    #     """
    #     embryo = EmbryoImage(img=frame)
    #     num, elements = counting_polar_corpuscles(frame, embryo.get_ZP())
    #     return num, elements