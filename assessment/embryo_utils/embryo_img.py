"""
Embryo Image Analysis Class

This module provides functionality for analyzing embryo images, including
segmentation, measurement of various morphological features, and grading.
"""

import cv2
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Any
from pathlib import Path

from assessment.embryo_utils.crop import crop
from assessment.embryo_utils.seg_model import predict_mask
from assessment.embryo_utils.functions import (hull, polar_img, higuchi_fractal_dimension, radius, thikness)


from markers.cell_count import detect
from markers.blastocyst_grading.inference import run
from config.config import PATH_MODELS, SEGMENTATION_VALUES, TMP_PATH


class EmbryoImage:
    """
    A class for analyzing embryo images with automated segmentation and measurement.
    
    This class handles embryo image processing, segmentation of different regions
    (ZP, TE, ICM), and calculation of various morphological metrics.
    
    Attributes:
        path (Optional[str]): Path to the image file
        image (Optional[np.ndarray]): Full input image
        cropped_embryo (Optional[np.ndarray]): Cropped embryo region
        segmentation (Optional[np.ndarray]): Segmentation mask
        tam (int): Target size for cropping (default: 500)
        
    Example:
        >>> embryo = EmbryoImage(path="embryo.jpg", compute=True)
        >>> metrics = embryo.get_all_values()
        >>> embryo.draw_prediction("output.png")
    """
    root_dir = '../../'
    root_dir = PATH_MODELS
    
    # Class constants for segmentation values
    SEGMENTATION_VALUES['THRESHOLD'] = 145
    PATH_SEGMENTATION_MODEL = root_dir +'/markers/blastocyst_segmentation/segmentation_hrnet.pth'
    PATH_FRAGMENTATION_MODEL = root_dir +'/markers/fragmentation/fragmentation_hrnet.pth'
    PATH_COUNTING_MODEL = root_dir +'/markers/cell_count/best.pt'
    PATH_EXPANSION_MODEL = root_dir +'/markers/blastocyst_grading/EXP_0.1754.pt'
    PATH_TE_MODEL = root_dir +'/markers/blastocyst_grading/TE_0.8447.pt'
    PATH_ICM_MODEL = root_dir +'/markers/blastocyst_grading/ICM_0.6777.pt'
    PATH_STAGE_CLASSIF = root_dir + 'markers/stage_classif/resnet34/resnet34_we15_e100.pth'
    TMP_PATH = root_dir + TMP_PATH+'markers'

    # Supported image formats
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    def __init__(self, path: Optional[str] = None, img: Optional[np.ndarray] = None, tam: int = 500, cropped_image: bool = False, compute: bool = False):
        """
        Initialize the EmbryoImage object.
        
        Args:
            path: Path to the image file
            img: Input image as numpy array
            tam: Target size for cropping
            cropped_image: Whether the input image is already cropped
            compute: Whether to compute all metrics immediately
            
        Raises:
            ValueError: If neither path nor img is provided, or if file format is unsupported
        """
        # Input parameters
        self.path = path
        self.tam = tam
        self.cropped_image = cropped_image
        
        # Image data
        self.image: Optional[np.ndarray] = None
        self.cropped_embryo: Optional[np.ndarray] = None
        self.segmentation: Optional[np.ndarray] = None
        
        # Cell classification
        self.number_of_cells: Optional[int] = None
        self.cleavage: Optional[int] = None  # Fixed typo: cleveage -> cleavage
        
        # Morphological measurements - Embryo
        self.diameter: Optional[float] = None
        self.blasto: Optional[np.ndarray] = None
        self.center_mass: Optional[Tuple[int, int]] = None
        self.area: Optional[float] = None
        
        # Zona Pellucida (ZP) measurements
        self.ZP: Optional[np.ndarray] = None
        self.ZP_R: Optional[float] = None  # Outer radius
        self.ZP_r: Optional[float] = None  # Inner radius
        self.ZP_thickness: Optional[float] = None
        self.ZP_area: Optional[float] = None
        self.ZP_symmetry: Optional[float] = None
        
        # Trophectoderm (TE) measurements
        self.TE: Optional[np.ndarray] = None
        self.TE_area: Optional[float] = None
        self.TE_area_ratio: Optional[float] = None
        self.TE_thickness: Optional[np.ndarray] = None
        self.TE_fractal_d: Optional[float] = None
        self.TE_morph: Optional[Any] = None
        
        # Inner Cell Mass (ICM) measurements
        self.ICM: Optional[np.ndarray] = None
        self.ICM_area: Optional[float] = None
        self.ICM_area_ratio: Optional[float] = None
        self.ICM_eccentricity: Optional[float] = None
        
        # Blastocoel Cavity (BC) measurements
        self.BC: Optional[np.ndarray] = None
        self.BC_area: Optional[float] = None
        self.BC_area_ratio: Optional[float] = None
        
        # Grading
        self.expansion: Optional[int] = None
        self.te_grading: Optional[str] = None
        self.icm_grading: Optional[str] = None
        self.grading: Optional[str] = None

        # Stage classification
        self.stage_classif: Optional[str] = None    

        # Fragmentation
        self.fragmentation: Optional[np.ndarray] = None
        self.fragmentation_idx: Optional[float] = None
        
        # Initialize image
        self._initialize_image(path, img, cropped_image, tam)
        
        # Optionally compute all values
        if compute:
            self.get_all_values()
    
    def _initialize_image(self, path: Optional[str], img: Optional[np.ndarray], cropped_image: bool, tam: int) -> None:
        """Initialize image from path or array."""
        if path is not None:
            self._load_from_path(path, cropped_image, tam)
        elif img is not None:
            self._load_from_array(img, cropped_image, tam)
        else:
            raise ValueError("Please provide either 'path' or 'img' parameter")
    
    def _load_from_path(self, path: str, cropped_image:bool,tam: int) -> None:
        """Load image from file path."""
        path = str(path)
        
        if not path.lower().endswith(self.SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported file format. Supported formats: {self.SUPPORTED_FORMATS}")
        
        self.path = path
        self.image = cv2.imread(self.path, cv2.IMREAD_COLOR)
        
        if self.image is None:
            raise ValueError(f"Error: Could not open the image at {self.path}")
        
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        if cropped_image:
            self.cropped_embryo = self.image
        else:
            self.cropped_embryo = crop(self.image, tam)
    
    def _load_from_array(self, img: np.ndarray, cropped_image: bool, tam: int) -> None:
        """Load image from numpy array."""
        self.image = img
        if not cropped_image:
            self.cropped_embryo = crop(self.image, tam)
        else:
            self.cropped_embryo = img
        
        # Save temporary image (used by YOLO-based cell counting)
        tmp_dir = Path(self.TMP_PATH)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        resized = cv2.resize(self.image, (640, 640), interpolation=cv2.INTER_AREA)
        tmp_path = tmp_dir / 'original_image.png'
        cv2.imwrite(str(tmp_path), resized)
        self.path = str(tmp_path)

    def get_values(
        self,
        *,
        include_blastocyst_structures: bool = True,
        include_cell_count: bool = True,
        include_fragmentation: bool = True,
        include_grading: bool = True,
        include_stage: bool = True,
    ) -> Dict[str, Any]:
        """Compute metrics with optional expensive components.

        Returns a dict with the same keys as `get_all_values()`, but can skip
        YOLO cell counting, fragmentation segmentation, and grading/classifiers.
        """
        # If blastocyst structures are disabled, skip segmentation-derived metrics entirely.
        if not include_blastocyst_structures:
            out: Dict[str, Any] = {
                'diameter': None,
                'area': None,
                'ZP_R': None,
                'ZP_r': None,
                'ZP_thickness': None,
                'ZP_area': None,
                'ZP_symmetry': None,
                'TE_area': None,
                'TE_fractal_d': None,
                'TE_mean_thickness': None,
                'TE_area_ratio': None,
                'ICM_area': None,
                'ICM_area_ratio': None,
                'ICM_eccentricity': None,
                'BC_area': None,
                'BC_area_ratio': None,
                'n_cells': None,
                'cleavage': None,
                'fragmentation_idx': None,
                'expansion': None,
                'te_grading': None,
                'icm_grading': None,
                'grading': None,
                'stage_classif': None,
            }

            if include_cell_count:
                out['n_cells'] = self.get_number_of_cells()
                out['cleavage'] = self.get_cleavage()
            if include_fragmentation:
                out['fragmentation_idx'] = self.get_fragmentation_idx()
            if include_grading:
                out['expansion'], out['te_grading'], out['icm_grading'] = self.get_grading()
            if include_stage:
                out['stage_classif'] = self.get_stage_classification()

            return out

        # Check if segmentation is valid
        if len(np.unique(self.get_blasto_seg())) == 1:
            return {
                'diameter': 0,
                'area': 0,
                'ZP_R': 0,
                'ZP_r': 0,
                'ZP_thickness': 0,
                'ZP_area': 0,
                'ZP_symmetry': 0,
                'TE_area': 0,
                'TE_fractal_d': 0,
                'TE_mean_thickness': 0,
                'TE_area_ratio': 0,
                'ICM_area': 0,
                'ICM_area_ratio': 0,
                'ICM_eccentricity': 0,
                'BC_area': 0,
                'BC_area_ratio': 0,
                'n_cells': 0 if include_cell_count else None,
                'cleavage': None,
                'fragmentation_idx': 0 if include_fragmentation else None,
                'expansion': None,
                'te_grading': None,
                'icm_grading': None,
                'grading': None,
                'stage_classif': None,
            }

        # Compute core metrics
        zp_r, zp_r_inner = self.get_ZP_R_r()
        out: Dict[str, Any] = {
            'diameter': self.get_diameter(),
            'area': self.get_area(),
            'ZP_R': zp_r,
            'ZP_r': zp_r_inner,
            'ZP_thickness': self.get_ZP_thickness(),
            'ZP_area': self.get_ZP_area(),
            'ZP_symmetry': self.get_ZP_symmetry(),
            'TE_area': self.get_TE_area(),
            'TE_fractal_d': self.get_TE_fractal_d(),
            'TE_mean_thickness': np.mean(self.TE_thickness) if self.TE_thickness is not None else 0,
            'TE_area_ratio': self.get_TE_area_ratio(),
            'ICM_area': self.get_ICM_area(),
            'ICM_area_ratio': self.get_ICM_area_ratio(),
            'ICM_eccentricity': self.get_ICM_eccentricity(),
            'BC_area': self.get_BC_area(),
            'BC_area_ratio': self.get_BC_area_ratio(),
            # Optional parts (filled below)
            'n_cells': None,
            'cleavage': None,
            'fragmentation_idx': None,
            'expansion': None,
            'te_grading': None,
            'icm_grading': None,
            'grading': None,
            'stage_classif': None,
        }

        if include_cell_count:
            out['n_cells'] = self.get_number_of_cells()
            out['cleavage'] = self.get_cleavage()

        if include_fragmentation:
            out['fragmentation_idx'] = self.get_fragmentation_idx()

        if include_grading:
            out['expansion'],  out['te_grading'], out['icm_grading'] = self.get_grading()

        if include_stage:
            out['stage_classif'] = self.get_stage_classification()

        return out
    
    # ==================== Image Access Methods ====================
    
    def get_image(self) -> np.ndarray:
        """Get the full input image."""
        return self.image
    
    def show_embryo(self) -> None:
        """Display the full embryo image."""
        plt.imshow(self.image)
        plt.axis('off')
        plt.title('Embryo Image')
        plt.show()
    
    def get_cropped_embryo(self) -> np.ndarray:
        """Get the cropped embryo region."""
        return self.cropped_embryo
    
    def show_cropped_embryo(self) -> None:
        """Display the cropped embryo."""
        plt.imshow(self.cropped_embryo)
        plt.axis('off')
        plt.title('Cropped Embryo')
        plt.show()
    
    def set_cropped_embryo(self, img: np.ndarray) -> None:
        """
        Set a new cropped embryo image and reset segmentation.
        
        Args:
            img: New cropped embryo image
        """
        self.cropped_embryo = img
        self.segmentation = None
    
    # ==================== Segmentation Methods ====================
    
    def get_blasto_seg(self) -> np.ndarray:
        """
        Get or compute the segmentation mask.
        
        Returns:
            Segmentation mask with different values for ZP, ICM, and TE
        """
        if self.segmentation is None:
            self.segmentation = predict_mask(img=self.get_cropped_embryo(), checkpoint=self.PATH_SEGMENTATION_MODEL , task='Segmentation')
        return self.segmentation
    
    def show_blasto_seg(self) -> None:
        """Display the segmentation mask."""
        plt.imshow(self.get_blasto_seg(), cmap='tab10')
        plt.axis('off')
        plt.title('Segmentation Mask')
        plt.colorbar()
        plt.show()
    
    # ==================== Cell Detection Methods ====================
    
    def get_number_of_cells(self) -> int:
        """
        Detect and count the number of cells in the embryo.
        
        Returns:
            Number of detected cells
        """
        if self.number_of_cells is not None:
            return self.number_of_cells
        
        class_map = {0: 2, 1: 0, 2: 1}
        path_tmp = self.TMP_PATH
        name = 'cell_count'
        cells_file = f'{path_tmp}/{name}/cells.txt'
        
        try:
            if os.path.exists(cells_file):
                os.remove(cells_file)
            detect.run(weights=self.PATH_COUNTING_MODEL, source=self.path, project=path_tmp,name = name, device = 0,save_txt=True)
                   
            if os.path.exists(cells_file):
                with open(cells_file, 'r') as f:
                    lines = f.readlines()
                    self.number_of_cells = len(lines)
                
                    if len(lines) > 0:
                        self.cleavage = class_map[int(lines[0].split()[0])]
            else:
                self.number_of_cells = 0 
                self.cleavage = None
                
        except Exception as e:
            print(f"Warning: Cell detection failed: {e}")
            self.number_of_cells = 0
            self.cleavage = None
        
        return self.number_of_cells
    
    def get_cleavage(self) -> Optional[int]:
        """
        Get the cleavage stage classification.
        
        Returns:
            Cleavage stage (0, 1, or 2) or None
        """
        if self.cleavage is None:
            self.get_number_of_cells()
        return self.cleavage
    
    # ==================== General Embryo Measurements ====================
    
    def get_diameter(self) -> float:
        """
        Get the diameter of the embryo (excluding ZP).
        
        Returns:
            Embryo diameter in pixels
        """
        if self.diameter is None:
            self.get_TE_fractal_d()  # This also computes diameter
        return self.diameter
    
    def set_diameter(self, d: float) -> None:
        """Set the embryo diameter."""
        self.diameter = d
    
    def get_blasto(self) -> np.ndarray:
        """
        Get the blastocyst region (convex hull excluding ZP).
        
        Returns:
            Binary mask of blastocyst region
        """
        if self.blasto is None:
            mask = self.get_blasto_seg().copy()
            mask[mask < SEGMENTATION_VALUES['THRESHOLD']] = 0
            mask[mask >= SEGMENTATION_VALUES['THRESHOLD']] = 255
            self.blasto = hull(mask)
        return self.blasto
    
    def get_center_mass(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the center of mass of the embryo.
        
        Returns:
            Tuple of (cx, cy) coordinates or (None, None) if calculation fails
        """
        if self.center_mass is None:
            try:
                te_mask = self.get_TE()
                cx = int(np.mean(np.where(te_mask > 0)[1]))
                cy = int(np.mean(np.where(te_mask > 0)[0]))
            except:
                try:
                    zp_mask = self.get_ZP()
                    cx = int(np.mean(np.where(zp_mask > 0)[1]))
                    cy = int(np.mean(np.where(zp_mask > 0)[0]))
                except:
                    cx, cy = None, None
            
            self.center_mass = (cx, cy)
        
        return self.center_mass
    
    def get_area(self) -> float:
        """
        Get the total embryo area (excluding ZP).
        
        Returns:
            Embryo area in pixels
        """
        if self.area is None:
            mask = self.get_blasto_seg().copy()
            mask[mask < SEGMENTATION_VALUES['THRESHOLD']] = 0
            mask[mask >= SEGMENTATION_VALUES['THRESHOLD']] = 255
            hull_image = hull(mask)
            self.area = np.sum(hull_image > 0)
        return self.area
    
    def set_area(self, a: float) -> None:
        """Set the embryo area."""
        self.area = a
    
    # ==================== Zona Pellucida (ZP) Methods ====================
    
    def get_ZP(self) -> np.ndarray:
        """
        Get the Zona Pellucida mask.
        
        Returns:
            Binary mask of ZP region
        """
        if self.ZP is None:
            mask = self.get_blasto_seg().copy()
            zp_value = SEGMENTATION_VALUES['ZP']
            mask[mask != zp_value] = 0
            mask[mask == zp_value] = 255
            self.ZP = mask
        return self.ZP
    
    def get_ZP_R_r(self) -> Tuple[float, float]:
        """
        Get the outer (R) and inner (r) radii of the ZP.
        
        Returns:
            Tuple of (outer_radius, inner_radius) in pixels
        """
        if self.ZP_R is not None and self.ZP_r is not None:
            return self.ZP_R, self.ZP_r
        
        cx, cy = self.get_center_mass()
        
        if cx is None or cy is None:
            self.ZP_R = 0
            self.ZP_r = 0
            self.ZP_thickness = 0
            return self.ZP_R, self.ZP_r
        
        # Process ZP mask
        ZP = self.get_ZP()
        ZP = cv2.erode(ZP, np.ones((3, 3), np.uint8), iterations=1)
        ZP = cv2.dilate(ZP, np.ones((3, 3), np.uint8), iterations=1)
        
        # Convert to polar coordinates
        points = np.argwhere(ZP > 0)
        theta = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        r = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        
        x_band = np.degrees(theta) + 180
        y_band = r
        
        polar_band = polar_img(x_band, y_band)
        
        if polar_band is None:
            self.ZP_R = 0
            self.ZP_r = 0
            self.ZP_thickness = 0
            return self.ZP_R, self.ZP_r

        thickness_values = np.array(thikness(polar_band))
        mean_t = thickness_values.mean() if len(thickness_values) > 0 else 0
        std_t = thickness_values.std() if len(thickness_values) > 0 else 0
        cv = std_t / mean_t if mean_t != 0 else 0

        self.ZP_thickness = mean_t
        self.ZP_R, self.ZP_r = radius(polar_band)
        self.ZP_symmetry = max(0, 1 - cv) 

        return self.ZP_R, self.ZP_r
    
    def get_ZP_thickness(self) -> float:
        """
        Get the average thickness of the ZP.
        
        Returns:
            ZP thickness in pixels
        """
        if self.ZP_thickness is None:
            self.get_ZP_R_r()
        return self.ZP_thickness
    
    def set_ZP_thickness(self, t: float) -> None:
        """Set the ZP thickness."""
        self.ZP_thickness = t
    
    def get_ZP_area(self) -> float:
        """
        Get the area of the ZP.
        
        Returns:
            ZP area in pixels
        """
        if self.ZP_area is None:
            self.ZP_area = np.sum(
                self.get_blasto_seg() == SEGMENTATION_VALUES['ZP']
            )
        return self.ZP_area
    
    def get_ZP_symmetry(self) -> float:
        """
        Calculate ZP symmetry computing the coefficient of variation.
        
        Returns:
            Symmetry score (0-1), where 1 is perfectly symmetric
        """
        if self.ZP_symmetry is None:
            self.get_ZP_R_r()  # This computes ZP_symmetry as well
        
        return self.ZP_symmetry
    
    # ==================== Trophectoderm (TE) Methods ====================
    
    def get_TE(self) -> np.ndarray:
        """
        Get the Trophectoderm mask.
        
        Returns:
            Binary mask of TE region
        """
        if self.TE is None:
            mask = self.get_blasto_seg().copy()
            te_value = SEGMENTATION_VALUES['TE']
            mask[mask != te_value] = 0
            self.TE = mask
        return self.TE
    
    def get_TE_area(self) -> float:
        """
        Get the area of the TE.
        
        Returns:
            TE area in pixels
        """
        if self.TE_area is None:
            self.TE_area = np.sum(
                self.get_blasto_seg() == SEGMENTATION_VALUES['TE']
            )
        return self.TE_area
    
    def set_TE_area(self, a: float) -> None:
        """Set the TE area."""
        self.TE_area = a
    
    def get_TE_area_ratio(self) -> float:
        """
        Get the ratio of TE area to total embryo area.
        
        Returns:
            TE area ratio (0-1)
        """
        if self.TE_area_ratio is None:
            try:
                self.TE_area_ratio = self.get_TE_area() / self.get_area()
            except:
                self.TE_area_ratio = 0
        return self.TE_area_ratio
    
    def get_TE_fractal_d(self) -> float:
        """
        Calculate the fractal dimension of the TE.
        
        This is a measure of TE complexity/irregularity.
        
        Returns:
            Fractal dimension value
        """
        if self.TE_fractal_d is not None:
            return self.TE_fractal_d
        
        # Process TE mask
        TE = self.get_TE()
        TE = cv2.erode(TE, np.ones((3, 3), np.uint8), iterations=1)
        TE = cv2.dilate(TE, np.ones((3, 3), np.uint8), iterations=1)
        
        cx, cy = self.get_center_mass()
        
        # Convert to polar coordinates
        points = np.argwhere(TE > 0)
        theta = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        r = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        
        x_band = np.degrees(theta) + 180
        y_band = r
        
        polar_band = polar_img(x_band, y_band)
        
        if polar_band is None:
            self.TE_fractal_d = 0
            self.TE_thickness = 0
            return self.TE_fractal_d
        
        # Calculate thickness and diameter
        self.TE_thickness = thikness(polar_band)
        
        # Calculate mean radius
        radii = [
            max(np.where(polar_band[:, x] == 255)[0])
            for x in range(360)
            if len(np.where(polar_band[:, x] == 255)[0]) > 0
        ]
        radius_mean = np.mean(radii) if radii else 1
        
        # Calculate diameter
        diameters = [
            min(np.where(polar_band[:, x] == 255)[0])
            for x in range(360)
            if len(np.where(polar_band[:, x] == 255)[0]) > 0
        ]
        self.diameter = 2 * np.mean(diameters) if diameters else 0
        
        # Calculate fractal dimension
        try:
            TE_thickness_norm = self.TE_thickness / radius_mean
            self.TE_fractal_d = higuchi_fractal_dimension(TE_thickness_norm, 100)
        except:
            self.TE_fractal_d = 0
        
        return self.TE_fractal_d
    
    # ==================== Inner Cell Mass (ICM) Methods ====================
    
    def get_ICM(self) -> np.ndarray:
        """
        Get the Inner Cell Mass mask.
        
        Returns:
            Binary mask of ICM region
        """
        if self.ICM is None:
            mask = self.get_blasto_seg().copy()
            icm_value = SEGMENTATION_VALUES['ICM']
            mask[mask != icm_value] = 0
            mask[mask == icm_value] = 255
            self.ICM = mask
        return self.ICM
    
    def get_ICM_area(self) -> float:
        """
        Get the area of the ICM.
        
        Returns:
            ICM area in pixels
        """
        if self.ICM_area is None:
            self.ICM_area = np.sum(
                self.get_blasto_seg() == SEGMENTATION_VALUES['ICM']
            )
        return self.ICM_area
    
    def set_ICM_area(self, a: float) -> None:
        """Set the ICM area."""
        self.ICM_area = a
    
    def get_ICM_area_ratio(self) -> float:
        """
        Get the ratio of ICM area to total embryo area.
        
        Returns:
            ICM area ratio (0-1)
        """
        if self.ICM_area_ratio is None:
            try:
                self.ICM_area_ratio = self.get_ICM_area() / self.get_area()
            except:
                self.ICM_area_ratio = 0
        return self.ICM_area_ratio
    
    
    def get_ICM_eccentricity(self) -> Optional[float]:
        """
        Calculate the eccentricity of the ICM.
        
        Eccentricity measures how elongated the ICM is (0 = circle, approaching 1 = ellipse).
        
        Returns:
            Eccentricity value (0-1) or None if ICM not found
        """
        if self.ICM_eccentricity is not None:
            return self.ICM_eccentricity
        
        contours, _ = cv2.findContours(
            self.get_ICM(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) >= 5:  # Need at least 5 points for ellipse fitting
            ellipse = cv2.fitEllipse(largest_contour)
            (x, y), (major_axis, minor_axis), angle = ellipse
            
            # Compute eccentricity
            a = max(major_axis, minor_axis) / 2  # Semi-major axis
            b = min(major_axis, minor_axis) / 2  # Semi-minor axis
            
            self.ICM_eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2))
        
        return self.ICM_eccentricity
    
    # ==================== Blastocoel Cavity (BC) Methods ====================
    
    def get_BC(self) -> np.ndarray:
        """
        Get the Blastocoel Cavity mask.
        
        Returns:
            Binary mask of BC region
        """
        if self.BC is None:
            self.BC = self.get_blasto() - self.get_ICM() - self.get_TE()
            self.BC = cv2.erode(self.BC, np.ones((3, 3), np.uint8), iterations=1)
            self.BC = cv2.dilate(self.BC, np.ones((3, 3), np.uint8), iterations=1)
        return self.BC
    
    def get_BC_area(self) -> float:
        """
        Get the area of the blastocoel cavity.
        
        Returns:
            BC area in pixels
        """
        if self.BC_area is None:
            self.BC_area = (
                self.get_area() - self.get_TE_area() - self.get_ICM_area()
            )
        return self.BC_area
    
    def set_BC_area(self, a: float) -> None:
        """Set the BC area."""
        self.BC_area = a
    
    def get_BC_area_ratio(self) -> float:
        """
        Get the ratio of BC area to total embryo area.
        
        Returns:
            BC area ratio (0-1)
        """
        if self.BC_area_ratio is None:
            try:
                self.BC_area_ratio = self.get_BC_area() / self.get_area()
            except:
                self.BC_area_ratio = 0
        return self.BC_area_ratio
    
    # ==================== Grading Methods ====================
    
    def get_expansion(self) -> Optional[int]:
        """
        Get the expansion stage of the blastocyst.
        
        Returns:
            Expansion stage (1-6) or None
        """
        if self.expansion is None:
            #print('Calculating expansion stage...')
            self.expansion = run(self.get_cropped_embryo(), checkpoint=self.PATH_EXPANSION_MODEL, task='EXP')
        
        return self.expansion
    
    def get_te_grading(self) -> Optional[str]:
        """
        Get the TE grading (A, B, C).
        
        Returns:
            TE grading string or None
        """
        if self.te_grading is None:
            self.te_grading = run(self.get_cropped_embryo(), checkpoint=self.PATH_TE_MODEL, task='TE')
        
        return self.te_grading
    
    def get_icm_grading(self) -> Optional[str]:
        """
        Get the ICM grading (A, B, C).
        
        Returns:
            ICM grading string or None
        """
        if self.icm_grading is None:
            self.icm_grading = run(self.get_cropped_embryo(), checkpoint=self.PATH_ICM_MODEL, task='ICM')
        
        return self.icm_grading
    

    def get_grading(self) -> Optional[str]:
        """
        Get the embryo quality grading.
        
        Returns:
            Grading string or None
        """
        if self.grading is None:
            if self.get_expansion()> 2:
                self.grading = [self.get_expansion(), self.get_te_grading(), self.get_icm_grading()]
            else:
                self.grading = [self.get_expansion(), None, None]
        return self.grading
    
    # ==================== Stage Classification Methods ====================
    def get_stage_classification(self) -> Optional[str]:
        """
        Get the developmental stage classification of the embryo.
        
        Returns:
            Stage classification string or None
        """
        if self.stage_classif is None:
            self.stage_classif = run(self.get_cropped_embryo(), checkpoint=self.PATH_STAGE_CLASSIF, task='STAGE')
        
        return self.stage_classif

    # ==================== Fragmentation Methods ====================

    def get_fragmentation(self) -> Optional[np.ndarray]:
        """
        Get the fragmentation mask.
        
        Returns:
            Binary mask of fragmented regions or None
        """
        if self.fragmentation is None:
            self.fragmentation = predict_mask(img=self.get_image(), task='Fragmentation', checkpoint=self.PATH_FRAGMENTATION_MODEL)

        return self.fragmentation
    
    def get_fragmentation_idx(self) -> Optional[float]:
        """
        Get the fragmentation index.
        
        Returns:
            Fragmentation index or None
        """
        if self.fragmentation_idx is None:
            self.fragmentation_idx =np.sum(self.get_fragmentation()==255) /self.get_area()

        return self.fragmentation_idx
    
    # ==================== Comprehensive Analysis ====================
    
    def get_all_values(self) -> Dict[str, Any]:
        """
        Compute and return all morphological measurements.
        
        Returns:
            Dictionary containing all measured metrics
        """
        return self.get_values(
            include_cell_count=True,
            include_fragmentation=True,
            include_grading=True,
            include_stage=True,
        )
    
    def reset_values(self) -> None:
        """Reset all computed values to None.""" 
        self.diameter = None 
        self.area = None 
        self.ZP_R = None 
        self.ZP_r = None 
        self.ZP_thickness = None 
        self.ZP_area = None 
        self.ZP_symmetry = None 
        self.TE_area = None 
        self.TE_area_ratio = None 
        self.TE_thickness = None 
        self.TE_fractal_d = None 
        self.TE_morph = None 
        self.ICM_area = None 
        self.ICM_area_ratio = None 
        self.ICM_eccentricity = None 
        self.BC_area = None 
        self.BC_area_ratio = None 
        self.number_of_cells = None 
        self.cleavage = None 
        self.fragmentation_idx = None 
        self.expansion = None 
        self.te_grading = None 
        self.icm_grading = None 
        self.grading = None
        self.stage_classif = None
    # ==================== Visualization Methods ====================
    
    def draw_prediction(self, filename: str = 'output/prediction.png') -> None:
        """
        Draw segmentation boundaries on the embryo image.
        
        Args:
            filename: Output file path
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        img = self.cropped_embryo.copy()
        
        # Draw ZP boundary in red
        mask = cv2.dilate(self.get_ZP(), np.ones((3, 3), np.uint8)) - self.get_ZP()
        img[mask == 255] = (255, 0, 0)
        
        # Draw TE boundary in green
        mask = cv2.dilate(self.get_TE(), np.ones((3, 3), np.uint8)) - self.get_TE()
        img[mask == 255] = (0, 255, 0)
        
        # Draw ICM boundary in blue
        mask = cv2.dilate(self.get_ICM(), np.ones((3, 3), np.uint8)) - self.get_ICM()
        img[mask == 255] = (0, 0, 255)
        
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    def color_prediction(self, filename: str = 'output/assessment/PloidIA/prediction_colored.png') -> None:
        """
        Create a colored overlay of the segmentation.
        
        Args:
            filename: Output file path
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        img = self.cropped_embryo.copy()
        mask = np.zeros_like(img)
        
        # Assign colors to each channel
        mask[:, :, 0] = self.get_ZP()      # Red channel
        mask[:, :, 1] = self.get_TE()      # Green channel  
        mask[:, :, 2] = self.get_ICM()     # Blue channel
        
        # Blend with original image
        img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # ==================== Export Methods ====================
    
    def export_values(self, filename: str = 'output/assessment/PloidIA/values.txt') -> None:
        """
        Export all measurements to a text file.
        
        Args:
            filename: Output file path
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        metrics = self.get_all_values()
        
        with open(filename, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all measurements to a dictionary.
        
        Returns:
            Dictionary of all metrics
        """
        return self.get_all_values()
    
    def __repr__(self) -> str:
        """String representation of the EmbryoImage object."""
        return (
            f"EmbryoImage(path={self.path}, "
            f"size={self.cropped_embryo.shape if self.cropped_embryo is not None else None}, "
            f"n_cells={self.number_of_cells})"
        )
