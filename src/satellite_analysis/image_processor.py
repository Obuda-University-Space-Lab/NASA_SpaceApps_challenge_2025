"""
Core satellite image processing functionality
Handles loading, preprocessing, and basic analysis of satellite imagery
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import warnings

# Optional imports
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

warnings.filterwarnings('ignore')


class SatelliteImageProcessor:
    """
    Main class for satellite image processing operations
    """
    
    def __init__(self):
        self.supported_formats = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load satellite image from file path
        
        Args:
            image_path: Path to the satellite image file
            
        Returns:
            numpy array of the loaded image
        """
        try:
            # Try loading with rasterio for geotiff files (if available)
            if HAS_RASTERIO and image_path.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(image_path) as src:
                    image = src.read()
                    # Convert from (bands, height, width) to (height, width, bands)
                    if len(image.shape) == 3:
                        image = np.transpose(image, (1, 2, 0))
                    return image
            else:
                # Use OpenCV for other formats
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Could not load image from {image_path}")
                    
        except Exception as e:
            print(f"Error loading image: {e}")
            # Fallback to PIL
            image = Image.open(image_path)
            return np.array(image)
    
    def preprocess_image(self, image: np.ndarray, 
                        target_size: Optional[Tuple[int, int]] = None,
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess satellite image for analysis
        
        Args:
            image: Input image array
            target_size: Target size (height, width) for resizing
            normalize: Whether to normalize pixel values to 0-1 range
            
        Returns:
            Preprocessed image array
        """
        processed_image = image.copy()
        
        # Handle different data types
        if processed_image.dtype == np.uint16:
            processed_image = (processed_image / 65535.0 * 255).astype(np.uint8)
        elif processed_image.dtype == np.float32 or processed_image.dtype == np.float64:
            processed_image = (processed_image * 255).astype(np.uint8)
        
        # Resize if target size is specified
        if target_size:
            processed_image = cv2.resize(processed_image, (target_size[1], target_size[0]))
        
        # Normalize to 0-1 range if requested
        if normalize:
            processed_image = processed_image.astype(np.float32) / 255.0
            
        return processed_image
    
    def enhance_contrast(self, image: np.ndarray, 
                        method: str = 'clahe') -> np.ndarray:
        """
        Enhance contrast of satellite image
        
        Args:
            image: Input image array
            method: Enhancement method ('clahe', 'histogram_eq', 'gamma')
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = (image * 255).astype(np.uint8)
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(l_channel)
        elif method == 'histogram_eq':
            enhanced = cv2.equalizeHist(l_channel)
        elif method == 'gamma':
            # Gamma correction with gamma = 0.7
            gamma = 0.7
            enhanced = np.power(l_channel / 255.0, gamma) * 255
            enhanced = enhanced.astype(np.uint8)
        else:
            enhanced = l_channel
        
        if len(image.shape) == 3:
            lab[:, :, 0] = enhanced
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return result.astype(np.float32) / 255.0
        else:
            return enhanced.astype(np.float32) / 255.0
    
    def calculate_indices(self, image: np.ndarray, 
                         bands: dict) -> dict:
        """
        Calculate vegetation and other spectral indices
        
        Args:
            image: Multi-band satellite image
            bands: Dictionary mapping band names to indices
                  e.g., {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
            
        Returns:
            Dictionary of calculated indices
        """
        indices = {}
        
        # NDVI (Normalized Difference Vegetation Index)
        if 'red' in bands and 'nir' in bands:
            red = image[:, :, bands['red']]
            nir = image[:, :, bands['nir']]
            ndvi = (nir - red) / (nir + red + 1e-8)
            indices['ndvi'] = ndvi
        
        # NDWI (Normalized Difference Water Index)
        if 'green' in bands and 'nir' in bands:
            green = image[:, :, bands['green']]
            nir = image[:, :, bands['nir']]
            ndwi = (green - nir) / (green + nir + 1e-8)
            indices['ndwi'] = ndwi
        
        # SAVI (Soil-Adjusted Vegetation Index)
        if 'red' in bands and 'nir' in bands:
            red = image[:, :, bands['red']]
            nir = image[:, :, bands['nir']]
            L = 0.5  # soil brightness correction factor
            savi = ((nir - red) * (1 + L)) / (nir + red + L + 1e-8)
            indices['savi'] = savi
            
        return indices
    
    def visualize_image(self, image: np.ndarray, 
                       title: str = "Satellite Image",
                       bands_to_display: Optional[list] = None,
                       figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize satellite image
        
        Args:
            image: Image array to visualize
            title: Title for the plot
            bands_to_display: List of band indices to display (for multi-band images)
            figsize: Figure size for matplotlib
        """
        plt.figure(figsize=figsize)
        
        if len(image.shape) == 2:
            # Single band image
            plt.imshow(image, cmap='gray')
        elif len(image.shape) == 3:
            if bands_to_display:
                # Display specific bands
                if len(bands_to_display) == 3:
                    rgb_image = image[:, :, bands_to_display]
                    plt.imshow(rgb_image)
                else:
                    # Single band from multi-band image
                    plt.imshow(image[:, :, bands_to_display[0]], cmap='gray')
            else:
                # Display as RGB
                if image.shape[2] >= 3:
                    plt.imshow(image[:, :, :3])
                else:
                    plt.imshow(image[:, :, 0], cmap='gray')
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_indices(self, indices: dict, 
                         figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize calculated spectral indices
        
        Args:
            indices: Dictionary of calculated indices
            figsize: Figure size for matplotlib
        """
        n_indices = len(indices)
        if n_indices == 0:
            return
        
        cols = min(3, n_indices)
        rows = (n_indices + cols - 1) // cols
        
        plt.figure(figsize=figsize)
        
        for i, (name, index) in enumerate(indices.items()):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(index, cmap='RdYlGn' if 'ndvi' in name.lower() or 'savi' in name.lower() 
                      else 'RdYlBu' if 'ndwi' in name.lower() else 'viridis')
            plt.title(f'{name.upper()}')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()