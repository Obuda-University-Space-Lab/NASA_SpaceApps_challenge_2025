"""
Data loading and preprocessing utilities for satellite image segmentation
"""

import numpy as np
import cv2
import os
from typing import Tuple, List, Generator, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

# Optional imports
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

warnings.filterwarnings('ignore')


class SatelliteDataLoader:
    """
    Data loader for satellite imagery and segmentation masks
    """
    
    def __init__(self, images_dir: str, masks_dir: str, 
                 image_size: Tuple[int, int] = (256, 256),
                 batch_size: int = 8):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Get list of image files
        self.image_files = self._get_image_files()
        self.mask_files = self._get_mask_files()
        
        # Ensure matching files
        self._validate_file_pairs()
        
    def _get_image_files(self) -> List[str]:
        """Get list of image files"""
        if not os.path.exists(self.images_dir):
            return []
        
        extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(self.images_dir) 
                         if f.lower().endswith(ext)])
        return sorted(files)
    
    def _get_mask_files(self) -> List[str]:
        """Get list of mask files"""
        if not os.path.exists(self.masks_dir):
            return []
        
        extensions = ['.tif', '.tiff', '.png']
        files = []
        for ext in extensions:
            files.extend([f for f in os.listdir(self.masks_dir) 
                         if f.lower().endswith(ext)])
        return sorted(files)
    
    def _validate_file_pairs(self):
        """Ensure image and mask files are properly paired"""
        if len(self.image_files) == 0:
            print("Warning: No image files found")
            return
            
        if len(self.mask_files) == 0:
            print("Warning: No mask files found")
            return
        
        # Create a mapping based on file names (without extensions)
        image_names = [os.path.splitext(f)[0] for f in self.image_files]
        mask_names = [os.path.splitext(f)[0] for f in self.mask_files]
        
        # Find common files
        common_names = set(image_names) & set(mask_names)
        
        if len(common_names) == 0:
            print("Warning: No matching image-mask pairs found")
            return
        
        # Filter files to only include matching pairs
        self.image_files = [f for f in self.image_files 
                           if os.path.splitext(f)[0] in common_names]
        self.mask_files = [f for f in self.mask_files 
                          if os.path.splitext(f)[0] in common_names]
        
        print(f"Found {len(self.image_files)} matching image-mask pairs")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load and preprocess a single mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        mask = cv2.resize(mask, self.image_size)
        mask = (mask > 127).astype(np.float32)  # Binarize mask
        
        return mask
    
    def create_data_generators(self, validation_split: float = 0.2, 
                             augment: bool = True) -> Tuple[Optional[object], Optional[object]]:
        """Create training and validation data generators"""
        if not HAS_TENSORFLOW:
            print("Warning: TensorFlow not available. Cannot create TensorFlow datasets.")
            return None, None
            
        if len(self.image_files) == 0:
            raise ValueError("No image files available for training")
        
        # Split data
        train_images, val_images, train_masks, val_masks = train_test_split(
            self.image_files, self.mask_files, 
            test_size=validation_split, 
            random_state=42
        )
        
        # Create datasets
        train_dataset = self._create_dataset(train_images, train_masks, augment=augment)
        val_dataset = self._create_dataset(val_images, val_masks, augment=False)
        
        return train_dataset, val_dataset
    
    def _create_dataset(self, image_files: List[str], mask_files: List[str], 
                       augment: bool = False) -> Optional[object]:
        """Create TensorFlow dataset from file lists"""
        if not HAS_TENSORFLOW:
            return None
            
        def generator():
            for img_file, mask_file in zip(image_files, mask_files):
                img_path = os.path.join(self.images_dir, img_file)
                mask_path = os.path.join(self.masks_dir, mask_file)
                
                try:
                    image = self.load_image(img_path)
                    mask = self.load_mask(mask_path)
                    
                    if augment:
                        image, mask = self._apply_augmentation(image, mask)
                    
                    yield image, mask
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape(self.image_size + (3,)),
                tf.TensorShape(self.image_size + (1,))
            )
        )
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _apply_augmentation(self, image: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        if not HAS_ALBUMENTATIONS:
            # Fallback to basic augmentation
            return self._basic_augmentation(image, mask)
            
        # Convert to uint8 for augmentation
        image_uint8 = (image * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Define augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
                A.HueSaturationValue(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
            ], p=0.2),
        ])
        
        try:
            # Apply transformation
            augmented = transform(image=image_uint8, mask=mask_uint8)
            
            # Convert back to float32
            aug_image = augmented['image'].astype(np.float32) / 255.0
            aug_mask = augmented['mask'].astype(np.float32) / 255.0
            
            return aug_image, aug_mask
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, mask
    
    def _basic_augmentation(self, image: np.ndarray, 
                           mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Basic augmentation without albumentations"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip  
        if np.random.random() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random brightness adjustment
        if np.random.random() > 0.7:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)
        
        return image, mask


class DataAugmentation:
    """
    Additional data augmentation utilities
    """
    
    @staticmethod
    def random_crop(image: np.ndarray, mask: np.ndarray, 
                   crop_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop with consistent position for image and mask"""
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        if h <= crop_h or w <= crop_w:
            return image, mask
        
        top = np.random.randint(0, h - crop_h)
        left = np.random.randint(0, w - crop_w)
        
        cropped_image = image[top:top+crop_h, left:left+crop_w]
        cropped_mask = mask[top:top+crop_h, left:left+crop_w]
        
        return cropped_image, cropped_mask
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, 
                          noise_factor: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_factor, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, 
                         brightness_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Adjust brightness randomly"""
        factor = np.random.uniform(brightness_range[0], brightness_range[1])
        bright_image = image * factor
        return np.clip(bright_image, 0, 1)


class SyntheticDataGenerator:
    """
    Generate synthetic satellite-like images for testing
    """
    
    def __init__(self, image_size: Tuple[int, int] = (256, 256)):
        self.image_size = image_size
    
    def generate_sample_image(self, num_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic satellite image with corresponding mask"""
        h, w = self.image_size
        
        # Create base image (ground/background)
        image = np.random.uniform(0.2, 0.4, (h, w, 3)).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Add random features (buildings, water bodies, vegetation)
        for _ in range(num_features):
            feature_type = np.random.choice(['building', 'water', 'vegetation'])
            
            # Random position and size
            center_x = np.random.randint(50, w - 50)
            center_y = np.random.randint(50, h - 50)
            size = np.random.randint(20, 80)
            
            if feature_type == 'building':
                # Gray rectangular buildings
                x1, y1 = max(0, center_x - size//2), max(0, center_y - size//2)
                x2, y2 = min(w, center_x + size//2), min(h, center_y + size//2)
                image[y1:y2, x1:x2] = [0.6, 0.6, 0.6]  # Gray color
                mask[y1:y2, x1:x2] = 1  # Mark as target class
                
            elif feature_type == 'water':
                # Blue circular water bodies
                y, x = np.ogrid[:h, :w]
                mask_circle = (x - center_x)**2 + (y - center_y)**2 <= (size//2)**2
                image[mask_circle] = [0.1, 0.3, 0.8]  # Blue color
                mask[mask_circle] = 1  # Mark as target class
                
            elif feature_type == 'vegetation':
                # Green irregular vegetation
                y, x = np.ogrid[:h, :w]
                mask_circle = (x - center_x)**2 + (y - center_y)**2 <= (size//2)**2
                # Add some randomness to make it more irregular
                noise = np.random.random((h, w)) > 0.3
                vegetation_mask = mask_circle & noise
                image[vegetation_mask] = [0.2, 0.7, 0.3]  # Green color
                mask[vegetation_mask] = 1  # Mark as target class
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1).astype(np.float32)
        
        return image, mask.astype(np.float32)
    
    def create_synthetic_dataset(self, num_samples: int = 100, 
                               save_path: Optional[str] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate a dataset of synthetic images and masks"""
        images, masks = [], []
        
        for i in range(num_samples):
            image, mask = self.generate_sample_image()
            images.append(image)
            masks.append(mask)
            
            if save_path:
                # Save synthetic data
                os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
                os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)
                
                # Convert to uint8 for saving
                image_uint8 = (image * 255).astype(np.uint8)
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                cv2.imwrite(os.path.join(save_path, 'images', f'synthetic_{i:04d}.png'), 
                           cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_path, 'masks', f'synthetic_{i:04d}.png'), mask_uint8)
        
        print(f"Generated {num_samples} synthetic samples")
        if save_path:
            print(f"Saved synthetic data to {save_path}")
        
        return images, masks