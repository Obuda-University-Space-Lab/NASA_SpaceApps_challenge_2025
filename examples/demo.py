#!/usr/bin/env python3
"""
Example usage of the satellite image analysis system
Demonstrates different features and capabilities
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.satellite_analysis.image_processor import SatelliteImageProcessor
from src.satellite_analysis.data_loader import SyntheticDataGenerator
from src.models.segmentation_models import SegmentationTrainer


def example_image_processing():
    """Example of basic image processing operations"""
    print("=== Image Processing Example ===")
    
    # Create synthetic data for demonstration
    generator = SyntheticDataGenerator()
    sample_image, sample_mask = generator.generate_sample_image()
    
    # Initialize processor
    processor = SatelliteImageProcessor()
    
    # Enhance contrast
    enhanced_image = processor.enhance_contrast(sample_image, method='clahe')
    
    # Calculate indices (simulate multi-band data)
    # Add a fake NIR band
    multispectral_image = np.concatenate([
        sample_image,
        np.random.uniform(0.3, 0.8, sample_image.shape[:2] + (1,))  # Fake NIR
    ], axis=2)
    
    bands = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
    indices = processor.calculate_indices(multispectral_image, bands)
    
    # Visualize results
    processor.visualize_image(sample_image, "Original Synthetic Image")
    processor.visualize_image(enhanced_image, "Enhanced Image")
    processor.visualize_indices(indices)
    
    print("Image processing example completed!")


def example_synthetic_data_generation():
    """Example of synthetic data generation"""
    print("=== Synthetic Data Generation Example ===")
    
    generator = SyntheticDataGenerator(image_size=(256, 256))
    
    # Generate a few samples
    images, masks = generator.create_synthetic_dataset(
        num_samples=10,
        save_path="examples/synthetic_demo"
    )
    
    print(f"Generated {len(images)} synthetic samples")
    print(f"Image shape: {images[0].shape}")
    print(f"Mask shape: {masks[0].shape}")
    
    return images, masks


def example_model_training():
    """Example of model training with synthetic data"""
    print("=== Model Training Example ===")
    
    # Generate training data
    images, masks = example_synthetic_data_generation()
    
    if len(images) == 0:
        print("No training data available")
        return
    
    # Initialize trainer
    trainer = SegmentationTrainer(model_type='unet')
    
    # Build model
    model = trainer.build_keras_unet(
        input_shape=(256, 256, 3),
        num_classes=1
    )
    
    # Compile model
    model = trainer.compile_model(model)
    
    print("Model created and compiled successfully!")
    print(f"Model has {model.count_params():,} parameters")
    
    # Note: Actual training would require proper data loaders
    # This is just for demonstration of model creation
    
    return model


def example_evaluation_metrics():
    """Example of evaluation metrics calculation"""
    print("=== Evaluation Metrics Example ===")
    
    # Generate sample predictions and ground truth
    height, width = 256, 256
    
    # Simulate ground truth mask
    gt_mask = np.random.choice([0, 1], size=(height, width), p=[0.7, 0.3])
    
    # Simulate prediction (with some noise)
    pred_mask = gt_mask.copy()
    noise_indices = np.random.choice(height * width, size=int(0.1 * height * width), replace=False)
    pred_mask.flat[noise_indices] = 1 - pred_mask.flat[noise_indices]
    
    # Calculate metrics manually
    intersection = np.sum(gt_mask * pred_mask)
    union = np.sum((gt_mask + pred_mask) > 0)
    iou = intersection / (union + 1e-8)
    
    dice = (2 * intersection) / (np.sum(gt_mask) + np.sum(pred_mask) + 1e-8)
    
    accuracy = np.sum(gt_mask == pred_mask) / (height * width)
    
    print(f"IoU Score: {iou:.4f}")
    print(f"Dice Score: {dice:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy
    }


def main():
    """Run all examples"""
    print("NASA SpaceApps 2025 - Satellite Image Analysis Examples")
    print("=" * 60)
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    try:
        # Run examples
        example_image_processing()
        print("\n" + "="*60 + "\n")
        
        example_synthetic_data_generation()
        print("\n" + "="*60 + "\n")
        
        example_model_training()
        print("\n" + "="*60 + "\n")
        
        example_evaluation_metrics()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()