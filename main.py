#!/usr/bin/env python3
"""
Main application for NASA SpaceApps 2025 - Satellite Image Analysis
AI-based Image Segmentation for Satellite Imagery
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Optional imports
try:
    from satellite_analysis.image_processor import SatelliteImageProcessor
    from satellite_analysis.data_loader import SatelliteDataLoader, SyntheticDataGenerator
    HAS_CORE_MODULES = True
except ImportError as e:
    print(f"Warning: Core modules not available: {e}")
    HAS_CORE_MODULES = False

try:
    from models.segmentation_models import SegmentationTrainer, DeepLabV3Plus
    import tensorflow as tf
    HAS_AI_MODULES = True
except ImportError as e:
    print(f"Warning: AI modules not available: {e}")
    HAS_AI_MODULES = False

import warnings
warnings.filterwarnings('ignore')


def create_sample_data(output_dir: str = "data/synthetic", num_samples: int = 50):
    """Create synthetic satellite data for demonstration"""
    if not HAS_CORE_MODULES:
        print("Error: Core modules not available. Please install required dependencies.")
        return None
        
    print("Creating synthetic satellite data for demonstration...")
    
    generator = SyntheticDataGenerator(image_size=(256, 256))
    images, masks = generator.create_synthetic_dataset(
        num_samples=num_samples,
        save_path=output_dir
    )
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'Synthetic Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(masks[i], cmap='gray')
        axes[1, i].set_title(f'Segmentation Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_data.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Sample data created and saved to {output_dir}")
    return output_dir


def train_segmentation_model(data_dir: str, model_type: str = "unet", 
                           epochs: int = 10, batch_size: int = 8):
    """Train a segmentation model"""
    if not HAS_AI_MODULES:
        print("Error: AI modules not available. Please install TensorFlow/PyTorch:")
        print("pip install tensorflow torch torchvision")
        return None
        
    print(f"Training {model_type} model...")
    
    # Initialize data loader
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    data_loader = SatelliteDataLoader(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=(256, 256),
        batch_size=batch_size
    )
    
    # Create data generators
    try:
        train_dataset, val_dataset = data_loader.create_data_generators(
            validation_split=0.2,
            augment=True
        )
    except ValueError as e:
        print(f"Error creating data generators: {e}")
        return None
    
    # Initialize trainer
    trainer = SegmentationTrainer(model_type=model_type)
    
    # Build model
    if model_type.lower() == 'unet':
        model = trainer.build_keras_unet(
            input_shape=(256, 256, 3),
            num_classes=1
        )
    elif model_type.lower() == 'deeplabv3plus':
        deeplabv3plus = DeepLabV3Plus(
            input_shape=(256, 256, 3),
            num_classes=1
        )
        model = deeplabv3plus.build_model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Compile model
    model = trainer.compile_model(model)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = trainer.train_model(
        model=model,
        train_generator=train_dataset,
        val_generator=val_dataset,
        epochs=epochs,
        patience=5
    )
    
    # Save model
    model_save_path = f"models/{model_type}_satellite_segmentation.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plot_training_history(history, model_type)
    
    return model, history


def plot_training_history(history, model_type: str):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Dice coefficient
    axes[0, 1].plot(history.history['dice_coefficient'], label='Training Dice')
    axes[0, 1].plot(history.history['val_dice_coefficient'], label='Validation Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    
    # IoU
    axes[1, 0].plot(history.history['iou_metric'], label='Training IoU')
    axes[1, 0].plot(history.history['val_iou_metric'], label='Validation IoU')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    
    # Accuracy
    axes[1, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_type}.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_satellite_image(image_path: str, model_path: Optional[str] = None):
    """Analyze a single satellite image"""
    if not HAS_CORE_MODULES:
        print("Error: Core modules not available. Please install required dependencies.")
        return None
        
    print(f"Analyzing satellite image: {image_path}")
    
    # Initialize processor
    processor = SatelliteImageProcessor()
    
    # Load and preprocess image
    try:
        image = processor.load_image(image_path)
        print(f"Loaded image with shape: {image.shape}")
        
        # Preprocess
        processed_image = processor.preprocess_image(
            image, 
            target_size=(256, 256), 
            normalize=True
        )
        
        # Enhance contrast
        enhanced_image = processor.enhance_contrast(processed_image, method='clahe')
        
        # Calculate indices if multi-band
        if len(processed_image.shape) == 3 and processed_image.shape[2] >= 4:
            # Assume standard band order: R, G, B, NIR
            bands = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
            indices = processor.calculate_indices(processed_image, bands)
        else:
            indices = {}
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(processed_image[:, :, :3] if processed_image.shape[2] >= 3 
                         else processed_image, cmap='gray' if len(processed_image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Enhanced image
        axes[0, 1].imshow(enhanced_image[:, :, :3] if enhanced_image.shape[2] >= 3 
                         else enhanced_image, cmap='gray' if len(enhanced_image.shape) == 2 else None)
        axes[0, 1].set_title('Enhanced Image')
        axes[0, 1].axis('off')
        
        # Segmentation prediction (if model is available)
        if model_path and os.path.exists(model_path) and HAS_AI_MODULES:
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Prepare image for prediction
                input_image = np.expand_dims(processed_image[:, :, :3], axis=0)
                prediction = model.predict(input_image)
                pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.float32)
                
                axes[0, 2].imshow(pred_mask, cmap='RdYlBu')
                axes[0, 2].set_title('Segmentation Prediction')
                axes[0, 2].axis('off')
            except Exception as e:
                print(f"Error loading model: {e}")
                axes[0, 2].text(0.5, 0.5, 'Model not available', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].axis('off')
        else:
            axes[0, 2].text(0.5, 0.5, 'Model not provided', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
        
        # Display indices
        if indices:
            idx_names = list(indices.keys())[:3]  # Show first 3 indices
            for i, name in enumerate(idx_names):
                axes[1, i].imshow(indices[name], cmap='RdYlGn' if 'ndvi' in name.lower() 
                                 else 'RdYlBu' if 'ndwi' in name.lower() else 'viridis')
                axes[1, i].set_title(f'{name.upper()}')
                axes[1, i].axis('off')
                
                # Add colorbar
                im = axes[1, i].get_images()[0]
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        else:
            for i in range(3):
                axes[1, i].text(0.5, 0.5, 'Multi-band data\nnot available', 
                               ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('satellite_analysis_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="NASA SpaceApps 2025 - Satellite Image Analysis with AI Segmentation"
    )
    
    parser.add_argument(
        'command',
        choices=['demo', 'train', 'analyze', 'create-data'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--image-path',
        type=str,
        help='Path to satellite image for analysis'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/synthetic',
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['unet', 'deeplabv3plus'],
        default='unet',
        help='Type of segmentation model to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model for inference'
    )
    
    args = parser.parse_args()
    
    print("NASA SpaceApps 2025 - Satellite Image Analysis")
    print("=" * 50)
    
    if args.command == 'create-data':
        create_sample_data(args.data_dir, num_samples=50)
        
    elif args.command == 'demo':
        # Run complete demo
        print("Running complete demonstration...")
        
        # Create sample data
        data_dir = create_sample_data(args.data_dir, num_samples=50)
        
        # Train model
        model, history = train_segmentation_model(
            data_dir=data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Analyze a sample image
        sample_image = os.path.join(data_dir, 'images', 'synthetic_0000.png')
        if os.path.exists(sample_image):
            model_path = f"models/{args.model_type}_satellite_segmentation.h5"
            analyze_satellite_image(sample_image, model_path)
        
        print("Demo completed successfully!")
        
    elif args.command == 'train':
        if not os.path.exists(args.data_dir):
            print(f"Data directory {args.data_dir} does not exist.")
            print("Please create training data first using 'create-data' command.")
            return
        
        train_segmentation_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
    elif args.command == 'analyze':
        if not args.image_path:
            print("Please provide --image-path for analysis")
            return
        
        if not os.path.exists(args.image_path):
            print(f"Image file {args.image_path} does not exist")
            return
        
        analyze_satellite_image(args.image_path, args.model_path)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()