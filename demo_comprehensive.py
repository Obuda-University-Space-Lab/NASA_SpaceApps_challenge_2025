#!/usr/bin/env python3
"""
Comprehensive demo of satellite image analysis functionality
Creates synthetic data and demonstrates key features
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from satellite_analysis.image_processor import SatelliteImageProcessor
from satellite_analysis.data_loader import SyntheticDataGenerator


def demo_synthetic_data_generation():
    """Demonstrate synthetic data generation"""
    print("=== Synthetic Data Generation Demo ===")
    
    generator = SyntheticDataGenerator(image_size=(256, 256))
    
    # Generate multiple samples
    images, masks = [], []
    for i in range(6):
        img, mask = generator.generate_sample_image(num_features=np.random.randint(3, 8))
        images.append(img)
        masks.append(mask)
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i in range(3):
        # Show image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Synthetic Image {i+1}')
        axes[i, 0].axis('off')
        
        # Show mask
        axes[i, 1].imshow(masks[i], cmap='RdYlBu')
        axes[i, 1].set_title(f'Segmentation Mask {i+1}')
        axes[i, 1].axis('off')
        
        # Show enhanced image
        processor = SatelliteImageProcessor()
        enhanced = processor.enhance_contrast(images[i])
        axes[i, 2].imshow(enhanced)
        axes[i, 2].set_title(f'Enhanced Image {i+1}')
        axes[i, 2].axis('off')
        
        # Show overlay
        overlay = images[i].copy()
        mask_colored = plt.cm.RdYlBu(masks[i])[:, :, :3]
        overlay = 0.7 * overlay + 0.3 * mask_colored
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Segmentation Overlay {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_synthetic_data.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Generated {len(images)} synthetic samples")
    return images, masks


def demo_spectral_analysis():
    """Demonstrate spectral analysis capabilities"""
    print("\n=== Spectral Analysis Demo ===")
    
    # Generate synthetic multispectral image
    generator = SyntheticDataGenerator(image_size=(200, 200))
    rgb_image, _ = generator.generate_sample_image()
    
    # Simulate NIR and SWIR bands
    nir_band = np.random.uniform(0.4, 0.9, rgb_image.shape[:2])
    swir_band = np.random.uniform(0.2, 0.6, rgb_image.shape[:2])
    
    # Add vegetation signature to NIR (higher values for green areas)
    green_mask = rgb_image[:, :, 1] > 0.6  # Green areas
    nir_band[green_mask] *= 1.8  # Higher NIR for vegetation
    
    # Add water signature (lower NIR for water)
    blue_mask = (rgb_image[:, :, 2] > 0.7) & (rgb_image[:, :, 0] < 0.3)  # Water areas
    nir_band[blue_mask] *= 0.3  # Lower NIR for water
    
    # Combine bands
    multispectral = np.stack([
        rgb_image[:, :, 0],  # Red
        rgb_image[:, :, 1],  # Green  
        rgb_image[:, :, 2],  # Blue
        nir_band,            # NIR
        swir_band            # SWIR
    ], axis=2)
    
    # Calculate spectral indices
    processor = SatelliteImageProcessor()
    bands = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3, 'swir1': 4}
    indices = processor.calculate_indices(multispectral, bands)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original RGB
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # NIR band
    axes[0, 1].imshow(nir_band, cmap='RdYlGn')
    axes[0, 1].set_title('NIR Band')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].get_images()[0], ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # False color composite (NIR, Red, Green)
    false_color = np.stack([nir_band, rgb_image[:, :, 0], rgb_image[:, :, 1]], axis=2)
    false_color = np.clip(false_color, 0, 1)
    axes[0, 2].imshow(false_color)
    axes[0, 2].set_title('False Color (NIR-R-G)')
    axes[0, 2].axis('off')
    
    # Spectral indices
    idx_names = ['ndvi', 'ndwi', 'savi']
    cmaps = ['RdYlGn', 'RdYlBu', 'RdYlGn']
    
    for i, (idx_name, cmap) in enumerate(zip(idx_names, cmaps)):
        if idx_name in indices:
            im = axes[1, i].imshow(indices[idx_name], cmap=cmap, vmin=-1, vmax=1)
            axes[1, i].set_title(f'{idx_name.upper()}')
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        else:
            axes[1, i].text(0.5, 0.5, f'{idx_name.upper()}\nNot Available', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_spectral_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Calculated spectral indices: {list(indices.keys())}")
    return multispectral, indices


def demo_image_processing():
    """Demonstrate image processing capabilities"""
    print("\n=== Image Processing Demo ===")
    
    # Generate test image
    generator = SyntheticDataGenerator(image_size=(200, 200))
    original_image, _ = generator.generate_sample_image()
    
    # Add some noise and artifacts
    noisy_image = original_image + np.random.normal(0, 0.1, original_image.shape)
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Apply different enhancement methods
    processor = SatelliteImageProcessor()
    
    methods = ['clahe', 'histogram_eq', 'gamma']
    enhanced_images = {}
    
    for method in methods:
        enhanced = processor.enhance_contrast(noisy_image, method=method)
        enhanced_images[method] = enhanced
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_image)
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    # Show two enhancement methods
    axes[1, 0].imshow(enhanced_images['clahe'])
    axes[1, 0].set_title('CLAHE Enhanced')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(enhanced_images['histogram_eq'])
    axes[1, 1].set_title('Histogram Equalized')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_image_processing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Applied enhancement methods: {methods}")
    return enhanced_images


def create_comprehensive_demo():
    """Create a comprehensive demonstration document"""
    print("\nNASA SpaceApps 2025 - Satellite Image Analysis")
    print("Comprehensive Functionality Demonstration")
    print("=" * 50)
    
    # Set matplotlib backend for headless environment
    plt.switch_backend('Agg')
    
    # Run all demos
    print("\n1. Generating synthetic satellite data...")
    images, masks = demo_synthetic_data_generation()
    
    print("\n2. Demonstrating spectral analysis...")
    multispectral, indices = demo_spectral_analysis()
    
    print("\n3. Showcasing image processing...")
    enhanced = demo_image_processing()
    
    # Create summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Generated {len(images)} synthetic image-mask pairs")
    print(f"Image dimensions: {images[0].shape}")
    print(f"Calculated spectral indices: {list(indices.keys())}")
    print(f"Applied {len(enhanced)} enhancement methods")
    
    # Save a sample dataset
    sample_dir = "demo_output"
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in range(min(3, len(images))):
        # Save as PNG (convert to uint8)
        img_uint8 = (images[i] * 255).astype(np.uint8)
        mask_uint8 = (masks[i] * 255).astype(np.uint8)
        
        plt.imsave(f"{sample_dir}/sample_image_{i:02d}.png", img_uint8)
        plt.imsave(f"{sample_dir}/sample_mask_{i:02d}.png", mask_uint8, cmap='gray')
    
    print(f"\nSample data saved to {sample_dir}/")
    print("Demonstration completed successfully!")
    
    # List generated files
    print("\nGenerated visualization files:")
    viz_files = ['demo_synthetic_data.png', 'demo_spectral_analysis.png', 'demo_image_processing.png']
    for f in viz_files:
        if os.path.exists(f):
            print(f"  - {f}")


if __name__ == "__main__":
    create_comprehensive_demo()