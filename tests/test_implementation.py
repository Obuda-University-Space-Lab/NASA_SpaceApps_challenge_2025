#!/usr/bin/env python3
"""
Simple test script to validate the satellite image analysis implementation
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_image_processor():
    """Test basic image processing functionality"""
    print("Testing SatelliteImageProcessor...")
    
    try:
        from satellite_analysis.image_processor import SatelliteImageProcessor
        
        processor = SatelliteImageProcessor()
        
        # Create a test image
        test_image = np.random.rand(100, 100, 3).astype(np.float32)
        
        # Test preprocessing
        processed = processor.preprocess_image(test_image, target_size=(64, 64))
        assert processed.shape == (64, 64, 3), f"Expected (64, 64, 3), got {processed.shape}"
        
        # Test contrast enhancement
        enhanced = processor.enhance_contrast(test_image)
        assert enhanced.shape == test_image.shape, f"Shape mismatch in contrast enhancement"
        
        # Test indices calculation
        multispectral = np.random.rand(100, 100, 4).astype(np.float32)
        bands = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}
        indices = processor.calculate_indices(multispectral, bands)
        
        assert 'ndvi' in indices, "NDVI not calculated"
        assert indices['ndvi'].shape == (100, 100), f"NDVI shape incorrect: {indices['ndvi'].shape}"
        
        print("✓ SatelliteImageProcessor tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SatelliteImageProcessor tests failed: {e}")
        return False


def test_synthetic_data_generator():
    """Test synthetic data generation"""
    print("Testing SyntheticDataGenerator...")
    
    try:
        from satellite_analysis.data_loader import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(image_size=(128, 128))
        
        # Generate a sample
        image, mask = generator.generate_sample_image()
        
        assert image.shape == (128, 128, 3), f"Image shape incorrect: {image.shape}"
        assert mask.shape == (128, 128), f"Mask shape incorrect: {mask.shape}"
        assert image.dtype == np.float32, f"Image dtype incorrect: {image.dtype}"
        assert mask.dtype == np.float32, f"Mask dtype incorrect: {mask.dtype}"
        
        # Check value ranges
        assert np.all((image >= 0) & (image <= 1)), "Image values not in [0, 1] range"
        assert np.all((mask >= 0) & (mask <= 1)), "Mask values not in [0, 1] range"
        
        print("✓ SyntheticDataGenerator tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SyntheticDataGenerator tests failed: {e}")
        return False


def test_segmentation_trainer():
    """Test segmentation model trainer"""
    print("Testing SegmentationTrainer...")
    
    try:
        from models.segmentation_models import SegmentationTrainer
        
        trainer = SegmentationTrainer(model_type='unet')
        
        # Build a small model for testing
        model = trainer.build_keras_unet(
            input_shape=(64, 64, 3),
            num_classes=1
        )
        
        assert model is not None, "Model not created"
        
        # Test metrics
        import tensorflow as tf
        
        # Create dummy data
        y_true = tf.constant([[1., 0.], [0., 1.]], dtype=tf.float32)
        y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8]], dtype=tf.float32)
        
        dice_score = trainer.dice_coefficient(y_true, y_pred)
        iou_score = trainer.iou_metric(y_true, y_pred)
        
        assert isinstance(dice_score.numpy(), (float, np.float32, np.float64)), "Dice score not numeric"
        assert isinstance(iou_score.numpy(), (float, np.float32, np.float64)), "IoU score not numeric"
        
        print("✓ SegmentationTrainer tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SegmentationTrainer tests failed: {e}")
        return False


def test_data_loader():
    """Test data loader (without actual files)"""
    print("Testing SatelliteDataLoader...")
    
    try:
        from satellite_analysis.data_loader import SatelliteDataLoader, DataAugmentation
        
        # Test data loader initialization (with non-existent directories)
        loader = SatelliteDataLoader(
            images_dir="./non_existent_images",
            masks_dir="./non_existent_masks",
            image_size=(128, 128),
            batch_size=4
        )
        
        assert loader.image_size == (128, 128), "Image size not set correctly"
        assert loader.batch_size == 4, "Batch size not set correctly"
        
        # Test augmentation utilities
        test_image = np.random.rand(100, 100, 3).astype(np.float32)
        test_mask = np.random.randint(0, 2, (100, 100)).astype(np.float32)
        
        # Test random crop
        cropped_img, cropped_mask = DataAugmentation.random_crop(
            test_image, test_mask, (64, 64)
        )
        
        assert cropped_img.shape == (64, 64, 3), f"Cropped image shape incorrect: {cropped_img.shape}"
        assert cropped_mask.shape == (64, 64), f"Cropped mask shape incorrect: {cropped_mask.shape}"
        
        # Test noise addition
        noisy_image = DataAugmentation.add_gaussian_noise(test_image)
        assert noisy_image.shape == test_image.shape, "Noisy image shape changed"
        
        print("✓ SatelliteDataLoader tests passed")
        return True
        
    except Exception as e:
        print(f"✗ SatelliteDataLoader tests failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("NASA SpaceApps 2025 - Satellite Image Analysis - Test Suite")
    print("=" * 60)
    
    tests = [
        test_image_processor,
        test_synthetic_data_generator,
        test_segmentation_trainer,
        test_data_loader,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
        
        print()  # Empty line between tests
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)