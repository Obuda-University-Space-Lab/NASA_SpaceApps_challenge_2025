#!/usr/bin/env python3
"""
Basic test script to validate core functionality without heavy dependencies
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


def test_data_augmentation():
    """Test data augmentation utilities"""
    print("Testing DataAugmentation...")
    
    try:
        from satellite_analysis.data_loader import DataAugmentation
        
        # Test data loader initialization (with non-existent directories)
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
        
        # Test brightness adjustment
        bright_image = DataAugmentation.adjust_brightness(test_image)
        assert bright_image.shape == test_image.shape, "Brightness adjusted image shape changed"
        
        print("✓ DataAugmentation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ DataAugmentation tests failed: {e}")
        return False


def test_main_script():
    """Test that main script can be imported"""
    print("Testing main script import...")
    
    try:
        # Change to parent directory temporarily
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, parent_dir)
        
        # Just test that we can import the main functions
        import main
        
        # Test that core functions exist
        assert hasattr(main, 'create_sample_data'), "create_sample_data function missing"
        assert hasattr(main, 'analyze_satellite_image'), "analyze_satellite_image function missing"
        assert hasattr(main, 'main'), "main function missing"
        
        print("✓ Main script tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Main script tests failed: {e}")
        return False


def run_basic_tests():
    """Run basic tests without heavy dependencies"""
    print("NASA SpaceApps 2025 - Satellite Image Analysis - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        test_image_processor,
        test_synthetic_data_generator,
        test_data_augmentation,
        test_main_script,
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
        print("🎉 All basic tests passed!")
        print("\nCore functionality is working correctly!")
        print("For full functionality, install all dependencies with:")
        print("pip install -r requirements.txt")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)