# Configuration file for satellite image analysis

# Model Configuration
MODEL_CONFIG = {
    "input_size": [256, 256, 3],
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
}

# Data Configuration
DATA_CONFIG = {
    "augmentation": True,
    "normalize": True,
    "supported_formats": ['.tif', '.tiff', '.png', '.jpg', '.jpeg'],
    "synthetic_samples": 100,
}

# Segmentation Models
MODELS = {
    "unet": {
        "description": "U-Net architecture for biomedical image segmentation",
        "input_channels": 3,
        "output_classes": 1,
    },
    "deeplabv3plus": {
        "description": "DeepLabV3+ with MobileNetV2 backbone",
        "input_channels": 3,
        "output_classes": 1,
        "backbone": "MobileNetV2",
    }
}

# Spectral Indices
SPECTRAL_INDICES = {
    "ndvi": {
        "name": "Normalized Difference Vegetation Index",
        "formula": "(NIR - Red) / (NIR + Red)",
        "range": [-1, 1],
        "interpretation": "Higher values indicate healthier vegetation"
    },
    "ndwi": {
        "name": "Normalized Difference Water Index",
        "formula": "(Green - NIR) / (Green + NIR)",
        "range": [-1, 1],
        "interpretation": "Higher values indicate water bodies"
    },
    "savi": {
        "name": "Soil-Adjusted Vegetation Index",
        "formula": "((NIR - Red) * (1 + L)) / (NIR + Red + L)",
        "range": [-1, 1],
        "interpretation": "Vegetation index that accounts for soil brightness"
    }
}

# Band Mappings for Common Satellites
SATELLITE_BANDS = {
    "landsat8": {
        "coastal": 0,    # Band 1
        "blue": 1,       # Band 2
        "green": 2,      # Band 3
        "red": 3,        # Band 4
        "nir": 4,        # Band 5
        "swir1": 5,      # Band 6
        "swir2": 6,      # Band 7
    },
    "sentinel2": {
        "blue": 0,       # B2
        "green": 1,      # B3
        "red": 2,        # B4
        "nir": 3,        # B8
        "swir1": 4,      # B11
        "swir2": 5,      # B12
    }
}