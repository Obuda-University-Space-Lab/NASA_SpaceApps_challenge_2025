"""
Unit tests for Terra (Earth) analysis module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from terra.earth_analysis import TerraDataProcessor


class TestTerraDataProcessor:
    """Test cases for Terra data processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TerraDataProcessor()
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.satellite_data == {}
        assert self.processor.climate_data == {}
        assert self.processor.geographical_data == {}
    
    def test_process_satellite_imagery(self):
        """Test satellite imagery processing."""
        result = self.processor.process_satellite_imagery("test_image.tif")
        
        assert "image_path" in result
        assert "processed_at" in result
        assert "bands_analyzed" in result
        assert "ndvi_calculated" in result
        assert "cloud_cover" in result
        assert "quality_score" in result
        
        assert result["image_path"] == "test_image.tif"
        assert result["ndvi_calculated"] is True
        assert 0 <= result["cloud_cover"] <= 100
        assert 0.7 <= result["quality_score"] <= 1.0
    
    def test_analyze_climate_patterns(self):
        """Test climate pattern analysis."""
        result = self.processor.analyze_climate_patterns("Europe", "2024-01")
        
        assert "region" in result
        assert "time_period" in result
        assert "temperature_trend" in result
        assert "precipitation_change" in result
        assert "extreme_events" in result
        assert "analysis_confidence" in result
        
        assert result["region"] == "Europe"
        assert result["time_period"] == "2024-01"
        assert -2 <= result["temperature_trend"] <= 5
        assert -20 <= result["precipitation_change"] <= 30
        assert 0.8 <= result["analysis_confidence"] <= 0.95
    
    def test_calculate_environmental_indices(self):
        """Test environmental indices calculation."""
        test_data = {"region": "test", "time": "2024-01"}
        result = self.processor.calculate_environmental_indices(test_data)
        
        assert "ndvi" in result
        assert "ndwi" in result
        assert "evi" in result
        assert "lst" in result
        assert "calculation_timestamp" in result
        
        assert 0.2 <= result["ndvi"] <= 0.8
        assert -0.3 <= result["ndwi"] <= 0.3
        assert 0.1 <= result["evi"] <= 0.7
        assert 280 <= result["lst"] <= 320