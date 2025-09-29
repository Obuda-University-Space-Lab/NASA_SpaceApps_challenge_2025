"""
Unit tests for the Analytics Core module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from analytics.core import AnalyticsEngine


class TestAnalyticsEngine:
    """Test cases for the main analytics engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = AnalyticsEngine()
    
    def test_initialization(self):
        """Test that the analytics engine initializes correctly."""
        assert self.engine.terra_data == {}
        assert self.engine.luna_data == {}
        assert self.engine.processed_datasets == {}
    
    def test_load_terra_data(self):
        """Test loading Terra data."""
        result = self.engine.load_terra_data("test_source", region="test_region")
        assert result is True
        assert "test_source" in self.engine.terra_data
        assert self.engine.terra_data["test_source"]["status"] == "loaded"
    
    def test_load_luna_data(self):
        """Test loading Luna data."""
        result = self.engine.load_luna_data("test_source", mission="test_mission")
        assert result is True
        assert "test_source" in self.engine.luna_data
        assert self.engine.luna_data["test_source"]["status"] == "loaded"
    
    def test_cross_platform_analysis(self):
        """Test cross-platform analysis."""
        # Load some test data
        self.engine.load_terra_data("terra_test")
        self.engine.load_luna_data("luna_test")
        
        # Run analysis
        results = self.engine.process_cross_platform_analysis()
        
        assert "terra_datasets" in results
        assert "luna_datasets" in results
        assert "analysis_timestamp" in results
        assert results["terra_datasets"] == 1
        assert results["luna_datasets"] == 1
    
    def test_get_data_summary(self):
        """Test getting data summary."""
        # Load test data
        self.engine.load_terra_data("terra_test")
        self.engine.load_luna_data("luna_test")
        
        summary = self.engine.get_data_summary()
        
        assert "terra_sources" in summary
        assert "luna_sources" in summary
        assert "processed_analyses" in summary
        assert "status" in summary
        assert len(summary["terra_sources"]) == 1
        assert len(summary["luna_sources"]) == 1