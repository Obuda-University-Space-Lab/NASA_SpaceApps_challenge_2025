"""
Unit tests for Luna (Moon) analysis module
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from luna.moon_analysis import LunaDataProcessor


class TestLunaDataProcessor:
    """Test cases for Luna data processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = LunaDataProcessor()
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.surface_data == {}
        assert self.processor.subsurface_data == {}
        assert self.processor.orbital_data == {}
    
    def test_analyze_surface_composition(self):
        """Test surface composition analysis."""
        coords = (45.0, -120.0)
        result = self.processor.analyze_surface_composition(coords)
        
        assert "coordinates" in result
        assert "regolith_depth" in result
        assert "mineral_composition" in result
        assert "water_ice_probability" in result
        assert "analysis_timestamp" in result
        
        assert result["coordinates"] == coords
        assert 1 <= result["regolith_depth"] <= 10
        assert 0 <= result["water_ice_probability"] <= 0.8
        
        # Check mineral composition
        minerals = result["mineral_composition"]
        assert "anorthite" in minerals
        assert "pyroxene" in minerals
        assert "olivine" in minerals
        assert "ilmenite" in minerals
        assert "other" in minerals
        
        # Check that percentages are reasonable (each mineral should be positive)
        for mineral, percentage in minerals.items():
            assert percentage > 0, f"{mineral} percentage should be positive"
    
    def test_process_crater_analysis(self):
        """Test crater analysis."""
        result = self.processor.process_crater_analysis("Mare Tranquillitatis")
        
        assert "region" in result
        assert "crater_count" in result
        assert "average_diameter" in result
        assert "age_distribution" in result
        assert "impact_frequency" in result
        assert "geological_insights" in result
        
        assert result["region"] == "Mare Tranquillitatis"
        assert 50 <= result["crater_count"] <= 500
        assert 0.5 <= result["average_diameter"] <= 50
        
        # Check age distribution
        age_dist = result["age_distribution"]
        assert "young" in age_dist
        assert "medium" in age_dist
        assert "old" in age_dist
        
        # Check that geological insights are provided
        assert len(result["geological_insights"]) > 0
    
    def test_calculate_landing_site_suitability(self):
        """Test landing site suitability calculation."""
        coords = (20.0, -30.0)
        result = self.processor.calculate_landing_site_suitability(coords)
        
        assert "coordinates" in result
        assert "terrain_slope" in result
        assert "boulder_density" in result
        assert "communication_visibility" in result
        assert "solar_exposure" in result
        assert "safety_score" in result
        assert "science_value" in result
        assert "resource_potential" in result
        assert "overall_score" in result
        
        assert result["coordinates"] == coords
        assert 0 <= result["terrain_slope"] <= 15
        assert 0 <= result["boulder_density"] <= 0.3
        assert 0.6 <= result["communication_visibility"] <= 1.0
        assert 0.7 <= result["solar_exposure"] <= 1.0
        assert 0 <= result["safety_score"] <= 1.0
        assert 0.5 <= result["science_value"] <= 1.0
        assert 0.3 <= result["resource_potential"] <= 0.9
        assert 0 <= result["overall_score"] <= 1.0