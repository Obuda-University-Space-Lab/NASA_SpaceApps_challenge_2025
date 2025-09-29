#!/usr/bin/env python3
"""
Terra & Luna Analytics - Demonstration Script
Shows key functionality of the NASA Space Apps 2025 project
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from analytics.core import AnalyticsEngine
from terra.earth_analysis import TerraDataProcessor
from luna.moon_analysis import LunaDataProcessor


def main():
    """Run demonstration of Terra & Luna Analytics."""
    print("=" * 60)
    print("🚀 Terra & Luna Analytics - NASA Space Apps 2025")
    print("📡 Óbuda University Space Lab")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 Initializing analytics components...")
    engine = AnalyticsEngine()
    terra_processor = TerraDataProcessor()
    luna_processor = LunaDataProcessor()
    print("✅ All components initialized successfully!")
    
    # Terra (Earth) Analytics Demo
    print("\n🌍 TERRA (EARTH) ANALYTICS DEMONSTRATION")
    print("-" * 40)
    
    # Process sample satellite imagery
    imagery_result = terra_processor.process_satellite_imagery("sample_landsat_scene.tif")
    print(f"📡 Satellite Imagery Processing:")
    print(f"   └ NDVI Calculated: {imagery_result['ndvi_calculated']}")
    print(f"   └ Cloud Cover: {imagery_result['cloud_cover']:.1f}%")
    print(f"   └ Quality Score: {imagery_result['quality_score']:.3f}")
    
    # Analyze climate patterns
    climate_result = terra_processor.analyze_climate_patterns("Europe", "2024")
    print(f"🌡️  Climate Pattern Analysis (Europe, 2024):")
    print(f"   └ Temperature Trend: {climate_result['temperature_trend']:.2f}°C")
    print(f"   └ Precipitation Change: {climate_result['precipitation_change']:.1f}%")
    print(f"   └ Extreme Events: {climate_result['extreme_events']}")
    
    # Calculate environmental indices
    env_indices = terra_processor.calculate_environmental_indices({"region": "sample"})
    print(f"📊 Environmental Indices:")
    print(f"   └ NDVI: {env_indices['ndvi']:.3f}")
    print(f"   └ NDWI: {env_indices['ndwi']:.3f}")
    print(f"   └ EVI: {env_indices['evi']:.3f}")
    print(f"   └ LST: {env_indices['lst']:.1f}K")
    
    # Luna (Moon) Analytics Demo
    print("\n🌙 LUNA (MOON) ANALYTICS DEMONSTRATION")
    print("-" * 40)
    
    # Analyze surface composition
    test_coords = (45.0, -120.0)
    surface_comp = luna_processor.analyze_surface_composition(test_coords)
    print(f"🔬 Surface Composition Analysis ({test_coords[0]}°, {test_coords[1]}°):")
    print(f"   └ Regolith Depth: {surface_comp['regolith_depth']:.2f} meters")
    print(f"   └ Water Ice Probability: {surface_comp['water_ice_probability']:.2%}")
    print("   └ Mineral Composition:")
    for mineral, percentage in surface_comp['mineral_composition'].items():
        print(f"      • {mineral.title()}: {percentage:.1f}%")
    
    # Process crater analysis
    crater_analysis = luna_processor.process_crater_analysis("Mare Tranquillitatis")
    print(f"🌑 Crater Analysis (Mare Tranquillitatis):")
    print(f"   └ Crater Count: {crater_analysis['crater_count']}")
    print(f"   └ Average Diameter: {crater_analysis['average_diameter']:.2f} km")
    print(f"   └ Impact Frequency: {crater_analysis['impact_frequency']:.2e} impacts/km²/year")
    
    # Landing site suitability
    landing_site = luna_processor.calculate_landing_site_suitability(test_coords)
    print(f"🚀 Landing Site Suitability ({test_coords[0]}°, {test_coords[1]}°):")
    print(f"   └ Safety Score: {landing_site['safety_score']:.3f}")
    print(f"   └ Science Value: {landing_site['science_value']:.3f}")
    print(f"   └ Resource Potential: {landing_site['resource_potential']:.3f}")
    print(f"   └ Overall Score: {landing_site['overall_score']:.3f}")
    
    # Cross-Platform Analysis Demo
    print("\n🔬 CROSS-PLATFORM ANALYTICS DEMONSTRATION")
    print("-" * 45)
    
    # Load sample data
    engine.load_terra_data("MODIS_NDVI", region="global", resolution="250m")
    engine.load_terra_data("Landsat_8", mission="OLI", coverage="continental")
    engine.load_luna_data("LRO_LROC", instrument="NAC", resolution="0.5m")
    engine.load_luna_data("Chang'e_4", landing_site="Von Kármán", instruments=["PCAM", "LPR"])
    
    # Perform cross-platform analysis
    cross_results = engine.process_cross_platform_analysis()
    print(f"📊 Cross-Platform Analysis Results:")
    print(f"   └ Terra Datasets: {cross_results['terra_datasets']}")
    print(f"   └ Luna Datasets: {cross_results['luna_datasets']}")
    print(f"   └ Analysis Timestamp: {cross_results['analysis_timestamp']}")
    print("   └ Key Insights:")
    for insight in cross_results['insights']:
        print(f"      • {insight}")
    
    # Data Summary
    summary = engine.get_data_summary()
    print(f"\n📈 DATA SUMMARY")
    print(f"-" * 20)
    print(f"Terra Sources: {', '.join(summary['terra_sources'])}")
    print(f"Luna Sources: {', '.join(summary['luna_sources'])}")
    print(f"Processed Analyses: {len(summary['processed_analyses'])}")
    print(f"System Status: {summary['status'].upper()}")
    
    print("\n" + "=" * 60)
    print("🎉 Demonstration completed successfully!")
    print("💡 To launch the interactive dashboard, run: python src/main.py")
    print("📚 Check out notebooks/getting_started.ipynb for detailed examples")
    print("=" * 60)


if __name__ == "__main__":
    main()