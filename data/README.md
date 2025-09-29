# Data Sources and APIs

This directory contains documentation and sample data for the Terra & Luna Analytics project.

## 🌍 Earth (Terra) Data Sources

### NASA Earth Observation Data

1. **MODIS (Moderate Resolution Imaging Spectroradiometer)**
   - Land surface temperature
   - Vegetation indices (NDVI, EVI)
   - Snow cover
   - Fire detection

2. **Landsat Program**
   - High-resolution multispectral imagery
   - Land use and land cover change
   - Urban development monitoring

3. **SRTM (Shuttle Radar Topography Mission)**
   - Global elevation data
   - Digital elevation models

### Copernicus Program (ESA)

1. **Sentinel-1**: SAR imaging for surface monitoring
2. **Sentinel-2**: Optical imaging for land monitoring
3. **Sentinel-3**: Ocean and land color monitoring

## 🌙 Moon (Luna) Data Sources

### NASA Lunar Data

1. **Lunar Reconnaissance Orbiter (LRO)**
   - LROC: High-resolution surface imaging
   - DIVINER: Temperature mapping
   - LOLA: Laser altimetry
   - Mini-RF: Radar imaging

2. **GRAIL Mission**
   - Gravity field mapping
   - Internal structure data

3. **ARTEMIS Program Data**
   - Landing site reconnaissance
   - Resource mapping

### International Lunar Data

1. **Chang'e Missions (CNSA)**
   - Surface composition analysis
   - Subsurface radar data

2. **Chandrayaan Missions (ISRO)**
   - Mineral mapping
   - Water ice detection

## 📡 API Endpoints and Access

### NASA APIs

- **NASA Open Data Portal**: https://data.nasa.gov/
- **Earthdata**: https://earthdata.nasa.gov/
- **PDS (Planetary Data System)**: https://pds.nasa.gov/

### Sample API Usage

```python
# Example Terra data access
from src.terra.earth_analysis import TerraDataProcessor

processor = TerraDataProcessor()
data = processor.process_satellite_imagery('path/to/landsat/scene')

# Example Luna data access
from src.luna.moon_analysis import LunaDataProcessor

luna_processor = LunaDataProcessor()
composition = luna_processor.analyze_surface_composition((lat, lon))
```

## 📁 Directory Structure

```
data/
├── terra/
│   ├── satellite_imagery/     # Earth observation imagery
│   ├── climate_data/         # Climate and weather data
│   └── environmental/        # Environmental monitoring data
├── luna/
│   ├── surface/              # Lunar surface data
│   ├── subsurface/           # Subsurface analysis data
│   └── orbital/              # Orbital mission data
├── processed/                # Processed analysis results
└── samples/                  # Sample datasets for development
```

## 🔄 Data Processing Pipeline

1. **Data Acquisition**: Automated downloads from NASA/ESA APIs
2. **Preprocessing**: Format conversion, quality control, calibration
3. **Analysis**: Scientific analysis using Terra/Luna processors
4. **Visualization**: Interactive charts and maps
5. **Storage**: Processed results stored for future use

## 📊 Sample Datasets

For development and testing, we provide sample datasets:

- `samples/terra_ndvi_sample.tif`: Sample NDVI data
- `samples/luna_elevation_sample.tif`: Sample lunar elevation data
- `samples/climate_timeseries.csv`: Sample climate time series data

## 🛡️ Data Usage Guidelines

- Respect NASA and ESA data usage policies
- Acknowledge data sources in publications
- Follow open data sharing principles
- Ensure proper data citation

## 📚 Additional Resources

- [NASA Earthdata User Guide](https://earthdata.nasa.gov/learn)
- [ESA Copernicus User Guide](https://scihub.copernicus.eu/userguide/)
- [Planetary Data System Guide](https://pds.nasa.gov/datastandards/)

---

For questions about data access or processing, please see the [Contributing Guidelines](../CONTRIBUTING.md).