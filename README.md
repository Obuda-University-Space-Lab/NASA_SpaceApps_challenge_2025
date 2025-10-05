<p align="center">
  <img src="https://assets.spaceappschallenge.org/media/images/Colorway2-Color_White3x.width-440.jpegquality-60.png" alt="NASA Space Apps Challenge" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Technologies](#technologies)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

**Inferno Watch** is a geospatial wildfire prediction system that combines meteorological data analysis with machine learning to forecast wildfire risk across user-defined geographic regions. The application provides an interactive web interface where users can select locations, specify prediction dates, and visualize fire risk predictions overlaid on an interactive map.

This project aims to support fire prevention efforts, emergency response planning, and environmental monitoring by providing accessible, data-driven wildfire risk assessments.

---

## Features

- **Interactive Map Interface**: Select any geographic location with latitude/longitude coordinates
- **Dynamic Area Selection**: Adjust prediction area size (30-200 km offset)
- **Real-time Weather Data Integration**: Fetches historical meteorological data from Open-Meteo API
- **Kriging Interpolation**: Generates smooth, continuous risk surfaces from discrete prediction points
- **Risk Visualization**: Color-coded markers and heat maps showing Low/Medium/High fire risk levels
- **Date-based Predictions**: Specify any date for wildfire risk forecasting
- **Responsive Web UI**: Built with Streamlit for intuitive user interaction

---

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
User Input → Weather Data Retrieval → Risk Prediction → Kriging Interpolation → Map Visualization
```

### Core Components

1. **`streamlit_app.py`** - Frontend application and user interface
2. **`logic.py`** - Business logic and prediction orchestration
3. **`meteo_data.py`** - Weather data acquisition from Open-Meteo API
4. **`kriging.py`** - Spatial interpolation and heat map generation

---

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Obuda-University-Space-Lab/NASA_SpaceApps_challenge_2025.git
   cd NASA_SpaceApps_challenge_2025
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Starting the Application

1. **Activate the virtual environment** (if not already activated)
   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the web interface**
   - The application will automatically open in your default browser
   - Default URL: `http://localhost:8501`

### Using the Interface

1. **Set Prediction Parameters**
   - **Longitude**: Enter longitude coordinate (-180.0 to 180.0)
   - **Latitude**: Enter latitude coordinate (-90.0 to 90.0)
   - **Date to Predict**: Select the date for fire risk prediction
   - **Area Offset**: Choose prediction area size (30-200 km)

2. **Generate Prediction**
   - Click the **"Predict"** button
   - The system will:
     - Fetch weather data for the past 31 days
     - Generate fire risk predictions
     - Create kriging interpolation
     - Display results on the interactive map

3. **Interpret Results**
   - **Green markers**: Low fire risk (<70%)
   - **Orange markers**: Medium fire risk (70-90%)
   - **Red flame icons**: High fire risk (>90%)
   - **Heat map overlay**: Continuous risk surface visualization

---

## Project Structure

```
NASA_SpaceApps_challenge_2025/
│
├── streamlit_app.py          # Main Streamlit web application
├── logic.py                  # Core prediction logic and orchestration
├── meteo_data.py             # Weather data retrieval from Open-Meteo API
├── kriging.py                # Spatial interpolation using PyKrige
├── requirements.txt          # Python package dependencies
│
├── data/                     # eg. Dataset directory
│   ├── greece_fire_places.csv
│   ├── greece_fire_dates.csv
│   ├── greece_fire_weather.csv
│   └── [other regional fire data]
│
├── weather_output/           # Generated weather data cache
├── design_elements/          # UI design assets
├── .streamlit/               # Streamlit configuration
└── venv/                     # Virtual environment (not in git)
```

### Key Files Description

#### `streamlit_app.py` (Main Application)
- Implements the web-based user interface
- Handles user input validation
- Manages map visualization with Folium
- Displays prediction results and risk markers
- Coordinates between UI and backend logic

#### `logic.py` (Business Logic)
- Orchestrates the prediction workflow
- Validates geographic bounds
- Manages data flow between components
- Generates prediction dataframes
- Integrates weather data with ML models

Key functions:
- `is_within_bounds()`: Validates coordinates within specified area
- `logic_func()`: Main prediction pipeline orchestrator

#### `meteo_data.py` (Weather Data Integration)
- Interfaces with Open-Meteo Archive API
- Fetches historical weather data for multiple locations
- Processes weather variables into structured dataframes
- Handles API requests and error management

Key functions:
- `fetch_point_history()`: Retrieves weather data for single coordinate
- `build_weather_dataframe()`: Aggregates data across multiple locations
- `meteo_data_extract()`: Main entry point for weather retrieval
- `load_coordinates_from_csv()`: Filters coordinates by bounds

Weather variables retrieved:
- Temperature (max, min, apparent)
- Precipitation and rainfall
- Wind speed and direction
- Solar radiation
- Evapotranspiration
- Daylight and sunshine duration

#### `kriging.py` (Spatial Interpolation)
- Implements Ordinary Kriging interpolation
- Generates smooth risk surfaces from discrete points
- Creates heat map visualizations
- Produces overlay images for Folium maps

Key functions:
- `krige_data()`: Performs kriging interpolation using spherical variogram
- `image_overlay()`: Creates Folium-compatible image overlay

---

## Data Sources

### Open-Meteo Archive API
The application uses the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) to retrieve meteorological data.

**API Endpoint**: `https://archive-api.open-meteo.com/v1/archive`

**Features**:
- Free, open-source weather API
- Historical data from 1940 to present
- High-resolution meteorological variables
- No API key required
- Worldwide coverage

**Retrieved Variables** (17 features):
- `temperature_2m_max` - Maximum 2m temperature
- `temperature_2m_min` - Minimum 2m temperature
- `apparent_temperature_max` - Maximum apparent temperature
- `apparent_temperature_min` - Minimum apparent temperature
- `sunrise` - Sunrise time
- `sunset` - Sunset time
- `daylight_duration` - Duration of daylight
- `sunshine_duration` - Duration of sunshine
- `precipitation_sum` - Total precipitation
- `rain_sum` - Total rainfall
- `snowfall_sum` - Total snowfall
- `precipitation_hours` - Hours with precipitation
- `wind_speed_10m_max` - Maximum wind speed at 10m
- `wind_gusts_10m_max` - Maximum wind gusts at 10m
- `wind_direction_10m_dominant` - Dominant wind direction
- `shortwave_radiation_sum` - Total solar radiation
- `et0_fao_evapotranspiration` - Reference evapotranspiration

### Fire Data
Historical fire occurrence data from NASA FIRMS (Fire Information for Resource Management System):
- Greece fire locations and dates (2000-2021)
- Italy fire data (2000-2021)
- Israel fire data (2000-2021)
- Tunisia fire data (2000-2021)

---

## Technologies

### Core Frameworks
- **Streamlit** (1.28.0+) - Web application framework
- **Folium** (0.14.0+) - Interactive map visualization
- **PyKrige** (1.7.0+) - Geostatistical analysis and kriging

### Data Processing
- **Pandas** (2.0.0+) - Data manipulation and analysis
- **NumPy** (1.24.0+) - Numerical computing
- **GeoPandas** (0.13.0+) - Geospatial data operations

### Machine Learning
- **scikit-learn** (1.3.0+) - Machine learning algorithms
- **joblib** - Model serialization

### Visualization
- **Matplotlib** (3.7.0+) - Plotting library
- **Seaborn** (0.12.0+) - Statistical visualization
- **Pillow** - Image processing

### Geospatial
- **Rasterio** (1.3.0+) - Raster data I/O
- **PyProj** (3.6.0+) - Cartographic projections
- **Cartopy** (0.21.0+) - Geospatial data processing

### Data Acquisition
- **Requests** (2.31.0+) - HTTP library for API calls
- **aiohttp** (3.9.0+) - Asynchronous HTTP client

---

## How It Works

### Prediction Pipeline

1. **User Input Collection**
   - Collects latitude, longitude, date, and area offset from UI
   - Calculates geographic bounds based on offset

2. **Historical Weather Retrieval** (`meteo_data.py`)
   - Fetches 31 days of historical weather data ending 1 day before prediction date
   - Retrieves data for all fire-prone locations within specified bounds
   - Caches results to `weather_output/all_locations_weather.csv`

3. **Fire Risk Prediction** (`logic.py`)
   - Loads pre-trained machine learning model (`model.pkl`)
   - Processes weather features for each location
   - Generates fire risk probability (0.0 to 1.0) for each coordinate
   - Filters predictions to locations within bounds

4. **Spatial Interpolation** (`kriging.py`)
   - Applies Ordinary Kriging with spherical variogram model
   - Creates continuous risk surface from discrete prediction points
   - Generates heat map visualization
   - Saves interpolation as `kriging_interpolation.png`

5. **Visualization** (`streamlit_app.py`)
   - Overlays kriging heat map on Folium map
   - Adds risk markers:
     - Green circles for low risk (<70%)
     - Orange circles for medium risk (70-90%)
     - Red flame icons for high risk (>90%)
   - Displays interactive map with zoom and pan capabilities

### Risk Classification

```python
if risk_level < 0.7:
    risk = "Low" (Green)
elif risk_level < 0.9:
    risk = "Medium" (Orange)
else:
    risk = "High" (Red)
```

---

## License

This project was developed for the NASA Space Apps Challenge 2025.

---

## Team

**Óbuda University Space Lab**

Developed by the team **Terra and Luna Analytics** for the NASA Space Apps Challenge 2025.

Team members:

- Bence Majer
- László Potyondi
- Sándor Burian
- Tamás Péter Kozma
- Yhair Sifuentes
- Zsombor Korb

---

## Acknowledgments

- **NASA** - For organizing the Space Apps Challenge
- **Open-Meteo** - For providing free historical weather data
- **NASA FIRMS** - For fire occurrence datasets
- **Streamlit** - For the excellent web framework

---

## Contact

For questions or feedback about this project, please open an issue on the [GitHub repository](https://github.com/Obuda-University-Space-Lab/NASA_SpaceApps_challenge_2025).

---
