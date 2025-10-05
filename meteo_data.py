import requests
import pandas as pd
import os
from typing import List, Dict, Any, Tuple

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_point_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: List[str],
    daily_vars: List[str],
    timezone: str = "auto"
) -> Dict[str, Any]:
    """Fetch weather history from Open-Meteo API for a given coordinate."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars) if hourly_vars else None,
        "daily": ",".join(daily_vars) if daily_vars else None,
        "timezone": timezone,
    }
    params = {k: v for (k, v) in params.items() if v is not None}

    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data


def make_daily_dataframe(
    location_id: int,
    lat: float,
    lon: float,
    data: Dict[str, Any],
    daily_vars: List[str]
) -> pd.DataFrame:
    """Build a daily DataFrame for one location, including metadata."""
    daily = data.get("daily", {})
    df = pd.DataFrame(daily)

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["time"]).dt.date

    # Add metadata columns
    df["location_id"] = location_id
    df["latitude"] = lat
    df["longitude"] = lon

    # Reorder columns
    cols = ["location_id", "time", "date", "latitude", "longitude"] + daily_vars
    df = df.reindex(columns=[c for c in cols if c in df.columns])

    return df

import pandas as pd
from typing import List, Tuple


def is_within_bounds(lat: float, lon: float, bounds: List[List[float]]) -> bool:
    """Check if (lat, lon) is inside rectangular bounds [[lat_min, lon_min], [lat_max, lon_max]]."""
    return (bounds[0][0] <= lat <= bounds[1][0]) and (bounds[0][1] <= lon <= bounds[1][1])


def load_coordinates_from_csv(filepath: str, bounds: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Read coordinates from CSV file and return only those within bounds.
    The CSV must have columns named 'latitude' and 'longitude'.
    """
    coords = []
    try:
        df = pd.read_csv(filepath)

        # Ensure required columns exist
        if not {"latitude", "longitude"}.issubset(df.columns):
            raise ValueError("CSV must contain 'latitude' and 'longitude' columns.")

        for _, row in df.iterrows():
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if is_within_bounds(lat, lon, bounds):
                coords.append((lat, lon))

    except Exception as e:
        print(f"Error loading CSV: {e}")

    return coords

def build_weather_dataframe(
    coords: List[Tuple[float, float]],
    start_date: str,
    end_date: str,
    hourly_vars: List[str],
    daily_vars: List[str]
) -> pd.DataFrame:
    """Fetch weather data for all coordinates and combine into a single DataFrame."""
    all_dfs = []
    loc_id = 1

    for lat, lon in coords:
        print(f"Fetching weather data for {lat}, {lon}...")
        try:
            data = fetch_point_history(lat, lon, start_date, end_date, hourly_vars, daily_vars)
            df = make_daily_dataframe(loc_id, lat, lon, data, daily_vars)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"Error fetching data for {lat},{lon}: {e}")
        loc_id += 1

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


# Example usage
def meteo_data_extract(lat, lon, start_date, end_date,bounds):



    # Load coordinates from txt file for only greece
    coords = load_coordinates_from_csv("./data/greece_fire_places.csv", bounds)

    # Define variables
    hourly_vars = ["weather_code"]
    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "sunrise",
        "sunset",
        "daylight_duration",
        "sunshine_duration",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration"

        #"weathercode",
        #"temperature_2m_mean",
        #"temperature_2m_max",
        #"temperature_2m_min",
        #"apparent_temperature_mean",
        #"apparent_temperature_max",
        #"apparent_temperature_min",
        #"daylight_duration",
        #"sunshine_duration",
        #"precipitation_sum",
        #"rain_sum",
        #"snowfall_sum",
        #"precipitation_hours",
        #"windspeed_10m_max",
        #"windgusts_10m_max",
        #"winddirection_10m_dominant",
        #"shortwave_radiation_sum",
        #"et0_fao_evapotranspiration"
    ]

    # Build dataframe
    weather_df = build_weather_dataframe(coords, start_date, end_date, hourly_vars, daily_vars)

    # Save to CSV
    if not weather_df.empty:
        os.makedirs("weather_output", exist_ok=True)
        weather_df.to_csv("weather_output/all_locations_weather.csv", index=False, encoding="utf-8")
        print("Weather data saved to weather_output/all_locations_weather.csv")
        return weather_df
    else:
        print("No data fetched.")
