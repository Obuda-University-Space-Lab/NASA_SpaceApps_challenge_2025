from typing import List, Tuple

def is_within_bounds(lat: float, lon: float, bounds: List[List[float]]) -> bool:
    return (bounds[0][0] <= lat <= bounds[1][0]) and (bounds[0][1] <= lon <= bounds[1][1])

def load_coordinates_from_txt(filepath: str, bounds: List[List[float]]) -> List[Tuple[float, float]]:
    """Read coordinates from txt file and return only those within bounds."""
    coords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lat, lon = map(float, line.split(","))
                    if is_within_bounds(lat, lon, bounds):
                        coords.append((lat, lon))
                except ValueError:
                    print(f"Skipping invalid line: {line}")
    return coords

bounds = [
            [latitude - offset_deg, longitude - offset_deg],
            [latitude + offset_deg, longitude + offset_deg]
        ]

def load_coordinates_from_txt(filepath: str) -> List[Tuple[float, float]]:
    """Read coordinates from txt file. Expected format: latitude,longitude per line."""
    coords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lat, lon = map(float, line.split(","))
                    coords.append((lat, lon))
                except ValueError:
                    print(f"Skipping invalid line: {line}")
    return coords