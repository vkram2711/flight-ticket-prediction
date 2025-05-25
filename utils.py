import pandas as pd
from geopy.distance import geodesic


def load_airport_data(file_path='merged_airports.csv'):
    """Load airport data from merged_airports.csv."""
    try:
        # Read CSV with correct column names
        airports_df = pd.read_csv(file_path, header=None,
                                  names=['icao', 'iata', 'name', 'city', 'subd', 'country', 'elevation', 'lat', 'lon', 'tz', 'lid'])
        return airports_df
    except Exception as e:
        print(f"Error loading airport data: {str(e)}")
        return None


def calculate_distance(departure_airport, arrival_airport, airports_df):
    """Calculate distance between two airports in kilometers."""
    # Find departure airport coordinates
    dep_airport = airports_df[
        (airports_df['lid'] == departure_airport) |
        (airports_df['iata'] == departure_airport) |
        (airports_df['icao'] == departure_airport)
        ]

    # Find arrival airport coordinates
    arr_airport = airports_df[
        (airports_df['lid'] == arrival_airport) |
        (airports_df['iata'] == arrival_airport) |
        (airports_df['icao'] == arrival_airport)
        ]

    # Check if both airports were found
    if dep_airport.empty or arr_airport.empty:
        return None

    # Get coordinates
    dep_coords = (dep_airport.iloc[0]['lat'], dep_airport.iloc[0]['lon'])
    arr_coords = (arr_airport.iloc[0]['lat'], arr_airport.iloc[0]['lon'])

    # Calculate distance
    distance = geodesic(dep_coords, arr_coords).kilometers
    return distance
