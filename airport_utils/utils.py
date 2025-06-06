import pandas as pd
from geopy.distance import geodesic
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from airport_utils.upload_airports import Airport
from config.database import Base, engine, SessionLocal


def load_airport_data():
    """
    Load airport data from the database.
    
    Returns:
        pandas.DataFrame: DataFrame containing airport information
    """
    try:
        # Create a new session
        session = SessionLocal()
        
        # Query all airports
        airports = session.query(Airport).all()
        
        # Convert to DataFrame
        airports_df = pd.DataFrame([{
            'icao': airport.icao,
            'iata': airport.iata,
            'name': airport.name,
            'city': airport.city,
            'subd': airport.subd,
            'country': airport.country,
            'elevation': airport.elevation,
            'lat': airport.lat,
            'lon': airport.lon,
            'tz': airport.tz,
            'lid': airport.lid
        } for airport in airports])
        
        return airports_df
    except Exception as e:
        print(f"Error loading airport data from database: {str(e)}")
        return None
    finally:
        session.close()


def calculate_distance(departure_airport, arrival_airport):
    """
    Calculate distance between two airports in nautical miles using direct database queries.
    
    Args:
        departure_airport (str): Departure airport code (IATA, ICAO, or LID)
        arrival_airport (str): Arrival airport code (IATA, ICAO, or LID)
        
    Returns:
        float: Distance in nautical miles, or None if airports not found
    """
    try:
        session = SessionLocal()
        
        # Find departure airport coordinates
        dep_airport = session.query(Airport).filter(
            (Airport.icao == departure_airport) |
            (Airport.iata == departure_airport) |
            (Airport.lid == departure_airport)
        ).first()
        
        # Find arrival airport coordinates
        arr_airport = session.query(Airport).filter(
            (Airport.icao == arrival_airport) |
            (Airport.iata == arrival_airport) |
            (Airport.lid == arrival_airport)
        ).first()
        
        # Check if both airports were found
        if not dep_airport or not arr_airport:
            return None
            
        # Get coordinates
        dep_coords = (dep_airport.lat, dep_airport.lon)
        arr_coords = (arr_airport.lat, arr_airport.lon)
        
        # Calculate distance
        distance = geodesic(dep_coords, arr_coords).nm
        return distance
        
    except Exception as e:
        print(f"Error calculating distance: {str(e)}")
        return None
    finally:
        session.close()


def check_airport_code(airport_code):
    """
    Check if an airport code exists in the database.
    
    Args:
        airport_code (str): The airport code to check (can be IATA, ICAO, or LID)
        
    Returns:
        bool: True if the airport code exists, False otherwise
    """
    try:
        session = SessionLocal()
        exists = session.query(Airport).filter(
            (Airport.icao == airport_code) |
            (Airport.iata == airport_code) |
            (Airport.lid == airport_code)
        ).first() is not None
        return exists
    except Exception as e:
        print(f"Error checking airport code: {str(e)}")
        return False
    finally:
        session.close()


def get_airport_by_code(airport_code):
    """
    Get airport information directly from the database using any airport code.
    
    Args:
        airport_code (str): The airport code to look up (can be IATA, ICAO, or LID)
        
    Returns:
        dict: Airport information or None if not found
    """
    try:
        session = SessionLocal()
        airport = session.query(Airport).filter(
            (Airport.icao == airport_code) |
            (Airport.iata == airport_code) |
            (Airport.lid == airport_code)
        ).first()
        
        if airport:
            return {
                'icao': airport.icao,
                'iata': airport.iata,
                'name': airport.name,
                'city': airport.city,
                'subd': airport.subd,
                'country': airport.country,
                'elevation': airport.elevation,
                'lat': airport.lat,
                'lon': airport.lon,
                'tz': airport.tz,
                'lid': airport.lid
            }
        return None
    except Exception as e:
        print(f"Error querying airport from database: {str(e)}")
        return None
    finally:
        session.close()


def get_airports_by_country(country):
    """
    Get all airports in a specific country.
    
    Args:
        country (str): Country name to search for
        
    Returns:
        list: List of airport dictionaries
    """
    try:
        session = SessionLocal()
        airports = session.query(Airport).filter(Airport.country == country).all()
        
        return [{
            'icao': airport.icao,
            'iata': airport.iata,
            'name': airport.name,
            'city': airport.city,
            'subd': airport.subd,
            'country': airport.country,
            'elevation': airport.elevation,
            'lat': airport.lat,
            'lon': airport.lon,
            'tz': airport.tz,
            'lid': airport.lid
        } for airport in airports]
    except Exception as e:
        print(f"Error querying airports by country: {str(e)}")
        return []
    finally:
        session.close()


def get_airports_by_city(city):
    """
    Get all airports in a specific city.
    
    Args:
        city (str): City name to search for
        
    Returns:
        list: List of airport dictionaries
    """
    try:
        session = SessionLocal()
        airports = session.query(Airport).filter(Airport.city == city).all()
        
        return [{
            'icao': airport.icao,
            'iata': airport.iata,
            'name': airport.name,
            'city': airport.city,
            'subd': airport.subd,
            'country': airport.country,
            'elevation': airport.elevation,
            'lat': airport.lat,
            'lon': airport.lon,
            'tz': airport.tz,
            'lid': airport.lid
        } for airport in airports]
    except Exception as e:
        print(f"Error querying airports by city: {str(e)}")
        return []
    finally:
        session.close()
