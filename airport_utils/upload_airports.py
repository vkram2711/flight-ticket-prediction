"""
Script to upload airports data from CSV to the database.

This script reads the merged_airports.csv file and uploads it to the database
using SQLAlchemy ORM. It creates an airports table if it doesn't exist and handles
the data upload with proper error handling and logging.
"""

import logging
import os
import sys
from datetime import datetime
import pandas as pd
from sqlalchemy import Column, Integer, String, Float, DateTime, text
from sqlalchemy.exc import SQLAlchemyError

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import Base, engine, SessionLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Airport(Base):
    """SQLAlchemy model for airports table."""
    __tablename__ = 'airports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    icao = Column(String(4), unique=True, index=True, nullable=True)  # Allow NULL values
    iata = Column(String(3), index=True, nullable=True)  # Allow NULL values
    name = Column(String(255))
    city = Column(String(255))
    subd = Column(String(255))  # Subdivision/State/Province
    country = Column(String(255))
    elevation = Column(Integer, nullable=True)  # Allow NULL values
    lat = Column(Float(precision=6), nullable=True)  # Allow NULL values
    lon = Column(Float(precision=6), nullable=True)  # Allow NULL values
    tz = Column(String(50))  # Timezone
    lid = Column(String(50))  # Location ID
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Airport(iata='{self.iata}', icao='{self.icao}', name='{self.name}')>"

def init_db():
    """
    Initialize the database and create tables.
    """
    try:
        # Drop existing table if it exists
        with engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS airports"))
            connection.commit()
            logger.info("Dropped existing airports table if it existed")
        
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_csv_path():
    """
    Get the absolute path to the merged_airports.csv file.
    
    Returns:
        str: Absolute path to the CSV file
        
    Raises:
        FileNotFoundError: If the CSV file cannot be found
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for the CSV file in the same directory as the script
    csv_path = os.path.join(script_dir, 'merged_airports.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    return csv_path

def clean_value(value, field_type=str):
    """
    Clean a value from the CSV file.
    
    Args:
        value: The value to clean
        field_type: The type to convert to (str, int, float)
        
    Returns:
        The cleaned value or None if empty/invalid
    """
    if pd.isna(value) or value == '':
        return None
    try:
        return field_type(value)
    except (ValueError, TypeError):
        return None

def merge_airport_records(df):
    """
    Merge duplicate airport records based on ICAO or IATA codes.
    
    Args:
        df: DataFrame containing airport records
        
    Returns:
        DataFrame with merged records
    """
    # Create a copy of the dataframe
    merged_df = df.copy()
    
    # First, merge based on ICAO codes
    logger.info("Merging records based on ICAO codes...")
    icao_groups = merged_df[merged_df['icao'].notna()].groupby('icao')
    merged_records = []
    
    for icao, group in icao_groups:
        if len(group) > 1:
            logger.info(f"Merging {len(group)} records for ICAO: {icao}")
            # Take the first non-null value for each column
            merged_record = group.agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None)
            merged_records.append(merged_record)
        else:
            merged_records.append(group.iloc[0])
    
    # Create a new dataframe with merged ICAO records
    merged_df = pd.DataFrame(merged_records)
    
    # Then, merge based on IATA codes for records without ICAO
    logger.info("Merging records based on IATA codes...")
    iata_groups = merged_df[merged_df['iata'].notna()].groupby('iata')
    merged_records = []
    
    for iata, group in iata_groups:
        if len(group) > 1:
            logger.info(f"Merging {len(group)} records for IATA: {iata}")
            # Take the first non-null value for each column
            merged_record = group.agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None)
            merged_records.append(merged_record)
        else:
            merged_records.append(group.iloc[0])
    
    # Create final merged dataframe
    merged_df = pd.DataFrame(merged_records)
    
    # Log merge statistics
    logger.info(f"Original record count: {len(df)}")
    logger.info(f"Merged record count: {len(merged_df)}")
    logger.info(f"Removed {len(df) - len(merged_df)} duplicate records")
    
    return merged_df

def upload_airports_data():
    """
    Upload airports data from CSV to the database using SQLAlchemy ORM.
    
    This function:
    1. Reads the merged_airports.csv file
    2. Creates the airports table if it doesn't exist
    3. Uploads the data to the database using ORM
    4. Handles any errors during the process
    """
    session = None
    
    try:
        # Initialize database
        init_db()
        session = SessionLocal()
        
        # Get CSV file path
        csv_path = get_csv_path()
        logger.info(f"Reading airports data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Merge duplicate records
        df = merge_airport_records(df)
        
        # Upload data in chunks
        chunk_size = 1000
        total_rows = len(df)
        
        for i in range(0, total_rows, chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            airports = []
            
            for _, row in chunk.iterrows():
                airport = Airport(
                    icao=clean_value(row['icao']),
                    iata=clean_value(row['iata']),
                    name=clean_value(row['name']),
                    city=clean_value(row['city']),
                    subd=clean_value(row['subd']),
                    country=clean_value(row['country']),
                    elevation=clean_value(row['elevation'], int),
                    lat=clean_value(row['lat'], float),
                    lon=clean_value(row['lon'], float),
                    tz=clean_value(row['tz']),
                    lid=clean_value(row['lid'])
                )
                airports.append(airport)
            
            # Bulk insert the chunk
            session.bulk_save_objects(airports)
            session.commit()
            
            logger.info(f"Uploaded {min(i + chunk_size, total_rows)} of {total_rows} airports")
        
        # Verify upload
        count = session.query(Airport).count()
        logger.info(f"Total airports in database: {count}")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        if session:
            session.rollback()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if session:
            session.rollback()
        sys.exit(1)
    finally:
        if session:
            session.close()

if __name__ == '__main__':
    logger.info("Starting airports data upload process")
    upload_airports_data()
    logger.info("Airports data upload process completed")
