from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from config.database import get_db, engine
from model.models import Base, Flight, RemovedFlight
from sqlalchemy import text

def create_tables():
    """Create database tables if they don't exist."""
    print("Creating tables in main database...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully")

def clean_dict_for_db(d: dict) -> dict:
    """Clean dictionary values for database insertion."""
    cleaned = {}
    for k, v in d.items():
        if pd.isna(v) or (isinstance(v, float) and np.isnan(v)) or v == '':
            cleaned[k] = None
        elif isinstance(v, pd.Timestamp):
            cleaned[k] = v.to_pydatetime()
        elif k == 'leg_Passenger_number_PAX' and isinstance(v, str):
            # Handle string values for PAX column
            try:
                cleaned[k] = int(v) if v.strip() else None
            except (ValueError, AttributeError):
                cleaned[k] = None
        elif k == 'tailNumber' and isinstance(v, str):
            # Truncate tailNumber to 20 characters
            cleaned[k] = v[:20] if v else None
        else:
            cleaned[k] = v
    return cleaned

def save_to_database(df: pd.DataFrame, removed_rows_list: list, db: Session):
    """Save processed data to database."""
    # Save removed flights
    for row_dict in removed_rows_list:
        # Filter out any keys that aren't in the RemovedFlight model
        valid_keys = {c.name for c in RemovedFlight.__table__.columns}
        filtered_dict = {k: v for k, v in row_dict.items() if k in valid_keys}
        cleaned_dict = clean_dict_for_db(filtered_dict)
        removed_flight = RemovedFlight(**cleaned_dict)
        db.add(removed_flight)
    
    # Save valid flights
    for _, row in df.iterrows():
        # Filter out any keys that aren't in the Flight model
        valid_keys = {c.name for c in Flight.__table__.columns}
        filtered_dict = {k: v for k, v in row.to_dict().items() if k in valid_keys}
        cleaned_dict = clean_dict_for_db(filtered_dict)
        flight = Flight(**cleaned_dict)
        db.add(flight)
    
    db.commit()

def get_all_flights(db: Session) -> pd.DataFrame:
    """Retrieve all flights from database."""
    flights = db.query(Flight).all()
    return pd.DataFrame([flight.__dict__ for flight in flights])

def get_removed_flights(db: Session) -> pd.DataFrame:
    """Retrieve all removed flights from database."""
    removed_flights = db.query(RemovedFlight).all()
    return pd.DataFrame([flight.__dict__ for flight in removed_flights])

def get_flights_by_category(db: Session, category: str) -> pd.DataFrame:
    """Retrieve flights for a specific category."""
    flights = db.query(Flight).filter(Flight.category == category).all()
    return pd.DataFrame([flight.__dict__ for flight in flights])

def get_flights_by_route(db: Session, departure: str, arrival: str) -> pd.DataFrame:
    """Retrieve flights for a specific route."""
    flights = db.query(Flight).filter(
        Flight.leg_Departure_Airport == departure,
        Flight.leg_Arrival_Airport == arrival
    ).all()
    return pd.DataFrame([flight.__dict__ for flight in flights])

def load_clean_one_data(db: Session) -> pd.DataFrame:
    """Load data from the CleanOne table."""
    query = text("SELECT * FROM CleanOne")
    result = db.execute(query)
    return pd.DataFrame(result.fetchall(), columns=result.keys()) 