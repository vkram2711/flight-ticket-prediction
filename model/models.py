from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base

class Flight(Base):
    __tablename__ = "model_input_flights"

    id = Column(Integer, primary_key=True, index=True)
    aircraftModel = Column(String(100))
    tailNumber = Column(String(20), nullable=True)
    price = Column(Float)
    priceUnit = Column(String(10))
    quoteDate = Column(DateTime)
    leg_Passenger_number_PAX = Column(Integer)
    leg_Departure_Date = Column(DateTime)
    leg_Arrival_Date = Column(DateTime)
    leg_Departure_Airport = Column(String(10))
    leg_Arrival_Airport = Column(String(10))
    leg_Departure_City = Column(String(100))
    leg_Arrival_City = Column(String(100))
    filename = Column(String(255))
    fileModificationTime = Column(DateTime)
    diffDays = Column(Integer)
    quoteDayOfWeek = Column(String(20))
    category = Column(String(50))
    airport_distance = Column(Float)
    price_per_mile = Column(Float)
    distance_bin = Column(String(20))
    route = Column(String(50))

class RemovedFlight(Base):
    __tablename__ = "removed_flights"

    id = Column(Integer, primary_key=True, index=True)
    aircraftModel = Column(String(100), nullable=True)
    tailNumber = Column(String(20), nullable=True)
    price = Column(Float, nullable=True)
    priceUnit = Column(String(10), nullable=True)
    quoteDate = Column(DateTime, nullable=True)
    leg_Passenger_number_PAX = Column(Integer, nullable=True)
    leg_Departure_Date = Column(DateTime, nullable=True)
    leg_Arrival_Date = Column(DateTime, nullable=True)
    leg_Departure_Airport = Column(String(10), nullable=True)
    leg_Arrival_Airport = Column(String(10), nullable=True)
    leg_Departure_City = Column(String(100), nullable=True)
    leg_Arrival_City = Column(String(100), nullable=True)
    filename = Column(String(255), nullable=True)
    fileModificationTime = Column(DateTime, nullable=True)
    diffDays = Column(Integer, nullable=True)
    quoteDayOfWeek = Column(String(20), nullable=True)
    category = Column(String(50), nullable=True)
    duplicate_key = Column(String(255), nullable=True)
    reason = Column(String(255))
    price_zscore = Column(Float, nullable=True)
    category_lower = Column(String(50), nullable=True)
    airport_distance = Column(Float, nullable=True)
    price_per_mile = Column(Float, nullable=True)
    distance_bin = Column(String(20), nullable=True)


# api doc
# model doc
#
#