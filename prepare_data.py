import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from geopy.distance import geodesic
import airportsdata
import os

def get_airport_coordinates(airport_code):
    """Get airport coordinates using airport code rules to determine the correct database."""
    try:
        # Rule 1: If code contains numbers, it's a FAA LID
        if any(char.isdigit() for char in airport_code):
            airports = airportsdata.load('LID')
            airport = airports.get(airport_code)
            if airport:
                return (airport['lat'], airport['lon'])
            print(f"FAA LID {airport_code} not found in database")
            return (0, 0)
            
        # Rule 2: If code is 4 letters, it's ICAO
        if len(airport_code) == 4 and airport_code.isalpha():
            airports = airportsdata.load('ICAO')
            airport = airports.get(airport_code)
            if airport:
                return (airport['lat'], airport['lon'])
            print(f"ICAO code {airport_code} not found in database")
            return (0, 0)
            
        # Rule 3: If code is 3 letters, it's IATA
        if len(airport_code) == 3 and airport_code.isalpha():
            airports = airportsdata.load('IATA')
            airport = airports.get(airport_code)
            if airport:
                return (airport['lat'], airport['lon'])
            print(f"IATA code {airport_code} not found in database")
            return (0, 0)
            
        print(f"Invalid airport code format: {airport_code}")
        return (0, 0)
    except Exception as e:
        print(f"Error getting coordinates for {airport_code}: {str(e)}")
        return (0, 0)

def calculate_distance(departure_airport, arrival_airport):
    """Calculate distance between two airports in kilometers."""
    dep_coords = get_airport_coordinates(departure_airport)
    arr_coords = get_airport_coordinates(arrival_airport)
    return geodesic(dep_coords, arr_coords).kilometers

def clean_and_prepare_data(df):
    """Clean and prepare the data for model training."""
    print("\nCleaning and preparing data...")
    
    # Convert price to numeric and handle any non-numeric values
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove rows with missing or invalid prices
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    
    # Handle price outliers using IQR method
    price_stats = df['price'].describe()
    Q1 = price_stats['25%']
    Q3 = price_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print("\nPrice cleaning statistics:")
    print(f"Original price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Price bounds: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    
    # Filter prices within bounds
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    print(f"Number of rows after price cleaning: {len(df)}")
    
    # Convert dates
    df['leg_Departure_Date'] = pd.to_datetime(df['leg_Departure_Date'])
    df['leg_Arrival_Date'] = pd.to_datetime(df['leg_Arrival_Date'], errors='coerce')
    
    # Remove rows with missing required fields
    required_columns = ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']
    df = df.dropna(subset=required_columns)
    
    # Calculate distances between airports
    print("Calculating airport distances...")
    df['airport_distance'] = df.apply(
        lambda row: calculate_distance(row['leg_Departure_Airport'], row['leg_Arrival_Airport']),
        axis=1
    )

    # Holiday features
    us_holidays = holidays.US()
    df['is_holiday'] = df['leg_Departure_Date'].apply(lambda x: x in us_holidays)

    # Weekend features
    df['is_weekend'] = df['leg_Departure_Date'].dt.dayofweek.isin([5, 6])

    # Route features
    df['route'] = df['leg_Departure_Airport'] + ' - ' + df['leg_Arrival_Airport']
    
    # Remove rows with zero distance (invalid airport codes)
    df = df[df['airport_distance'] > 0]
    
    print(f"Data shape after cleaning: {df.shape}")
    return df

def main():
    # Create output directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv('CleanOne.csv')
    print(f"Original data shape: {df.shape}")
    
    # Clean and prepare data
    df = clean_and_prepare_data(df)
    
    # Save processed data
    output_file = 'processed_data/model_input_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total number of records: {len(df)}")
    print("\nPrice Statistics:")
    print(df['price'].describe())
    print("\nDistance Statistics:")
    print(df['airport_distance'].describe())
    print("\nUnique values in categorical columns:")
    print(f"Aircraft Models: {df['aircraftModel'].nunique()}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Departure Airports: {df['leg_Departure_Airport'].nunique()}")
    print(f"Arrival Airports: {df['leg_Arrival_Airport'].nunique()}")
    print(f"Routes: {df['route'].nunique()}")
    
    # Print sample of processed data
    print("\nSample of processed data:")
    print(df[['aircraftModel', 'category', 'price', 'airport_distance', 'is_holiday', 'is_weekend']].head())

if __name__ == "__main__":
    main() 