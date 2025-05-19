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
    
    # Calculate price statistics
    price_stats = df['price'].describe()
    print("\nInitial price statistics:")
    print(price_stats)
    
    # Calculate z-scores for prices
    mean_price = df['price'].mean()
    std_price = df['price'].std()
    df['price_zscore'] = (df['price'] - mean_price) / std_price
    
    # Remove extreme outliers using z-scores (values beyond 3 standard deviations)
    df = df[abs(df['price_zscore']) <= 3]
    print(f"\nPrice range after z-score filtering: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Number of rows after z-score filtering: {len(df)}")
    
    # Additional filtering using percentiles (remove top and bottom 1%)
    lower_percentile = df['price'].quantile(0.01)
    upper_percentile = df['price'].quantile(0.99)
    df = df[(df['price'] >= lower_percentile) & (df['price'] <= upper_percentile)]
    
    print("\nFinal price statistics:")
    print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Number of rows after percentile filtering: {len(df)}")
    
    # Drop the temporary z-score column
    df = df.drop('price_zscore', axis=1)
    
    # Convert dates
    df['leg_Departure_Date'] = pd.to_datetime(df['leg_Departure_Date'])
    df['leg_Arrival_Date'] = pd.to_datetime(df['leg_Arrival_Date'], errors='coerce')
    
    # Remove rows with missing required fields
    required_columns = ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']
    df = df.dropna(subset=required_columns)
    
    # Print unique categories before removal
    print("\nCategories before removal:")
    print(df['category'].unique())
    
    # Remove specified categories (case-insensitive)
    categories_to_remove = ['Unknown', 'Airliner', 'Helicopter']
    df['category_lower'] = df['category'].str.lower()
    df = df[~df['category_lower'].isin([cat.lower() for cat in categories_to_remove])]
    df = df.drop('category_lower', axis=1)
    
    print("\nCategories after removal:")
    print(df['category'].unique())
    print(f"Number of rows after removing specified categories: {len(df)}")
    
    # Calculate distances between airports
    print("Calculating airport distances...")
    df['airport_distance'] = df.apply(
        lambda row: calculate_distance(row['leg_Departure_Airport'], row['leg_Arrival_Airport']),
        axis=1
    )
    
    # Remove rows with zero distance (invalid airport codes)
    df = df[df['airport_distance'] > 0]
    
    # Create distance bins for more granular analysis
    df['distance_bin'] = pd.qcut(df['airport_distance'], q=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    
    print("\nRemoving outliers by category and distance...")
    # Group by category and distance bin to remove outliers within each group
    filtered_dfs = []
    for (category, distance_bin), group in df.groupby(['category', 'distance_bin']):
        if len(group) > 10:  # Only process groups with enough data points
            # Calculate z-scores for this group
            mean_price = group['price'].mean()
            std_price = group['price'].std()
            group['price_zscore'] = (group['price'] - mean_price) / std_price
            
            # Remove outliers (beyond 2.5 standard deviations)
            filtered_group = group[abs(group['price_zscore']) <= 2.5]
            filtered_dfs.append(filtered_group.drop('price_zscore', axis=1))
        else:
            filtered_dfs.append(group)
    
    # Combine all filtered groups
    df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"Number of rows after category and distance-based filtering: {len(df)}")
    
    # Print statistics by category and distance
    print("\nPrice statistics by category and distance:")
    stats_by_category = df.groupby(['category', 'distance_bin'])['price'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats_by_category)
    
    # Holiday features
    us_holidays = holidays.US()
    df['is_holiday'] = df['leg_Departure_Date'].apply(lambda x: x in us_holidays)

    # Weekend features
    df['is_weekend'] = df['leg_Departure_Date'].dt.dayofweek.isin([5, 6])

    # Route features
    df['route'] = df['leg_Departure_Airport'] + ' - ' + df['leg_Arrival_Airport']
    
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