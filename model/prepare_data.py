import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from airport_utils.utils import load_airport_data, calculate_distance
from model.utils import CATEGORY_LIMITS


def load_or_create_distance_cache():
    """Load existing distance cache or create a new one."""
    cache_file = '../cache/distance_cache.pkl'

    # Create cache directory if it doesn't exist
    os.makedirs('../cache', exist_ok=True)

    # Try to load existing cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print("Loading existing distance cache...")
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cache: {str(e)}")

    # Create new cache if loading failed or file doesn't exist
    print("Creating new distance cache...")
    return {}


def save_distance_cache(cache):
    """Save distance cache to file."""
    cache_file = '../cache/distance_cache.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print("Distance cache saved.")


def clean_and_prepare_data(df):
    """Clean and prepare the data for model training."""
    print("\nCleaning and preparing data...")
    print(f"Initial data shape: {df.shape}")
    
    # Create a list to track removed rows
    removed_rows_list = []
    initial_count = len(df)
    
    # Remove duplicates based on similar quotes
    print("\nRemoving duplicate quotes...")
    # Convert quoteDate to datetime
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    # Sort by quoteDate to ensure we process in chronological order
    df = df.sort_values('quoteDate')
    
    # Create a key for identifying potential duplicates (excluding time)
    def create_duplicate_key(row):
        # Base key components that are always present
        key_parts = [
            str(row['aircraftModel']),
            str(row['leg_Departure_Airport']),
            str(row['leg_Arrival_Airport']),
            str(row['price'])
        ]
        # Add tailNumber only if it's present and not empty
        if pd.notna(row['tailNumber']) and str(row['tailNumber']).strip():
            key_parts.append(str(row['tailNumber']))
        return '_'.join(key_parts)
    
    df['duplicate_key'] = df.apply(create_duplicate_key, axis=1)
    
    # Find duplicates within 5-minute windows
    duplicates = []
    for key in df['duplicate_key'].unique():
        # Get all rows with this key
        key_rows = df[df['duplicate_key'] == key].copy()
        if len(key_rows) > 1:
            # Sort by time
            key_rows = key_rows.sort_values('quoteDate')
            # Get the first row's time
            first_time = key_rows.iloc[0]['quoteDate']
            # Find rows within 5 minutes of the first row
            time_diff = (key_rows['quoteDate'] - first_time).dt.total_seconds() / 60
            # Keep only the first row, mark others as duplicates
            duplicates.extend(key_rows[time_diff <= 5].iloc[1:].index.tolist())
    
    # Record and remove duplicates
    if duplicates:
        for idx in duplicates:
            row_dict = df.loc[idx].to_dict()
            row_dict['reason'] = 'Duplicate quote (same aircraft, route, price, within 5 minutes)'
            removed_rows_list.append(row_dict)
        
        df = df.drop(duplicates)
        print(f"Removed {len(duplicates)} duplicate quotes")
    
    # Remove the temporary duplicate_key column
    df = df.drop('duplicate_key', axis=1)
    print(f"Rows after removing duplicates: {len(df)}")
    
    # Convert price to numeric and handle any non-numeric values
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    removed = df[df['price'].isna()]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Invalid price (non-numeric)'
            removed_rows_list.append(row_dict)
    df = df.dropna(subset=['price'])
    print(f"Rows after price conversion: {len(df)}")
    
    # Remove rows with missing or invalid prices
    removed = df[df['price'] <= 0]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Invalid price (zero or negative)'
            removed_rows_list.append(row_dict)
    df = df[df['price'] > 0]
    print(f"Rows after removing invalid prices: {len(df)}")
    
    # Calculate price statistics
    price_stats = df['price'].describe()
    print("\nInitial price statistics:")
    print(price_stats)
    
    # Calculate z-scores for prices
    mean_price = df['price'].mean()
    std_price = df['price'].std()
    df['price_zscore'] = (df['price'] - mean_price) / std_price
    
    # Remove extreme outliers using z-scores
    removed = df[abs(df['price_zscore']) > 3]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Price outlier (z-score > 3)'
            removed_rows_list.append(row_dict)
    df = df[abs(df['price_zscore']) <= 3]
    print(f"\nPrice range after z-score filtering: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Number of rows after z-score filtering: {len(df)}")
    
    # Additional filtering using percentiles
    lower_percentile = df['price'].quantile(0.01)
    upper_percentile = df['price'].quantile(0.99)
    removed = df[(df['price'] < lower_percentile) | (df['price'] > upper_percentile)]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Price outside 1-99 percentile range'
            removed_rows_list.append(row_dict)
    df = df[(df['price'] >= lower_percentile) & (df['price'] <= upper_percentile)]
    
    print("\nFinal price statistics:")
    print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Number of rows after percentile filtering: {len(df)}")
    
    # Drop the temporary z-score column
    df = df.drop('price_zscore', axis=1)
    
    # Remove rows with missing required fields
    required_columns = ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']
    removed = df[df[required_columns].isna().any(axis=1)]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Missing required fields'
            removed_rows_list.append(row_dict)
    df = df.dropna(subset=required_columns)
    print(f"Rows after removing missing required fields: {len(df)}")
    
    # Print unique categories before removal
    print("\nCategories before removal:")
    print(df['category'].unique())
    
    # Remove specified categories
    categories_to_remove = ['Unknown', 'Airliner', 'Helicopter']
    df['category_lower'] = df['category'].str.lower()
    removed = df[df['category_lower'].isin([cat.lower() for cat in categories_to_remove])]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = f'Removed categories: {", ".join(categories_to_remove)}'
            removed_rows_list.append(row_dict)
    df = df[~df['category_lower'].isin([cat.lower() for cat in categories_to_remove])]
    df = df.drop('category_lower', axis=1)
    
    print("\nCategories after removal:")
    print(df['category'].unique())
    print(f"Number of rows after removing specified categories: {len(df)}")
    
    # Load airport data
    airports_df = load_airport_data()
    print(f"Loaded {len(airports_df)} airports from CSV")
    
    # Calculate distances between airports
    print("\nCalculating airport distances...")
    df['airport_distance'] = df.apply(
        lambda row: calculate_distance(
            row['leg_Departure_Airport'], 
            row['leg_Arrival_Airport'],
            airports_df
        ),
        axis=1
    )
    
    # Remove rows with zero distance (invalid airport codes)
    removed = df[df['airport_distance'] == 0]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Invalid airport codes (distance = 0)'
            removed_rows_list.append(row_dict)
    df = df[df['airport_distance'] > 0]
    print(f"Rows after removing invalid distances: {len(df)}")
    
    # Calculate price per mile
    df['price_per_mile'] = df['price'] / df['airport_distance']
    
    # Remove extreme price per mile values by category
    print("\nRemoving extreme price per mile values by category...")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        lower_percentile = cat_data['price_per_mile'].quantile(0.01)
        upper_percentile = cat_data['price_per_mile'].quantile(0.99)
        
        removed = df[(df['category'] == category) & 
                    ((df['price_per_mile'] < lower_percentile) | 
                     (df['price_per_mile'] > upper_percentile))]
        
        if not removed.empty:
            for _, row in removed.iterrows():
                row_dict = row.to_dict()
                row_dict['reason'] = f'Extreme price per mile for {category}'
                removed_rows_list.append(row_dict)
        
        df = df[~((df['category'] == category) & 
                  ((df['price_per_mile'] < lower_percentile) | 
                   (df['price_per_mile'] > upper_percentile)))]
    
    print(f"Number of rows after removing extreme price per mile values: {len(df)}")
    
    # Create distance bins by category
    print("\nCreating distance bins by category...")
    distance_bins = {
        'Piston': [0, 270, 540, 810, 1080],
        'Turbo prop': [0, 270, 540, 810, 1080],
        'Light jet': [0, 540, 1080, 1620, 2160],
        'Entry level jet (VLJ)': [0, 540, 1080, 1620, 2160],
        'Super light jet': [0, 540, 1080, 1620, 2160],
        'Midsize jet': [0, 810, 1620, 2430, 3240],
        'Super midsize jet': [0, 810, 1620, 2430, 3240],
        'Ultra long range': [0, 1080, 2160, 3240, 4320],
        'Heavy jet': [0, 1080, 2160, 3240, 4320]
    }
    
    df['distance_bin'] = df.apply(
        lambda row: pd.cut(
            [row['airport_distance']], 
            bins=distance_bins[row['category']], 
            labels=[f'bin_{i+1}' for i in range(len(distance_bins[row['category']])-1)]
        )[0],
        axis=1
    )
    
    # Apply distance restrictions based on category
    print("\nApplying distance restrictions by category...")
    
    # First remove all flights shorter than 55 miles
    removed = df[df['airport_distance'] < 55]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Distance less than 55 miles'
            removed_rows_list.append(row_dict)
    df = df[df['airport_distance'] >= 55]
    print(f"Number of rows after removing flights < 55 miles: {len(df)}")
    
    # Filter by category limits
    for category, limit in CATEGORY_LIMITS.items():
        removed = df[(df['category'] == category) & (df['airport_distance'] > limit)]
        if not removed.empty:
            for _, row in removed.iterrows():
                row_dict = row.to_dict()
                row_dict['reason'] = f'{category} flights exceeding {limit} miles'
                removed_rows_list.append(row_dict)
        df = df[~((df['category'] == category) & (df['airport_distance'] > limit))]
    
    print(f"Number of rows after applying category distance limits: {len(df)}")
    
    # Route features
    df['route'] = df['leg_Departure_Airport'] + ' - ' + df['leg_Arrival_Airport']
    
    print(f"\nFinal data shape after cleaning: {df.shape}")
    
    # Convert removed rows list to DataFrame and save
    removed_rows_df = pd.DataFrame(removed_rows_list)
    if not removed_rows_df.empty:
        removed_rows_df.to_csv('processed_data/removed_rows.csv', index=False)
        print(f"\nRemoved {len(removed_rows_df)} rows. Details saved to processed_data/removed_rows.csv")
    
    return df


def plot_price_per_mile(df):
    """Plot price per mile distributions by category."""
    # Create output directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    # Plot price per mile distributions
    for category in df['category'].unique():
        plt.figure(figsize=(10, 6))
        cat_data = df[df['category'] == category]
        plt.hist(cat_data['price_per_mile'], bins=50)
        plt.title(f'Price per Mile Distribution - {category}')
        plt.xlabel('Price per Mile ($)')
        plt.ylabel('Frequency')
        plt.savefig(f'processed_data/price_per_mile_{category.lower().replace(" ", "_")}.png')
        plt.close()


def main():
    # Create output directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    # Load the data
    print("Loading data...")
    df = pd.read_csv('../CleanOne.csv')
    
    # Clean and prepare the data
    df = clean_and_prepare_data(df)
    
    # Save the processed data
    print("\nSaving processed data...")
    df.to_csv('processed_data/model_input_data.csv', index=False)
    print("Data saved to processed_data/model_input_data.csv")
    
    # Plot price per mile distributions
    print("\nGenerating price per mile plots...")
    plot_price_per_mile(df)
    
    # Calculate and save route statistics
    print("\nCalculating route statistics...")
    route_stats = df.groupby('route').agg({
        'price': ['mean', 'std', 'count'],
        'airport_distance': 'mean'
    }).round(2)
    route_stats.columns = ['avg_price', 'price_std', 'flight_count', 'avg_distance']
    route_stats = route_stats.sort_values('flight_count', ascending=False)
    route_stats.to_csv('processed_data/route_statistics.csv')
    print("Route statistics saved to processed_data/route_statistics.csv")
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
