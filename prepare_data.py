import pandas as pd
import numpy as np
from datetime import datetime
import holidays
from geopy.distance import geodesic
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_airport_data():
    """Load airport data from merged_airports.csv."""
    try:
        airports_df = pd.read_csv('merged_airports.csv')
        # Create a dictionary for faster lookups
        airports_dict = {}
        for _, row in airports_df.iterrows():
            airports_dict[row['code']] = {
                'lat': row['latitude'],
                'lon': row['longitude']
            }
        return airports_dict
    except Exception as e:
        print(f"Error loading airport data: {str(e)}")
        return {}

def calculate_distance(departure_airport, arrival_airport, airports_dict, distance_cache=None):
    """Calculate distance between two airports in kilometers."""
    # Check if distance is in cache
    if distance_cache is not None:
        cache_key = f"{departure_airport}_{arrival_airport}"
        if cache_key in distance_cache:
            return distance_cache[cache_key]
    
    # Get coordinates from airports dictionary
    dep_coords = airports_dict.get(departure_airport, {'lat': 0, 'lon': 0})
    arr_coords = airports_dict.get(arrival_airport, {'lat': 0, 'lon': 0})
    
    # Calculate distance
    distance = geodesic(
        (dep_coords['lat'], dep_coords['lon']),
        (arr_coords['lat'], arr_coords['lon'])
    ).kilometers
    
    # Add to cache if cache is provided
    if distance_cache is not None:
        distance_cache[cache_key] = distance
    
    return distance

def load_or_create_distance_cache():
    """Load existing distance cache or create a new one."""
    cache_file = 'cache/distance_cache.pkl'
    
    # Create cache directory if it doesn't exist
    os.makedirs('cache', exist_ok=True)
    
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
    cache_file = 'cache/distance_cache.pkl'
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print("Distance cache saved.")

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
    
    # Load airport data and distance cache
    airports_dict = load_airport_data()
    distance_cache = load_or_create_distance_cache()
    
    # Calculate distances between airports
    print("Calculating airport distances...")
    df['airport_distance'] = df.apply(
        lambda row: calculate_distance(
            row['leg_Departure_Airport'], 
            row['leg_Arrival_Airport'],
            airports_dict,
            distance_cache
        ),
        axis=1
    )
    
    # Save updated cache
    save_distance_cache(distance_cache)
    
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

def plot_price_per_mile(df):
    """Plot price per mile analysis for different categories."""
    # Calculate price per mile
    df['price_per_mile'] = df['price'] / df['airport_distance']
    
    # Get unique categories
    categories = df['category'].unique()
    
    # Create a figure for each category
    for category in categories:
        # Filter data for this category
        cat_data = df[df['category'] == category]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot
        plt.scatter(cat_data['airport_distance'], cat_data['price_per_mile'], 
                   alpha=0.5, label='Data points')
        
        # Add trend line
        z = np.polyfit(cat_data['airport_distance'], cat_data['price_per_mile'], 1)
        p = np.poly1d(z)
        plt.plot(cat_data['airport_distance'], p(cat_data['airport_distance']), 
                "r--", label='Trend line')
        
        # Customize plot
        plt.title(f'Price per Mile vs Distance for {category}')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Price per Mile ($/mile)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats = cat_data['price_per_mile'].describe()
        stats_text = f"""
        Mean: ${stats['mean']:.2f}/mile
        Median: ${stats['50%']:.2f}/mile
        Std Dev: ${stats['std']:.2f}/mile
        Min: ${stats['min']:.2f}/mile
        Max: ${stats['max']:.2f}/mile
        """
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'processed_data/price_per_mile_{category.lower().replace(" ", "_")}.png')
        plt.close()
    
    print("\nPrice per Mile Statistics by Category:")
    stats = df.groupby('category')['price_per_mile'].agg(['mean', 'median', 'std', 'min', 'max'])
    print(stats.round(2))

def main():
    # Create output directory if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv('CleanOne.csv')
    print(f"Original data shape: {df.shape}")
    
    # Clean and prepare data
    df = clean_and_prepare_data(df)
    
    # Plot price per mile analysis
    plot_price_per_mile(df)
    
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