import os
import pickle
import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_airport_data, calculate_distance


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
    print(f"Initial data shape: {df.shape}")
    
    # Create a list to track removed rows
    removed_rows_list = []
    initial_count = len(df)
    
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
        'Piston': [0, 500, 1000, 1500, 2000],
        'Turbo prop': [0, 500, 1000, 1500, 2000],
        'Light jet': [0, 1000, 2000, 3000, 4000],
        'Entry level jet (VLJ)': [0, 1000, 2000, 3000, 4000],
        'Super light jet': [0, 1000, 2000, 3000, 4000],
        'Midsize jet': [0, 1500, 3000, 4500, 6000],
        'Super midsize jet': [0, 1500, 3000, 4500, 6000],
        'Ultra long range': [0, 2000, 4000, 6000, 8000],
        'Heavy jet': [0, 2000, 4000, 6000, 8000]
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
    
    # First remove all flights shorter than 100 miles
    removed = df[df['airport_distance'] < 100]
    if not removed.empty:
        for _, row in removed.iterrows():
            row_dict = row.to_dict()
            row_dict['reason'] = 'Distance less than 100 miles'
            removed_rows_list.append(row_dict)
    df = df[df['airport_distance'] >= 100]
    print(f"Number of rows after removing flights < 100 miles: {len(df)}")
    
    # Apply category-specific distance limits
    category_limits = {
        'Piston': 2000,
        'Turboprop': 2000,
        'Light Jet': 4000,
        'Entry level jet (VLJ)': 4000,
        'Super light jet': 4000,
        'Midsize': 6000,
        'Super midsize jet': 6000
        # Ultralong and Heavy have no limits
    }
    
    # Filter by category limits
    for category, limit in category_limits.items():
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
    print(df[['aircraftModel', 'category', 'price', 'airport_distance']].head())


if __name__ == "__main__":
    main()
