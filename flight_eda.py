import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import holidays
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """Load and perform initial cleaning of the data."""
    df = pd.read_csv(file_path)
    
    # Convert price to numeric and handle 'TBA' values
    df['price'] = pd.to_numeric(df['price'].replace('TBA', np.nan))
    df = df[df['price'] > 0]
    
    # Convert date columns
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['leg_Departure_Date'] = pd.to_datetime(df['leg_Departure_Date'])
    df['leg_Arrival_Date'] = pd.to_datetime(df['leg_Arrival_Date'])
    
    return df

def analyze_price_distribution(df):
    """Analyze and visualize price distribution."""
    plt.figure(figsize=(12, 6))
    
    # Plot price distribution
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Flight Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.savefig('price_distribution_analysis.png')
    plt.close()
    
    # Calculate and print price statistics
    price_stats = df['price'].describe()
    print("\nPrice Statistics:")
    print(price_stats)
    
    # Calculate price by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='category', y='price', data=df)
    plt.title('Price Distribution by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_by_category.png')
    plt.close()

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data."""
    # Monthly price trends
    df['month'] = df['leg_Departure_Date'].dt.month
    monthly_prices = df.groupby('month')['price'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_prices.plot(kind='line', marker='o')
    plt.title('Average Price by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.savefig('monthly_price_trends.png')
    plt.close()
    
    # Day of week patterns
    df['day_of_week'] = df['leg_Departure_Date'].dt.dayofweek
    dow_prices = df.groupby('day_of_week')['price'].mean()
    
    plt.figure(figsize=(12, 6))
    dow_prices.plot(kind='bar')
    plt.title('Average Price by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Price')
    plt.xticks(range(7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.savefig('dow_price_patterns.png')
    plt.close()

def create_advanced_features(df):
    """Create advanced features for the model."""
    # Calculate days until departure
    df['days_until_departure'] = (df['leg_Departure_Date'] - df['quoteDate']).dt.days
    
    # Add holiday features
    us_holidays = holidays.US()
    df['is_holiday'] = df['leg_Departure_Date'].apply(lambda x: x in us_holidays)
    
    # Add season features
    df['season'] = df['leg_Departure_Date'].dt.month.map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    
    # Calculate trip duration
    df['trip_duration'] = (df['leg_Arrival_Date'] - df['leg_Departure_Date']).dt.days
    
    # Add weekend features
    df['is_weekend'] = df['leg_Departure_Date'].dt.dayofweek.isin([5, 6])
    
    # Add time of day features
    df['departure_hour'] = df['leg_Departure_Date'].dt.hour
    df['is_peak_hour'] = df['departure_hour'].apply(
        lambda x: (x >= 7 and x <= 9) or (x >= 16 and x <= 19)
    )
    
    return df

def analyze_airport_patterns(df):
    """Analyze patterns related to airports."""
    # Calculate average prices by departure airport
    plt.figure(figsize=(15, 6))
    airport_prices = df.groupby('leg_Departure_Airport')['price'].mean().sort_values(ascending=False)
    airport_prices.head(10).plot(kind='bar')
    plt.title('Average Price by Departure Airport (Top 10)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('airport_price_patterns.png')
    plt.close()
    
    # Analyze popular routes
    df['route'] = df['leg_Departure_Airport'] + ' - ' + df['leg_Arrival_Airport']
    route_counts = df['route'].value_counts().head(10)
    
    plt.figure(figsize=(15, 6))
    route_counts.plot(kind='bar')
    plt.title('Most Popular Routes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('popular_routes.png')
    plt.close()

def main():
    # Load and clean data
    df = load_and_clean_data('CleanOne.csv')
    
    # Perform EDA
    analyze_price_distribution(df)
    analyze_temporal_patterns(df)
    analyze_airport_patterns(df)
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Save processed data
    df.to_csv('processed_flight_data.csv', index=False)
    print("\nEDA completed and new features created. Results saved in CSV and visualization files.")

if __name__ == "__main__":
    main() 