import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model.utils import load_model_components
from airport_utils.utils import load_airport_data, calculate_distance

# Get the absolute path to the testing directory
TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR_PATH)

# Create output directories relative to the testing directory
VISUALIZATIONS_DIR = os.path.join(TEST_DIR_PATH, 'visualizations')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
PROCESSED_DATA_DIR = os.path.join(MODEL_DIR, 'processed_data')

def find_common_routes(df, top_n=5):
    """Find the most common routes for each category."""
    common_routes = {}

    for category in df['category'].unique():
        category_routes = df[df['category'] == category]['route'].value_counts().head(top_n)
        common_routes[category] = category_routes.index.tolist()

    return common_routes


def calculate_route_statistics(df, common_routes):
    """Calculate statistics for common routes."""
    route_stats = []

    for category, routes in common_routes.items():
        for route in routes:
            route_data = df[(df['category'] == category) & (df['route'] == route)]
            if not route_data.empty:
                # Get the most common aircraft model for this route
                most_common_aircraft = route_data['aircraftModel'].mode().iloc[0]

                stats = {
                    'route': route,
                    'category': category,
                    'aircraft_model': most_common_aircraft,
                    'median_price': route_data['price'].median(),
                    'mean_price': route_data['price'].mean(),
                    'std_price': route_data['price'].std(),
                    'min_price': route_data['price'].min(),
                    'max_price': route_data['price'].max(),
                    'sample_count': len(route_data)
                }
                route_stats.append(stats)

    return pd.DataFrame(route_stats)


def analyze_route_performance(model, scaler, encoders, feature_names, input_data, airports_df):
    """Analyze model performance for a specific route."""
    try:
        # Create features
        features = {}
        
        # Encode categorical features
        for col in ['aircraftModel', 'category']:
            if col in input_data:
                try:
                    features[f'{col}_encoded'] = encoders[col].transform([input_data[col]])[0]
                except ValueError as e:
                    if 'unseen' in str(e):
                        raise ValueError(f"Unseen value for {col}: {input_data[col]}")
                    raise e

        # Calculate airport distance
        distance = calculate_distance(
            input_data['leg_Departure_Airport'],
            input_data['leg_Arrival_Airport'],
            airports_df
        )
        if distance is None:
            raise ValueError(f"Could not calculate distance between {input_data['leg_Departure_Airport']} and {input_data['leg_Arrival_Airport']}")
        features['airport_distance'] = distance

        # Create route feature
        route = f"{input_data['leg_Departure_Airport']} - {input_data['leg_Arrival_Airport']}"
        try:
            features['route_encoded'] = encoders['route'].transform([route])[0]
        except ValueError:
            # If route is not in training data, use a default encoding
            features['route_encoded'] = 0

        # Create DataFrame with features in the correct order
        feature_df = pd.DataFrame([features])
        feature_df = feature_df[feature_names]  # Ensure correct feature order

        # Scale features
        scaled_features = scaler.transform(feature_df)

        # Make prediction (model was trained on log-transformed prices)
        log_prediction = model.predict(scaled_features)[0]
        predicted_price = np.expm1(log_prediction)  # Transform back from log

        return {
            'predicted_price': predicted_price,
            'distance': distance,
            'price_per_mile': predicted_price / distance
        }

    except Exception as e:
        print(f"Error analyzing route: {str(e)}")
        return None


def plot_route_analysis(results_df, output_dir):
    """Create visualizations for route analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='predicted_price', bins=30)
    plt.title('Distribution of Predicted Prices')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'price_distribution.png'))
    plt.close()

    # Plot 2: Price vs Distance
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['distance'], results_df['predicted_price'], alpha=0.5)
    plt.title('Price vs Distance')
    plt.xlabel('Distance (miles)')
    plt.ylabel('Predicted Price ($)')
    plt.savefig(os.path.join(output_dir, 'price_vs_distance.png'))
    plt.close()

    # Plot 3: Price per Mile Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='price_per_mile', bins=30)
    plt.title('Distribution of Price per Mile')
    plt.xlabel('Price per Mile ($/mile)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'price_per_mile_distribution.png'))
    plt.close()


def main():
    try:
        # Create output directories
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

        # Load input data first
        print("Loading input data...")
        input_data_path = os.path.join(PROCESSED_DATA_DIR, 'model_input_data.csv')
        input_df = pd.read_csv(input_data_path)
        print(f"Loaded {len(input_df)} records")

        # Create route statistics
        print("\nCreating route statistics...")
        common_routes = find_common_routes(input_df)
        route_stats_df = calculate_route_statistics(input_df, common_routes)
        
        # Save route statistics
        route_stats_path = os.path.join(PROCESSED_DATA_DIR, 'route_statistics.csv')
        route_stats_df.to_csv(route_stats_path, index=False)
        print(f"Route statistics saved to {route_stats_path}")

        # Load model components for prediction analysis
        print("\nLoading model components...")
        model, scaler, encoders, feature_names = load_model_components()

        # Load airport data
        print("Loading airport data...")
        airports_df = load_airport_data()
        if airports_df is None:
            raise ValueError("Failed to load airport data")

        # Analyze routes with model predictions
        print("Analyzing routes with model predictions...")
        results = []
        for _, row in input_df.iterrows():
            input_data = {
                'aircraftModel': row['aircraftModel'],
                'category': row['category'],
                'leg_Departure_Airport': row['leg_Departure_Airport'],
                'leg_Arrival_Airport': row['leg_Arrival_Airport']
            }
            
            result = analyze_route_performance(model, scaler, encoders, feature_names, input_data, airports_df)
            if result:
                result['route'] = f"{input_data['leg_Departure_Airport']} - {input_data['leg_Arrival_Airport']}"
                result['category'] = input_data['category']
                result['aircraft_model'] = input_data['aircraftModel']
                results.append(result)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        if results_df.empty:
            print("No valid results to analyze")
            return

        # Save prediction results
        results_path = os.path.join(TEST_DIR_PATH, 'route_analysis_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nPrediction results saved to {results_path}")

        # Create visualizations
        print("\nCreating visualizations...")
        plot_route_analysis(results_df, VISUALIZATIONS_DIR)

        # Print summary statistics
        print("\nRoute Analysis Summary:")
        print(f"Total routes analyzed: {len(results_df)}")
        print("\nPrice Statistics:")
        print(f"Mean predicted price: ${results_df['predicted_price'].mean():,.2f}")
        print(f"Median predicted price: ${results_df['predicted_price'].median():,.2f}")
        print(f"Mean price per mile: ${results_df['price_per_mile'].mean():,.2f}")
        print(f"Median price per mile: ${results_df['price_per_mile'].median():,.2f}")

        print("\nTop 5 most expensive routes:")
        print(results_df.nlargest(5, 'predicted_price')[['route', 'category', 'predicted_price', 'price_per_mile']])

        print("\nTop 5 most expensive routes per mile:")
        print(results_df.nlargest(5, 'price_per_mile')[['route', 'category', 'predicted_price', 'price_per_mile']])

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
