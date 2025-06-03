import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from airport_utils.utils import load_airport_data, calculate_distance
from model.utils import load_model_components

# Get the absolute path to the testing directory
TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR_PATH)

# Create output directories relative to the testing directory
VISUALIZATIONS_DIR = os.path.join(TEST_DIR_PATH, 'visualizations')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
PROCESSED_DATA_DIR = os.path.join(MODEL_DIR, 'processed_data')


def create_features(input_data, encoders, airports_df):
    """Create features for prediction."""
    features = {}

    # Encode categorical features
    for col in ['aircraftModel', 'category']:
        if col in input_data:
            try:
                features[f'{col}_encoded'] = encoders[col].transform([input_data[col]])[0]
            except ValueError as e:
                if 'unseen' in str(e):
                    print(f"Warning: Unseen value for {col}: {input_data[col]}")
                    # Use a default encoding for unseen values
                    features[f'{col}_encoded'] = 0
                else:
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
        print(f"Warning: Unseen route: {route}")
        features['route_encoded'] = 0

    return features


def test_model_predictions(model, scaler, encoders, route_stats_df, airports_df, feature_names):
    """Test model predictions against average prices."""
    results = []
    
    # Print available columns for debugging
    print("\nAvailable columns in route statistics:")
    print(route_stats_df.columns.tolist())
    
    # Verify required columns exist
    required_columns = ['route', 'mean_price', 'sample_count']
    missing_columns = [col for col in required_columns if col not in route_stats_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in route statistics: {missing_columns}")

    print(f"\nProcessing {len(route_stats_df)} routes...")
    successful_predictions = 0
    failed_predictions = 0

    for idx, route_data in route_stats_df.iterrows():
        try:
            # Split route into departure and arrival airports
            dep_airport, arr_airport = route_data['route'].split(' - ')

            # Create input data with default values
            input_data = {
                'aircraftModel': route_data['aircraft_model'],  # Use actual aircraft model from route stats
                'category': route_data['category'],             # Use actual category from route stats
                'leg_Departure_Airport': dep_airport,
                'leg_Arrival_Airport': arr_airport
            }

            # Create features
            features = create_features(input_data, encoders, airports_df)

            # Create DataFrame with features in the correct order
            feature_df = pd.DataFrame([features])
            feature_df = feature_df[feature_names]  # Ensure correct feature order

            # Scale features
            scaled_features = scaler.transform(feature_df)

            # Make prediction (model was trained on log-transformed prices)
            log_prediction = model.predict(scaled_features)[0]
            predicted_price = np.expm1(log_prediction)  # Transform back from log

            # Calculate deviation from average price
            avg_price = route_data['mean_price']
            deviation = ((predicted_price - avg_price) / avg_price) * 100

            results.append({
                'route': route_data['route'],
                'category': input_data['category'],
                'aircraft_model': input_data['aircraftModel'],
                'avg_price': avg_price,
                'predicted_price': predicted_price,
                'deviation_percent': deviation,
                'flight_count': route_data['sample_count']
            })
            successful_predictions += 1

        except Exception as e:
            failed_predictions += 1
            print(f"Error processing route {route_data['route']}: {str(e)}")
            continue

    print(f"\nPrediction Summary:")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")

    if not results:
        raise ValueError("No valid predictions were generated. Check the error messages above for details.")

    return pd.DataFrame(results)


def plot_results(results_df):
    """Create visualizations of the test results."""
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    if results_df.empty:
        print("No results to plot")
        return

    # Plot 1: Box plot of deviations by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='category', y='deviation_percent')
    plt.xticks(rotation=45)
    plt.title('Price Deviation Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Deviation from Average Price (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'deviation_by_category.png'))
    plt.close()

    # Plot 2: Scatter plot of predicted vs average prices
    plt.figure(figsize=(10, 10))
    plt.scatter(results_df['avg_price'], results_df['predicted_price'], alpha=0.5)
    plt.plot([0, results_df['avg_price'].max()], [0, results_df['avg_price'].max()], 'r--')
    plt.title('Predicted vs Average Prices')
    plt.xlabel('Average Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'predicted_vs_avg.png'))
    plt.close()

    # Plot 3: Deviation vs flight count
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['flight_count'], results_df['deviation_percent'], alpha=0.5)
    plt.title('Price Deviation vs Flight Count')
    plt.xlabel('Number of Flights')
    plt.ylabel('Deviation from Average Price (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'deviation_vs_flights.png'))
    plt.close()


def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

    try:
        # Load model components
        print("Loading model components...")
        model, scaler, encoders, feature_names = load_model_components()

        # Load route statistics
        print("Loading route statistics...")
        route_stats_path = os.path.join(PROCESSED_DATA_DIR, 'route_statistics.csv')
        route_stats_df = pd.read_csv(route_stats_path)
        print(f"Loaded {len(route_stats_df)} routes")

        # Load airport data
        print("Loading airport data...")
        airports_df = load_airport_data()
        if airports_df is None:
            raise ValueError("Failed to load airport data")

        # Test model predictions
        print("Testing model predictions...")
        results_df = test_model_predictions(model, scaler, encoders, route_stats_df, airports_df, feature_names)
        if results_df.empty:
            print("No results to analyze")
            return

        # Plot results
        print("\nCreating visualizations...")
        plot_results(results_df)

        # Save detailed results
        results_path = os.path.join(TEST_DIR_PATH, 'test_results.csv')
        results_df.to_csv(results_path, index=False)

        # Print summary
        print("\nTest Results Summary:")
        print(f"Total routes tested: {len(results_df)}")

        print("\nPrice Statistics:")
        print(f"Mean predicted price: ${results_df['predicted_price'].mean():,.2f}")
        print(f"Mean average price: ${results_df['avg_price'].mean():,.2f}")
        print(f"Mean absolute deviation: {results_df['deviation_percent'].abs().mean():.1f}%")

        print("\nTop 5 routes with largest deviations:")
        print(results_df.nlargest(5, 'deviation_percent')[['route', 'category', 'deviation_percent', 'flight_count']])

        print("\nResults saved to test_results.csv")
        print(f"Visualizations saved to {VISUALIZATIONS_DIR}/")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
