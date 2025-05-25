import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_airport_data, calculate_distance

# Constants
MODEL_DIR = 'model_files'
VISUALIZATIONS_DIR = 'visualizations'

def load_model_components():
    """Load all model components."""
    # Load model
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load encoders
    with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    
    # Load feature names
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(MODEL_DIR, 'model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, encoders, feature_names, metadata

def create_features(input_data, encoders, airports_df):
    """Create features for prediction."""
    features = {}
    
    # Encode categorical features
    for col in ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']:
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
    
    return features

def find_common_routes(df, top_n=5):
    """Find the most common routes for each category."""
    common_routes = {}
    
    for category in df['category'].unique():
        category_routes = df[df['category'] == category]['route'].value_counts().head(top_n)
        common_routes[category] = category_routes.index.tolist()
    
    return common_routes

def calculate_route_medians(df, common_routes):
    """Calculate median prices and other statistics for common routes."""
    route_stats = {}
    
    for category, routes in common_routes.items():
        for route in routes:
            route_data = df[(df['category'] == category) & (df['route'] == route)]
            if not route_data.empty:
                # Get the most common aircraft model for this route
                most_common_aircraft = route_data['aircraftModel'].mode().iloc[0]
                
                route_stats[route] = {
                    'category': category,
                    'aircraft_model': most_common_aircraft,  # Add the most common aircraft model
                    'median_price': route_data['price'].median(),
                    'mean_price': route_data['price'].mean(),
                    'std_price': route_data['price'].std(),
                    'min_price': route_data['price'].min(),
                    'max_price': route_data['price'].max(),
                    'sample_count': len(route_data)
                }
    
    return route_stats

def test_model_predictions(model, scaler, encoders, route_stats, airports_df, feature_names):
    """Test model predictions against median prices."""
    results = []
    
    for route, stats in route_stats.items():
        dep_airport, arr_airport = route.split(' - ')
        
        # Create input data
        input_data = {
            'aircraftModel': stats['aircraft_model'],
            'category': stats['category'],
            'leg_Departure_Airport': dep_airport,
            'leg_Arrival_Airport': arr_airport
        }
        
        try:
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
            
            # Calculate deviation from median
            deviation = ((predicted_price - stats['median_price']) / stats['median_price']) * 100
            
            # Debug output for extreme deviations
            if abs(deviation) > 50:  # Print details for large deviations
                print(f"\nLarge deviation detected for route {route}:")
                print(f"Category: {stats['category']}")
                print(f"Aircraft: {stats['aircraft_model']}")
                print(f"Median price: ${stats['median_price']:,.2f}")
                print(f"Predicted price: ${predicted_price:,.2f}")
                print(f"Deviation: {deviation:.2f}%")
                print(f"Sample count: {stats['sample_count']}")
                print(f"Feature values:")
                for feature in feature_names:
                    print(f"  {feature}: {feature_df[feature].iloc[0]}")
            
            results.append({
                'route': route,
                'category': stats['category'],
                'aircraft_model': stats['aircraft_model'],
                'median_price': stats['median_price'],
                'predicted_price': predicted_price,
                'deviation_percent': deviation,
                'sample_count': stats['sample_count']
            })
            
        except Exception as e:
            print(f"Error processing route {route}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nPrediction Statistics:")
    print(f"Mean predicted price: ${results_df['predicted_price'].mean():,.2f}")
    print(f"Median predicted price: ${results_df['predicted_price'].median():,.2f}")
    print(f"Min predicted price: ${results_df['predicted_price'].min():,.2f}")
    print(f"Max predicted price: ${results_df['predicted_price'].max():,.2f}")
    
    print("\nActual Price Statistics:")
    print(f"Mean actual price: ${results_df['median_price'].mean():,.2f}")
    print(f"Median actual price: ${results_df['median_price'].median():,.2f}")
    print(f"Min actual price: ${results_df['median_price'].min():,.2f}")
    print(f"Max actual price: ${results_df['median_price'].max():,.2f}")
    
    return results_df

def plot_results(results_df):
    """Create visualizations of the test results."""
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    if results_df.empty:
        print("No results to plot")
        return
    
    # Plot 1: Box plot of deviations by category
    plt.figure(figsize=(12, 6))
    try:
        sns.boxplot(data=results_df, x='category', y='deviation_percent')
        plt.xticks(rotation=45)
        plt.title('Price Deviation Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Deviation from Median Price (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'deviation_by_category.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating box plot: {str(e)}")
    
    # Plot 2: Scatter plot of predicted vs median prices
    plt.figure(figsize=(10, 10))
    try:
        plt.scatter(results_df['median_price'], results_df['predicted_price'], alpha=0.5)
        plt.plot([0, results_df['median_price'].max()], [0, results_df['median_price'].max()], 'r--')
        plt.title('Predicted vs Median Prices')
        plt.xlabel('Median Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'predicted_vs_median.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {str(e)}")
    
    # Plot 3: Deviation vs sample count
    plt.figure(figsize=(10, 6))
    try:
        plt.scatter(results_df['sample_count'], results_df['deviation_percent'], alpha=0.5)
        plt.title('Price Deviation vs Sample Count')
        plt.xlabel('Number of Samples')
        plt.ylabel('Deviation from Median Price (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'deviation_vs_samples.png'))
        plt.close()
    except Exception as e:
        print(f"Error creating sample count plot: {str(e)}")

def main():
    # Load model components
    print("Loading model components...")
    model, scaler, encoders, feature_names, metadata = load_model_components()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('processed_data/model_input_data.csv')
    
    # Load airport data
    print("Loading airport data...")
    airports_df = load_airport_data()
    if airports_df is None:
        raise ValueError("Failed to load airport data")
    
    # Find common routes
    print("Finding common routes...")
    common_routes = find_common_routes(df)
    
    # Calculate route medians
    print("Calculating route statistics...")
    route_stats = calculate_route_medians(df, common_routes)
    
    # Test model predictions
    print("Testing model predictions...")
    results_df = test_model_predictions(model, scaler, encoders, route_stats, airports_df, feature_names)
    
    if results_df.empty:
        print("No results to analyze")
        return
    
    # Debug output
    print("\nResults DataFrame columns:")
    print(results_df.columns.tolist())
    print("\nFirst few rows of results:")
    print(results_df.head())
    
    # Plot results
    print("\nCreating visualizations...")
    plot_results(results_df)
    
    # Save detailed results
    results_df.to_csv('test_results.csv', index=False)
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"Total routes tested: {len(results_df)}")
    if not results_df.empty:
        print("\nTop 10 routes with largest deviations:")
        print(results_df.nlargest(10, 'deviation_percent')[['route', 'category', 'deviation_percent', 'sample_count']])
    
    print("\nResults saved to test_results.csv")
    print(f"Visualizations saved to {VISUALIZATIONS_DIR}/")

if __name__ == "__main__":
    main() 