import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_airport_data, calculate_distance

# Constants
MODEL_DIR = 'model_files'
VISUALIZATIONS_DIR = 'visualizations'

def load_xgboost_components():
    """Load XGBoost model components."""
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

def load_neural_components():
    """Load Neural Network model components."""
    # Load model
    from train_neural_model import FlightPriceNN
    model = FlightPriceNN(6)  # 6 features
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'neural_model.pth')))
    model.eval()
    
    # Load scaler
    with open(os.path.join(MODEL_DIR, 'neural_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load encoders
    with open(os.path.join(MODEL_DIR, 'neural_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    
    # Load feature names
    with open(os.path.join(MODEL_DIR, 'neural_feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load metadata
    with open(os.path.join(MODEL_DIR, 'neural_model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, encoders, feature_names, metadata

def create_features(input_data, encoders, airports_df):
    """Create features for prediction."""
    features = {}
    
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
    
    # Encode categorical features
    for col in ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']:
        value = input_data[col]
        if value not in encoders[col].classes_:
            raise ValueError(f"Unseen value '{value}' for {col}. Please select a different option.")
        features[f'{col}_encoded'] = encoders[col].transform([value])[0]
    
    # Handle route encoding
    if route not in encoders['route'].classes_:
        features['route_encoded'] = 0
    else:
        features['route_encoded'] = encoders['route'].transform([route])[0]
    
    return features

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
                    'aircraft_model': most_common_aircraft,
                    'median_price': route_data['price'].median(),
                    'mean_price': route_data['price'].mean(),
                    'std_price': route_data['price'].std(),
                    'min_price': route_data['price'].min(),
                    'max_price': route_data['price'].max(),
                    'sample_count': len(route_data)
                }
    
    return route_stats

def test_model_predictions(model, scaler, encoders, route_stats, airports_df, feature_names, is_neural=False):
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
            
            # Make prediction
            if is_neural:
                with torch.no_grad():
                    log_prediction = model(torch.FloatTensor(scaled_features)).numpy()[0][0]
            else:
                log_prediction = model.predict(scaled_features)[0]
            
            predicted_price = np.expm1(log_prediction)  # Transform back from log
            
            # Calculate deviation from median
            deviation = ((predicted_price - stats['median_price']) / stats['median_price']) * 100
            
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
            continue
    
    return pd.DataFrame(results)

def plot_comparison_results(xgb_results, neural_results):
    """Create comparison visualizations."""
    print("\nCreating comparison visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # 1. Deviation Distribution Comparison
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=xgb_results['deviation_percent'], label='XGBoost', alpha=0.5)
    sns.kdeplot(data=neural_results['deviation_percent'], label='Neural Network', alpha=0.5)
    plt.title('Distribution of Price Deviation from Median')
    plt.xlabel('Deviation from Median Price (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'deviation_distribution_comparison.png'))
    plt.close()
    
    # 2. Scatter Plot of Predictions
    plt.figure(figsize=(10, 10))
    plt.scatter(xgb_results['predicted_price'], neural_results['predicted_price'], alpha=0.5)
    plt.plot([0, max(xgb_results['predicted_price'])], 
             [0, max(xgb_results['predicted_price'])], 'r--', lw=2)
    plt.title('XGBoost vs Neural Network Predictions')
    plt.xlabel('XGBoost Predicted Price ($)')
    plt.ylabel('Neural Network Predicted Price ($)')
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'prediction_comparison.png'))
    plt.close()
    
    # 3. Performance Metrics Comparison
    metrics = pd.DataFrame({
        'Model': ['XGBoost', 'Neural Network'],
        'Mean Absolute Deviation (%)': [
            xgb_results['deviation_percent'].abs().mean(),
            neural_results['deviation_percent'].abs().mean()
        ],
        'Median Absolute Deviation (%)': [
            xgb_results['deviation_percent'].abs().median(),
            neural_results['deviation_percent'].abs().median()
        ],
        'Standard Deviation of Deviation (%)': [
            xgb_results['deviation_percent'].std(),
            neural_results['deviation_percent'].std()
        ]
    })
    
    plt.figure(figsize=(12, 6))
    metrics_melted = pd.melt(metrics, id_vars=['Model'], 
                            value_vars=['Mean Absolute Deviation (%)', 
                                      'Median Absolute Deviation (%)',
                                      'Standard Deviation of Deviation (%)'])
    sns.barplot(data=metrics_melted, x='variable', y='value', hue='Model')
    plt.title('Model Performance Metrics Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'metrics_comparison.png'))
    plt.close()
    
    return metrics

def main():
    # Load model components
    print("Loading XGBoost model components...")
    xgb_model, xgb_scaler, xgb_encoders, xgb_feature_names, xgb_metadata = load_xgboost_components()
    
    print("Loading Neural Network model components...")
    neural_model, neural_scaler, neural_encoders, neural_feature_names, neural_metadata = load_neural_components()
    
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
    common_routes = {
        category: df[df['category'] == category]['route'].value_counts().head(10).index.tolist()
        for category in df['category'].unique()
    }
    
    # Calculate route medians
    print("Calculating route statistics...")
    route_stats = calculate_route_medians(df, common_routes)
    
    # Test model predictions
    print("Testing XGBoost model predictions...")
    xgb_results = test_model_predictions(xgb_model, xgb_scaler, xgb_encoders, 
                                       route_stats, airports_df, xgb_feature_names)
    
    print("Testing Neural Network model predictions...")
    neural_results = test_model_predictions(neural_model, neural_scaler, neural_encoders, 
                                          route_stats, airports_df, neural_feature_names, 
                                          is_neural=True)
    
    if xgb_results.empty or neural_results.empty:
        print("No results to analyze")
        return
    
    # Plot comparison results
    metrics = plot_comparison_results(xgb_results, neural_results)
    
    # Save detailed results
    xgb_results.to_csv('xgb_test_results.csv', index=False)
    neural_results.to_csv('neural_test_results.csv', index=False)
    metrics.to_csv('model_comparison_metrics.csv', index=False)
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"Total routes tested: {len(xgb_results)}")
    
    print("\nXGBoost Model Performance:")
    print(f"Mean Absolute Deviation: {xgb_results['deviation_percent'].abs().mean():.2f}%")
    print(f"Median Absolute Deviation: {xgb_results['deviation_percent'].abs().median():.2f}%")
    print(f"Standard Deviation: {xgb_results['deviation_percent'].std():.2f}%")
    
    print("\nNeural Network Model Performance:")
    print(f"Mean Absolute Deviation: {neural_results['deviation_percent'].abs().mean():.2f}%")
    print(f"Median Absolute Deviation: {neural_results['deviation_percent'].abs().median():.2f}%")
    print(f"Standard Deviation: {neural_results['deviation_percent'].std():.2f}%")
    
    print("\nResults saved to:")
    print("- xgb_test_results.csv")
    print("- neural_test_results.csv")
    print("- model_comparison_metrics.csv")
    print(f"Visualizations saved to {VISUALIZATIONS_DIR}/")

if __name__ == "__main__":
    main() 