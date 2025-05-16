import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directories
MODEL_DIR = 'model_files'
VIZ_DIR = 'visualizations'
CACHE_DIR = 'cache'


def create_output_dirs():
    """Create directories for model files and visualizations if they don't exist."""
    for directory in [MODEL_DIR, VIZ_DIR, CACHE_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def load_distance_cache():
    """Load the distance cache from file."""
    cache_file = os.path.join(CACHE_DIR, 'distance_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}


def save_distance_cache(cache):
    """Save the distance cache to file."""
    cache_file = os.path.join(CACHE_DIR, 'distance_cache.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)


def validate_and_clean_data(df):
    """Validate and clean the input data."""
    print("\nValidating and cleaning data...")
    
    # Convert price to numeric and handle any non-numeric values
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Remove rows with missing or invalid prices
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    
    # Convert dates
    df['leg_Departure_Date'] = pd.to_datetime(df['leg_Departure_Date'])
    
    # Handle missing arrival dates
    df['leg_Arrival_Date'] = pd.to_datetime(df['leg_Arrival_Date'], errors='coerce')
    
    # Remove rows with missing required fields
    required_columns = ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']
    df = df.dropna(subset=required_columns)
    
    print(f"Data shape after cleaning: {df.shape}")
    return df


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
    """Calculate distance between two airports in kilometers using cache."""
    # Load cache
    cache = load_distance_cache()
    
    # Create cache key
    cache_key = f"{departure_airport}-{arrival_airport}"
    
    # Check if distance is in cache
    if cache_key in cache:
        return cache[cache_key]
    
    # Calculate distance if not in cache
    dep_coords = get_airport_coordinates(departure_airport)
    arr_coords = get_airport_coordinates(arrival_airport)
    distance = geodesic(dep_coords, arr_coords).kilometers
    
    # Store in cache
    cache[cache_key] = distance
    save_distance_cache(cache)
    
    return distance


def create_features(df):
    """Create features for the model."""
    print("\nCreating features...")

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

    return df


def encode_features(df):
    """Encode categorical features."""
    print("\nEncoding features...")

    # Create and fit encoders
    encoders = {}
    categorical_columns = ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport', 'route']

    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def prepare_training_data(df):
    """Prepare data for model training."""
    print("\nPreparing training data...")

    # Define feature columns
    feature_columns = [
        # Aircraft features
        'aircraftModel_encoded',
        'category_encoded',

        # Airport features
        'leg_Departure_Airport_encoded',
        'leg_Arrival_Airport_encoded',
        'route_encoded',
        'airport_distance',

        # Temporal features
        'is_holiday',
        'is_weekend'
    ]

    # Prepare X and y
    X = df[feature_columns]
    y = np.log1p(df['price'])  # Log transform the prices

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_columns


def train_model(X, y):
    """Train the XGBoost model with hyperparameter tuning."""
    print("\nTraining model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create and train model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Get best model
    model = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)

    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${np.sqrt(mse):,.2f}")
    print(f"R² Score: {r2:.2f}")

    return model, mae, mse, r2


def save_model_components(model, scaler, encoders, feature_columns, mae, mse, r2):
    """Save all model components."""
    print("\nSaving model components...")

    # Save model
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # Save scaler
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save encoders
    with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)

    # Save feature names
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_columns, f)

    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'categorical_columns': list(encoders.keys()),
        'model_performance': {
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2)
        }
    }
    with open(os.path.join(MODEL_DIR, 'model_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("All model components saved successfully!")


def plot_model_insights(model, feature_columns, X_test, y_test, y_pred):
    """Create and save model insights visualizations."""
    print("\nCreating model insights visualizations...")

    # Feature importance plot
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'feature_importance.png'))
    plt.close()

    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price (log scale)')
    plt.ylabel('Predicted Price (log scale)')
    plt.savefig(os.path.join(VIZ_DIR, 'actual_vs_predicted.png'))
    plt.close()


def main():
    # Create output directories
    create_output_dirs()

    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv('processed_data/model_input_data.csv')
    print(f"Loaded {len(df)} records")

    # Encode features
    df, encoders = encode_features(df)

    # Prepare training data
    X, y, scaler, feature_columns = prepare_training_data(df)

    # Train model
    model, mae, mse, r2 = train_model(X, y)

    # Save model components
    save_model_components(model, scaler, encoders, feature_columns, mae, mse, r2)

    # Create visualizations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    plot_model_insights(model, feature_columns, X_test, y_test, y_pred)


if __name__ == "__main__":
    main()
