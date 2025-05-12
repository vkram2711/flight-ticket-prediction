import os
import pickle

import airportsdata
import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from geopy.distance import geodesic
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

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


def load_and_clean_data(file_path):
    """Load and clean the flight data."""
    print("Loading and cleaning data...")

    # Load data
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")

    # Clean price data
    df['price'] = pd.to_numeric(df['price'].replace('TBA', np.nan))
    df = df[df['price'] > 0]

    # Calculate price bounds using IQR method
    price_stats = df['price'].describe()
    Q1 = price_stats['25%']
    Q3 = price_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter prices within bounds
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

    print("\nPrice cleaning statistics:")
    print(f"Original price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Price bounds: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    print(f"Number of rows after price cleaning: {len(df)}")

    # Convert date columns
    df['quoteDate'] = pd.to_datetime(df['quoteDate'])
    df['leg_Departure_Date'] = pd.to_datetime(df['leg_Departure_Date'])
    df['leg_Arrival_Date'] = pd.to_datetime(df['leg_Arrival_Date'])

    # Clean passenger number
    df['leg_Passenger_number_PAX'] = pd.to_numeric(df['leg_Passenger_number_PAX'], errors='coerce')
    df = df.dropna(subset=['leg_Passenger_number_PAX'])

    return df


def get_airport_coordinates(airport_code):
    """Get airport coordinates from FAA Location Identifier (LID) using airportsdata."""
    try:
        # Load airport data using LID dataset
        airports = airportsdata.load('LID')
        airport = airports.get(airport_code)
        if airport:
            return (airport['lat'], airport['lon'])
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

    # Basic date features
    df['quote_month'] = df['quoteDate'].dt.month
    df['quote_day'] = df['quoteDate'].dt.day
    df['quote_dayofweek'] = df['quoteDate'].dt.dayofweek

    df['departure_month'] = df['leg_Departure_Date'].dt.month
    df['departure_day'] = df['leg_Departure_Date'].dt.day
    df['departure_dayofweek'] = df['leg_Departure_Date'].dt.dayofweek
    df['departure_hour'] = df['leg_Departure_Date'].dt.hour

    # Advanced features
    df['days_until_departure'] = (df['leg_Departure_Date'] - df['quoteDate']).dt.days
    df['trip_duration'] = (df['leg_Arrival_Date'] - df['leg_Departure_Date']).dt.days

    # Calculate distances between airports
    print("Calculating airport distances...")
    df['airport_distance'] = df.apply(
        lambda row: calculate_distance(row['leg_Departure_Airport'], row['leg_Arrival_Airport']),
        axis=1
    )

    # Holiday features
    us_holidays = holidays.US()
    df['is_holiday'] = df['leg_Departure_Date'].apply(lambda x: x in us_holidays)

    # Season features
    df['season'] = df['leg_Departure_Date'].dt.month.map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })

    # Weekend features
    df['is_weekend'] = df['leg_Departure_Date'].dt.dayofweek.isin([5, 6])

    # Peak hour features
    df['is_peak_hour'] = df['departure_hour'].apply(
        lambda x: (x >= 7 and x <= 9) or (x >= 16 and x <= 19)
    )

    # Route features
    df['route'] = df['leg_Departure_Airport'] + ' - ' + df['leg_Arrival_Airport']

    return df


def encode_features(df):
    """Encode categorical features."""
    print("\nEncoding features...")

    # Create and fit encoders
    encoders = {}
    categorical_columns = ['aircraftModel', 'category', 'leg_Departure_Airport',
                           'leg_Arrival_Airport', 'route', 'season']

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
        # Basic features
        'aircraftModel_encoded', 'category_encoded',
        'leg_Passenger_number_PAX',

        # Temporal features
        'quote_month', 'quote_day', 'quote_dayofweek',
        'departure_month', 'departure_day', 'departure_dayofweek',
        'departure_hour', 'days_until_departure',

        # Holiday and season features
        'is_holiday', 'season_encoded',

        # Trip features
        'trip_duration', 'is_weekend', 'is_peak_hour',
        'airport_distance',  # Added distance feature

        # Route features
        'leg_Departure_Airport_encoded', 'leg_Arrival_Airport_encoded',
        'route_encoded'
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

    # Load and clean data
    df = load_and_clean_data('CleanOne.csv')

    # Create features
    df = create_features(df)

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
