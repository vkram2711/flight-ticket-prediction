import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# Load the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Clean price data
def clean_prices(df):
    # Convert to numeric and replace 'TBA' with NaN
    df['price'] = pd.to_numeric(df['price'].replace('TBA', np.nan))
    
    # Remove zero prices
    df = df[df['price'] > 0]
    
    # Calculate price statistics
    price_stats = df['price'].describe()
    Q1 = price_stats['25%']
    Q3 = price_stats['75%']
    IQR = Q3 - Q1
    
    # Define reasonable price bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter prices within bounds
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    
    print("\nPrice cleaning statistics:")
    print(f"Original price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")
    print(f"Price bounds: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    print(f"Number of rows after price cleaning: {len(df)}")
    
    return df


# Preprocess the data
def preprocess_data(df):
    print("Original data shape:", df.shape)
    
    # Clean price data
    df = clean_prices(df)
    
    print("\nPrice statistics after cleaning:")
    print(df['price'].describe())
    
    # Convert date columns to datetime, handling errors
    df['quoteDate'] = pd.to_datetime(df['quoteDate'], errors='coerce')
    df['leg_Departure_Date'] = pd.to_datetime(df['leg_Departure_Date'], errors='coerce')
    df['leg_Arrival_Date'] = pd.to_datetime(df['leg_Arrival_Date'], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['quoteDate', 'leg_Departure_Date', 'leg_Arrival_Date'])
    
    # Extract features from dates
    df['quote_month'] = df['quoteDate'].dt.month
    df['quote_day'] = df['quoteDate'].dt.day
    df['quote_dayofweek'] = df['quoteDate'].dt.dayofweek
    
    df['departure_month'] = df['leg_Departure_Date'].dt.month
    df['departure_day'] = df['leg_Departure_Date'].dt.day
    df['departure_dayofweek'] = df['leg_Departure_Date'].dt.dayofweek
    
    # Calculate trip duration in days
    df['trip_duration'] = (df['leg_Arrival_Date'] - df['leg_Departure_Date']).dt.days
    
    # Clean passenger number - convert to numeric and handle any non-numeric values
    df['leg_Passenger_number_PAX'] = pd.to_numeric(df['leg_Passenger_number_PAX'], errors='coerce')
    df = df.dropna(subset=['leg_Passenger_number_PAX'])
    
    # Encode categorical variables
    le = LabelEncoder()
    df['aircraftModel_encoded'] = le.fit_transform(df['aircraftModel'])
    df['category_encoded'] = le.fit_transform(df['category'])
    df['departure_airport_encoded'] = le.fit_transform(df['leg_Departure_Airport'])
    df['arrival_airport_encoded'] = le.fit_transform(df['leg_Arrival_Airport'])
    
    print(f"\nData shape after preprocessing: {df.shape}")
    
    # Plot price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.savefig('price_distribution.png')
    plt.close()
    
    return df


# Prepare features for training
def prepare_features(df):
    feature_columns = [
        'aircraftModel_encoded', 'category_encoded', 'quote_month', 'quote_day',
        'quote_dayofweek', 'departure_month', 'departure_day', 'departure_dayofweek',
        'trip_duration', 'leg_Passenger_number_PAX', 'departure_airport_encoded',
        'arrival_airport_encoded'
    ]
    
    X = df[feature_columns]
    y = np.log1p(df['price'])  # Log transform the prices
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    
    return X_scaled, y, scaler, feature_columns


# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid for XGBoost
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create the XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Perform grid search
    print("\nPerforming hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    model = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Convert predictions back from log scale
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${np.sqrt(mse):,.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return model


# Predict price for a new flight
def predict_price(model, feature_names, new_flight_data, scaler):
    # Scale the new features
    new_features = pd.DataFrame([new_flight_data], columns=feature_names)
    new_features_scaled = scaler.transform(new_features)
    new_features_scaled = pd.DataFrame(new_features_scaled, columns=feature_names)
    
    # Predict and convert back from log scale
    predicted_price_log = model.predict(new_features_scaled)[0]
    predicted_price = np.expm1(predicted_price_log)
    
    return predicted_price


def main():
    # Load and preprocess the data
    df = load_data('CleanOne.csv')
    df = preprocess_data(df)
    
    # Prepare features and train the model
    X, y, scaler, feature_names = prepare_features(df)
    model = train_model(X, y)
    
    # Example of predicting a new flight price
    new_flight = {
        'aircraftModel_encoded': 0,  # Replace with actual encoded value
        'category_encoded': 0,  # Replace with actual encoded value
        'quote_month': 1,
        'quote_day': 15,
        'quote_dayofweek': 2,
        'departure_month': 2,
        'departure_day': 20,
        'departure_dayofweek': 3,
        'trip_duration': 1,
        'leg_Passenger_number_PAX': 2,
        'departure_airport_encoded': 0,  # Replace with actual encoded value
        'arrival_airport_encoded': 1  # Replace with actual encoded value
    }
    
    predicted_price = predict_price(model, feature_names, new_flight, scaler)
    print(f"\nPredicted price for the new flight: ${predicted_price:,.2f}")


if __name__ == "__main__":
    main()
