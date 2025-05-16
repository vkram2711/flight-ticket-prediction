import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
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
