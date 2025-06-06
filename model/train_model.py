import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from config.database import get_db
from model.db_operations import get_all_flights

# Get the absolute path to the model directory
MODEL_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# Create output directories relative to the model directory
MODEL_DIR = os.path.join(MODEL_DIR_PATH, 'model_files')
VIZ_DIR = os.path.join(MODEL_DIR_PATH, 'visualizations')
CACHE_DIR = os.path.join(os.path.dirname(MODEL_DIR_PATH), 'cache')


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

        # Route features
        'route_encoded',
        'airport_distance'
    ]

    # Prepare X and y
    X = df[feature_columns]
    y = np.log1p(df['price'])  # Log transform the prices

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_columns


def plot_feature_importance(model, feature_columns, X_train, X_test, y_test):
    """Create and save feature importance visualizations using SHAP and permutation importance."""
    print("\nCreating feature importance visualizations...")

    # 1. SHAP Global Feature Importance
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_columns, show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'shap_global_importance.png'))
    plt.close()

    # SHAP dependence plots for top 3 features
    feature_importance = np.abs(shap_values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-3:]
    
    for idx in top_features_idx:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_columns, show=False)
        plt.title(f'SHAP Dependence Plot for {feature_columns[idx]}')
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'shap_dependence_{feature_columns[idx]}.png'))
        plt.close()

    # 2. Permutation Importance
    print("Calculating permutation importance...")
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Sort features by importance
    sorted_idx = result.importances_mean.argsort()
    
    # Plot permutation importance
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        labels=[feature_columns[i] for i in sorted_idx],
        vert=False
    )
    plt.title('Permutation Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'permutation_importance.png'))
    plt.close()

    # Save importance scores
    importance_scores = {
        'shap_importance': dict(zip(feature_columns, feature_importance)),
        'permutation_importance': dict(zip(feature_columns, result.importances_mean))
    }
    
    with open(os.path.join(MODEL_DIR, 'feature_importance_scores.pkl'), 'wb') as f:
        pickle.dump(importance_scores, f)

    return importance_scores


def train_model(X, y):
    """Train the XGBoost model with hyperparameter tuning."""
    print("\nTraining model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    # Create and train model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
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

    # Calculate training set metrics
    y_train_pred = model.predict(X_train)
    y_train_original = np.expm1(y_train)
    y_train_pred_original = np.expm1(y_train_pred)

    train_mae = mean_absolute_error(y_train_original, y_train_pred_original)
    train_mse = mean_squared_error(y_train_original, y_train_pred_original)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_original, y_train_pred_original)
    train_mape = np.mean(np.abs((y_train_original - y_train_pred_original) / y_train_original)) * 100

    # Calculate test set metrics
    y_test_pred = model.predict(X_test)
    y_test_original = np.expm1(y_test)
    y_test_pred_original = np.expm1(y_test_pred)

    test_mae = mean_absolute_error(y_test_original, y_test_pred_original)
    test_mse = mean_squared_error(y_test_original, y_test_pred_original)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_original, y_test_pred_original)
    test_mape = np.mean(np.abs((y_test_original - y_test_pred_original) / y_test_original)) * 100

    # Print performance metrics
    print("\nModel Performance:")
    print("\nTraining Set Performance:")
    print(f"Mean Absolute Error: ${train_mae:,.2f}")
    print(f"Root Mean Squared Error: ${train_rmse:,.2f}")
    print(f"R² Score: {train_r2:.4f}")
    print(f"Mean Absolute Percentage Error: {train_mape:.2f}%")

    print("\nTest Set Performance:")
    print(f"Mean Absolute Error: ${test_mae:,.2f}")
    print(f"Root Mean Squared Error: ${test_rmse:,.2f}")
    print(f"R² Score: {test_r2:.4f}")
    print(f"Mean Absolute Percentage Error: {test_mape:.2f}%")

    # Plot training performance
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_original, y_train_pred_original, alpha=0.5, label='Training')
    plt.plot([y_train_original.min(), y_train_original.max()],
             [y_train_original.min(), y_train_original.max()], 'r--', lw=2)
    plt.title('Training Set: Actual vs Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.savefig(os.path.join(VIZ_DIR, 'training_performance.png'))
    plt.close()

    # Plot test performance
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_test_pred_original, alpha=0.5, label='Test')
    plt.plot([y_test_original.min(), y_test_original.max()],
             [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
    plt.title('Test Set: Actual vs Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.savefig(os.path.join(VIZ_DIR, 'test_performance.png'))
    plt.close()

    # Plot comparison of training and test performance
    plt.figure(figsize=(12, 6))
    metrics = ['MAE', 'RMSE', 'R²', 'MAPE']
    train_scores = [train_mae, train_rmse, train_r2, train_mape]
    test_scores = [test_mae, test_rmse, test_r2, test_mape]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Training')
    plt.bar(x + width/2, test_scores, width, label='Test')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Training vs Test Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig(os.path.join(VIZ_DIR, 'performance_comparison.png'))
    plt.close()

    return model, test_mae, test_mse, test_r2, X_test, y_test, y_test_pred, X_train


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

    # Load data from database
    print("Loading data from model_input_flights table...")
    try:
        # Get database session
        db = next(get_db())
        try:
            # Load data from model_input_flights table
            df = get_all_flights(db)
            print(f"Loaded {len(df)} records from database")
        finally:
            db.close()
    except Exception as e:
        print(f"Error loading data from database: {str(e)}")
        return

    # Encode features
    df, encoders = encode_features(df)

    # Prepare training data
    X, y, scaler, feature_columns = prepare_training_data(df)

    # Train model
    model, test_mae, test_mse, test_r2, X_test, y_test, y_test_pred, X_train = train_model(X, y)

    # Save model components
    save_model_components(model, scaler, encoders, feature_columns, test_mae, test_mse, test_r2)

    # Create visualizations
    plot_model_insights(model, feature_columns, X_test, y_test, y_test_pred)
    
    # Create feature importance visualizations
    importance_scores = plot_feature_importance(model, feature_columns, X_train, X_test, y_test)
    
    # Print top features by importance
    print("\nTop 5 Features by SHAP Importance:")
    shap_importance = pd.Series(importance_scores['shap_importance']).sort_values(ascending=False)
    print(shap_importance.head())
    
    print("\nTop 5 Features by Permutation Importance:")
    perm_importance = pd.Series(importance_scores['permutation_importance']).sort_values(ascending=False)
    print(perm_importance.head())


if __name__ == "__main__":
    main()
