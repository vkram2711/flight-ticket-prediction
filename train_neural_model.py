import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
MODEL_DIR = 'model_files'
VIZ_DIR = 'visualizations'
CACHE_DIR = 'cache'

class FlightDataset(Dataset):
    def __init__(self, features, targets):
        # Convert to numpy arrays first, then to tensors
        self.features = torch.FloatTensor(features.values if isinstance(features, pd.DataFrame) else features)
        self.targets = torch.FloatTensor(targets.values if isinstance(targets, pd.Series) else targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class FlightPriceNN(nn.Module):
    def __init__(self, input_size):
        super(FlightPriceNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

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
        'aircraftModel_encoded',
        'category_encoded',
        'leg_Departure_Airport_encoded',
        'leg_Arrival_Airport_encoded',
        'route_encoded',
        'airport_distance'
    ]

    # Prepare X and y
    X = df[feature_columns]
    y = np.log1p(df['price'].values)  # Convert to numpy array and log transform the prices

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, feature_columns

def train_model(X, y, batch_size=32, epochs=100, learning_rate=0.001):
    """Train the neural network model."""
    print("\nTraining neural network model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = FlightDataset(X_train, y_train)
    test_dataset = FlightDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FlightPriceNN(X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'neural_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}')

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'neural_model.pth')))
    model.eval()
    
    # Evaluate model
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
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
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'neural_model.pth'))

    # Save scaler
    with open(os.path.join(MODEL_DIR, 'neural_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save encoders
    with open(os.path.join(MODEL_DIR, 'neural_encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)

    # Save feature names
    with open(os.path.join(MODEL_DIR, 'neural_feature_names.pkl'), 'wb') as f:
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
    with open(os.path.join(MODEL_DIR, 'neural_model_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("All model components saved successfully!")

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

if __name__ == "__main__":
    main() 