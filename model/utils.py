import os
import pickle

# Constants
MODEL_DIR = '../model/model_files'
VISUALIZATIONS_DIR = '../visualizations'

CATEGORIES = [
    'Piston',
    'Turbo prop',
    'Light jet',
    'Entry level jet (VLJ)',
    'Super light jet',
    'Midsize jet',
    'Super midsize jet',
    'Heavy jet',
    'Ultra long range'
]

# Category distance limits in nautical miles
CATEGORY_LIMITS = {
    'Piston': 1000,
    'Turbo prop': 1000,
    'Light jet': 2150,
    'Entry level jet (VLJ)': 2150,
    'Super light jet': 2150,
    'Midsize jet': 3200,
    'Super midsize jet': 3200
    # Ultralong and Heavy have no limits
}


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

    return model, scaler, encoders, feature_names
