import os
import pickle
import streamlit as st
# Constants
MODEL_DIR = '../model/model_files'
VISUALIZATIONS_DIR = '../visualizations'

# Category distance limits in kilometers
CATEGORY_LIMITS = {
    'Piston': 2000,
    'Turboprop': 2000,
    'Light Jet': 4000,
    'Entry level jet (VLJ)': 4000,
    'Super light jet': 4000,
    'Midsize': 6000,
    'Super midsize jet': 6000
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
