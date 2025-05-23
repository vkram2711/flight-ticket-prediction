import streamlit as st
import pandas as pd
import numpy as np
import pickle
import holidays
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import airportsdata

# Constants
MODEL_DIR = 'model_files'
CACHE_DIR = 'cache'

@st.cache_resource
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

@st.cache_data
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

def get_available_options(encoders):
    """Get available options for each categorical feature."""
    return {
        'aircraftModel': encoders['aircraftModel'].classes_,
        'category': encoders['category'].classes_,
        'leg_Departure_Airport': encoders['leg_Departure_Airport'].classes_,
        'leg_Arrival_Airport': encoders['leg_Arrival_Airport'].classes_
    }

def create_route_encoder(encoders, new_route):
    """Create a new route encoder that can handle unseen routes."""
    route_encoder = LabelEncoder()
    all_routes = list(encoders['route'].classes_)
    if new_route not in all_routes:
        all_routes.append(new_route)
    route_encoder.fit(all_routes)
    return route_encoder

def get_airport_coordinates(airport_code):
    """Get airport coordinates using airport code rules to determine the correct database."""
    try:
        # Rule 1: If code contains numbers, it's a FAA LID
        if any(char.isdigit() for char in airport_code):
            airports = airportsdata.load('LID')
            airport = airports.get(airport_code)
            if airport:
                return (airport['lat'], airport['lon'])
            st.warning(f"FAA LID {airport_code} not found in database")
            return (0, 0)
            
        # Rule 2: If code is 4 letters, it's ICAO
        if len(airport_code) == 4 and airport_code.isalpha():
            airports = airportsdata.load('ICAO')
            airport = airports.get(airport_code)
            if airport:
                return (airport['lat'], airport['lon'])
            st.warning(f"ICAO code {airport_code} not found in database")
            return (0, 0)
            
        # Rule 3: If code is 3 letters, it's IATA
        if len(airport_code) == 3 and airport_code.isalpha():
            airports = airportsdata.load('IATA')
            airport = airports.get(airport_code)
            if airport:
                return (airport['lat'], airport['lon'])
            st.warning(f"IATA code {airport_code} not found in database")
            return (0, 0)
            
        st.warning(f"Invalid airport code format: {airport_code}")
        return (0, 0)
    except Exception as e:
        st.warning(f"Error getting coordinates for {airport_code}: {str(e)}")
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

def create_features(input_data, encoders):
    """Create features for prediction."""
    # Convert dates
    departure_date = pd.to_datetime(input_data['leg_Departure_Date'])
    
    # Basic features
    features = {}
    
    # Calculate airport distance
    features['airport_distance'] = calculate_distance(
        input_data['leg_Departure_Airport'],
        input_data['leg_Arrival_Airport']
    )
    
    # Holiday features
    us_holidays = holidays.US()
    features['is_holiday'] = departure_date in us_holidays
    
    # Weekend features
    features['is_weekend'] = departure_date.dayofweek in [5, 6]
    
    # Route features
    route = f"{input_data['leg_Departure_Airport']} - {input_data['leg_Arrival_Airport']}"
    
    # Create new route encoder for this route
    route_encoder = create_route_encoder(encoders, route)
    
    # Encode categorical features
    for col in ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']:
        value = input_data[col]
        features[f'{col}_encoded'] = encoders[col].transform([value])[0]
    
    # Encode route using the new encoder
    features['route_encoded'] = route_encoder.transform([route])[0]
    
    return features

def main():
    st.title("Flight Price Prediction")
    
    try:
        # Load model components
        model, scaler, encoders, feature_names, metadata = load_model_components()
        available_options = get_available_options(encoders)
        
        # Sidebar for model information
        st.sidebar.title("Model Information")
        st.sidebar.write("Model Performance:")
        st.sidebar.write(f"MAE: ${metadata['model_performance']['mae']:,.2f}")
        st.sidebar.write(f"RMSE: ${metadata['model_performance']['rmse']:,.2f}")
        st.sidebar.write(f"R² Score: {metadata['model_performance']['r2']:.2f}")
        
        # Input form
        st.header("Flight Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Aircraft information
            aircraft_model = st.selectbox('Aircraft Model', available_options['aircraftModel'])
            category = st.selectbox('Category', available_options['category'])
            
            # Airport information
            departure_airport = st.selectbox('Departure Airport', available_options['leg_Departure_Airport'])
            arrival_airport = st.selectbox('Arrival Airport', available_options['leg_Arrival_Airport'])
        
        with col2:
            # Date information (only needed for holiday/weekend check)
            departure_date = st.date_input('Departure Date', datetime.now() + timedelta(days=7))
        
        # Create input data dictionary
        input_data = {
            'aircraftModel': aircraft_model,
            'category': category,
            'leg_Departure_Airport': departure_airport,
            'leg_Arrival_Airport': arrival_airport,
            'leg_Departure_Date': departure_date
        }
        
        # Create features
        features = create_features(input_data, encoders)
        
        # Prepare input for prediction
        X = pd.DataFrame([features])
        X = X[feature_names]  # Ensure correct feature order
        X_scaled = scaler.transform(X)
        
        # Make prediction
        if st.button('Predict Price'):
            with st.spinner('Calculating price...'):
                # Get prediction
                log_prediction = model.predict(X_scaled)[0]
                prediction = np.expm1(log_prediction)
                
                # Display prediction
                st.success(f"Predicted Price: ${prediction:,.2f}")
                
                # Display feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                })
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                st.bar_chart(feature_importance.set_index('feature'))
                
                # Display derived features
                st.subheader("Derived Features")
                derived_features = {
                    'Route': f"{departure_airport} - {arrival_airport}",
                    'Distance': f"{features['airport_distance']:,.1f} km",
                    'Is holiday': features['is_holiday'],
                    'Is weekend': features['is_weekend']
                }
                st.write(derived_features)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all model files are present in the model_files directory.")

if __name__ == "__main__":
    main() 