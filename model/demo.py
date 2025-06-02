import streamlit as st
import pandas as pd
import numpy as np
from airport_utils.utils import load_airport_data, calculate_distance

# Constants
MODEL_DIR = 'model_files'
CACHE_DIR = '../cache'



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
        raise ValueError(f"Could not calculate distance between {input_data['leg_Departure_Airport']} and {input_data['leg_Arrival_Airport']}. Please select different airports.")
    
    features['airport_distance'] = distance
    
    # Route features
    route = f"{input_data['leg_Departure_Airport']} - {input_data['leg_Arrival_Airport']}"
    
    # Encode categorical features
    for col in ['aircraftModel', 'category', 'leg_Departure_Airport', 'leg_Arrival_Airport']:
        value = input_data[col]
        if value not in encoders[col].classes_:
            raise ValueError(f"Unseen value '{value}' for {col}. Please select a different option.")
        features[f'{col}_encoded'] = encoders[col].transform([value])[0]
    
    # Handle route encoding
    if route not in encoders['route'].classes_:
        # If route is not in training data, use a default route encoding
        features['route_encoded'] = 0
    else:
        features['route_encoded'] = encoders['route'].transform([route])[0]
    
    return features

def main():
    st.title("Flight Price Prediction")
    
    try:
        # Load model components
        model, scaler, encoders, feature_names, metadata = load_model_components()
        available_options = get_available_options(encoders)
        
        # Load airport data
        airports_df = load_airport_data()
        if airports_df is None:
            st.error("Failed to load airport data. Please check if merged_airports.csv exists.")
            return
        
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
        
        # Create input data dictionary
        input_data = {
            'aircraftModel': aircraft_model,
            'category': category,
            'leg_Departure_Airport': departure_airport,
            'leg_Arrival_Airport': arrival_airport
        }
        
        # Create features
        try:
            features = create_features(input_data, encoders, airports_df)
            
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
                        'Distance': f"{features['airport_distance']:,.1f} km"
                    }
                    st.write(derived_features)
        
        except ValueError as e:
            st.error(str(e))
            st.info("Please select different options from the dropdown menus.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all model files are present in the model_files directory.")

def get_available_options(encoders):
    """Get available options for each categorical feature."""
    return {
        'aircraftModel': encoders['aircraftModel'].classes_,
        'category': encoders['category'].classes_,
        'leg_Departure_Airport': encoders['leg_Departure_Airport'].classes_,
        'leg_Arrival_Airport': encoders['leg_Arrival_Airport'].classes_
    }

if __name__ == "__main__":
    main() 