import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, scaler, metadata

# Title and description
st.title("✈️ Flight Price Predictor")
st.markdown("""
This app predicts flight prices based on your travel details. 
Enter your travel information below to get an estimate.
""")

# Load model and metadata
model, scaler, metadata = load_model()

# Get available options from label encoders
aircraft_options = list(metadata['label_encoders']['aircraft']['mapping'].keys())
category_options = list(metadata['label_encoders']['category']['mapping'].keys())
departure_options = list(metadata['label_encoders']['departure']['mapping'].keys())
arrival_options = list(metadata['label_encoders']['arrival']['mapping'].keys())

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    # Origin and Destination
    st.subheader("Route Information")
    departure_airport = st.selectbox("Departure Airport", departure_options)
    arrival_airport = st.selectbox("Arrival Airport", arrival_options)
    
    # Date selection
    st.subheader("Travel Dates")
    departure_date = st.date_input(
        "Departure Date",
        min_value=datetime.now(),
        max_value=datetime.now() + timedelta(days=365)
    )
    
    # Time selection
    departure_time = st.time_input("Departure Time")

with col2:
    # Additional flight details
    st.subheader("Flight Details")
    aircraft_model = st.selectbox("Aircraft Model", aircraft_options)
    category = st.selectbox("Class", category_options)
    
    stops = st.selectbox(
        "Number of Stops",
        ["0", "1", "2+"]
    )
    
    passengers = st.number_input("Number of Passengers", min_value=1, max_value=9, value=1)

# Add a predict button
if st.button("Predict Price", type="primary"):
    try:
        # Encode the categorical variables
        aircraft_encoded = metadata['label_encoders']['aircraft']['mapping'][aircraft_model]
        category_encoded = metadata['label_encoders']['category']['mapping'][category]
        departure_encoded = metadata['label_encoders']['departure']['mapping'][departure_airport]
        arrival_encoded = metadata['label_encoders']['arrival']['mapping'][arrival_airport]
        
        # Prepare the input data with exact feature names and order
        input_data = pd.DataFrame({
            'aircraftModel_encoded': [aircraft_encoded],
            'category_encoded': [category_encoded],
            'quote_month': [datetime.now().month],
            'quote_day': [datetime.now().day],
            'quote_dayofweek': [datetime.now().weekday()],
            'departure_month': [departure_date.month],
            'departure_day': [departure_date.day],
            'departure_dayofweek': [departure_date.weekday()],
            'trip_duration': [1],  # Assuming one-way trip
            'leg_Passenger_number_PAX': [passengers],
            'departure_airport_encoded': [departure_encoded],
            'arrival_airport_encoded': [arrival_encoded]
        })
        
        # Ensure the columns are in the correct order
        input_data = input_data[metadata['feature_columns']]
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Convert from log scale
        price = np.expm1(prediction[0])
        
        # Display the prediction
        st.success(f"Estimated Price: ${price:,.2f}")
        
        # Add confidence interval
        st.info(f"""
        Note: This is an estimate with a typical error margin of ±$3,470.83
        The actual price may vary based on:
        - Time of booking
        - Seasonality
        - Airline pricing strategies
        - Competition on the route
        """)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("""
        Please make sure you have run save_model.py first to generate the model files.
        """)

# Add some additional information
st.markdown("---")
st.markdown("""
### About the Model
- Mean Absolute Error: $3,470.83
- Root Mean Squared Error: $5,124.88
- R² Score: 0.75
""") 