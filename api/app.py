"""
Flight Price Prediction API

This API provides endpoints for predicting private jet flight prices based on various parameters
such as aircraft category, departure and arrival airports. It uses a machine learning model
trained on historical flight data to make predictions.

The API includes endpoints for:
- Price prediction
- Health status checking
- Airport listing
- Aircraft category information

Author: Your Name
Version: 1.0
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource
import pandas as pd
import numpy as np

from airport_utils.utils import (
    calculate_distance, 
    check_airport_code, 
    get_airport_by_code,
    get_airports_by_country
)
from model.utils import CATEGORY_LIMITS, load_model_components, CATEGORIES
from docs import init_docs

# Initialize Flask application
app = Flask(__name__)

# Initialize Flask-RESTX API with metadata
api = Api(
    app,
    version='1.0',
    title='Flight Price Prediction API',
    description='''
    API for predicting private jet flight prices.
    
    This API provides endpoints to:
    - Predict flight prices for different aircraft categories
    - Get available airports and aircraft categories
    - Check API health status
    
    The prediction model takes into account:
    - Aircraft category and model
    - Departure and arrival airports
    - Distance between airports
    
    All prices are predicted in the same currency unit as the training data.
    ''',
    doc='/',  # Serve Swagger UI at root URL
    default='Flight Prediction',  # Default namespace
    default_label='Flight Prediction Operations'
)

# Create namespaces
ns = api.namespace('api', description='Flight prediction operations')

# Initialize API documentation and models
models = init_docs(api)

# Constants
MODEL_DIR = '../model/model_files'

def validate_airports(dep_airport, arr_airport):
    """
    Validate that airports exist in the database.
    
    Args:
        dep_airport (str): Departure airport IATA code
        arr_airport (str): Arrival airport IATA code
        
    Raises:
        ValueError: If either airport is not found or if they are the same
    """
    if not check_airport_code(dep_airport):
        raise ValueError(f"Departure airport '{dep_airport}' not found in database")
    if not check_airport_code(arr_airport):
        raise ValueError(f"Arrival airport '{arr_airport}' not found in database")
    if dep_airport == arr_airport:
        raise ValueError("Departure and arrival airports cannot be the same")

def validate_category_distance(category, distance):
    """
    Validate that the distance is within category limits and minimum distance.
    
    Args:
        category (str): Aircraft category
        distance (float): Distance in kilometers
        
    Raises:
        ValueError: If distance is less than minimum or exceeds category limit
    """
    if distance < 55:
        raise ValueError(f"Distance ({distance:.0f} km) is less than minimum allowed distance (55 km)")
    if category in CATEGORY_LIMITS:
        max_distance = CATEGORY_LIMITS[category]
        if distance > max_distance:
            raise ValueError(
                f"Distance ({distance:.0f} km) exceeds maximum allowed distance "
                f"({max_distance} km) for {category} aircraft"
            )

def create_features(input_data, encoders):
    """
    Create features for prediction from input data.
    
    Args:
        input_data (dict): Dictionary containing input parameters
        encoders (dict): Dictionary of encoders for categorical features
        
    Returns:
        dict: Dictionary of features ready for model prediction
        
    Raises:
        ValueError: If input validation fails
    """
    features = {}
    
    # Validate airports
    validate_airports(
        input_data['leg_Departure_Airport'],
        input_data['leg_Arrival_Airport']
    )
    
    # Encode categorical features
    for col in ['aircraftModel', 'category']:
        if col in input_data:
            try:
                features[f'{col}_encoded'] = encoders[col].transform([input_data[col]])[0]
            except ValueError as e:
                if 'unseen' in str(e):
                    raise ValueError(f"Unseen value for {col}: {input_data[col]}")
                raise e
    
    # Calculate airport distance
    distance = calculate_distance(
        input_data['leg_Departure_Airport'],
        input_data['leg_Arrival_Airport']
    )
    if distance is None:
        raise ValueError(f"Could not calculate distance between {input_data['leg_Departure_Airport']} and {input_data['leg_Arrival_Airport']}")
    
    # Validate distance against category limits
    validate_category_distance(input_data['category'], distance)
    
    features['airport_distance'] = distance
    
    # Create route feature
    route = f"{input_data['leg_Departure_Airport']} - {input_data['leg_Arrival_Airport']}"
    try:
        features['route_encoded'] = encoders['route'].transform([route])[0]
    except ValueError:
        # If route is not in training data, use a default encoding
        features['route_encoded'] = 0
    
    return features

# Load model components at startup
print("Loading model components...")
model, scaler, encoders, feature_names = load_model_components()

# Define available aircraft models for each category
CATEGORY_MODELS = {
    'Piston': ['Cessna 402C', 'Cirrus SR 22', 'Piper Aerostar 600'],
    'Turbo prop': ['Piper M600', 'Pilatus PC12 NGX', 'King Air 350'],
    'Light jet': ['Beechjet 400A', 'Citation II', 'Learjet 35A'],
    'Entry level jet (VLJ)': ['Citation CJ1', 'HondaJet', 'Phenom 100'],
    'Super light jet': ['Learjet 45', 'Phenom 300', 'Citation Excel'],
    'Midsize jet': ['Citation XLS', 'Hawker 800XP', 'Learjet 60'],
    'Super midsize jet': ['Falcon 50 EX', 'Citation X', 'Hawker 800XP'],
    'Heavy jet': ['Gulfstream G-IVSP', 'Learjet 60', 'Legacy 600'],
    'Ultra long range': ['Gulfstream G550', 'Global 6500', 'Falcon 7X']
}

@ns.route('/predict')
class Prediction(Resource):
    @ns.expect(models['prediction_request'])
    @ns.response(200, 'Success', models['prediction_response'])
    @ns.response(400, 'Bad Request', models['error_response'])
    def post(self):
        """
        Make a price prediction for a flight route.
        
        This endpoint predicts the price for a flight between two airports using a specific
        aircraft category. The prediction is made for all available aircraft models within
        the specified category.
        
        The prediction takes into account:
        - Distance between airports
        - Aircraft category and specific models
        - Historical route data
        
        Returns predictions for all available aircraft models in the specified category.
        """
        try:
            # Get input data from request
            input_data = request.get_json()
            if not input_data:
                return {'error': 'No input data provided'}, 400
            
            # Validate required fields
            required_fields = ['category', 'leg_Departure_Airport', 'leg_Arrival_Airport']
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                return {
                    'error': 'Missing required fields',
                    'missing_fields': missing_fields
                }, 400
            
            # Validate field types
            if not all(isinstance(input_data[field], str) for field in required_fields):
                return {
                    'error': 'All input fields must be strings',
                    'invalid_fields': [field for field in required_fields if not isinstance(input_data[field], str)]
                }, 400
            
            predictions = []
            distance = None
            if not input_data['category'] in CATEGORIES:
                return {
                    'error': f'Invalid category: {input_data["category"]}',
                    'available_categories': CATEGORIES
                }, 400

            for aircraftModel in CATEGORY_MODELS[input_data['category']]:
                input_data["aircraftModel"] = aircraftModel

                # Create features
                features = create_features(input_data, encoders)
                distance = float(features['airport_distance'])
                # Create DataFrame with features in the correct order
                feature_df = pd.DataFrame([features])
                feature_df = feature_df[feature_names]  # Ensure correct feature order

                # Scale features
                scaled_features = scaler.transform(feature_df)

                # Make prediction (model was trained on log-transformed prices)
                log_prediction = model.predict(scaled_features)[0]
                predicted_price = np.expm1(log_prediction)  # Transform back from log

                # Prepare response
                predictions.append({
                    'predicted_price': round(float(predicted_price), 2),
                    'aircraft_model': aircraftModel,
                })
            del input_data['aircraftModel']

            response = {
                "distance_nm": round(float(distance), 2),
                "predictions": predictions
            }
            return response
        
        except ValueError as e:
            return {'error': str(e)}, 400

@ns.route('/health')
class Health(Resource):
    @ns.response(200, 'Success', models['health_response'])
    def get(self):
        """
        Check API health status.
        
        This endpoint provides information about the current state of the API,
        including whether the prediction model and airport database are properly loaded.
        
        Returns:
            dict: Health status information
        """
        return {
            'status': 'healthy',
            'model_loaded': model is not None,
            'airports_loaded': True  # Database is always available
        }

@ns.route('/airports')
class Airports(Resource):
    @ns.response(200, 'Success', models['airports_response'])
    def get(self):
        """
        List all available airports.
        
        This endpoint returns a list of all airports that are available
        in the database for making predictions.
        
        Returns:
            dict: List of airport information
        """
        # Get all US airports as an example (you might want to modify this based on your needs)
        airports = get_airports_by_country('United States')
        return {'airports': airports}

@ns.route('/categories')
class Categories(Resource):
    @ns.response(200, 'Success', models['categories_response'])
    def get(self):
        """
        List all available aircraft categories and their distance limits.
        
        This endpoint returns information about all available aircraft categories
        and their maximum allowed flight distances.
        
        Returns:
            dict: List of categories and their distance limits
        """
        return {
            'categories': list(CATEGORIES),
            'distance_limits': CATEGORY_LIMITS
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 