from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from airport_utils.utils import load_airport_data, calculate_distance, check_airport_code
from model.utils import CATEGORY_LIMITS, load_model_components, CATEGORIES

app = Flask(__name__)

# Constants
MODEL_DIR = '../model/model_files'


def validate_airports(dep_airport, arr_airport, airports_df):
    """Validate that airports exist in the database."""
    if not check_airport_code(dep_airport, airports_df):
        raise ValueError(f"Departure airport '{dep_airport}' not found in database")
    if not check_airport_code(arr_airport, airports_df):
        raise ValueError(f"Arrival airport '{arr_airport}' not found in database")
    if dep_airport == arr_airport:
        raise ValueError("Departure and arrival airports cannot be the same")


def validate_category_distance(category, distance):
    """Validate that the distance is within category limits and minimum distance."""
    if distance < 55:
        raise ValueError(f"Distance ({distance:.0f} km) is less than minimum allowed distance (55 km)")
    if category in CATEGORY_LIMITS:
        max_distance = CATEGORY_LIMITS[category]
        if distance > max_distance:
            raise ValueError(
                f"Distance ({distance:.0f} km) exceeds maximum allowed distance "
                f"({max_distance} km) for {category} aircraft"
            )

def create_features(input_data, encoders, airports_df):
    """Create features for prediction."""
    features = {}
    
    # Validate airports
    validate_airports(
        input_data['leg_Departure_Airport'],
        input_data['leg_Arrival_Airport'],
        airports_df
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
        input_data['leg_Arrival_Airport'],
        airports_df
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

# Load model components and airport data at startup
print("Loading model components...")
model, scaler, encoders, feature_names = load_model_components()

print("Loading airport data...")
airports_df = load_airport_data()
if airports_df is None:
    raise ValueError("Failed to load airport data")


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


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for price prediction."""
    try:
        # Get input data from request
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Validate required fields
        required_fields = ['category', 'leg_Departure_Airport', 'leg_Arrival_Airport']
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Validate field types
        if not all(isinstance(input_data[field], str) for field in required_fields):
            return jsonify({
                'error': 'All input fields must be strings',
                'invalid_fields': [field for field in required_fields if not isinstance(input_data[field], str)]
            }), 400
        predictions = []
        distance = None
        if not input_data['category'] in CATEGORIES:
            return jsonify({
                'error': f'Invalid category: {input_data['category']}',
                'available_categories': CATEGORIES
            }), 400

        for aircraftModel in CATEGORY_MODELS[input_data['category']]:
            input_data["aircraftModel"] = aircraftModel

            # Create features
            features = create_features(input_data, encoders, airports_df)
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
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    #except Exception as e:
    #    app.logger.error(f"Unexpected error: {str(e)}")
    #    return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'airports_loaded': airports_df is not None
    })

@app.route('/airports', methods=['GET'])
def list_airports():
    """List all available airports."""
    return jsonify({
        'airports': airports_df['IATA'].tolist()
    })

@app.route('/categories', methods=['GET'])
def list_categories():
    """List all available aircraft categories and their distance limits."""
    return jsonify({
        'categories': list(CATEGORIES),
        'distance_limits': CATEGORY_LIMITS
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 