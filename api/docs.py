"""
API Documentation and Models

This module contains all the Swagger documentation and models for the Flight Price Prediction API.
It defines the request/response models and their schemas for API documentation.
"""

from flask_restx import fields, Api
from model.utils import CATEGORIES

def init_docs(api: Api):
    """Initialize API documentation and models."""
    
    # Define models for request/response documentation
    prediction_request = api.model('PredictionRequest', {
        'category': fields.String(
            required=True,
            description='Aircraft category (e.g., "Light jet", "Heavy jet")',
            enum=list(CATEGORIES),
            example='Light jet'
        ),
        'leg_Departure_Airport': fields.String(
            required=True,
            description='IATA code of the departure airport (e.g., "JFK", "LAX")',
            example='JFK'
        ),
        'leg_Arrival_Airport': fields.String(
            required=True,
            description='IATA code of the arrival airport (e.g., "LAX", "SFO")',
            example='LAX'
        )
    })

    prediction_response = api.model('PredictionResponse', {
        'distance_nm': fields.Float(
            description='Distance between airports in nautical miles',
            example=2475.5
        ),
        'predictions': fields.List(
            fields.Nested(api.model('Prediction', {
                'predicted_price': fields.Float(
                    description='Predicted price for the flight',
                    example=15000.00
                ),
                'aircraft_model': fields.String(
                    description='Specific aircraft model used for prediction',
                    example='Citation II'
                )
            }))
        )
    })

    error_response = api.model('ErrorResponse', {
        'error': fields.String(
            description='Detailed error message explaining what went wrong',
            example='Invalid airport code: XYZ'
        ),
        'missing_fields': fields.List(
            fields.String,
            description='List of required fields that were not provided',
            example=['category', 'leg_Departure_Airport']
        ),
        'invalid_fields': fields.List(
            fields.String,
            description='List of fields with invalid values',
            example=['category']
        )
    })

    health_response = api.model('HealthResponse', {
        'status': fields.String(
            description='Current health status of the API',
            example='healthy'
        ),
        'model_loaded': fields.Boolean(
            description='Whether the prediction model is successfully loaded',
            example=True
        ),
        'airports_loaded': fields.Boolean(
            description='Whether the airport database is successfully loaded',
            example=True
        )
    })

    airports_response = api.model('AirportsResponse', {
        'airports': fields.List(
            fields.Nested(api.model('Airport', {
                'icao': fields.String(description='ICAO code'),
                'iata': fields.String(description='IATA code'),
                'name': fields.String(description='Airport name'),
                'city': fields.String(description='City'),
                'country': fields.String(description='Country')
            }))
        )
    })

    categories_response = api.model('CategoriesResponse', {
        'categories': fields.List(
            fields.String,
            description='List of all available aircraft categories',
            example=['Light jet', 'Heavy jet', 'Midsize jet']
        ),
        'distance_limits': fields.Raw(
            description='Maximum allowed distance (in km) for each aircraft category',
            example={'Light jet': 2000, 'Heavy jet': 5000}
        )
    })

    return {
        'prediction_request': prediction_request,
        'prediction_response': prediction_response,
        'error_response': error_response,
        'health_response': health_response,
        'airports_response': airports_response,
        'categories_response': categories_response
    } 