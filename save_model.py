import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from flight_price_prediction import load_data, preprocess_data, prepare_features, train_model

# Load and preprocess the data
df = load_data('CleanOne.csv')
df = preprocess_data(df)

# Create and fit label encoders
le_aircraft = LabelEncoder()
le_category = LabelEncoder()
le_departure = LabelEncoder()
le_arrival = LabelEncoder()

df['aircraftModel_encoded'] = le_aircraft.fit_transform(df['aircraftModel'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['departure_airport_encoded'] = le_departure.fit_transform(df['leg_Departure_Airport'])
df['arrival_airport_encoded'] = le_arrival.fit_transform(df['leg_Arrival_Airport'])

# Save the label encoders and their mappings
label_encoders = {
    'aircraft': {
        'encoder': le_aircraft,
        'mapping': dict(zip(le_aircraft.classes_, le_aircraft.transform(le_aircraft.classes_)))
    },
    'category': {
        'encoder': le_category,
        'mapping': dict(zip(le_category.classes_, le_category.transform(le_category.classes_)))
    },
    'departure': {
        'encoder': le_departure,
        'mapping': dict(zip(le_departure.classes_, le_departure.transform(le_departure.classes_)))
    },
    'arrival': {
        'encoder': le_arrival,
        'mapping': dict(zip(le_arrival.classes_, le_arrival.transform(le_arrival.classes_)))
    }
}

# Prepare features and train model
X, y, scaler, feature_columns = prepare_features(df)
model = train_model(X, y)

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metadata including feature columns and label encoders
metadata = {
    'feature_columns': feature_columns,
    'label_encoders': label_encoders
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Model and related files have been saved successfully!") 