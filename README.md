# Flight Price Prediction

A machine learning application that predicts flight prices based on various features such as route, dates, aircraft type, and other relevant factors. The project includes both a machine learning model for price prediction and a user-friendly web interface built with Streamlit.

## Features

- Flight price prediction using XGBoost model
- Interactive web interface for easy price estimation
- Support for various flight parameters:
  - Departure and arrival airports
  - Travel dates
  - Aircraft model
  - Number of passengers
  - Flight category
- Advanced feature engineering including:
  - Airport distance calculations
  - Holiday detection
  - Season analysis
  - Peak hour identification
  - Weekend detection

## Project Structure

```
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── flight_eda.py          # Exploratory data analysis
├── requirements.txt       # Project dependencies
├── model_files/          # Directory for trained models
├── cache/                # Cache directory for distance calculations
├── visualizations/       # Directory for data visualizations
└── processed_flight_data.csv  # Processed dataset
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ticketprediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model (if needed):
```bash
python train_model.py
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.2
- xgboost >= 1.4.2
- matplotlib >= 3.4.2
- seaborn >= 0.11.1
- holidays >= 0.14
- geopy >= 2.2.0
- streamlit >= 1.22.0
- airportsdata

## Model Performance

The model's performance metrics are displayed in the web application's sidebar, including:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
