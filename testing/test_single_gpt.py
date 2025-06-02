import pandas as pd
from gpt.gpt_utils import (
    initialize_gpt_client,
    create_assistant,
    get_or_create_thread,
    create_gpt_prompt,
    get_gpt_prediction
)

def main():
    # Initialize OpenAI client
    client = initialize_gpt_client()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('test_results.csv')
    
    # Get a single route for testing
    test_route = df['route'].value_counts().index[0]  # Get the most common route
    route_data = df[df['route'] == test_route].iloc[0]  # Get first instance of this route
    
    # Create input data
    input_data = {
        'aircraftModel': route_data['aircraft_model'],
        'category': route_data['category'],
        'leg_Departure_Airport': route_data['route'].split(' - ')[0],
        'leg_Arrival_Airport': route_data['route'].split(' - ')[1]
    }
    
    print(f"\nTesting prediction for route: {route_data['route']}")
    print(f"Category: {input_data['category']}")
    print(f"Aircraft Model: {input_data['aircraftModel']}")
    print(f"Sample Count: {route_data['sample_count']}")
    print(f"Median Price: ${route_data['median_price']:,.2f}")
    
    # Create or retrieve the assistant
    print("\nSetting up assistant...")
    assistant = create_assistant(client)
    
    # Get or create a thread
    thread, is_new = get_or_create_thread(client)
    if is_new:
        print("Created new conversation thread")
    else:
        print("Using existing conversation thread")
    
    # Create prompt and get prediction
    prompt = create_gpt_prompt(input_data)
    prediction = get_gpt_prediction(client, assistant, thread, prompt)
    
    if prediction:
        print("\nPrediction Results:")
        print(f"Predicted Price: ${prediction['predicted_price']:,.2f}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Reasoning: {prediction['reasoning']}")
        
        # Calculate deviation from median price
        deviation = ((prediction['predicted_price'] - route_data['median_price']) / route_data['median_price']) * 100
        print(f"\nDeviation from Median Price: {deviation:,.1f}%")
        print(f"Original Deviation: {route_data['deviation_percent']:,.1f}%")
        
        # Save thread ID for future use
        print(f"\nThread ID: {thread.id}")
        print("You can use this ID to continue the conversation in future runs")
    else:
        print("\nFailed to get prediction")

if __name__ == "__main__":
    main() 