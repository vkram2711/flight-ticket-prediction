import os
import json
import time
from openai import OpenAI

def create_assistant(client):
    """Create or retrieve the flight price prediction assistant."""
    # List existing assistants
    assistants = client.beta.assistants.list()
    
    # Check if our assistant already exists
    for assistant in assistants.data:
        if assistant.name == "Flight Price Predictor":
            return assistant
    
    # Create new assistant if it doesn't exist
    assistant = client.beta.assistants.create(
        name="Flight Price Predictor",
        instructions="""You are an expert in flight price prediction. Your task is to predict flight prices based on route information and aircraft type.
        Always provide your predictions in the exact JSON format requested, including:
        - predicted_price: a numerical value
        - confidence: a number between 0 and 1
        - reasoning: a brief explanation of your prediction
        
        Consider factors such as:
        - Route popularity and demand
        - Aircraft type and capacity
        - Historical price patterns
        - Price ranges and volatility
        - Market conditions
        
        IMPORTANT: Your response must be ONLY the JSON object, with no additional text or explanation.""",
        model="gpt-4-turbo-preview",
        tools=[{"type": "code_interpreter"}]
    )
    return assistant

def get_or_create_thread(client, thread_id=None):
    """Get an existing thread or create a new one.
    
    Args:
        client: OpenAI client instance
        thread_id: Optional thread ID to retrieve. If None, creates a new thread.
    
    Returns:
        tuple: (thread, is_new_thread)
    """
    if thread_id:
        try:
            thread = client.beta.threads.retrieve(thread_id)
            return thread, False
        except Exception as e:
            print(f"Could not retrieve thread {thread_id}: {str(e)}")
            print("Creating new thread...")
    
    thread = client.beta.threads.create()
    return thread, True

def create_gpt_prompt(input_data, route_stats=None):
    """Create a prompt for GPT model."""
    if route_stats is not None:
        # For batch processing with route statistics
        stats_text = f"""
Historical Statistics for this route:
- Median Price: ${route_stats['median_price']:,.2f}
- Mean Price: ${route_stats['mean_price']:,.2f}
- Price Range: ${route_stats['min_price']:,.2f} - ${route_stats['max_price']:,.2f}
- Standard Deviation: ${route_stats['std_price']:,.2f}
- Number of Samples: {route_stats['sample_count']}
"""
    else:
        stats_text = ""
    
    prompt = f"""Based on this flight information, predict a price. Return ONLY a JSON object with no additional text:

Route: {input_data['leg_Departure_Airport']} to {input_data['leg_Arrival_Airport']}
Aircraft Model: {input_data['aircraftModel']}
Category: {input_data['category']}{stats_text}

Required JSON format:
{{
    "predicted_price": <number>,
    "confidence": <number between 0 and 1>,
    "reasoning": "<brief explanation>"
}}"""
    return prompt

def get_gpt_prediction(client, assistant, thread, prompt, verbose=True):
    """Get prediction from GPT model using Assistants API."""
    try:
        if verbose:
            print("\nSending prompt to assistant...")
            print("Prompt content:")
            print("-" * 50)
            print(prompt)
            print("-" * 50)
        
        # Add the message to the thread
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        
        if verbose:
            print("Starting assistant run...")
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for the run to complete
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if verbose:
                print(f"Run status: {run_status.status}")
            if run_status.status == 'completed':
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                raise Exception(f"Run failed with status: {run_status.status}")
            time.sleep(1)
        
        if verbose:
            print("Retrieving messages...")
        # Get the messages
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Get the assistant's response
        for message in messages.data:
            if message.role == "assistant":
                # Extract the text content
                content = message.content[0].text.value
                if verbose:
                    print("\nAssistant's response:")
                    print("-" * 50)
                    print(content)
                    print("-" * 50)
                
                # Try to find JSON in the response
                try:
                    # First try direct JSON parsing
                    prediction = json.loads(content)
                except json.JSONDecodeError:
                    # If that fails, try to extract JSON from the text
                    try:
                        # Look for JSON-like structure in the text
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            prediction = json.loads(json_str)
                        else:
                            raise ValueError("No JSON structure found in response")
                    except Exception as e:
                        print(f"Failed to parse JSON from response: {content}")
                        raise e
                
                # Validate the prediction format
                required_keys = ['predicted_price', 'confidence', 'reasoning']
                if not all(key in prediction for key in required_keys):
                    raise ValueError(f"Missing required keys in prediction. Got: {prediction.keys()}")
                
                # Validate the types
                if not isinstance(prediction['predicted_price'], (int, float)):
                    raise ValueError(f"predicted_price must be a number, got {type(prediction['predicted_price'])}")
                if not isinstance(prediction['confidence'], (int, float)):
                    raise ValueError(f"confidence must be a number, got {type(prediction['confidence'])}")
                if not isinstance(prediction['reasoning'], str):
                    raise ValueError(f"reasoning must be a string, got {type(prediction['reasoning'])}")
                
                return prediction
        
        raise ValueError("No assistant response found")
    
    except Exception as e:
        print(f"\nError getting GPT prediction: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        return None

def initialize_gpt_client():
    """Initialize OpenAI client and check API key."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    return client 