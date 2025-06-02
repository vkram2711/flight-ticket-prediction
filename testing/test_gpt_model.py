import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gpt.gpt_utils import (
    initialize_gpt_client,
    create_assistant,
    get_or_create_thread,
    create_gpt_prompt,
    get_gpt_prediction
)


def test_gpt_predictions(route_stats_df, thread_id=None):
    """
    Test GPT model predictions against median prices
    """
    print("Initializing GPT client...")
    client = initialize_gpt_client()

    # Create or retrieve the assistant
    print("Setting up assistant...")
    assistant = create_assistant(client)

    # Get or create thread
    thread, is_new = get_or_create_thread(client, thread_id)
    if is_new:
        print("Created new conversation thread")
    else:
        print("Using existing conversation thread")

    results = []
    total_routes = len(route_stats_df)

    for idx, row in route_stats_df.iterrows():
        print(f"\nProcessing route {idx + 1}/{total_routes}: {row['route']}")

        # Create input data
        input_data = {
            'aircraftModel': row['aircraft_model'],
            'category': row['category'],
            'leg_Departure_Airport': row['route'].split(' - ')[0],
            'leg_Arrival_Airport': row['route'].split(' - ')[1]
        }

        # Create prompt and get prediction
        prompt = create_gpt_prompt(input_data, route_stats=row)
        prediction = get_gpt_prediction(client, assistant, thread, prompt, verbose=False)

        if prediction:
            # Calculate deviation from median price
            deviation = ((prediction['predicted_price'] - row['median_price']) / row['median_price']) * 100

            results.append({
                'route': row['route'],
                'category': row['category'],
                'median_price': row['median_price'],
                'predicted_price': prediction['predicted_price'],
                'confidence': prediction['confidence'],
                'deviation_percent': deviation,
                'reasoning': prediction['reasoning']
            })
            print(f"Predicted: ${prediction['predicted_price']:,.2f} (Deviation: {deviation:,.1f}%)")
        else:
            print("Failed to get prediction")

    # Save thread ID for future use
    print(f"\nThread ID: {thread.id}")
    print("You can use this ID to continue the conversation in future runs")

    return pd.DataFrame(results)


def plot_results(results_df):
    """
    Create visualizations of the results
    """
    # Create visualizations directory if it doesn't exist
    os.makedirs('../visualizations', exist_ok=True)

    # 1. Box plot of deviations by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='category', y='deviation_percent')
    plt.title('Price Deviation Distribution by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/gpt_deviations_by_category.png')
    plt.close()

    # 2. Scatter plot of predicted vs median prices
    plt.figure(figsize=(10, 10))
    plt.scatter(results_df['median_price'], results_df['predicted_price'], alpha=0.5)
    plt.plot([0, results_df['median_price'].max()], [0, results_df['median_price'].max()], 'r--')
    plt.xlabel('Median Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs Median Prices')
    plt.tight_layout()
    plt.savefig('visualizations/gpt_predicted_vs_median.png')
    plt.close()

    # 3. Confidence vs Absolute Deviation
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['confidence'], abs(results_df['deviation_percent']), alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Absolute Deviation (%)')
    plt.title('Confidence vs Absolute Deviation')
    plt.tight_layout()
    plt.savefig('visualizations/gpt_confidence_vs_deviation.png')
    plt.close()


def main():
    # Load route statistics
    print("Loading route statistics...")
    route_stats_df = pd.read_csv('../model/processed_data/route_statistics.csv')

    # Test GPT predictions
    results_df = test_gpt_predictions(route_stats_df)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'gpt_test_results_{timestamp}.csv', index=False)

    # Create visualizations
    plot_results(results_df)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Routes Tested: {len(results_df)}")
    print(f"Mean Predicted Price: ${results_df['predicted_price'].mean():,.2f}")
    print(f"Mean Median Price: ${results_df['median_price'].mean():,.2f}")
    print(f"Mean Absolute Deviation: {abs(results_df['deviation_percent']).mean():,.1f}%")
    print(f"Mean Confidence: {results_df['confidence'].mean():.2%}")

    print("\nTop 5 Routes with Largest Deviations:")
    top_deviations = results_df.nlargest(5, 'deviation_percent')
    for _, row in top_deviations.iterrows():
        print(f"\nRoute: {row['route']}")
        print(f"Category: {row['category']}")
        print(f"Median Price: ${row['median_price']:,.2f}")
        print(f"Predicted Price: ${row['predicted_price']:,.2f}")
        print(f"Deviation: {row['deviation_percent']:,.1f}%")
        print(f"Confidence: {row['confidence']:.2%}")
        print(f"Reasoning: {row['reasoning']}")


if __name__ == "__main__":
    main()
