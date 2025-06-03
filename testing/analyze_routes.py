import os
import pandas as pd

# Get the absolute path to the testing directory
TEST_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR_PATH)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'model', 'processed_data')

def find_common_routes(df, top_n=5):
    """Find the most common routes for each category."""
    common_routes = {}
    
    for category in df['category'].unique():
        category_routes = df[df['category'] == category]['route'].value_counts().head(top_n)
        common_routes[category] = category_routes.index.tolist()
    
    return common_routes

def calculate_route_statistics(df, common_routes):
    """Calculate statistics for common routes."""
    route_stats = []
    
    for category, routes in common_routes.items():
        for route in routes:
            route_data = df[(df['category'] == category) & (df['route'] == route)]
            if not route_data.empty:
                # Get the most common aircraft model for this route
                most_common_aircraft = route_data['aircraftModel'].mode().iloc[0]
                
                stats = {
                    'route': route,
                    'category': category,
                    'aircraft_model': most_common_aircraft,
                    'avg_price': route_data['price'].mean(),
                    'median_price': route_data['price'].median(),
                    'std_price': route_data['price'].std(),
                    'min_price': route_data['price'].min(),
                    'max_price': route_data['price'].max(),
                    'flight_count': len(route_data)
                }
                route_stats.append(stats)
    
    return pd.DataFrame(route_stats)

def main():
    try:
        # Load input data
        print("Loading input data...")
        input_data_path = os.path.join(PROCESSED_DATA_DIR, 'model_input_data.csv')
        df = pd.read_csv(input_data_path)
        print(f"Loaded {len(df)} records")

        # Create route column if it doesn't exist
        if 'route' not in df.columns:
            df['route'] = df['leg_Departure_Airport'] + ' - ' + df['leg_Arrival_Airport']

        # Find most common routes for each category
        print("\nFinding most common routes...")
        common_routes = find_common_routes(df)
        
        # Calculate statistics for common routes
        print("Calculating route statistics...")
        route_stats_df = calculate_route_statistics(df, common_routes)
        
        # Save route statistics
        route_stats_path = os.path.join(PROCESSED_DATA_DIR, 'route_statistics.csv')
        route_stats_df.to_csv(route_stats_path, index=False)
        print(f"Route statistics saved to {route_stats_path}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
