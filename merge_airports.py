import pandas as pd
import numpy as np

# Define the correct headers for airports2.csv
airports2_columns = ["icao","iata","name","city","subd","country","elevation","lat","lon","tz","lid"]

# Read the CSV files
print("Reading input files...")
df1 = pd.read_csv('airports.csv')
df2 = pd.read_csv('airports2.csv', names=airports2_columns, header=None)

print(f"First file has {len(df1)} airports")
print(f"Second file has {len(df2)} airports")

# Print column names for debugging
print("\nColumns in first file:", df1.columns.tolist())
print("Columns in second file:", df2.columns.tolist())

# Map columns from df1 to the required format
df1_mapped = pd.DataFrame()
df1_mapped['icao'] = df1['icao_code']
df1_mapped['iata'] = df1['iata_code']
df1_mapped['name'] = df1['name']
df1_mapped['city'] = df1['municipality']
df1_mapped['subd'] = df1['region']
df1_mapped['country'] = df1['country']
df1_mapped['elevation'] = np.nan  # Not available in first file
df1_mapped['lat'] = df1['latitude_deg']
df1_mapped['lon'] = df1['longitude_deg']
df1_mapped['tz'] = np.nan  # Not available in first file
df1_mapped['lid'] = df1['local_code']  # Map local_code to lid (FAA)

# df2 already has the correct format, so we can use it as is
print("\nCombining dataframes...")
combined_df = pd.concat([df1_mapped, df2], ignore_index=True)
print(f"Total rows after initial combination: {len(combined_df)}")

# Create a unique identifier for each airport using all three codes
combined_df['unique_id'] = combined_df.apply(
    lambda row: '_'.join(filter(None, [
        str(row['icao']) if pd.notna(row['icao']) else '',
        str(row['iata']) if pd.notna(row['iata']) else '',
        str(row['lid']) if pd.notna(row['lid']) else ''
    ])), 
    axis=1
)

# Remove duplicates while keeping the most complete record
print("\nRemoving duplicates...")
# First, count non-null values for each row
combined_df['non_null_count'] = combined_df.notna().sum(axis=1)

# Sort by non-null count in descending order to keep the most complete records
combined_df = combined_df.sort_values('non_null_count', ascending=False)

# Remove duplicates based on unique_id, keeping the first (most complete) record
combined_df = combined_df.drop_duplicates(subset=['unique_id'], keep='first')

# Remove the temporary columns
combined_df = combined_df.drop(['unique_id', 'non_null_count'], axis=1)

print(f"Rows after removing duplicates: {len(combined_df)}")

# Verify we haven't lost any airports from either source
df1_airports = set(df1_mapped['icao'].fillna('') + '_' + df1_mapped['iata'].fillna('') + '_' + df1_mapped['lid'].fillna(''))
df2_airports = set(df2['icao'].fillna('') + '_' + df2['iata'].fillna('') + '_' + df2['lid'].fillna(''))
combined_airports = set(combined_df['icao'].fillna('') + '_' + combined_df['iata'].fillna('') + '_' + combined_df['lid'].fillna(''))

print(f"\nVerification:")
print(f"Airports from first file: {len(df1_airports)}")
print(f"Airports from second file: {len(df2_airports)}")
print(f"Airports in combined file: {len(combined_airports)}")

# Sort by country and city for better organization
combined_df = combined_df.sort_values(['country', 'city'])

# Save the combined dataframe
print("\nSaving to merged_airports.csv...")
combined_df.to_csv('merged_airports.csv', index=False, quoting=1)  # quoting=1 ensures all fields are quoted

print(f"Final file created with {len(combined_df)} unique airports") 