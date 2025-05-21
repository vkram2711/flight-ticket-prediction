import csv

# File paths
file1_path = "airports.csv"
file2_path = "airports2.csv"
output_path = "merged_airports.csv"

# Output columns
output_columns = ["icao", "iata", "name", "city", "subd", "country", "lat", "lon", "lid"]

# Load file2 into lookup maps
icao_map = {}
iata_map = {}
lid_map = {}

with open(file2_path, newline='', encoding='utf-8') as f2:
    reader = csv.DictReader(f2)
    for row in reader:
        if row.get("icao"):
            icao_map[row["icao"]] = row
        if row.get("iata"):
            iata_map[row["iata"]] = row
        if row.get("lid"):
            lid_map[row["lid"]] = row

# Process file1 line by line and write output
with open(file1_path, newline='', encoding='utf-8') as f1, \
     open(output_path, mode='w', newline='', encoding='utf-8') as fout:

    reader = csv.DictReader(f1)
    writer = csv.DictWriter(fout, fieldnames=output_columns)
    writer.writeheader()

    for row in reader:
        merged = {key: "" for key in output_columns}

        # Try matches in order: ICAO, IATA, LID
        match = (
            icao_map.get(row.get("icao_code")) or
            iata_map.get(row.get("iata_code")) or
            lid_map.get(row.get("local_code"))
        )

        if match:
            for col in output_columns:
                merged[col] = match.get(col, "")

        # Fallback from file1 if no match or missing field
        merged["icao"] = merged["icao"] or row.get("icao_code", "")
        merged["iata"] = merged["iata"] or row.get("iata_code", "")
        merged["lid"] = merged["lid"] or row.get("local_code", "")
        merged["name"] = merged["name"] or row.get("name", "")
        merged["city"] = merged["city"] or row.get("municipality", "")
        merged["subd"] = merged["subd"] or row.get("iso_region", "")
        merged["country"] = merged["country"] or row.get("iso_country", "")
        merged["lat"] = merged["lat"] or row.get("latitude_deg", "")
        merged["lon"] = merged["lon"] or row.get("longitude_deg", "")

        writer.writerow(merged)

print("✅ Wrote merged_airports.csv successfully (no pandas, no RAM blowup).")
