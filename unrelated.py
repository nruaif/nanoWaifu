import os
import pandas as pd
import json

# Configuration
PARQUET_DIR = r"C:\Users\nRuaif\Downloads\danb"
OUTPUT_DIR = r"C:\Users\nRuaif\Downloads\danb_filtered"
JSON_FILE = "tags.txt"
MIN_IMAGE_COUNT = 20

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the valid artists into a Set for fast lookup
print("Loading artist counts...")
with open(JSON_FILE, "r", encoding="utf-8") as f:
    artist_data = json.load(f)

# Create a set of artists who have > 20 images
# Sets are O(1) for lookups, making the filtering much faster than lists
valid_artists = {artist for artist, count in artist_data.items() if count > MIN_IMAGE_COUNT}

print(f"Found {len(valid_artists)} unique artists with > {MIN_IMAGE_COUNT} images.")

# 2. Collect parquet files
parquet_files = [f for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet")]

# 3. Process files one by one
print(f"Processing {len(parquet_files)} files...")

for i, file_name in enumerate(parquet_files):
    input_path = os.path.join(PARQUET_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name)

    try:
        # Read the file
        df = pd.read_parquet(input_path)

        # Drop rows where artist string is missing/NaN
        df = df.dropna(subset=["tag_string_artist"])


        # --- The Filtering Logic ---
        # We define a helper function to check if the row contains a valid artist
        def has_valid_artist(artist_string):
            # Split the string into individual artists
            current_artists = artist_string.split()
            # Check if there is any intersection between this row's artists and our valid set
            return not valid_artists.isdisjoint(current_artists)


        # Apply the filter
        # valid_artists.isdisjoint() returns True if NO match, so we negate it
        mask = df["tag_string_artist"].apply(has_valid_artist)
        filtered_df = df[mask]

        # Save only if we have data left
        if not filtered_df.empty:
            filtered_df.to_parquet(output_path, index=False)
            print(f"[{i + 1}/{len(parquet_files)}] Saved {len(filtered_df)} rows to {file_name}")
        else:
            print(f"[{i + 1}/{len(parquet_files)}] Skipped {file_name} (no matching artists)")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("Filtering complete!")