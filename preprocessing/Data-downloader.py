import pandas as pd
import requests
import os
from urllib.parse import urlparse
from tqdm import tqdm  # Import tqdm for the progress bar

# Read the CSV file
df = pd.read_csv('../Datasets/Inat/observations-all-info-research-grade.csv/observations-598126.csv')

# Create a directory to save the sound files
os.makedirs('frog_sounds', exist_ok=True)

# Helper function to generate a unique filename
def get_unique_filename(directory, base_name, extension):
    full_path = os.path.join(directory, base_name + extension)
    counter = 1
    while os.path.exists(full_path):
        full_path = os.path.join(directory, f"{base_name}_{counter}{extension}")
        counter += 1
    return full_path

# Create a progress bar for the total number of downloads
with tqdm(total=len(df), desc="Downloading frog sounds") as pbar:
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        sound_url = row['sound_url']
        common_name = row['common_name']
        
        if pd.notna(sound_url) and pd.notna(common_name):
            try:
                # Extract the file extension from the URL
                parsed_url = urlparse(sound_url)
                filename = os.path.basename(parsed_url.path)
                extension = os.path.splitext(filename)[1]
                
                # Clean the common name to make it a valid filename
                clean_name = "".join(c if c.isalnum() or c in (' ', '-') else '_' for c in common_name).strip()
                clean_name = clean_name.replace(' ', '_')  # Optional: replace spaces with underscores
                
                # Generate a unique filename
                unique_filename = get_unique_filename('frog_sounds', clean_name, extension)
                
                # Download the file
                response = requests.get(sound_url, stream=True)
                if response.status_code == 200:
                    with open(unique_filename, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    print(f"Downloaded: {unique_filename}")
                else:
                    print(f"Failed to download: {sound_url}")
            except Exception as e:
                print(f"Error processing {sound_url}: {e}")

            # Update the progress bar after each download
            pbar.update(1)

print("Download complete.")
