import os
import csv
from collections import defaultdict
from pathlib import Path

def analyze_frog_recordings(directory_path, min_threshold=100000):
    """
    Analyze frog recordings to find species with few recordings.
    
    Args:
        directory_path: Path to directory containing audio files
        min_threshold: Minimum number of recordings to not be considered "few"
    
    Returns:
        dict: {frog_species: count} for species with fewer than min_threshold recordings
    """
    # Initialize dictionary to store counts
    frog_counts = defaultdict(int)
    
    # Supported audio file extensions
    audio_extensions = {'.wav'}
    
    # Count recordings per frog species
    for file in Path(directory_path).iterdir():
        if file.suffix.lower() in audio_extensions:
            # Extract frog species name (handling formats like "common_name_number" or "common_name_number_primary")
            parts = file.stem.rsplit('_', 2)
            
            # Case 1: common_name_number
            if len(parts) >= 2 and parts[-1].isdigit():
                frog_species = '_'.join(parts[:-1])
                frog_counts[frog_species] += 1
            # Case 2: common_name_number_primary/secondary
            elif len(parts) == 3 and parts[1].isdigit() and parts[2] in ['primary', 'secondary']:
                frog_species = parts[0]
                frog_counts[frog_species] += 1

            # print(frog_species)
    
    # Filter for species with few recordings
    few_recordings = {
        species: count 
        for species, count in frog_counts.items() 
        # if count < min_threshold
    }
    
    return few_recordings

def print_frog_report(few_recordings, output_csv=None):
    """Print a formatted report of frogs with few recordings and optionally save to CSV."""
    if not few_recordings:
        print("All frog species have sufficient recordings (10 or more each).")
        if output_csv:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Species", "Recording Count"])
                writer.writerow(["All species have sufficient recordings", ""])
        return
    
    # Print console report
    print("Frog species with fewer than 10 recordings:")
    print("-" * 45)
    for species, count in sorted(few_recordings.items(), key=lambda item: (item[1], item[0])):
        print(f"{species.ljust(30)}: {count} recording{'s' if count != 1 else ''}")
    print("-" * 45)
    print(f"Total under-represented species: {len(few_recordings)}")
    
    # Save to CSV if requested
    if output_csv:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Species", "Recording Count"])
            for species, count in sorted(few_recordings.items(), key=lambda item: (item[1], item[0])):
                writer.writerow([species, count])


# Example usage
if __name__ == "__main__":
    directory_path = "./Datasets/Raw-audio/frog_wav/"  # Change this to your directory
    output_csv = "frog_recording_report.csv"  # Output CSV filename
    few_recordings = analyze_frog_recordings(directory_path, min_threshold=10)
    print_frog_report(few_recordings)