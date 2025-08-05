import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

def create_even_test_sample(wav_source_dir, target_dir, samples_per_type=10, seed=42):
    """
    Create a balanced sample of WAV files with equal number from each type,
    where type is determined by the first part of the filename (before first '_')
    
    Args:
        wav_source_dir: Directory containing original WAV files
        target_dir: Where to copy the sample files
        samples_per_type: Number of files to sample from each type
        seed: Random seed for reproducibility
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Group files by the first part of their filename (before first '_')
    type_files = defaultdict(list)
    for file_path in Path(wav_source_dir).rglob('*.wav'):
        # Extract type from filename (part before first '_')
        filename = file_path.stem  # Get filename without extension
        type_name = filename.split('_')[0] if '_' in filename else filename
        type_files[type_name].append(file_path)
    
    if not type_files:
        print("No WAV files found in the source directory!")
        return []
    
    print("\nFound the following distribution of files:")
    for type_name, files in sorted(type_files.items()):
        print(f"- {type_name}: {len(files)} files")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    sampled_files = []
    # Sample from each type
    for type_name, files in type_files.items():
        # If there aren't enough files for this type, take what's available
        n_samples = min(samples_per_type, len(files))
        if len(files) < samples_per_type:
            print(f"Warning: Type '{type_name}' only has {len(files)} files (requested {samples_per_type})")
        sampled_files.extend(random.sample(files, n_samples))
    
    # Copy sampled files to target directory
    for src_path in sampled_files:
        dst_path = os.path.join(target_dir, src_path.name)
        shutil.copy2(src_path, dst_path)
    
    print(f"\nCopied {len(sampled_files)} files ({len(type_files)} types) to {target_dir}")
    print(f"Average {len(sampled_files)//len(type_files)} files per type")
    
    # Return the sampled filenames for later reference
    return [f.name for f in sampled_files]

# Usage
sampled_files = create_even_test_sample(
    wav_source_dir='./Datasets/clipped-4s/',
    target_dir='./Datasets/testset_even_ver/',
    samples_per_type=50  # Take 10 samples from each type
)