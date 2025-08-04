import os
import random
import shutil
from pathlib import Path

def create_fixed_test_sample(wav_source_dir, target_dir, sample_ratio=0.1, seed=42):
    """
    Create a representative sample of WAV files for manual verification
    
    Args:
        wav_source_dir: Directory containing original WAV files
        target_dir: Where to copy the sample files
        sample_ratio: Percentage of files to sample (0.1 = 10%)
        seed: Random seed for reproducibility
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all WAV files
    all_files = list(Path(wav_source_dir).rglob('*.wav'))
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample files
    sample_size = int(len(all_files) * sample_ratio)
    sampled_files = random.sample(all_files, sample_size)
    
    # Copy sampled files to target directory
    for src_path in sampled_files:
        dst_path = os.path.join(target_dir, src_path.name)
        shutil.copy2(src_path, dst_path)
    
    print(f"Copied {len(sampled_files)} files to {target_dir}")
    
    # Return the sampled filenames for later reference
    return [f.name for f in sampled_files]

# Usage
sampled_files = create_fixed_test_sample(
    wav_source_dir='./Datasets/clipped-4s/',
    target_dir='./Datasets/testset_ver/',
    sample_ratio=0.1
)