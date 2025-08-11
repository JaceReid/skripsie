import os
import librosa
import soundfile as sf
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, AddBackgroundNoise
from tqdm import tqdm

# Define the path to your frog bioacoustics dataset (input and output directories)
input_dir = './Datasets/Raw-audio/aug-test/'  # Directory containing original frog call audio files
output_dir = './Datasets/Raw-audio/aug-test/augmented-2/'  # Directory where augmented audio will be saved
os.makedirs(output_dir, exist_ok=True)

# Define the augmentation pipeline using Audiomentations
augmentation_pipeline = Compose([
    AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.6, p=0.5),  # Add random noise
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),  # Time-stretching (changing speed without altering pitch)
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),  # Pitch shifting
    AddBackgroundNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5)


])

# Function to load an audio file, apply augmentation, and save the augmented file
def augment_audio_file(file_path, output_path, augment):
    # Load the audio file (librosa)
    y, sr = librosa.load(file_path, sr=22050)  # Resample to 22050 Hz
    
    # Apply augmentation
    augmented_audio = augment(samples=y,sample_rate=sr)
    
    # Save the augmented audio file (Soundfile)
    sf.write(output_path, augmented_audio, sr)
    print(f"Saved augmented file: {output_path}")

# Get a list of all frog call audio files in the input directory
audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]

# Apply augmentation to each audio file and save it
for audio_file in tqdm(audio_files, desc="Augmenting audio files"):
    input_audio_path = os.path.join(input_dir, audio_file)
    output_audio_path = os.path.join(output_dir, f"aug_{audio_file}")
    
    # Augment the audio file and save it
    augment_audio_file(input_audio_path, output_audio_path, augmentation_pipeline)

print("Augmentation complete!")
