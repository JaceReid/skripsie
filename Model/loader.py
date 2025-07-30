import numpy as np
import matplotlib.pyplot as plt
import librosa
import h5py

# Configuration (same as before)
IMG_SIZE = (1028, 1028)  
SAMPLE_RATE = 22050    
HOP_LENGTH = 256       

# Load the HDF5 file
with h5py.File('./Datasets/FD-0.2/spectrograms.h5', 'r') as hf:
    # Get all keys (clip names)
    keys = list(hf.keys())
    print("Available clips:", keys)  # e.g., ['clip1', 'clip2', ...]

    # Load the first spectrogram
    S_dB = hf[keys[0]][:]  # [:] loads the full array into memory

# Visualize (same as your original code)
plt.figure(figsize=(IMG_SIZE[0]/100, IMG_SIZE[1]/100), dpi=100)  
plt.axis('off')  
librosa.display.specshow(S_dB, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, cmap='viridis')
plt.tight_layout()
plt.savefig(keys[0] + '.png', bbox_inches='tight', pad_inches=0)
plt.close()