import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
import noisereduce as nr

# Configuration
AUDIO_DIR = "./Datasets/Raw-audio/frog_wav/"
OUTPUT_DIR = "./Datasets/Spectograms/128-128/"
IMG_SIZE = (128, 128)  
SAMPLE_RATE = 22050    
N_FFT = 2048           
HOP_LENGTH = 256       
N_MELS = 256        

SAMPLE_RATE = 22050
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length
N_MELS = 128  # Number of Mel bands

# Desired spectrogram size
TARGET_HEIGHT = 128
TARGET_WIDTH = 128

AUDIO_EXTENSIONS = ('.wav')

os.makedirs(OUTPUT_DIR, exist_ok=True)
spectrogram_dict = {}

def print_grams(audio_path, output_dir):
    # audio_path = os.path.join(AUDIO_DIR, audio_path)
    # y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    # y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    # S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    # S_dB = librosa.power_to_db(S, ref=np.max)

    audio_path = os.path.join(AUDIO_DIR, file)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
    
    # Generate the Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    
    # Convert to dB scale
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Resize the spectrogram to 128x128
    S_resized = librosa.util.fix_length(S_dB, size=TARGET_WIDTH, axis=-1)[:TARGET_HEIGHT, :]
    
    key = os.path.splitext(file)[0]
    # return key, S_resized
    
    plt.figure(figsize=(IMG_SIZE[0]/100, IMG_SIZE[1]/100), dpi=100)  
    plt.axis('off')  
    librosa.display.specshow(S_resized, sr=sr, hop_length=HOP_LENGTH, cmap='viridis',)
    plt.tight_layout()
    
    output_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(audio_path))[0]
    )
    
    plt.savefig(output_path + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def audio_to_mel_spectrogram(file):
    audio_path = os.path.join(AUDIO_DIR, file)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
    S_dB = librosa.power_to_db(S, ref=np.max)
    key = os.path.splitext(file)[0]
    spectrogram_dict[key] = S_dB

for file in tqdm(os.listdir(AUDIO_DIR)):
    if 'weale' in file.lower() and file.lower().endswith(AUDIO_EXTENSIONS):
        # audio_to_mel_spectrogram(file)
        print_grams(file, OUTPUT_DIR)

# np.savez_compressed(OUTPUT_DIR + 'all_spectrograms.npz', **spectrogram_dict)
print(f"Spectrograms saved to {OUTPUT_DIR}")