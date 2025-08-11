import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

# Configuration
AUDIO_DIR = "./Datasets/Raw-audio/frog_wav/"
OUTPUT_DIR = "./Datasets/Spectograms/weale/"
IMG_SIZE = (1028, 1028)  
SAMPLE_RATE = 22050    
N_FFT = 2048           
HOP_LENGTH = 256       
N_MELS = 256        

AUDIO_EXTENSIONS = ('.wav')

os.makedirs(OUTPUT_DIR, exist_ok=True)
spectrogram_dict = {}

def print_grams(audio_path, output_dir):
    audio_path = os.path.join(AUDIO_DIR, audio_path)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(IMG_SIZE[0]/100, IMG_SIZE[1]/100), dpi=100)  
    plt.axis('off')  
    librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH, cmap='viridis',)
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

np.savez_compressed(OUTPUT_DIR + 'all_spectrograms.npz', **spectrogram_dict)
print(f"Spectrograms saved to {OUTPUT_DIR}")