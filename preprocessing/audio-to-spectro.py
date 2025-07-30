import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

# Configuration
AUDIO_DIR = "frog_sounds/"
OUTPUT_DIR = "../Datasets/Inat/compressed/"
IMG_SIZE = (512, 512)  
SAMPLE_RATE = 22050    
N_FFT = 2048           
HOP_LENGTH = 256       
N_MELS = 256        


os.makedirs(OUTPUT_DIR, exist_ok=True)
spectrogram_dict = {}

def print_grams(audio_path, output_dir):
   
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
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
    if file.endswith(".mp3"):
        audio_to_mel_spectrogram(file)
        # print_grams(file,OUTPUT_DIR)


np.savez_compressed(OUTPUT_DIR + 'all_spectrograms.npz', **spectrogram_dict)
print(f"Spectrograms saved to {OUTPUT_DIR}")

# spectrogram_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
# labels = [i for i in range(len(spectrogram_files))]  # Or use custom labels

# df = pd.DataFrame({"file_path": spectrogram_files, "label": labels})
# df.to_csv("spectrogram_labels.csv", index=False)