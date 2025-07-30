import os
import numpy as np
import librosa
from joblib import Parallel, delayed
import multiprocessing
import h5py
from tqdm import tqdm

# Configuration
AUDIO_DIR = "./preprocessing/frog_sounds_wav/"
OUTPUT_DIR = "./Datasets/FD-0.3/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128

def process_file(file):
    try:
        if file.lower().endswith('.wav'):
            audio_path = os.path.join(AUDIO_DIR, file)
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast', mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            S_dB = librosa.power_to_db(S, ref=np.max)
            key = os.path.splitext(file)[0]
            return key, S_dB
    except Exception as e:
        print(f"\nSkipped {file}: {str(e)}")
        return None

# Get list of files first for accurate tqdm progress
files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
print(f"Processing {len(files)} audio files...")


batch_size = 500
for i in tqdm(range(0, len(files), batch_size)):
    batch = files[i:i+batch_size]
    results = Parallel(n_jobs=2)(delayed(process_file)(f) for f in batch)
    with h5py.File(os.path.join(OUTPUT_DIR, 'spectrograms.h5'), 'a') as hf:  # 'a' for append
        for result in results:
            if result is not None:
                key, S_dB = result
                hf.create_dataset(key, data=S_dB)


print(f"\nDone! Spectrograms saved to {OUTPUT_DIR}")