import numpy as np 
from pathlib import Path
import os
import librosa
from scipy.io import wavfile
import random
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def augment(directory, num_new_files, number_calls=2):
    for filepath in Path(directory).glob('*.wav'):
        try:
            # Load original file
            sample_rate, data = wavfile.read(filepath)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            data_abs = np.abs(data)
            sorted_indices = np.argsort(data_abs)
            max_idx = sorted_indices[-1]
            
            
            
            # Create new 4-second clips
            for i in range(num_new_files):

                random_num_calls = random.randint(1,number_calls)

                # Extract frog call (window)
                random_window = random.uniform(0.5, 2)
                half_window = int((random_window/2) * sample_rate)
                frog_call = data[max(0, max_idx - half_window):min(len(data), max_idx + half_window)]
                # Create empty 4-second array
                new_audio = np.zeros(2 * sample_rate)
                
                # Generate random position for frog call (ensuring it fits completely)
                call_length = len(frog_call)
                max_start = len(new_audio) - call_length
                random_point = random.randint(0, max_start)
                
                # Insert frog call at random position
                new_audio[random_point:random_point+call_length] = frog_call
                
                # Add noise to the rest of the clip
                noise_indices = np.ones(len(new_audio), dtype=bool)
                noise_indices[random_point:random_point+call_length] = False
                
                # Generate noise with similar amplitude to original background
                background_level = np.percentile(data_abs, 50)  # Median amplitude as noise level
                new_audio[noise_indices] = np.random.normal(0, background_level, size=np.sum(noise_indices))
                num_spikes = 20
                spike_indices = np.random.choice(len(new_audio), num_spikes, replace=False)
                spike_amplitudes = np.random.uniform(0.5, 1, num_spikes) * np.random.randint(10,20)

                new_audio[spike_indices] = spike_amplitudes
                new_audio = np.append(new_audio, new_audio)
                
                # Save new file
                new_filename = f"{filepath.stem}_aug_{i}.wav"
                new_filepath = filepath.parent / new_filename
                wavfile.write(new_filepath, sample_rate, new_audio.astype(data.dtype))
                
        except Exception as e:
            print(f"Error processing {filepath.name}: {str(e)}")
            continue

augment("./Datasets/Raw-audio/aug-test/", num_new_files=5)