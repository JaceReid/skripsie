import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

sample_rate = 44100
duration = 10.0
num_samples = int(sample_rate * duration)
output = np.ones(num_samples)
# background_level = np.percentile(output, 10)  # Median amplitude as noise level
# output= np.random.normal(0, background_level, len(output))

num_blocks = np.random.randint(5, 15)
block_durations = np.random.uniform(0.3, 4.0, num_blocks)
block_starts = np.random.uniform(0, duration - max(block_durations), num_blocks)

# Add fade in/out to avoid clicks
def apply_fade(sound, fade_samples=1000):
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    sound[:fade_samples] *= fade_in
    sound[-fade_samples:] *= fade_out
    return sound

for i in range(num_blocks):
    center_freq = np.random.uniform(20, 20000)
    bandwidth = np.random.uniform(50, 4000)
    lowcut = max(20, center_freq - bandwidth/2)
    highcut = min(20000, center_freq + bandwidth/2)
    
    block_samples = int(block_durations[i] * sample_rate)
    noise = np.random.normal(0, 0.3, block_samples)
    filtered_noise = bandpass_filter(noise, lowcut, highcut, sample_rate, order=4)
    
    # Apply fade
    filtered_noise = apply_fade(filtered_noise, fade_samples=500)
    
    start_idx = int(block_starts[i] * sample_rate)
    end_idx = start_idx + block_samples
    
    # Crossfade with existing audio
    crossfade = 100  # samples
    if start_idx > crossfade:
        output[start_idx-crossfade:start_idx] = output[start_idx-crossfade:start_idx] * np.linspace(1, 0, crossfade)
        output[start_idx:end_idx] += filtered_noise * np.linspace(0, 1, block_samples)
    else:
        output[start_idx:end_idx] += filtered_noise

output = np.clip(output * 0.8, -1, 1)  # Headroom
wavfile.write('smooth_freq_blocks.wav', sample_rate, output)

# Plot spectrogram
plt.specgram(output, Fs=sample_rate, NFFT=1024)
plt.colorbar()
plt.title("Spectrogram of Random Frequency Blocks")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.show()