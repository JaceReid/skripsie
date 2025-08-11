import numpy as np 
from pathlib import Path
import random
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from tqdm import tqdm

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Ensure frequencies are within valid range
    low = max(0.001, min(0.999, low))
    high = max(0.001, min(0.999, high))
    if low >= high:
        high = min(0.999, low + 0.001)  # Ensure at least 0.001 difference
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    except ValueError as e:
        print(f"Filter error with lowcut={lowcut}, highcut={highcut}: {str(e)}")
        return data  # Return original data if filtering fails

def apply_fade(sound, fade_samples=1000):
    if len(sound) == 0:
        return sound
    fade_samples = min(fade_samples, len(sound)//2)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    sound[:fade_samples] *= fade_in
    sound[-fade_samples:] *= fade_out
    return sound

def generate_random_freq_background(length_samples, sample_rate):
    if length_samples == 0:
        return np.zeros(0)
    
    output = np.zeros(length_samples)
    duration = length_samples / sample_rate
    
    num_blocks = np.random.randint(5, 15)
    block_durations = np.random.uniform(0.5, 4.0, num_blocks)
    block_starts = np.random.uniform(0, max(0.1, duration - max(block_durations)), num_blocks)

    for i in range(num_blocks):
        center_freq = np.random.uniform(20, 6000)  # More reasonable frequency range
        bandwidth = np.random.uniform(50, 800)
        lowcut = max(20, center_freq - bandwidth/2)
        highcut = min(sample_rate/2 - 1, center_freq + bandwidth/2)
        
        block_samples = int(block_durations[i] * sample_rate)
        if block_samples == 0:
            continue
            
        noise = np.random.normal(0, 0.1, block_samples)  # Reduced amplitude
        filtered_noise = bandpass_filter(noise, lowcut, highcut, sample_rate, order=4)
        
        if len(filtered_noise) == 0:
            continue
            
        # Apply fade
        filtered_noise = apply_fade(filtered_noise, fade_samples=min(500, block_samples//2))
        
        start_idx = int(block_starts[i] * sample_rate)
        end_idx = min(start_idx + block_samples, length_samples)
        
        # Crossfade with existing audio
        crossfade = min(100, block_samples//2)  # samples
        if start_idx > crossfade and (end_idx - start_idx) > 0:
            output[start_idx-crossfade:start_idx] = output[start_idx-crossfade:start_idx] * np.linspace(1, 0, crossfade)
            output[start_idx:end_idx] += filtered_noise[:end_idx-start_idx] * np.linspace(0, 1, end_idx-start_idx)

    return np.clip(output * 0.5, -1, 1)  # Reduced headroom

def insert_frog_calls(new_audio, frog_call, sample_rate, num_calls):
    call_length = len(frog_call)
    max_start = len(new_audio) - call_length
    
    if max_start <= 0:
        print(f"Call too long for target audio")
        return new_audio
    
    # Calculate minimum spacing between calls (0.5 seconds)
    min_spacing = int(0.5 * sample_rate)
    available_length = len(new_audio) - (num_calls * (call_length + min_spacing))
    
    if available_length < 0:
        num_calls = len(new_audio) // (call_length + min_spacing)
        if num_calls == 0:
            num_calls = 1
    
    # Distribute call positions evenly with random variation
    if num_calls > 1:
        segment_size = len(new_audio) // num_calls
        start_positions = [i * segment_size + random.randint(-segment_size//4, segment_size//4) for i in range(num_calls)]
        start_positions = [max(0, min(pos, len(new_audio) - call_length)) for pos in start_positions]
    else:
        start_positions = [random.randint(0, max_start)]
    
    # Insert each call
    for pos in start_positions:
        end_pos = pos + call_length
        if end_pos > len(new_audio):
            end_pos = len(new_audio)
            pos = end_pos - call_length
        new_audio[pos:end_pos] = frog_call[:end_pos-pos]
    
    return new_audio, start_positions

def augment(directory, output_dir, num_new_files, min_calls=1, max_calls=3):
    for filepath in tqdm(Path(directory).glob('*.wav')):
        try:
            # Load original file
            sample_rate, data = wavfile.read(filepath)
            output_path = Path(output_dir)
    
            output_path.mkdir(exist_ok=True)
            
            # Convert to mono if stereo and ensure proper type
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            
            data_abs = np.abs(data)
            if len(data_abs) == 0:
                print(f"Skipping empty file: {filepath.name}")
                continue
                
            sorted_indices = np.argsort(data_abs)
            max_idx = sorted_indices[-1]
            
            # Create new 4-second clips
            for i in range(num_new_files):
                # Determine number of calls for this file
                num_calls = random.randint(min_calls, max_calls)

                # Extract frog call (window)
                random_window = random.uniform(0.25, max_calls/3)
                half_window = int((random_window/2) * sample_rate)
                start_idx = max(0, max_idx - half_window)
                end_idx = min(len(data), max_idx + half_window)
                frog_call = data[start_idx:end_idx]
                
                if len(frog_call) == 0:
                    print(f"Empty frog call in {filepath.name}")
                    continue
                    
                # Create empty 4-second array
                new_audio = np.zeros(4 * sample_rate)
                
                # Insert multiple frog calls
                new_audio, call_positions = insert_frog_calls(new_audio, frog_call, sample_rate, num_calls)
                
                # Add random frequency blocks as background
                background = generate_random_freq_background(len(new_audio), sample_rate)
                
                # Create mask for background (only where there's no frog call)
                mask = np.ones(len(new_audio), dtype=bool)
                for pos in call_positions:
                    call_end = pos + len(frog_call)
                    mask[pos:call_end] = False
                
                new_audio[mask] += background[mask]
                
                # Add some random spikes (fewer spikes when more calls present)
                max_spikes = max(5, 20 // num_calls)
                num_spikes = min(max_spikes, len(new_audio)//10)
                if num_spikes > 0:
                    spike_indices = np.random.choice(len(new_audio), num_spikes, replace=False)
                    spike_amplitudes = np.random.uniform(0.1, 0.5, num_spikes) * np.random.choice([-1, 1], num_spikes)
                    new_audio[spike_indices] += spike_amplitudes
                
                # Normalize and save
                new_audio = np.clip(new_audio, -1, 1)
                new_filename = f"{filepath.stem}_aug_{i}_calls{num_calls}.wav"
                new_filepath = output_path / new_filename
                wavfile.write(new_filepath, sample_rate, (new_audio * 32767).astype(np.int16))
                
        except Exception as e:
            print(f"Error processing {filepath.name}: {str(e)}")
            continue

augment("./Datasets/Raw-audio/aug-test/", "./Datasets/Raw-audio/aug-test/augmented/",num_new_files=500, min_calls=1, max_calls=5)