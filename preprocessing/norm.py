import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_to_spectrogram_ready(input_dir, output_dir, sample_rate=22050, mono=True):
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    audio_files = []
    extensions = ('.wav', '.mp3', '.mp4', '.m4a', '.aac', '.flac', '.ogg', '.mpga')
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            # if file.lower().endswith(extensions):
                audio_files.append(Path(root) / file)
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for input_file in tqdm(audio_files, desc="Converting for spectrograms"):
        output_file = Path(output_dir) / f"{input_file.stem}.wav"
        
        cmd = [
            'ffmpeg',
            '-i', str(input_file),
            '-ac', '1' if mono else '2',  # mono recommended for spectrograms
            '-ar', str(sample_rate),  # 22050 is standard for many ML models
            '-sample_fmt', 's16',  # 16-bit PCM
            '-acodec', 'pcm_s16le',  # Standard WAV format
            '-y',
            str(output_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"\nError processing {input_file}: {e.stderr.decode('utf-8')}")
    
    print(f"\nConversion complete. Files ready for spectrogram generation in {output_dir}")

if __name__ == "__main__":
    convert_to_spectrogram_ready(
        input_dir="../Datasets/calls/",
        output_dir="./frog_sounds_wav/"
    )