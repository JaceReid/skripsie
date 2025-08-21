from pydub import AudioSegment
import os

def slice_wav_file(input_file, output_folder, segment_length=4000):
    """
    Slice a WAV file into segments of specified length (in milliseconds)
    
    Args:
        input_file (str): Path to the input WAV file
        output_folder (str): Folder to save the output segments
        segment_length (int): Length of each segment in milliseconds (default: 4000 = 4 seconds)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    
    # Calculate total duration and number of segments
    duration = len(audio)
    num_segments = duration // segment_length
    
    # Extract the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Slice the audio into segments
    for i in range(num_segments + 1):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        
        # For the last segment, take whatever is left
        if end_time > duration:
            end_time = duration
        
        # Extract segment
        segment = audio[start_time:end_time]
        
        # Skip segments that are too short (less than 1 second)
        if len(segment) < 2000:
            continue
        
        # Save segment
        output_path = os.path.join(output_folder, f"{base_name}_segment_{i+1}.wav")
        segment.export(output_path, format="wav")
        print(f"Saved segment {i+1} to {output_path}")

# Example usage
if __name__ == "__main__":
    input_dir = "./Datasets/Raw-audio/calls-clipped/"  # Directory containing WAV files
    output_folder = "./Datasets/Raw-audio/calls-clipped-3/"  # Output for sliced files
    
    # Process all WAV files in input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            input_file = os.path.join(input_dir, filename)
            slice_wav_file(input_file, output_folder, segment_length=3050)