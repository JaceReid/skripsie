import pygame
import os
import random
import librosa
import numpy as np
import noisereduce as nr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Configuration
AUDIO_DIR = "../Datasets/clipped-4s/"  # Directory containing WAV files
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 256

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1024, 768))  # Screen size
pygame.display.set_caption('Spectrogram Viewer')

# Fonts
font = pygame.font.SysFont('Arial', 24)

# Function to generate Mel Spectrogram
def generate_spectrogram(y, sr, is_denoised=False):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

# Function to load random audio file
def load_random_audio_file():
    files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav')]
    selected_file = random.choice(files)
    audio_path = os.path.join(AUDIO_DIR, selected_file)
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
    return y, sr, selected_file

# Function to apply noise reduction
def denoise_audio(y, sr):
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    return y_denoised

# Function to plot the spectrogram
def plot_spectrogram(S_dB, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    # ax._colorbars(format='%+2.0f dB')

    # Convert the Matplotlib plot to a Pygame image
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    surface = pygame.image.fromstring(canvas.tostring_rgb(), (int(width), int(height)), "RGB")
    return surface

# Main loop
def main():
    y, sr, selected_file = load_random_audio_file()
    y_denoised = denoise_audio(y, sr)
    
    noisy_spectrogram = generate_spectrogram(y, sr)
    denoised_spectrogram = generate_spectrogram(y_denoised, sr, is_denoised=True)

    noisy_image = plot_spectrogram(noisy_spectrogram, "Noisy Spectrogram")
    denoised_image = plot_spectrogram(denoised_spectrogram, "Denoised Spectrogram")

    while True:
        screen.fill((255, 255, 255))  # Clear screen

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Display noisy and denoised spectrograms
        screen.blit(noisy_image, (20, 50))
        screen.blit(denoised_image, (500, 50))

        # Display file info
        text = font.render(f"File: {selected_file}", True, (0, 0, 0))
        screen.blit(text, (20, 20))

        # Update the screen
        pygame.display.update()

# Run the program
if __name__ == '__main__':
    main()
