import os
import random
import csv
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame, Toplevel, StringVar, Radiobutton
from pathlib import Path
import shutil
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import spectrogram
from scipy.io import wavfile

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2) 
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(256, 480, kernel_size=3, stride=1, padding=1)

        # Block 4
        self.conv4 = nn.Conv2d(480, 480, kernel_size=3, stride=1, padding=1)

        # Block 5
        self.conv5 = nn.Conv2d(480, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        # Global Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.conv3(x))
        
        # Block 4
        x = F.relu(self.conv4(x))
        
        # Block 5
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        
        # Head
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AudioClassifier:
    def __init__(self, model_path, label_encoder_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpectrogramCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Load label encoder classes
        self.classes = np.load(label_encoder_path)
        
    def _audio_to_spectrogram(self, audio_path, target_height=128, target_width=345):
        """Convert audio file to spectrogram with target dimensions"""
        try:
            # Read audio file
            # sample_rate, samples = wavfile.read(audio_path)
            
            # Convert stereo to mono if needed
            # if len(samples.shape) > 1:
            #     samples = samples.mean(axis=1)
                
            # Generate spectrogram
            # f, t, Sxx = spectrogram(samples, fs=sample_rate, nperseg=256, noverlap=128)
            # spectrogram_data = 10 * np.log10(Sxx + 1e-10)  # Convert to dB

            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast', mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            spectrogram_data= librosa.power_to_db(S, ref=np.max)
            
            # Adjust size to target dimensions
            current_height, current_width = spectrogram_data.shape
            
            if current_height > target_height:
                spectrogram_data = spectrogram_data[:target_height, :]
            elif current_height < target_height:
                pad_height = target_height - current_height
                spectrogram_data = np.pad(spectrogram_data, ((0, pad_height), (0, 0)),
                                        mode='constant', constant_values=spectrogram_data.min())
            
            if current_width > target_width:
                spectrogram_data = spectrogram_data[:, :target_width]
            elif current_width < target_width:
                pad_width = target_width - current_width
                spectrogram_data = np.pad(spectrogram_data, ((0, 0), (0, pad_width)),
                                        mode='constant', constant_values=spectrogram_data.min())
                
            return spectrogram_data
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    def predict(self, audio_path):
        """Predict class for given audio file"""
        spectrogram = self._audio_to_spectrogram(audio_path)
        if spectrogram is None:
            return "Error processing file"
            
        # Convert to tensor
        spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(spectrogram_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = self.classes[predicted.item()]
            
        return predicted_class

class RandomFilePlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Random File Player with Spectrogram and Classifier")
        self.root.geometry("800x650")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Initialize classifier
        self.classifier = AudioClassifier('./Model/saves/BestModel_My_model_1.4.py_FD_3.0.h5(08-08-17H-04M).pth', 'label_encoder_classes.npy')
        
        # Variables
        self.current_file = None
        self.flagged_files = set()
        self.total_processed = 0
        self.load_stats()
        
        # Create main frames
        self.control_frame = Frame(root)
        self.control_frame.pack(pady=10)
        
        self.spectrogram_frame = Frame(root)
        self.spectrogram_frame.pack(fill='both', expand=True)
        
        # UI Elements
        self.file_label = Label(self.control_frame, text="No file selected", wraplength=700)
        self.file_label.pack(pady=5)
        
        self.prediction_label = Label(self.control_frame, text="Prediction: ", font=('Arial', 12, 'bold'))
        self.prediction_label.pack(pady=5)
        
        self.stats_label = Label(self.control_frame, text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
        self.stats_label.pack(pady=5)
        
        self.select_button = Button(self.control_frame, text="Select Directory", command=self.select_directory)
        self.select_button.pack(pady=10)
        
        # Action buttons (initially hidden)
        self.button_frame = Frame(self.control_frame)
        self.button_frame.pack(pady=10)
        
        self.flag_button = Button(self.button_frame, text="Flag File", command=self.show_flag_options, state="disabled")
        self.flag_button.pack(side="left", padx=10)
        
        self.replay_button = Button(self.button_frame, text="Replay", command=self.replay_file, state="disabled")
        self.replay_button.pack(side="left", padx=10)
        
        self.next_button = Button(self.button_frame, text="Next File", command=self.play_random_file, state="normal")
        self.next_button.pack(side="left", padx=10)
        
        # Spectrogram display
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.spectrogram_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Track when playback ends
        self.check_playback()
    
    def load_stats(self):
        """Load flagged files and total processed count from CSV if it exists"""
        if os.path.exists("file_stats.csv"):
            with open("file_stats.csv", "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    # First row is flagged files
                    self.flagged_files = set(rows[0])
                    # Second row is total processed count
                    if len(rows) > 1:
                        self.total_processed = int(rows[1][0]) if rows[1][0].isdigit() else 0
    
    def save_stats(self):
        """Save flagged files and total processed count to CSV"""
        with open("file_stats.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Write flagged files as first row
            writer.writerow(list(self.flagged_files))
            # Write total processed count as second row
            writer.writerow([self.total_processed])
    
    def move_to_checked(self, filename):
        """Move the given file to a 'checked' subdirectory"""
        if not hasattr(self, 'directory'):
            return
            
        checked_dir = os.path.join(self.directory, "checked")
        os.makedirs(checked_dir, exist_ok=True)
        
        source_path = os.path.join(self.directory, filename)
        dest_path = os.path.join(checked_dir, filename)
        
        try:
            shutil.move(source_path, dest_path)
            print(f"Moved {filename} to checked directory")
        except Exception as e:
            print(f"Error moving file to checked directory: {e}")
    
    def select_directory(self):
        """Let user select a directory with files to play"""
        directory = filedialog.askdirectory()
        if directory:
            self.directory = directory
            self.files = [f for f in os.listdir(directory) 
                         if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.wav')]  # Only WAV for now
            if not self.files:
                messagebox.showwarning("No Files", "The selected directory contains no supported audio files.")
                return
            
            # Show action buttons
            self.flag_button.config(state="normal")
            self.replay_button.config(state="normal")
            self.next_button.config(state="normal")
            
            # Start playing
            self.play_random_file()
    
    def show_spectrogram(self, file_path):
        """Generate and display spectrogram for the audio file"""
        try:
            # Clear previous spectrogram
            self.ax.clear()
            
            # Read audio file
            sample_rate, samples = wavfile.read(file_path)
                
            # Convert stereo to mono if needed
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)
            
            # Generate spectrogram
            self.ax.specgram(samples, Fs=sample_rate, cmap='viridis')
            self.ax.set_title(f"Spectrogram: {os.path.basename(file_path)}")
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Frequency (Hz)')
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error generating spectrogram: {e}")
    
    def play_random_file(self):
        """Play a random file from the directory"""
        if not hasattr(self, 'files') or not self.files:
            return
            
        # Force-stop any currently playing file
        pygame.mixer.music.stop()
        
        # Rest of the method remains the same...
        self.current_file = random.choice(self.files)
        file_path = os.path.join(self.directory, self.current_file)
        self.file_label.config(text=f"Now playing: {self.current_file}")
        
        # Get prediction
        prediction = self.classifier.predict(file_path)
        self.prediction_label.config(text=f"Prediction: {prediction}", fg="blue")
        
        self.total_processed += 1
        self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
        self.save_stats()
        
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.show_spectrogram(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not play file: {e}")
            self.play_random_file()  # Skip to next file on error
    
    def replay_file(self):
        """Replay the current file"""
        if self.current_file:
            file_path = os.path.join(self.directory, self.current_file)
            try:
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                
                # Update spectrogram
                self.show_spectrogram(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not replay file: {e}")
    
    def show_flag_options(self):
        """Show popup with rename options for flagged file"""
        if not self.current_file:
            return
            
        popup = Toplevel(self.root)
        popup.title("Flag File Options")
        popup.geometry("300x200")
        
        Label(popup, text="Choose rename option:").pack(pady=10)
        
        rename_option = StringVar(value="none")
        
        Radiobutton(popup, text="none frog", variable=rename_option, value="none").pack(anchor='w', padx=20)
        Radiobutton(popup, text="other frog", variable=rename_option, value="other").pack(anchor='w', padx=20)
        Radiobutton(popup, text="mountain rain frog", variable=rename_option, value="Mountain").pack(anchor='w', padx=20)
        Radiobutton(popup, text="clicking stream frog", variable=rename_option, value="Clicking").pack(anchor='w', padx=20)
        
        def apply_rename():
            self.flag_and_rename_file(rename_option.get())
            popup.destroy()
        
        Button(popup, text="Apply", command=apply_rename).pack(pady=10)
    
    def flag_and_rename_file(self, prefix):
        """Flag the current file and rename it with the given prefix and a unique number"""
        if not self.current_file:
            return
        
        file_path = os.path.join(self.directory, self.current_file)
        file_ext = os.path.splitext(self.current_file)[1]
    
        # Create checked directory if it doesn't exist
        checked_dir = os.path.join(self.directory, "checked")
        os.makedirs(checked_dir, exist_ok=True)
    
        # Find the next available number in the checked directory
        i = 1
        while True:
            new_name = f"{prefix}_{i}{file_ext}"
            new_path = os.path.join(checked_dir, new_name)
            if not os.path.exists(new_path):
                break
            i += 1
    
        try:
            # Move and rename the file directly to the checked directory
            shutil.move(file_path, new_path)
            self.flagged_files.add(self.current_file)
            if self.current_file in self.files:
                self.files.remove(self.current_file)
            self.current_file = new_name  # Update current file reference
            self.save_stats()
            self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
        
            messagebox.showinfo("File Flagged", f"File renamed to '{new_name}' and moved to checked directory.")
        
            # Play next file automatically
            self.play_random_file()
        except Exception as e:
            messagebox.showerror("Error", f"Could not rename and move file: {e}")
    
    def check_playback(self):
        """Check if playback has finished to enable buttons"""
        if not pygame.mixer.music.get_busy() and hasattr(self, 'current_file') and self.current_file:
            # Enable buttons when playback is done
            self.flag_button.config(state="normal")
            self.replay_button.config(state="normal")
        else:
            # Disable only Flag and Replay during playback
            self.flag_button.config(state="disabled")
            self.replay_button.config(state="disabled")
        
        # Next button is always enabled
        self.next_button.config(state="normal")
        
        # Check again in 500ms
        self.root.after(500, self.check_playback)

if __name__ == "__main__":
    root = Tk()
    app = RandomFilePlayer(root)
    root.mainloop()