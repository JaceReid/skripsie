import os
import random
import csv
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame, Toplevel, StringVar, Radiobutton
from pathlib import Path
import shutil
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
import h5py  # NEW

# ====== Config ======
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128

TARGET_H = 128
TARGET_W = 345

MODEL_PATH = './Model/saves/BestModel_My_model_1.4.py_FD_3.0.h5(08-08-17H-04M).pth'
LABELS_PATH = 'label_encoder_classes.npy'

# ====== Helpers ======
def resize_to_target(arr, th=TARGET_H, tw=TARGET_W):
    """Resize/pad/crop a 2D spectrogram array to (th, tw)."""
    h, w = arr.shape
    # height
    if h > th:
        arr = arr[:th, :]
    elif h < th:
        pad = th - h
        arr = np.pad(arr, ((0, pad), (0, 0)), mode="constant", constant_values=arr.min())
    # width
    h, w = arr.shape
    if w > tw:
        arr = arr[:, :tw]
    elif w < tw:
        pad = tw - w
        arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=arr.min())
    return arr

# ====== Model ======
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
        
        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x)); x = self.pool3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class AudioClassifier:
    def __init__(self, model_path, label_encoder_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpectrogramCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.classes = np.load(label_encoder_path)

    def _audio_to_spectrogram(self, audio_path, target_height=TARGET_H, target_width=TARGET_W):
        """Load .wav and convert to log-mel spectrogram resized to target dims."""
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast', mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
            S_db = librosa.power_to_db(S, ref=np.max)
            return resize_to_target(S_db, target_height, target_width)
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None

    def predict_from_audio(self, audio_path):
        spec = self._audio_to_spectrogram(audio_path)
        if spec is None:
            return "Error processing file"
        return self._predict_from_spec_array(spec)

    def predict_from_spectrogram(self, spec_2d_array):
        """Classify a precomputed 2D spectrogram (dB)."""
        spec = resize_to_target(spec_2d_array, TARGET_H, TARGET_W)
        return self._predict_from_spec_array(spec)

    def _predict_from_spec_array(self, spec):
        tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes[predicted.item()]

# ====== UI App ======
class RandomFilePlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Random File Checker (WAV or H5)")
        self.root.geometry("900x700")

        # Init pygame for WAV playback
        pygame.mixer.init()

        # Classifier
        self.classifier = AudioClassifier(MODEL_PATH, LABELS_PATH)

        # State
        self.mode = None              # "wav" or "h5"
        self.directory = None         # for wav mode
        self.files = []               # for wav mode
        self.h5_path = None           # for h5 mode
        self.h5_keys = []             # for h5 mode
        self.current_file = None
        self.flagged_files = set()
        self.total_processed = 0
        self.load_stats()

        # Layout
        self.control_frame = Frame(root); self.control_frame.pack(pady=10)
        self.spectrogram_frame = Frame(root); self.spectrogram_frame.pack(fill='both', expand=True)

        # Labels
        self.file_label = Label(self.control_frame, text="No source selected", wraplength=800)
        self.file_label.pack(pady=5)

        self.prediction_label = Label(self.control_frame, text="Prediction: ", font=('Arial', 12, 'bold'))
        self.prediction_label.pack(pady=5)

        self.stats_label = Label(
            self.control_frame,
            text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}"
        )
        self.stats_label.pack(pady=5)

        # Source buttons
        self.btn_row = Frame(self.control_frame); self.btn_row.pack(pady=6)
        self.select_wav_btn = Button(self.btn_row, text="Select Directory (.wav)", command=self.select_directory)
        self.select_wav_btn.pack(side="left", padx=6)
        self.select_h5_btn = Button(self.btn_row, text="Select .h5 (spectrograms)", command=self.select_h5_file)
        self.select_h5_btn.pack(side="left", padx=6)

        # Action buttons
        self.button_frame = Frame(self.control_frame); self.button_frame.pack(pady=10)

        self.flag_button = Button(self.button_frame, text="Flag File", command=self.show_flag_options, state="disabled")
        self.flag_button.pack(side="left", padx=10)

        self.replay_button = Button(self.button_frame, text="Replay", command=self.replay_file, state="disabled")
        self.replay_button.pack(side="left", padx=10)

        self.next_button = Button(self.button_frame, text="Next", command=self.next_item, state="disabled")
        self.next_button.pack(side="left", padx=10)

        # Spectrogram display
        self.fig, self.ax = plt.subplots(figsize=(9, 4.8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.spectrogram_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Playback watcher for wav mode
        self.check_playback()

    # ====== Stats persist ======
    def load_stats(self):
        if os.path.exists("file_stats.csv"):
            with open("file_stats.csv", "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows:
                    self.flagged_files = set(rows[0])
                    if len(rows) > 1:
                        self.total_processed = int(rows[1][0]) if rows[1] and rows[1][0].isdigit() else 0

    def save_stats(self):
        with open("file_stats.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(self.flagged_files))
            writer.writerow([self.total_processed])

    # ====== Source selection ======
    def select_directory(self):
        directory = filedialog.askdirectory()
        if not directory:
            return
        files = [f for f in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.wav')]
        if not files:
            messagebox.showwarning("No WAV files", "The selected directory contains no .wav files.")
            return
        self.mode = "wav"
        self.directory = directory
        self.files = files
        self.h5_path = None
        self.h5_keys = []
        self.file_label.config(text=f"Directory selected: {directory}")
        self.flag_button.config(state="normal")
        self.replay_button.config(state="disabled")  # enabled after a file is played
        self.next_button.config(state="normal")
        self.next_item()

    def select_h5_file(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5 *.hdf5")])
        if not path:
            return
        try:
            with h5py.File(path, 'r') as hf:
                keys = list(hf.keys())
            if not keys:
                messagebox.showwarning("Empty H5", "No datasets found in the selected HDF5.")
                return
            self.mode = "h5"
            self.h5_path = path
            self.h5_keys = keys
            self.directory = None
            self.files = []
            self.file_label.config(text=f"H5 selected: {path} ({len(keys)} items)")
            # No audio to replay/flag
            self.flag_button.config(state="disabled")
            self.replay_button.config(state="disabled")
            self.next_button.config(state="normal")
            self.next_item()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open H5: {e}")

    # ====== Rendering ======
    def show_wav_spectrogram(self, file_path):
        try:
            self.ax.clear()
            sr, samples = wavfile.read(file_path)
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)
            # Matplotlib’s specgram
            self.ax.specgram(samples, Fs=sr, cmap='viridis')
            self.ax.set_title(f"Spectrogram: {os.path.basename(file_path)}")
            self.ax.set_xlabel('Time (s)'); self.ax.set_ylabel('Frequency (Hz)')
            self.canvas.draw()
        except Exception as e:
            print(f"Error generating spectrogram: {e}")

    def show_h5_spectrogram(self, spec2d, title="H5 spectrogram"):
        try:
            self.ax.clear()
            im = self.ax.imshow(spec2d, aspect='auto', origin='lower')
            self.ax.set_title(title)
            self.ax.set_xlabel('Time frames'); self.ax.set_ylabel('Mel bins')
            self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
            self.canvas.draw()
        except Exception as e:
            print(f"Error displaying H5 spectrogram: {e}")

    # ====== Actions ======
    def next_item(self):
        if self.mode == "wav":
            self.play_random_wav()
        elif self.mode == "h5":
            self.show_random_h5()
        else:
            messagebox.showinfo("Select a source", "Choose a WAV directory or an H5 file first.")

    def play_random_wav(self):
        if not self.files:
            return
        pygame.mixer.music.stop()
        self.current_file = random.choice(self.files)
        file_path = os.path.join(self.directory, self.current_file)
        self.file_label.config(text=f"Now playing: {self.current_file}")

        # Predict
        pred = self.classifier.predict_from_audio(file_path)
        self.prediction_label.config(text=f"Prediction: {pred}", fg="blue")

        self.total_processed += 1
        self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
        self.save_stats()

        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.replay_button.config(state="normal")
            self.flag_button.config(state="normal")
            self.show_wav_spectrogram(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not play file: {e}")
            # try next file
            self.play_random_wav()

    def show_random_h5(self):
        if not self.h5_keys:
            return
        key = random.choice(self.h5_keys)
        try:
            with h5py.File(self.h5_path, 'r') as hf:
                spec = hf[key][()]  # 2D array
            # Predict
            pred = self.classifier.predict_from_spectrogram(spec)
            self.file_label.config(text=f"H5 item: {key}")
            self.prediction_label.config(text=f"Prediction: {pred}", fg="blue")

            self.total_processed += 1
            self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
            self.save_stats()

            # Show spectrogram
            self.show_h5_spectrogram(resize_to_target(spec), title=f"{key}")
            # Disable audio-specific buttons
            self.replay_button.config(state="disabled")
            self.flag_button.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Error reading H5 dataset '{key}': {e}")

    def replay_file(self):
        if self.mode != "wav" or not self.current_file:
            return
        file_path = os.path.join(self.directory, self.current_file)
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.show_wav_spectrogram(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not replay file: {e}")

    def show_flag_options(self):
        if self.mode != "wav" or not self.current_file:
            return
        popup = Toplevel(self.root)
        popup.title("Flag File Options")
        popup.geometry("300x220")

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
        if not self.current_file or self.mode != "wav":
            return
        file_path = os.path.join(self.directory, self.current_file)
        file_ext = os.path.splitext(self.current_file)[1]
        checked_dir = os.path.join(self.directory, "checked")
        os.makedirs(checked_dir, exist_ok=True)

        i = 1
        while True:
            new_name = f"{prefix}_{i}{file_ext}"
            new_path = os.path.join(checked_dir, new_name)
            if not os.path.exists(new_path):
                break
            i += 1

        try:
            shutil.move(file_path, new_path)
            self.flagged_files.add(self.current_file)
            if self.current_file in self.files:
                self.files.remove(self.current_file)
            self.current_file = new_name
            self.save_stats()
            self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
            messagebox.showinfo("File Flagged", f"File renamed to '{new_name}' and moved to checked/")
            self.next_item()
        except Exception as e:
            messagebox.showerror("Error", f"Could not rename/move file: {e}")

    # ====== Playback watcher (for wav mode) ======
    def check_playback(self):
        if self.mode == "wav":
            playing = pygame.mixer.music.get_busy()
            self.flag_button.config(state="normal" if not playing else "disabled")
            self.replay_button.config(state="normal" if not playing and self.current_file else "disabled")
            self.next_button.config(state="normal")
        else:
            # In H5 mode, no audio
            self.flag_button.config(state="disabled")
            self.replay_button.config(state="disabled")
            self.next_button.config(state="normal" if self.mode == "h5" else "disabled")
        self.root.after(500, self.check_playback)

# ====== Main ======
if __name__ == "__main__":
    root = Tk()
    app = RandomFilePlayer(root)
    root.mainloop()
