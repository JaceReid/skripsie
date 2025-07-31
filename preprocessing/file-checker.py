import os
import random
import csv
import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame, Toplevel, StringVar, Radiobutton

class RandomFilePlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Random File Player with Spectrogram")
        self.root.geometry("800x600")
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
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
        self.file_label.pack(pady=10)
        
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
        
        self.next_button = Button(self.button_frame, text="Next File", command=self.play_random_file, state="disabled")
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
    
    def select_directory(self):
        """Let user select a directory with files to play"""
        directory = filedialog.askdirectory()
        if directory:
            self.directory = directory
            self.files = [f for f in os.listdir(directory) 
                         if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.wav', '.mp3', '.ogg'))]
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
            if file_path.lower().endswith('.wav'):
                sample_rate, samples = wavfile.read(file_path)
            else:
                # For non-WAV files, we'll use pygame to load and convert
                sound = pygame.mixer.Sound(file_path)
                samples = pygame.sndarray.array(sound)
                sample_rate = 44100  # Default assumption
                
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
            
        # Stop any currently playing file
        pygame.mixer.music.stop()
        
        # Select a random file
        self.current_file = random.choice(self.files)
        file_path = os.path.join(self.directory, self.current_file)
        
        # Update UI
        self.file_label.config(text=f"Now playing: {self.current_file}")
        self.total_processed += 1
        self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
        self.save_stats()
        
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Show spectrogram
            self.show_spectrogram(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not play file: {e}")
            # Try another file
            self.play_random_file()
    
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
        
        rename_option = StringVar(value="none_frog")
        
        Radiobutton(popup, text="none_frog", variable=rename_option, value="none_frog").pack(anchor='w', padx=20)
        Radiobutton(popup, text="other_frog", variable=rename_option, value="other_frog").pack(anchor='w', padx=20)
        
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
        
        # Find the next available number
        i = 1
        while True:
            new_name = f"{prefix}_{i}{file_ext}"
            new_path = os.path.join(self.directory, new_name)
            if not os.path.exists(new_path):
                break
            i += 1
        
        try:
            os.rename(file_path, new_path)
            self.flagged_files.add(self.current_file)
            self.files.remove(self.current_file)
            self.files.append(new_name)  # Add the new name to the files list
            self.current_file = new_name  # Update current file reference
            self.save_stats()
            self.stats_label.config(text=f"Flagged: {len(self.flagged_files)} | Processed: {self.total_processed}")
            messagebox.showinfo("File Flagged", f"File renamed to '{new_name}' and flagged.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not rename file: {e}")
    
    def check_playback(self):
        """Check if playback has finished to enable buttons"""
        if not pygame.mixer.music.get_busy() and hasattr(self, 'current_file') and self.current_file:
            # Enable buttons when playback is done
            self.flag_button.config(state="normal")
            self.replay_button.config(state="normal")
            self.next_button.config(state="normal")
        else:
            # Disable buttons during playback
            self.flag_button.config(state="disabled")
            self.replay_button.config(state="disabled")
            self.next_button.config(state="disabled")
        
        # Check again in 500ms
        self.root.after(500, self.check_playback)

if __name__ == "__main__":
    root = Tk()
    app = RandomFilePlayer(root)
    root.mainloop()