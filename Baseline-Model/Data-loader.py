import torch
from torchvision.models import convnext_tiny
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class SpectrogramDataset(Dataset):
    def __init__(self, h5_file_path, target_height=128, target_width=345, transform=None):
        self.h5_file_path = h5_file_path
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        
        with h5py.File(h5_file_path, 'r') as file:
            self.keys = list(file.keys())
            self.labels = [key.split('_')[0] for key in self.keys]
        
        self.le = LabelEncoder()
        self.le.fit(self.labels)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as file:
            spectrogram = file[self.keys[idx]][()]
            spectrogram = self._adjust_spectrogram_size(spectrogram)
            spectrogram = torch.from_numpy(spectrogram).float().unsqueeze(0)
            
            label = self.le.transform([self.labels[idx]])[0]
            label = torch.tensor(label, dtype=torch.long)
            
            if self.transform:
                spectrogram = self.transform(spectrogram)
                
            return spectrogram, label

    def _adjust_spectrogram_size(self, spectrogram):
        current_height, current_width = spectrogram.shape
        
        if current_height > self.target_height:
            spectrogram = spectrogram[:self.target_height, :]
        elif current_height < self.target_height:
            pad_height = self.target_height - current_height
            spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)),
                               mode='constant', constant_values=spectrogram.min())
        
        if current_width > self.target_width:
            spectrogram = spectrogram[:, :self.target_width]
        elif current_width < self.target_width:
            pad_width = self.target_width - current_width
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)),
                                mode='constant', constant_values=spectrogram.min())
        return spectrogram

    def get_label_mapping(self):
        """Returns a dictionary mapping original labels to encoded values."""
        unique_labels = self.le.classes_
        encoded_values = self.le.transform(unique_labels)
        return dict(zip(unique_labels, encoded_values))


def load_data(h5_file_path, test_size=0.2):
    dataset = SpectrogramDataset(h5_file_path)
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


# Example usage to output the labels
if __name__ == "__main__":
    h5_file_path = "./Datasets/FD-0.3/spectrograms.h5"  # Replace with your H5 file path
    dataset = SpectrogramDataset(h5_file_path)
    
    # Get and print the label mapping
    label_mapping = dataset.get_label_mapping()
    print("Label Mapping (Original Label -> Encoded Value):")
    for label, encoded in label_mapping.items():
        print(f"{label}: {encoded}")
    
    # Print the first few labels in the dataset
    print("\nFirst 10 labels in the dataset:")
    for i in range(min(10, len(dataset))):
        _, label = dataset[i]
        original_label = dataset.le.inverse_transform([label.numpy()])[0]
        print(f"Sample {i}: Original Label: {original_label}, Encoded Value: {label.item()}")