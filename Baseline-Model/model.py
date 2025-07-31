import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class SpectrogramDataset(Dataset):
    def __init__(self, h5_file_path, target_height=128, target_width=345, transform=None):
        """
        Args:
            h5_file_path (string): Path to the HDF5 file
            target_height: Target height for spectrograms
            target_width: Target width for spectrograms (time dimension)
            transform (callable, optional): Optional transform to be applied
        """
        self.h5_file_path = h5_file_path
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        
        with h5py.File(h5_file_path, 'r') as file:
            self.keys = list(file.keys())
            self.labels = [key.split('_')[0] for key in self.keys]
        
        # Initialize label encoder
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        np.save("label_encoder_classes.npy", self.le.classes_)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as file:
            # Load and ensure we have a 2D array
            spectrogram = file[self.keys[idx]][()]  # Shape: (height, width)
            
            # Pad or truncate to target size
            spectrogram = self._adjust_spectrogram_size(spectrogram)
            
            # Add channel dimension and convert to tensor
            spectrogram = torch.from_numpy(spectrogram).float().unsqueeze(0)  # (1, H, W)
            
            # Encode label
            label = self.le.transform([self.labels[idx]])[0]
            label = torch.tensor(label, dtype=torch.long)
            
            if self.transform:
                spectrogram = self.transform(spectrogram)
                
            return spectrogram, label

    def _adjust_spectrogram_size(self, spectrogram):
        """Ensure spectrogram matches target dimensions, truncating if needed"""
        current_height, current_width = spectrogram.shape
        
        # Handle height dimension
        if current_height > self.target_height:
            spectrogram = spectrogram[:self.target_height, :]  # Truncate
        elif current_height < self.target_height:
            pad_height = self.target_height - current_height
            spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)),
                                mode='constant', constant_values=spectrogram.min())
        
        # Handle width dimension (time)
        if current_width > self.target_width:
            spectrogram = spectrogram[:, :self.target_width]  # Truncate
        elif current_width < self.target_width:
            pad_width = self.target_width - current_width
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)),
                                mode='constant', constant_values=spectrogram.min())
        
        return spectrogram

def load_data(h5_file_path):
    return SpectrogramDataset(h5_file_path)

# Load data
h5_file_path = './Datasets/FD_0.3.h5'
dataset = load_data(h5_file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
weights = ConvNeXt_Tiny_Weights.DEFAULT
model = convnext_tiny(weights=weights)
num_classes = len(dataset.le.classes_)
model.classifier[2] = nn.Linear(
    in_features=model.classifier[2].in_features,
    out_features=num_classes
)

# Adjust first layer for single-channel input
model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_inputs, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()