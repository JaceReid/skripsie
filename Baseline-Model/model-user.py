import torch
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
def load_model(model_path, num_classes):
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=None)  # Disable pretrained weights
    
    # Modify first layer for single-channel input (as in training)
    model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    
    # Modify classifier (as in training)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

# Convert audio file to spectrogram
def audio_to_spectrogram(audio_path, target_height=128, target_width=345):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_height)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Resize spectrogram to match training dimensions
    current_height, current_width = spectrogram.shape
    
    if current_height > target_height:
        spectrogram = spectrogram[:target_height, :]
    elif current_height < target_height:
        pad_height = target_height - current_height
        spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), 
                            mode='constant', constant_values=spectrogram.min())
    
    if current_width > target_width:
        spectrogram = spectrogram[:, :target_width]
    elif current_width < target_width:
        pad_width = target_width - current_width
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), 
                            mode='constant', constant_values=spectrogram.min())
    
    return spectrogram

# Predict class for a given audio file
def predict_audio(model, audio_path, label_encoder):
    # Convert audio to spectrogram
    spectrogram = audio_to_spectrogram(audio_path)
    
    # Convert to tensor and add batch dimension
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(spectrogram_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = label_encoder.inverse_transform([predicted.item()])[0]
    
    return predicted_class

# Example usage
if __name__ == "__main__":
    # Load the model (replace num_classes with your actual number of classes)
    num_classes = 10  # Adjust based on your dataset
    model = load_model("best_model.pth", num_classes)
    
    # Load label encoder (assuming it was saved during training)
    # If not saved, you need to recreate it with the same classes as training
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("label_encoder_classes.npy")  # Save this during training
    
    # Select an audio file for prediction
    audio_path = ""  # Replace with your audio file
    
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found!")
    else:
        predicted_class = predict_audio(model, audio_path, label_encoder)
        print(f"Predicted class: {predicted_class}")