import numpy as np
import matplotlib.pyplot as plt
import librosa

IMG_SIZE = (1028, 1028)  
sr  = 22050    
HOP_LENGTH = 256       

data = np.load('../Datasets/calls-specto/one/all_spectrograms.npz')

# Get list of keys
print(list(data.keys()))  # e.g., ['clip1', 'clip2', ...]


S_dB = data[list(data.keys())[0]]

plt.figure(figsize=(IMG_SIZE[0]/100, IMG_SIZE[1]/100), dpi=100)  
plt.axis('off')  
librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH)
plt.tight_layout()
plt.savefig(list(data.keys())[0] + '.png', bbox_inches='tight', pad_inches=0)
plt.close()