import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, csv
import h5py
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
import random
from torch.cuda.amp import GradScaler, autocast

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

class SpectrogramDataset(Dataset):
    def __init__(self, h5_path, target_height=128, target_width=345, transform=None, shard_dev_num=0):
        self.h5_file_path = h5_path
        self.target_height = target_height
        self.target_width = target_width
        self.transform = transform
        
        with h5py.File(self.h5_file_path, 'r') as file:
            self.keys = list(file.keys())
            self.labels = [key.split('_')[0] for key in self.keys]
        
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        np.save("label_encoder_classes.npy", self.le.classes_) 

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


class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=15, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, 2) 
        
        # Block 2
        self.conv2 = nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(3, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)

        # Block 4
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)

        # Block 5
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        # Global Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
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


def load_data(h5_file_path, shard_dev_num = 0):

    shard_files = [f for f in os.listdir(h5_file_path) if f.startswith("shard") and f.endswith(".h5")]
    test_file = [f for f in os.listdir(h5_file_path) if f.startswith("test") and f.endswith(".h5")]
    

    shard_files.sort()
    # Pick the dev shard explicitly
    # dev_shard = shard_files[shard_dev_num]
    # train_shards = [f for i, f in enumerate(shard_files) if i != (shard_dev_num)]
    dev_shard = os.path.join(h5_file_path, shard_files[shard_dev_num])
    train_shards = [os.path.join(h5_file_path, f) for i, f in enumerate(shard_files) if i != shard_dev_num]


    train_datasets = [
    SpectrogramDataset(fname, target_height=128, target_width=345)
    for fname in train_shards
]

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = SpectrogramDataset(dev_shard)
    # test_dataset = SpectrogramDataset(test_file)
    test_dataset = SpectrogramDataset(os.path.join(h5_file_path, test_file[0]))

    
    
    return train_dataset, val_dataset, test_dataset

# Data loading with optimizations
h5_file_path = './Datasets/folds_10/'



def train_model(num_epochs=60, lr = 3e-4, ealry_stop=5,batch_size=128,dev_shard_num =0):

    # train_dataset,val_dataset, test_dataset = load_data(h5_file_path, test_size=0.2)
    train_dataset,val_dataset, test_dataset = load_data(h5_file_path, dev_shard_num)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("*"*20)
    print(f"USING DEV SHARD: {dev_shard_num}")
    print("*"*20)
    print("\n\n")
    # Model setup
    model = SpectrogramCNN()

    # Move entire model to device
    model = model.to(device)

    # Training setup
    num_epochs = 60
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=4e-4)
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4e-4,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    # Set path to save best_model
    save_dir = 'Model/saves'
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"BestModel_{__file__.split('/')[-1]}_{h5_file_path.split('/')[-1]}({time.strftime('%m-%d-%HH-%MM')}).pth")

    metrics_csv = os.path.join(save_dir, f"metrics_dev{dev_shard_num}.csv")
    write_header = not os.path.exists(metrics_csv)
    csv_f = open(metrics_csv, "a", newline="")
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","best_val_loss","dev_shard"])

    # Training loop
    best_accuracy = 0.0
    best_acc_epoch = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
            optimizer.zero_grad(set_to_none=True)
        
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

    
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
    
        with torch.no_grad():
            for inputs, labels in val_loader:  # Use val_loader here
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
    
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
    
        # if val_acc > best_accuracy:
        #     best_accuracy = val_acc
        #     torch.save(model.state_dict(), model_save_path)
        #     patience_counter = 0
        #     best_acc_epoch = epoch
        #     print(f"Best model saved {best_acc_epoch}\n")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            best_acc_epoch = epoch
            # print(f"Best model saved {best_acc_epoch}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
    
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        csv_w.writerow([epoch+1, f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}",
                f"{best_val_loss:.6f}", dev_shard_num])
        csv_f.flush()


    # Load best model and evaluate
    print(f"\nEvaluating best model(@epoch: {best_acc_epoch}) on test set...")
    model.load_state_dict(torch.load(model_save_path))
    # model.load_state_dict(torch.load("./Model/saves/BestModel_My_model_1.4.py_FD_2.0.h5(08-06-12H-11M).pth"))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

    # Calculate precision, recall, and F1 (macro/micro averaged)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # precision_micro = precision_score(all_labels, all_preds, average='micro')
    # recall_micro = recall_score(all_labels, all_preds, average='micro')
    # f1_micro = f1_score(all_labels, all_preds, average='micro')

    print(f"\nMacro-averaged Metrics:")
    print(f"Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    csv_f.close()




for dev_shard_num in range(0,10):
    train_model(dev_shard_num=dev_shard_num)








































# print(f"\nMicro-averaged Metrics:")
# print(f"Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")

# Detailed per-class report
# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds, target_names=test_dataset.le.classes_))

# cm = confusion_matrix(all_labels, all_preds)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title(f"Confusion Matrix: {__file__.split('/')[-1]}_{h5_file_path.split('/')[-1]}(acc: {test_accuracy})")
# plt.xlabel('Predicted')
# plt.ylabel('True')
# # plt.show()
# # plt.savefig(f"FIG_{__file__.split('/')[-1]}_{h5_file_path.split('/')[-1]}({test_accuracy:.4}).png")

# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.figure(figsize=(10, 5))
# plt.plot(train_accuracies, label='Training acc')
# plt.plot(val_accuracies, label='Validation acc')
# plt.title('Training and Validation accuracy Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()