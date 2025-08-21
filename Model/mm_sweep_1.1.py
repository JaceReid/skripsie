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
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=0)
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
    dev_shard = os.path.join(h5_file_path, shard_files[shard_dev_num])
    train_shards = [os.path.join(h5_file_path, f) for i, f in enumerate(shard_files) if i != shard_dev_num]

    train_datasets = [
    SpectrogramDataset(fname, target_height=128, target_width=345)
    for fname in train_shards
]

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = SpectrogramDataset(dev_shard)
    test_dataset = SpectrogramDataset(os.path.join(h5_file_path, test_file[0]))

    return train_dataset, val_dataset, test_dataset

# Data loading with optimizations

from itertools import product
import csv, os, json, time
import torch
run_num = 0
# --- Add optimizers and schedulers to hyperparameter grid ---
optimizers = [optim.AdamW, optim.RMSprop] 
schedulers = [torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.StepLR]

# --- Modify the train_model function to accept optimizer and scheduler ---
def train_model(run_num, num_epochs=60, lr=3e-4, early_stop=15, batch_size=64,
                weight_decay=1e-4, dropout=0.5, dev_shard_num=0, h5_file_path = './Datasets/folds_10_3.2.0/', alpha=0.1, 
                optimizer_type=optim.AdamW, scheduler_type=torch.optim.lr_scheduler.OneCycleLR):

    # load the right dev shard
    train_dataset, val_dataset, test_dataset = load_data(h5_file_path, shard_dev_num=dev_shard_num)

    # respect batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

    # Assuming you are looping through different hyperparameter combinations
    total_runs = len(list(product(epochs_list, lrs, batch_sizes, weight_decays, dropouts, smoothing, dev_shards, optimizers, schedulers)))  


    # model with configurable dropout
    model = SpectrogramCNN()
    if hasattr(model, "dropout"):
        model.dropout.p = float(dropout)
    model = model.to(device)

    criterion_train = nn.CrossEntropyLoss(label_smoothing=alpha)
    criterion_val = nn.CrossEntropyLoss()

    # Dynamically choose optimizer
    if optimizer_type == optim.AdamW:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == optim.RMSprop:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

 
    if scheduler_type == torch.optim.lr_scheduler.CosineAnnealingLR:
        
        scheduler = scheduler_type(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6      
        )
    elif scheduler_type == torch.optim.lr_scheduler.OneCycleLR:
        
        scheduler = scheduler_type(
            optimizer,
            max_lr=lr,  
            steps_per_epoch=len(train_loader),
            epochs=num_epochs
        )

    elif scheduler_type == torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler = scheduler_type(
            
            optimizer,
            mode='min',           # Reduce LR when validation loss is not decreasing
            factor=0.5,           # Reduce learning rate by a factor of 0.5
            patience=5,           # Wait 5 epochs before reducing LR
            verbose=True,
            min_lr=1e-6           # Lower bound for the learning rate
        )

    scaler = torch.amp.GradScaler()
    model = model.to(device, memory_format=torch.channels_last)

    best_val_loss = float('inf')
    patience_counter = 0
    best_acc_epoch = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # unique save name per config
    save_dir = 'Model/saves'
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime('%m-%d-%HH-%MM')
    model_save_path = os.path.join(save_dir, f"Best_ep{num_epochs}_lr{lr}_bs{batch_size}_wd{weight_decay}_drop{dropout}_dev{dev_shard_num}_{stamp}.pth")

    print("*"*100)
    print(f"Run {run_num}/{total_runs} | DEV SHARD: {dev_shard_num} | epochs={num_epochs} lr={lr} bs={batch_size} "
        f"wd={weight_decay} drop={dropout} alpha={alpha} optimizer={optimizer.__class__.__name__} "
        f"scheduler={scheduler.__class__.__name__}")
    print("*"*100, "\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion_train(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        # validation
        model.eval()
        vloss, vcorrect, vtotal = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                loss = criterion_val(out, labels)
                vloss += loss.item()
                _, vpred = out.max(1)
                vtotal += labels.size(0)
                vcorrect += vpred.eq(labels).sum().item()

        val_loss = vloss / max(1, len(val_loader))
        val_acc = vcorrect / max(1, vtotal)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accuracies.append(train_acc); val_accuracies.append(val_acc)

        # early stop on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_acc_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{num_epochs} | {dt:.2f}s | "
              f"Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

    # test the best checkpoint
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(inputs); _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    test_acc = accuracy_score(all_labels, all_preds)

    return {
        "dev_shard": dev_shard_num,
        "epochs": num_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "alpha": alpha,
        "best_epoch": best_acc_epoch+1,
        "best_val_loss": float(best_val_loss),
        "test_accuracy": float(test_acc),
        "model_path": model_save_path,
    }

# --- hyperparameter grid & sweep ---
epochs_list = [80]
lrs = [3e-4]
batch_sizes = [88, 148, 180]
optimizers = [optim.AdamW, optim.RMSprop]  # Example optimizers
schedulers = [torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.CosineAnnealingLR]  # Example schedulers
smoothing = [0.1]
weight_decays = [1e-4]
dropouts = [0.5]
dev_shards = list(range(0, 10))  # your 10-fold dev shards

results = []
os.makedirs("Model/sweeps", exist_ok=True)
csv_path = os.path.join("Model/sweeps", f"sweep_{time.strftime('%m-%d-%HH-%MM')}.csv")

with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dev_shard","epochs","lr","batch_size","weight_decay","dropout","optimizer","scheduler",
                "best_epoch","best_val_loss","test_accuracy","model_path"])

    for (ep, lr, bs, wd, dr, sm, dev_i, optimizer, scheduler) in product(epochs_list, lrs, batch_sizes, weight_decays, dropouts, smoothing, dev_shards, optimizers, schedulers):
        run_num = run_num + 1
        run = train_model(run_num=run_num,num_epochs=ep, lr=lr, early_stop=8, batch_size=bs,
                          weight_decay=wd, dropout=dr, dev_shard_num=dev_i, alpha=sm,
                          optimizer_type=optimizer, scheduler_type=scheduler)
        results.append(run)
        w.writerow([run["dev_shard"], run["epochs"], run["lr"], run["batch_size"],
                    run["weight_decay"], run["dropout"], optimizer.__name__, scheduler.__name__,
                    run["best_epoch"], f"{run['best_val_loss']:.6f}", f"{run['test_accuracy']:.6f}",
                    run["model_path"]])
        f.flush()

# optional: JSON dump
with open(os.path.join("Model/sweeps", "sweep_results.json"), "w") as jf:
    json.dump(results, jf, indent=2)
print(f"Wrote results to {csv_path}")
