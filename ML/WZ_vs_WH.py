from math import log
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


# Set a device to run the model on
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print('Running on', device)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Add dropout layer with dropout probability of 0.2
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),  # Add dropout layer with dropout probability of 0.2
            nn.Linear(256, 1),
            # nn.Sigmoid()
            # This last sigmoid function depends on the loss function used
        )
    
    def forward(self, x):
        return self.network(x)


class NtupleDataclass(Dataset):
    def __init__(self, csv_paths, transform=None):
        self.data_frames = [pd.read_csv(path) for path in csv_paths]
        self.transform = transform

        # Concatenate all data frames into one
        self.data_frame = pd.concat(self.data_frames, ignore_index=True)
        # Shuffle all data to mix different classes
        self.data_frame.sample(frac=1).reset_index(drop=True)

        train_df, test_df = train_test_split(self.data_frame, test_size=0.2, random_state=42)
        
        # Store training and testing data frames
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)


    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # Select all columns except the last one as features
        features = self.train_df.iloc[idx, :-1].values.astype('float32')
        try:
            # Select the last column as the label
            label = self.train_df.iloc[idx, -1].astype('float32')
        except ValueError as e:
            raise Exception(f"Error converting label to float at index {idx}. Error: {e}")

             
        if self.transform:
            features = self.transform(features)
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return features, label

    def get_test_data(self):
        features = self.test_df.iloc[:, :-1].values.astype('float32')
        labels = self.test_df.iloc[:, -1].values.astype('float32')

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return features, labels


# Define a dataset class that ensures all dataframes have the same number of rows
class SameSizeDataclass(Dataset):
    def __init__(self, csv_paths, transform=None):
        self.transform = transform
        
        data_frames = [pd.read_csv(path).sample(frac=1).reset_index(drop=True) for path in csv_paths]
        
        num_lines = [len(df) for df in data_frames]
        min_lines = min(num_lines)
        
        self.data_frames = [df.iloc[:min_lines].copy() for df in data_frames]

        self.data_frame = pd.concat(self.data_frames, ignore_index=True)
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Select all columns except the last one as features
        features = self.data_frame.iloc[idx, :-1].values.astype('float32')
        try:
            # Select the last column as the label
            label = self.data_frame.iloc[idx, -1].astype('float32')
        except ValueError as e:
            raise Exception(f"Error converting label to float at index {idx}. Error: {e}")

        if self.transform:
            features = self.transform(features)
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return features, label


def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(train_loader.dataset)

    
def evaluate_model(valid_loader, model, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            # Ensure outputs and labels are treated as batches
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            # Apply sigmoid to outputs to get predictions as BCEWithLogitsLoss expects unscaled outputs
            pred = (torch.sigmoid(outputs) > 0.75).float()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(valid_loader.dataset), accuracy


def plot_training_log(log_data, fold_num):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot([entry['epoch'] for entry in log_data], [entry['train_loss'] for entry in log_data], label='Train Loss')
    ax[0].plot([entry['epoch'] for entry in log_data], [entry['val_loss'] for entry in log_data], label='Val Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot([entry['epoch'] for entry in log_data], [entry['val_accuracy'] for entry in log_data], label='Val Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.savefig(f'/work/ehettwer/HiggsMewMew/ML/plots/fold_{fold_num}_training_log_weighted_256_256_128batches_lr0025_thr75.png')


def kfold_cross_validation(dataset, k=5, num_epochs=20, batch_size=128, learning_rate=0.005):
    kfold = KFold(n_splits=k, shuffle=True)

    best_model = None
    best_accuracy = 0.0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{k}')

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Compute class weights
        labels = [dataset[i][1].item() for i in train_idx]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(device)

        model = BinaryClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        log_data = []

        for epoch in range(num_epochs):
            train_loss = train_model(train_loader, model, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate_model(val_loader, model, criterion, device)

            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            log_data.append(epoch_data)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model.state_dict()
        
        torch.save(model.state_dict(), f'/work/ehettwer/HiggsMewMew/ML/model_cache/weighted_256_256_128batches_lr0025_thr75.pth')
        plot_training_log(log_data, fold)

    return best_model
        

def final_performance_test(best_model_state_dict, test_features, test_labels):
    model = BinaryClassifier().to(device)
    model.load_state_dict(best_model_state_dict)
    model.eval()

    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy = evaluate_model(test_loader, model, criterion, device)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


csv_paths = [
'/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', 
'/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
'/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
]


k = 4
num_epochs = 80
batch_size = 128
learning_rate = 0.0025

# Load Data once
dataset = NtupleDataclass(csv_paths)

# Perform kfold cross validation and return the best model
best_model_state_dict = kfold_cross_validation(dataset, k, num_epochs, batch_size, learning_rate)

# Get test data
test_features, test_labels = dataset.get_test_data()

# Test the best model on the test data
final_performance_test(best_model_state_dict, test_features, test_labels)