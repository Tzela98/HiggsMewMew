import pandas as pd
from sympy import Li
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
import mplhep as hep  # for plotting in HEP style
from torch import rand
from zmq import DeviceType  # for random number generation

# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
# Place the CMS label at location 0 (top-left corner)
hep.cms.label(loc=0)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.network(x)


class NtupleDataclass(Dataset):
    def __init__(self, csv_paths: list, device='cuda', transform=None):
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

        ic('training df:', self.train_df)
        ic('testing df', self.test_df)

        self.train_df.to_csv('/work/ehettwer/HiggsMewMew/ML/tmp/train_L2.csv', index=False)
        self.test_df.to_csv('/work/ehettwer/HiggsMewMew/ML/tmp/test_L2.csv', index=False)

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df.iloc[:, -1]),
            y=self.train_df.iloc[:, -1]
        )       
        
        self.pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32, device=device)

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


def train_model(train_loader, model, criterion, optimizer, device, clip_grad=None):
    model.train()
    running_loss = 0.0
    processed_samples = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).squeeze()

        # Ensure the outputs and labels are compatible for loss calculation
        if outputs.shape != labels.shape:
            raise ValueError(f"Shape mismatch: outputs.shape {outputs.shape} does not match labels.shape {labels.shape}")

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping if necessary
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Update statistics
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        processed_samples += batch_size

        progress_bar.set_postfix(loss=running_loss / processed_samples)

    avg_loss = running_loss / processed_samples
    return avg_loss


def evaluate_model(valid_loader, model, criterion, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    processed_samples = 0

    # Use tqdm for the progress bar
    progress_bar = tqdm(valid_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()
            all_outputs.extend((torch.sigmoid(outputs)).cpu().numpy())
            
            # Ensure outputs and labels are treated as batches
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            processed_samples += inputs.size(0)
            
            # Apply sigmoid to outputs to get predictions for binary classification
            preds = (torch.sigmoid(outputs) > threshold).float()

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update the progress bar
            progress_bar.set_postfix(loss=running_loss / processed_samples)
    
    # Compute the average loss and accuracy
    avg_loss = running_loss / processed_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_outputs, all_labels


def plot_training_log(log_data, epoch):
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

    plt.savefig(f'/work/ehettwer/HiggsMewMew/ML/plots/WH_vs_Background_L2/training_log_epoch{epoch + 1}.png')
    plt.close()


def plot_histogram(valid_outputs, valid_labels, epoch):
    fig, ax = plt.subplots()

    # Initialize the lists to store outputs based on the label
    valid_outputs_true = []
    valid_outputs_false = []
    
    # Iterate over both lists simultaneously
    for output, label in zip(valid_outputs, valid_labels):
        # Append the output to the appropriate list based on the label
        if label == 1:
            valid_outputs_true.append(output)
        elif label == 0:
            valid_outputs_false.append(output)
        else:
            # If the label is not 0 or 1, raise a ValueError
            raise ValueError("Labels should be either 0 or 1")
    
    hist_background, bin_edges = np.histogram(valid_outputs_false, bins=30, range=(0, 1))
    hist_signal, _ = np.histogram(valid_outputs_true, bins=30, range=(0, 1))

    hep.histplot([hist_background, hist_signal], bins=bin_edges, stack=False, label=['Background', 'Signal'], ax=ax)
    plt.legend()
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')

    plt.savefig(f'/work/ehettwer/HiggsMewMew/ML/plots/WH_vs_Background_L2/histogram_epoch{epoch + 1}.png')
    plt.close()


def plot_roc_curve(valid_outputs, valid_labels, epoch):
    # Calculate the false positive rate, true positive rate, and threshold values
    fpr, tpr, thresholds = roc_curve(valid_labels, valid_outputs)
    
    # Calculate the Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'/work/ehettwer/HiggsMewMew/ML/plots/WH_vs_Background_L2/roc_curve_epoch{epoch + 1}.png')
    plt.close()



def main():

    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print('Running on', device)

    # Example file paths (replace with actual CSV file paths)
    csv_paths = [
    '/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', 
    '/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    ]

    # Dataset and DataLoader
    dataset = NtupleDataclass(csv_paths, device=device)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Test data
    test_features, test_labels = dataset.get_test_data()
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model, loss function, optimizer
    model = BinaryClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    # Training loop
    num_epochs = 100
    log_data = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")

        valid_loss, accuracy, valid_output, valid_labels = evaluate_model(test_loader, model, criterion, device)
        print(f"Validation loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")

        epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': valid_loss,
                'val_accuracy': accuracy
            }
        
        log_data.append(epoch_data)

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            plot_histogram(valid_output, valid_labels, epoch)
            plot_roc_curve(valid_output, valid_labels, epoch)
            plot_training_log(log_data, epoch)

            torch.save(model.state_dict(), f'/work/ehettwer/HiggsMewMew/ML/model_cache/WH_vs_Background_L2_epoch{epoch+1}.pth')


if __name__ == "__main__":
    main()
