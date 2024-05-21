import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score


# Set a device to run the model on
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print('Running on', device)


class NtupleDataclass(Dataset):
    def __init__(self, csv_paths, transform=None):
        self.data_frames = [pd.read_csv(path) for path in csv_paths]
        self.transform = transform

        # Concatenate all data frames into one
        self.data_frame = pd.concat(self.data_frames, ignore_index=True)
        # Shuffle all data to mix different classes
        self.data_frame.sample(frac=1).reset_index(drop=True)

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


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout layer with dropout probability of 0.2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout layer with dropout probability of 0.2
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


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

    
def evaluate_model(valid_loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            pred = (outputs > 0.5).float()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def kfold_cross_validation(csv_paths: list, k=5, num_epochs=20, batch_size=64, learning_rate=0.005):
    dataset = NtupleDataclass(csv_paths)
    kfold = KFold(n_splits=k, shuffle=True)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{k}')

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = BinaryClassifier().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_loss = train_model(train_loader, model, criterion, optimizer, device)
            val_accuracy = evaluate_model(val_loader, model, device)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        torch.save(model.state_dict(), f'fold_{fold}_model_lr=0.005_batchsize=64.pth')

        fold_accuracies.append(val_accuracy)
    
    print(f'Average Accuracy: {sum(fold_accuracies)/len(fold_accuracies):.4f}')


csv_paths = [
'/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', 
'/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
'/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
]

kfold_cross_validation(csv_paths)