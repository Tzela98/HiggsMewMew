import pandas as pd
from sympy import Li
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm


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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.network(x)


class NtupleDataclass(Dataset):
    def __init__(self, csv_paths: list, transform=None):
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

        self.train_df.to_csv('/work/ehettwer/HiggsMewMew/ML/tmp/train2.csv', index=False)
        self.test_df.to_csv('/work/ehettwer/HiggsMewMew/ML/tmp/test2.csv', index=False)

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df.iloc[:, -1]),
            y=self.train_df.iloc[:, -1]
        )       
        
        self.pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32).to(device)

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


class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device, clip_grad=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.clip_grad = clip_grad
        self.training_loss_history = []
        self.validation_loss_history = []

    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        processed_samples = 0

        progress_bar = tqdm(train_loader, desc='Training', leave=False)

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs).squeeze()

            # Ensure the outputs and labels are compatible for loss calculation
            if outputs.shape != labels.shape:
                raise ValueError(f"Shape mismatch: outputs.shape {outputs.shape} does not match labels.shape {labels.shape}")

            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping if necessary
            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

            # Update statistics
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            processed_samples += batch_size

            progress_bar.set_postfix(loss=running_loss / processed_samples)

        avg_loss = running_loss / processed_samples
        self.training_loss_history.append(avg_loss)
        return avg_loss

    def evaluate(self, validation_loader):
        self.model.eval()
        running_loss = 0.0
        processed_samples = 0
        outputs_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs).squeeze()

                if outputs.shape != labels.shape:
                    raise ValueError(f"Shape mismatch: outputs.shape {outputs.shape} does not match labels.shape {labels.shape}")

                loss = self.criterion(outputs, labels)
                
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                processed_samples += batch_size

                outputs_list.append(outputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        avg_loss = running_loss / processed_samples
        self.validation_loss_history.append(avg_loss)
        return avg_loss, outputs_list, labels_list


    def histogram_output(self, outputs_list, labels_list):
        outputs_flat = [item for sublist in outputs_list for item in sublist]
        labels_flat = [item for sublist in labels_list for item in sublist]

        plt.figure(figsize=(10, 5))

        plt.hist(outputs_flat, bins=50, alpha=0.5, label='Model Outputs')
        plt.hist(labels_flat, bins=50, alpha=0.5, label='True Labels')

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Model Outputs and True Labels')
        plt.legend()
        plt.savefig('/work/ehettwer/HiggsMewMew/ML/plots/test2.png')


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example file paths (replace with actual CSV file paths)
    csv_paths = [
    '/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', 
    '/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    ]

    # Dataset and DataLoader
    dataset = NtupleDataclass(csv_paths)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Test data
    test_features, test_labels = dataset.get_test_data()
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model, loss function, optimizer
    model = BinaryClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Training
    trainer = ModelTrainer(model, criterion, optimizer, device)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = trainer.train(train_loader)
        print(f"Training loss: {train_loss:.4f}")

        valid_loss, valid_output, valid_labels = trainer.evaluate(test_loader)
        print(f"Validation loss: {valid_loss:.4f}")

        trainer.histogram_output(valid_output, valid_labels)

    
if __name__ == '__main__':
    main()

