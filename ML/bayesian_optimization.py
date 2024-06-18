from cgi import test
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dataclass import NtupleDataclass
from model import BinaryClassifier
from training import train_model, evaluate_model
from plotting import plot_training_log, plot_histogram, plot_feature_importance_autograd, ROCPlotter
from utils import create_directory, save_log_data, get_input
from skopt import gp_minimize
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class CSVDataset(Dataset):
    def __init__(self, csv_file, device='cuda', transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.data_frame.iloc[:, -1]),
            y=self.data_frame.iloc[:, -1]
        )       
        
        self.pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32, device=device)

    
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


def objective(params):
    # Unpack the parameters
    learning_rate, L2_regularisation = params
    
    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    training_data = CSVDataset('/work/ehettwer/HiggsMewMew/ML/projects/bayesian_optimisation/bayesian_train.csv')
    test_data = CSVDataset('/work/ehettwer/HiggsMewMew/ML/projects/bayesian_optimisation/bayesian_test.csv')

    training_loader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


    # Model, loss function, optimizer
    model = BinaryClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=training_data.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    # Training loop
    num_epochs = 80
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_model(training_loader, model, criterion, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")

        valid_loss, accuracy, valid_output, valid_labels = evaluate_model(test_loader, model, criterion, device)
        print(f"Validation loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")

        epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': valid_loss,
                'val_accuracy': accuracy
            }

    # Return validation loss as the objective
    return valid_loss

def main():
    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    # Define the search space for Bayesian optimization
    space = [Real(1e-5, 1e-2, name='learning_rate'),
             Real(1e-6, 1e-3, name='L2_regularisation')]
    
    # Perform Bayesian optimization
    res = gp_minimize(objective, space, n_calls=10, random_state=42)
    print(res)

    # Get the optimal hyperparameters
    best_params = res.x

    # Print the best parameters found
    print("Best parameters:", best_params)

if __name__ == "__main__":
    main()
