from cgi import test
from re import T
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
        self.num_features = self.data_frame.shape[1] - 1

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

    def get_num_features(self):
        return self.num_features


def objective(params):
    # Unpack the parameters
    learning_rate, batch_size, L2_regularisation = params
    batch_size = int(batch_size)
    
    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    training_data = CSVDataset('/work/ehettwer/HiggsMewMew/ML/projects/bayesian_optimisation/WH_vs_WZ_corrected_DO05_train.csv')
    test_data = CSVDataset('/work/ehettwer/HiggsMewMew/ML/projects/bayesian_optimisation/WH_vs_WZ_corrected_DO05_test.csv')

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # Number of features
    num_features = training_data.get_num_features()

    # Model, loss function, optimizer
    model = BinaryClassifier(num_features, 256, 128, 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=training_data.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    # Training loop
    # Initialize early stopping variables
    smallest_loss = float('inf')
    early_stop_counter = 0
    patience = 10

    num_epochs = 80

    print('------------------------------------')
    print('ATTENTION: TRAINING HAS STARTED.')
    print(f"Training with learning rate: {learning_rate}, batch size: {batch_size}", f"L2 regularisation: {L2_regularisation}")
    print('------------------------------------')

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

        # Early stopping
        if valid_loss < smallest_loss:
            smallest_loss = valid_loss
            early_stop_counter = 0  # Reset counter if validation loss improves
        else:
            early_stop_counter += 1  # Increment counter if no improvement
        
        # Check if early stopping condition is met
        if early_stop_counter >= patience:
            print('------------------------------------')
            print("ATTENTION: THE TRAINING HAS BEEN STOPPED EARLY DUE TO NO IMPROVEMENT IN VALIDATION LOSS.")
            print('Please refer to the training log for more details.')
            print('------------------------------------')
            break  # Exit the training loop

    # Return validation loss as the objective
    return smallest_loss

def main():
    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    # Define the search space for Bayesian optimization
    space = [Real(0.0001, 0.001, name='learning_rate'),
             Integer(64, 256, name='batch_size'),
             Real(0.00001, 0.001, name='L2_regularisation')]

    x0 = [0.004, 170, 0.00001]
    
    # Perform Bayesian optimization
    res = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=True)
    print(res)

    # Get the optimal hyperparameters
    best_params = res.x

    # Print the best parameters found
    print('------------------------------------')
    print("ATTENTION, THE BEST PARAMETERS ARE: ", best_params)
    print('------------------------------------')

    # Write results to a .txt file
    with open("optimization_results.txt", "w") as file:
        file.write(f"Optimization Result: {res}\n")
        file.write('------------------------------------\n')
        file.write(f"ATTENTION, THE BEST PARAMETERS ARE: {best_params}\n")
        file.write('------------------------------------\n')

if __name__ == "__main__":
    main()
