import numpy as np
import matplotlib.pyplot as plt
from sympy import plot, true
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from model import BinaryClassifier
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tqdm import tqdm
import torch.nn as nn


class ModelEvaluator:
    def __init__(self, model_class, model_path, device='cpu'):
        """
        Initialize the ModelEvaluator with the model class and model path.
        
        Args:
            model_class: The class of the model to be loaded.
            model_path: Path to the model state dictionary.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model = self.load_model(model_class, model_path)
        self.device = device
        self.model.to(self.device)
        
    def load_model(self, model_class, model_path):
        """
        Load the model from the given path.

        Args:
            model_class: The class of the model to be loaded.
            model_path: Path to the model state dictionary.

        Returns:
            model: Loaded model.
        """
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print("Model loaded successfully.")
        return model
    
    def load_validation_data(self, file_path, batch_size=32):
        """
        Load validation data from a CSV file.

        Args:
            file_path: Path to the CSV file containing validation data.
            batch_size: Batch size for DataLoader.

        Returns:
            dataloader: DataLoader for the validation data.
            feature_columns: List of feature column names.
            label_column: Name of the label column.
        """
        data = pd.read_csv(file_path)

        # Drop unwanted columns -> Do not shufle data afterwards!
        columns_to_drop = ['weights']
        weights = data['weights']
        data.drop(columns=columns_to_drop, inplace=True)
        
        label_column = 'is_wh'
        feature_columns = [col for col in data.columns if col != label_column]
        
        features = data[feature_columns].values
        labels = data[label_column].values
        
        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print("Validation data loaded successfully.")
        return dataloader, feature_columns, label_column, weights
    
    def forward_pass(self, dataloader, device):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_outputs = []
        running_loss = 0.0
        processed_samples = 0

        # Use tqdm for the progress bar
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = self.model(inputs).squeeze()
                all_outputs.extend((torch.sigmoid(outputs)).cpu().numpy())
                
                # Ensure outputs and labels are treated as batches
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)

                # Flatten labels to match the shape of outputs
                labels = labels.view(-1)
                
                # Compute loss
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                processed_samples += inputs.size(0)

                # Collect predictions and labels
                all_labels.extend(labels.cpu().numpy())

                # Update the progress bar
                progress_bar.set_postfix(loss=running_loss / processed_samples)
        
        # Compute the average loss and accuracy
        avg_loss = running_loss / processed_samples
        
        return avg_loss, all_outputs, all_labels



def main():
    # Define the path to the model state dictionary
    model_path = "/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05_run2/WH_vs_WZ_corrected_optimal_DO05_run2_epoch120.pth"

    # Define the path to the validation data
    data_path = "/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05_run2/WH_vs_WZ_corrected_optimal_DO05_run2_test.csv"

    # Create a ModelEvaluator object
    evaluator = ModelEvaluator(BinaryClassifier, model_path)

    # Load the validation data
    dataloader, feature_columns, label_column, weigths  = evaluator.load_validation_data(data_path)

    # Perform a forward pass on the model
    loss, predictions, true_labels = evaluator.forward_pass(dataloader, evaluator.device)

    df = pd.DataFrame({'predictions': predictions, 'true_labels': true_labels, 'weights': weigths})

    # Create a DataFrame with only true labels
    df_true_labels = df[df['true_labels'] == True]

    # Create a DataFrame with only false labels
    df_false_labels = df[df['true_labels'] == False]


    # Histogram of model predictions
    n, bins, patches = plt.hist(df_true_labels['predictions'], weights=df_true_labels['weights']*250, bins=10, range=(0, 1) ,histtype='step', alpha=1, label='Signal x 50')
    np.savetxt('/work/ehettwer/HiggsMewMew/ML/projects/test_two_backgrounds/signal.txt', n)
    n, bins, patches = plt.hist(df_false_labels['predictions'], weights=df_false_labels['weights']*5 ,bins=10, range=(0,1), histtype='step', alpha=1, label='Background')
    np.savetxt('/work/ehettwer/HiggsMewMew/ML/projects/test_two_backgrounds/background.txt', n)
    np.savetxt('/work/ehettwer/HiggsMewMew/ML/projects/test_two_backgrounds/bins.txt', bins)

    plt.xlabel('Predicted Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05_run2/predictions_histogramx50.png')

    print('All done!')


if __name__ == "__main__":
    main()