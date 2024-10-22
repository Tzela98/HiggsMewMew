import numpy as np
import matplotlib.pyplot as plt
from sympy import plot, true
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from model import BinaryClassifier, BinaryClassifierCopy
from sklearn.metrics import roc_curve, auc
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

import mplhep as hep
hep.style.use(hep.style.CMS)

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
                if outputs.dim() == 0:  # Check if the tensor is 0-dimensional
                    print("Output tensor is 0-dimensional. Reshaping...")
                    all_outputs.append(torch.sigmoid(outputs).cpu().numpy())
                else:
                    all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                
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
    model_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/all_backgrounds_final_run_epoch110.pth"

    # Define the path to the validation data
    data_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/all_backgrounds_final_run_test.csv"

    # Create a ModelEvaluator object
    evaluator = ModelEvaluator(BinaryClassifierCopy, model_path)

    # Load the validation data
    dataloader, feature_columns, label_column, weigths  = evaluator.load_validation_data(data_path)

    # Perform a forward pass on the model
    loss, predictions, true_labels = evaluator.forward_pass(dataloader, evaluator.device)

    df = pd.DataFrame({'predictions': predictions, 'true_labels': true_labels, 'weights': weigths})

    # Create a DataFrame with only true labels
    df_true_labels = df[df['true_labels'] == True]
    print(df_true_labels.head(10))

    # Create a DataFrame with only false labels
    df_false_labels = df[df['true_labels'] == False]
    print(df_false_labels.head(10))


    # Histogram of model predictions
    plt.figure(figsize=(10, 8))
    n, bins = np.histogram(df_false_labels['predictions'], weights=df_false_labels['weights']*5, bins=10, range=(0, 1))
    hep.histplot(n, bins, label='Background', histtype='fill', alpha=0.3, color='navy')
    hep.histplot(n, bins, histtype='step', lw=0.7, alpha=1, color='black')
    #n, bins, patches = plt.hist(df_false_labels['predictions'], weights=df_false_labels['weights']*5 ,bins=10, range=(0,1), histtype='step', alpha=1, label='Background')
    np.savetxt('/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/background_ML.txt', n)
    np.savetxt('/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/bins_ML.txt', bins)

    n, bins = np.histogram(df_true_labels['predictions'],  weights=df_true_labels['weights']*100,  bins=10, range=(0, 1))
    hep.histplot(n, bins, label='Signal x 20', histtype='step', alpha=1, color='orangered')
    #n, bins, patches = plt.hist(df_true_labels['predictions'], weights=df_true_labels['weights']*5, bins=10, range=(0, 1) ,histtype='step', alpha=1, label='Signal')
    np.savetxt('/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/signal_ML.txt', n)

    plt.xlabel('Neural Network Output')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=24)
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig('/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/wh_neural_network_output.png', bbox_inches='tight')

    print('All done!')


if __name__ == "__main__":
    main()