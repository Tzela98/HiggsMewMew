import numpy as np
import matplotlib.pyplot as plt
from sympy import plot
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from model import BinaryClassifier
import pandas as pd
from model import BinaryClassifier, BinaryClassifierCopy

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
        labels = data.iloc[:, -1].values
        features = data.iloc[:, :-1].values

        tensor_features = torch.tensor(features, dtype=torch.float32)
        tensor_labels = torch.tensor(labels, dtype=torch.float32)
        dataset = TensorDataset(tensor_features, tensor_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return dataloader, data.columns[:-1], data.columns[-1]
    
    def evaluate(self, test_loader, criterion):
        """
        Evaluate the model on the test data.

        Args:
            test_loader: DataLoader for the test data.
            criterion: Loss function.

        Returns:
            valid_loss: Average loss on the test data.
            accuracy: Accuracy on the test data.
            valid_output: Model predictions on the test data.
            valid_labels: Actual labels of the test data.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        valid_output = []
        valid_labels = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels.unsqueeze(1))  # Adjusting for the shape [batch_size, 1]
                total_loss += loss.item() * features.size(0)
                predicted = torch.sigmoid(outputs).round()
                correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
                total_samples += labels.size(0)

                valid_output.extend(predicted.cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())

        valid_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        return valid_loss, accuracy, valid_output, valid_labels


# Example usage (assuming you have defined the necessary classes and functions):
model = BinaryClassifier()
evaluator = ModelEvaluator(model, '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05/WH_vs_WZ_corrected_optimal_DO05_epoch100.pth', device='cpu')
test_loader, feature_columns, label_column = evaluator.load_validation_data('/work/ehettwer/HiggsMewMew/data/bug_fix/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')

valid_loss, accuracy, valid_output, valid_labels = evaluator.evaluate(test_loader, criterion)
print(f"Validation loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")