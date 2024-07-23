import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime


# Deep Neural Network to classify signal vs background
# The model is a binary classifier with 19 input features. 
# This can be varied as the number of features in the dataset might change.
# It includes Dropout and Batch Normalization layers to prevent overfitting.


class BinaryClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_1=256, hidden_2=128, output_size=1):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_2, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def log_model_details(self, path, filename="model_details.txt"):
        # Method to log the model architecture and parameters
        details = []

        details.append("Model Architecture:\n")
        
        details.append(f"Input size: {self.fc1.in_features}")
        details.append(f"Layer 1: Linear({self.fc1.in_features}, {self.fc1.out_features})")
        details.append(f"BatchNorm1d({self.bn1.num_features})")
        if hasattr(self, 'dropout1'):
            details.append(f"Dropout(p={self.dropout1.p})")
        
        details.append(f"Layer 2: Linear({self.fc2.in_features}, {self.fc2.out_features})")
        details.append(f"BatchNorm1d({self.bn2.num_features})")
        if hasattr(self, 'dropout2'):
            details.append(f"Dropout(p={self.dropout2.p})")
        
        details.append(f"Output layer: Linear({self.fc3.in_features}, {self.fc3.out_features})")
        
        details.append("\nParameters and Shapes:\n")
        details.append(f"Flatten: Output shape depends on input shape")
        details.append(f"Layer 1: {list(self.fc1.weight.shape)}")
        details.append(f"Layer 2: {list(self.fc2.weight.shape)}")
        details.append(f"Output layer: {list(self.fc3.weight.shape)}")
        
        details.append("\nTotal number of parameters:")
        total_params = sum(p.numel() for p in self.parameters())
        details.append(f"{total_params}")

        save_path = path + filename
        with open(save_path, "w") as file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f"Timestamp: {timestamp}\n")
            for line in details:
                file.write(line + "\n")


class BinaryClassifierCopy(nn.Module):
    def __init__(self, input_size=22, hidden_1=256, hidden_2=128, output_size=1):
        super(BinaryClassifierCopy, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_2, output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def log_model_details(self, path, filename="model_details.txt"):
        # Method to log the model architecture and parameters
        details = []

        details.append("Model Architecture:\n")
        
        details.append(f"Input size: {self.fc1.in_features}")
        details.append(f"Layer 1: Linear({self.fc1.in_features}, {self.fc1.out_features})")
        details.append(f"BatchNorm1d({self.bn1.num_features})")
        if hasattr(self, 'dropout1'):
            details.append(f"Dropout(p={self.dropout1.p})")
        
        details.append(f"Layer 2: Linear({self.fc2.in_features}, {self.fc2.out_features})")
        details.append(f"BatchNorm1d({self.bn2.num_features})")
        if hasattr(self, 'dropout2'):
            details.append(f"Dropout(p={self.dropout2.p})")
        
        details.append(f"Output layer: Linear({self.fc3.in_features}, {self.fc3.out_features})")
        
        details.append("\nParameters and Shapes:\n")
        details.append(f"Flatten: Output shape depends on input shape")
        details.append(f"Layer 1: {list(self.fc1.weight.shape)}")
        details.append(f"Layer 2: {list(self.fc2.weight.shape)}")
        details.append(f"Output layer: {list(self.fc3.weight.shape)}")
        
        details.append("\nTotal number of parameters:")
        total_params = sum(p.numel() for p in self.parameters())
        details.append(f"{total_params}")

        save_path = path + filename
        with open(save_path, "w") as file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f"Timestamp: {timestamp}\n")
            for line in details:
                file.write(line + "\n")

