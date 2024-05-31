import pandas as pd
import os
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
from datetime import datetime


# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
# Place the CMS label at location 0 (top-left corner)
hep.cms.label(loc=0)


# Deep Neural Network to classify signal vs background
# The model is a binary classifier with 19 input features. 
# This can be varied as the number of features in the dataset might change.
# It includes Dropout and Batch Normalization layers to prevent overfitting.
class BinaryClassifier(nn.Module):
    def __init__(self, input_size=19, hidden_1=256, hidden_2=128, output_size=1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_2, output_size),
        )
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.bn2 = nn.BatchNorm1d(hidden_2)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.bn1(nn.functional.relu(self.network[0](x)))
        x = self.bn2(nn.functional.relu(self.network[3](x)))
        x = self.network[6](x)
        return x

# NtupleDataclass is a custom Dataset class that reads data from CSV files.
# It can be used to create a DataLoader for training and testing datasets.
# It also calculates class weights for imbalanced datasets.
# The class is designed to work with the BinaryClassifier model.
# The last column of the CSV file is assumed to be the label column.
# The class assumes that the data is preprocessed and ready for training.
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
        # Data needs to be preprocessed before training
        features = self.train_df.iloc[idx, :-6].values.astype('float32')
        nfeatures = len(features)

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
        feature_names = self.test_df.columns[:-6]
        features = self.test_df.iloc[:, :-6].values.astype('float32')
        labels = self.test_df.iloc[:, -1].values.astype('float32')

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return feature_names, features, labels


# train_model function trains the model for one epoch using the training data.
# returns: average loss for the epoch.
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


# evaluate_model function evaluates the model using the validation data.
# returns: average loss, accuracy, and predictions for the validation data.
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


# plot_training_log function plots the training log data.
# log_data: list of dictionaries containing training log data.
def plot_training_log(log_data, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/'):
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

    plt.savefig(save_path + f'training_log_epoch{epoch + 1}.png')
    plt.close()


# plot_histogram function plots the histogram of model outputs for signal and background events.
# valid_outputs: list of model outputs for validation data (float).
# valid_labels: list of labels for validation data (0 or 1).
def plot_histogram(valid_outputs, valid_labels, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/'):
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

    plt.savefig(save_path + f'histogram_epoch{epoch + 1}.png')
    plt.close()


# plot_roc_curve function plots the Receiver Operating Characteristic (ROC) curve.
# valid_outputs: list of model outputs for validation data (float).
# valid_labels: list of labels for validation data (0 or 1).
def plot_roc_curve(valid_outputs, valid_labels, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/'):
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
    plt.savefig(save_path + f'roc_curve_epoch{epoch + 1}.png')
    plt.close()


# plot_feature_importance_autograd function calculates and plots the feature importance using autograd.
# model: trained model.
# feature_names: list of feature names (from test data data class).
# test_data: test data tensor.
def plot_feature_importance_autograd(model, feature_names, test_data, device, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/'):    

    test_data = test_data.to(device)
    test_data.requires_grad_(True)

    # Calculate baseline prediction

    with torch.no_grad():
        baseline_pred = model(test_data).mean().item()

    feature_importance = []

    # Iterate over each feature and evaluate importance
    for i in range(test_data.shape[1]):
        perturbed_X_test = test_data.clone()
        perturbed_X_test[:, i] = 0  # Perturb the feature to 0
        
        with torch.enable_grad():
            perturbed_pred = model(perturbed_X_test).mean()
            gradient = torch.autograd.grad(perturbed_pred, test_data)[0]
            importance = torch.abs(gradient).mean().item() * np.abs(baseline_pred - perturbed_pred.item())
            feature_importance.append(importance)

    scaled_feature_importance = [importance / sum(feature_importance) for importance in feature_importance]

    # Move scaled feature importance to CPU and convert to numpy array
    scaled_feature_importance_cpu = torch.tensor(scaled_feature_importance).cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, scaled_feature_importance_cpu)
    plt.xlabel('Features', fontsize=10)
    plt.ylabel('Scaled Feature Importance', fontsize=12)
    plt.title('Scaled Feature Importance', fontsize=14)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.tight_layout()  # Adjust layout to fit labels
    plt.savefig(save_path + f'feature_importance_epoch{epoch + 1}.png')
    plt.close()


# create_directory function creates a directory if it does not exist.
def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory to'{path}' was created successfully.")
    except OSError as error:
        print(f"Error creating directory '{path}': {error}")


# log_training_details function logs the training details to a file.
def log_training_details(save_path, model_name, batch_size, num_epochs, learning_rate, L2_regularisation):
    with open(save_path + model_name + '_log', 'a') as file:
        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create the log entry
        log_entry = (f"\nTimestamp: {timestamp}\n"
                     f"Model Name: {model_name}\n"
                     f"Batch Size: {batch_size}\n"
                     f"Number of Epochs: {num_epochs}\n"
                     f"Learning Rate: {learning_rate}\n"
                     f"L2 Regularisation: {L2_regularisation}\n"
                     f"{'-'*40}\n")
        
        # Write the log entry to the file
        file.write(log_entry)


# get_input function gets user input and returns the default value if the input is invalid.
def get_input(prompt, default_value, value_type):
    user_input = input(prompt).strip()
    if not user_input:
        print(f"Using default value: {default_value}")
        return default_value
    if user_input.lower() == 'no':
        print(f"Using default value: {default_value}")
        return default_value
    try:
        return value_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default value: {default_value}")
        return default_value



def main():
    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )
    ic('Running on', device)

    # Default values
    default_model_name = 'WH_vs_Background_test'
    default_model_path = '/work/ehettwer/HiggsMewMew/ML/'
    default_batch_size = 128
    default_num_epochs = 100
    default_learning_rate = 0.002
    default_L2_regularisation = 1e-4

    use_defaults = input("Do you want to use default values? (y/n): ").strip().lower() == 'y'

    if use_defaults:
        model_name = default_model_name
        model_path = default_model_path
        num_epochs = default_num_epochs
        batch_size = default_batch_size
        learning_rate = default_learning_rate
        L2_regularisation = default_L2_regularisation
    else:
        model_name = get_input("Enter model name: ", default_model_name, str)
        model_path = default_model_path
        num_epochs = get_input("Enter number of epochs: ", default_num_epochs, int)
        batch_size = get_input("Enter batch size: ", default_batch_size, int)
        learning_rate = get_input("Enter learning rate: ", default_learning_rate, float)
        L2_regularisation = get_input("Enter L2 regularisation: ", default_L2_regularisation, float)


    # Create a directory to store the model and plots
    create_path = model_path + model_name + '/'
    create_directory(create_path)

    # Print the parameters to confirm
    print(f'Model Name: {model_name}')
    print(f'Model Path: {model_path}')
    print(f'Batch Size: {batch_size}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Learning Rate: {learning_rate}')
    print(f'L2 Regularisation: {L2_regularisation}')

    # Log the training details
    log_training_details(create_path, model_name, batch_size, num_epochs, learning_rate, L2_regularisation)

    # Example file paths (replace with actual CSV file paths)
    csv_paths = [
    '/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', 
    '/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    ]

    print('Sourcing the training data from the following CSV files:')
    ic(csv_paths)

    # Dataset and DataLoader
    dataset = NtupleDataclass(csv_paths, device=device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Test data
    feature_names, test_features, test_labels = dataset.get_test_data()
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss function, optimizer
    model = BinaryClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    # Training loop
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
            plot_histogram(valid_output, valid_labels, epoch, save_path=create_path)
            plot_roc_curve(valid_output, valid_labels, epoch, save_path=create_path)
            plot_training_log(log_data, epoch, save_path=create_path)
            plot_feature_importance_autograd(model, feature_names, test_features, device, epoch, save_path=create_path)

            torch.save(model.state_dict(), create_path + f'{model_name}_epoch{epoch + 1}.pth')


if __name__ == "__main__":
    main()
