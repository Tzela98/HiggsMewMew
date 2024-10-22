from tkinter import font
from turtle import color
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

hep.style.use(hep.style.CMS)


class ROCPlotter:
    def __init__(self, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
        self.save_path = save_path
        self.fig, self.ax = plt.subplots()
        
        self.fpr_list = []
        self.tpr_list = []
        self.auc_list = []
        self.epoch_list = []

    def calculate_ROC(self, valid_outputs, valid_labels, epoch):
        # Calculate the false positive rate, true positive rate, and threshold values
        fpr, tpr, thresholds = roc_curve(valid_labels, valid_outputs)
        self.fpr_list.append(fpr)
        self.tpr_list.append(tpr)
        
        # Calculate the Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)

        self.auc_list.append(roc_auc)
        self.epoch_list.append(int(epoch + 1))

    def plot_roc_curve(self, epoch):
        # Get the most recent ROC curve data
        fpr = self.fpr_list[-1]
        tpr = self.tpr_list[-1]
        auc = self.auc_list[-1]
        latest_epoch = self.epoch_list[-1]

        # Plot the most recent ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, lw=1.5, label=f'AUC = {auc:.4f} after {latest_epoch} epochs', color='navy')

        # Plot the diagonal line
        plt.plot([0, 1], [0, 1], color='orangered', lw=1.5, linestyle='--')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
        plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)
        plt.legend(loc="lower right", fontsize=22)

        plt.savefig(self.save_path + 'ROC_Curves.png')
        plt.close()

        self.fig, self.ax = plt.subplots()  # Reset the figure for the next epoch

    def plot_auc(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.epoch_list, self.auc_list, marker='o', lw=1.5, color='navy')
        plt.xlabel('Epoch', fontsize=22)
        plt.ylabel('AUC', fontsize=22)
        plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
        plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)
        plt.savefig(self.save_path + 'AUC_vs_Epoch.png', bbox_inches='tight')
        plt.close()


# plot_training_log function plots the training log data.
# log_data: list of dictionaries containing training log data.

def plot_training_log(log_data, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    ax[0].plot([entry['epoch'] for entry in log_data], [entry['train_loss'] for entry in log_data], label='Train Loss', color='navy')
    ax[0].plot([entry['epoch'] for entry in log_data], [entry['val_loss'] for entry in log_data], label='Val Loss', color='orangered')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss', labelpad=10)  # Adjust labelpad to provide more space
    ax[0].legend()

    ax[1].plot([entry['epoch'] for entry in log_data], [entry['val_accuracy'] for entry in log_data], label='Val Accuracy', color='orangered')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy', labelpad=10)  # Adjust labelpad to provide more space
    ax[1].legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    ax[0].set_title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    ax[1].set_title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)
    plt.savefig(save_path + f'training_log_epoch.png', bbox_inches="tight")
    plt.close()



# plot_histogram function plots the histogram of model outputs for signal and background events.
# valid_outputs: list of model outputs for validation data (float).
# valid_labels: list of labels for validation data (0 or 1).

def plot_histogram(valid_outputs, valid_labels, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
    plt.figure(figsize=(10, 8))

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

    # Normalize histograms
    hist_background_normalized = hist_background / hist_background.sum()
    hist_signal_normalized = hist_signal / hist_signal.sum()

    hep.histplot([hist_background_normalized, hist_signal_normalized], bins=bin_edges, stack=False, label=['Background', 'Signal'], color=['navy', 'orangered'])
    plt.legend()
    plt.xlabel('Prediction')
    plt.ylabel('Frequency')

    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)
    plt.savefig(save_path + f'histogram_epoch{epoch + 1}.png', bbox_inches="tight")
    plt.close()


# plot_roc_curve function plots the Receiver Operating Characteristic (ROC) curve.
# valid_outputs: list of model outputs for validation data (float).
# valid_labels: list of labels for validation data (0 or 1).

def plot_roc_curve(valid_outputs, valid_labels, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
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
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)
    plt.savefig(save_path + f'roc_curve_epoch{epoch + 1}.png', bbox_inches="tight")
    plt.close()


# plot_feature_importance_autograd function calculates and plots the feature importance using autograd.
# model: trained model.
# feature_names: list of feature names (from test data data class).
# test_data: test data tensor.

def plot_feature_importance_autograd(model, feature_names, test_data, device, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):    

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
    plt.savefig(save_path + f'feature_importance_epoch{epoch + 1}.png', bbox_inches="tight")
    plt.close()