import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


class ROCPlotter:
    def __init__(self, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
        self.save_path = save_path
        self.fig, self.ax = plt.subplots()
        
        self.fpr_list = []
        self.tpr_list = []
        self.auc_list = []

    def calculate_ROC(self, valid_outputs, valid_labels):
        # Calculate the false positive rate, true positive rate, and threshold values
        fpr, tpr, thresholds = roc_curve(valid_labels, valid_outputs)
        self.fpr_list.append(fpr)
        self.tpr_list.append(tpr)
        
        # Calculate the Area Under the Curve (AUC)
        roc_auc = auc(fpr, tpr)
        self.auc_list.append(roc_auc)

    def plot_roc_curve(self, epoch):
        # Clear the current axes to avoid overlapping plots
        self.ax.cla()

        # Plot each ROC curve with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.fpr_list) + 1))
        for i in range(len(self.fpr_list)):
            # Plot the ROC curve with color gradients
            self.ax.plot(self.fpr_list[i], self.tpr_list[i], color=colors[i], lw=2, 
                         label=f'Epoch {epoch+1} (area = {self.auc_list[i]:.4f})')

        # Plot the diagonal line
        self.ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.1)
        self.ax.set_xlabel('False Positive Rate')
        self.ax.set_ylabel('True Positive Rate')
        self.ax.set_title('Receiver Operating Characteristic')
        self.ax.legend(loc="lower right")

        plt.savefig(self.save_path + f'ROC_Curves_{epoch+1}_epochs.png', bbox_inches='tight')
        plt.close()

        self.fig, self.ax = plt.subplots()  # Reset the figure for the next epoch

    def plot_auc(self):
        plt.figure()
        plt.plot(range(1, len(self.auc_list) + 1), self.auc_list, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC vs Epoch')
        plt.savefig(self.save_path + 'AUC_vs_Epoch.png', bbox_inches='tight')
        plt.close()


# plot_training_log function plots the training log data.
# log_data: list of dictionaries containing training log data.

def plot_training_log(log_data, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
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

    plt.savefig(save_path + f'training_log_epoch{epoch + 1}.png', bbox_inches="tight")
    plt.close()


# plot_histogram function plots the histogram of model outputs for signal and background events.
# valid_outputs: list of model outputs for validation data (float).
# valid_labels: list of labels for validation data (0 or 1).

def plot_histogram(valid_outputs, valid_labels, epoch, save_path='/work/ehettwer/HiggsMewMew/ML/projects/test/'):
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