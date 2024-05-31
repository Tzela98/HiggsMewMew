import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WZ_vs_WH_test import BinaryClassifier


device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

def evaluate_feature_importance(model, test_data_path, device):
    # Load test data
    test_data = pd.read_csv(test_data_path)
    features = test_data.columns.drop('is_wh').values
    X_test = torch.tensor(test_data.drop(columns=['is_wh']).values, dtype=torch.float32, requires_grad=True, device=device)
    
    # Calculate baseline prediction
    with torch.no_grad():
        baseline_pred = model(X_test).mean().item()

    feature_importance = []

    # Iterate over each feature and evaluate importance
    for i in range(X_test.shape[1]):
        perturbed_X_test = X_test.clone()
        perturbed_X_test[:, i] = 0  # Perturb the feature to 0
        
        with torch.enable_grad():
            perturbed_pred = model(perturbed_X_test).mean()
            gradient = torch.autograd.grad(perturbed_pred, X_test)[0]
            importance = torch.abs(gradient).mean().item() * np.abs(baseline_pred - perturbed_pred.item())
            feature_importance.append(importance)

    return features, feature_importance

# Example usage:
if __name__ == "__main__":
    model = BinaryClassifier().to(device)
    model_path = '/work/ehettwer/HiggsMewMew/ML/model_cache/WH_vs_Background_L2_epoch60.pth'
    load_model(model, model_path)

    test_data_path = '/work/ehettwer/HiggsMewMew/ML/tmp/test_L2.csv'
    features, feature_importance = evaluate_feature_importance(model, test_data_path, device)

    scaled_feature_importance = [importance / sum(feature_importance) for importance in feature_importance]
    
    print("Features:", features)
    print("Feature Importance:", scaled_feature_importance)

    plt.figure(figsize=(10, 6))
    plt.bar(features, scaled_feature_importance)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Scaled Feature Importance', fontsize=12)
    plt.title('Feature Importance', fontsize=14)
    plt.xticks(rotation=90)  # Rotate x-axis labels vertically
    plt.tight_layout()  # Adjust layout to fit labels
    plt.savefig('/work/ehettwer/HiggsMewMew/ML/plots/WH_vs_Background_L2/feature_importance_epoch60.png')

