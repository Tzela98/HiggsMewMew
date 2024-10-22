import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from model import BinaryClassifier, BinaryClassifierCopy

import mplhep as hep

hep.style.use(hep.style.CMS)

class ModelEvaluator:
    def __init__(self, model_class, model_path, device='cpu'):
        self.model = self.load_model(model_class, model_path)
        self.device = device
        self.model.to(self.device)
        
    def load_model(self, model_class, model_path):
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print("Model loaded successfully.")
        return model
    
    def load_validation_data(self, file_path, batch_size=32):
        data = pd.read_csv(file_path)
        columns_to_drop = ['weights', 'type']
        weights = data['weights']
        data.drop(columns=columns_to_drop, inplace=True)
        
        label_column = 'is_wh'
        feature_columns = [col for col in data.columns if col != label_column]
        
        features = data[feature_columns].values
        labels = data[label_column].values
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print("Validation data loaded successfully.")
        return dataloader, feature_columns, label_column, weights
    
    def forward_pass(self, dataloader, device):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_outputs = []
        all_indices = []  # Track indices
        running_loss = 0.0
        processed_samples = 0

        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs).squeeze()
                all_outputs.extend((torch.sigmoid(outputs)).cpu().numpy())
                
                if outputs.ndim == 0:
                    outputs = outputs.unsqueeze(0)
                if labels.ndim == 0:
                    labels = labels.unsqueeze(0)

                labels = labels.view(-1)
                
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                processed_samples += inputs.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_indices.extend(range(batch_idx * len(inputs), (batch_idx + 1) * len(inputs)))
                
                progress_bar.set_postfix(loss=running_loss / processed_samples)
        
        avg_loss = running_loss / processed_samples
        return avg_loss, all_outputs, all_labels, all_indices

def main():
    model_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/all_backgrounds_final_run_epoch110.pth"
    data_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/all_backgrounds_final_run_test.csv"
    output_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/all_backgrounds_final_run_test_with_predictions.csv"
    base_path = '/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_final_run/'

    evaluator = ModelEvaluator(BinaryClassifier, model_path)
    dataloader, feature_columns, label_column, weights = evaluator.load_validation_data(data_path)

    loss, predictions, true_labels, indices = evaluator.forward_pass(dataloader, evaluator.device)

    # Load the original data
    original_data = pd.read_csv(data_path)

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame({
        'predictions': predictions
    }, index=indices)

    # Combine the original data with predictions
    combined_data = original_data.join(predictions_df)
    combined_data.to_csv(output_path, index=False)

    print("Predictions added to the CSV file successfully.")

    # Create DataFrame with true and false labels
    df = pd.DataFrame({
        'predictions': combined_data['predictions'], 
        'true_labels': combined_data['is_wh'], 
        'weights': combined_data['weights'],
        'type': combined_data['type']
    })

    # Scale weights for type WZ as it is counted double
    df.loc[df['type'] == 'WZ', 'weights'] = df.loc[df['type'] == 'WZ', 'weights'] * 0.471
    
    df_true_labels = df[df['true_labels'] == True]
    df_false_labels = df[df['true_labels'] == False]

    plt.figure(figsize=(10, 8), dpi=600)

    # Background histogram with errors
    n_bkg, bins_bkg = np.histogram(df_false_labels['predictions'], weights=df_false_labels['weights']*5, bins=10, range=(0, 1))
    err_bkg = np.sqrt(np.histogram(df_false_labels['predictions'], weights=(df_false_labels['weights']*5)**2, bins=10, range=(0, 1))[0])
    hep.histplot(n_bkg, bins_bkg, label='Background', histtype='fill', alpha=0.3, color='navy')
    hep.histplot(n_bkg, bins_bkg, yerr=err_bkg, histtype='step', lw=1, alpha=1, color='black')
    np.savetxt(base_path + 'background_ML.txt', n_bkg)
    np.savetxt(base_path + 'bins_ML.txt', bins_bkg)

    # Signal histogram with errors
    scale_signal = 20  # Scaling factor for signal
    n_sig, bins_sig = np.histogram(df_true_labels['predictions'], weights=df_true_labels['weights']*5*scale_signal, bins=10, range=(0, 1))
    err_sig = np.sqrt(np.histogram(df_true_labels['predictions'], weights=(df_true_labels['weights']*5*scale_signal)**2, bins=10, range=(0, 1))[0])
    hep.histplot(n_sig, bins_sig, yerr=err_sig, label=f'Signal x {scale_signal}', histtype='step', alpha=1, color='orangered')
    np.savetxt(base_path + 'signal_ML.txt', n_sig)

    plt.xlabel('Neural Network Output')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathbf{CMS}\ \mathbf{Simulation}$', loc='left', pad=10, fontsize=24)
    plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=18)
    plt.xlim(0, 1)

    plt.tick_params(axis='both', pad=15)

    plt.legend()
    plt.savefig(base_path + 'neural_network_output_vh.png', bbox_inches='tight')

    print('All done!')



if __name__ == "__main__":
    main()
