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
        columns_to_drop = ['weights', 'm_H', 'type']
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
    base_path = '/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_no_mH_run3/'
    model_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_no_mH_run3/all_backgrounds_no_mH_run3_epoch110.pth"
    data_path = "/work/ehettwer/HiggsMewMew/ML/projects/all_backgrounds_no_mH_run3/all_backgrounds_no_mH_run3_test.csv"
    output_path = base_path + 'all_backgrounds_no_mH_run3_test_with_predictions.csv'

    cut = 0.5

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

    filtered_df = combined_data[combined_data['predictions'] >= cut]

    # Create DataFrame with true and false labels
    df = pd.DataFrame({
        'predictions': filtered_df['predictions'], 
        'true_labels': filtered_df['is_wh'], 
        'weights': filtered_df['weights'],
        'm_H': filtered_df['m_H'],
        'type': filtered_df['type']
    })

    # Scale weights for type WZ as it is counted double
    df.loc[df['type'] == 'WZ', 'weights'] = df.loc[df['type'] == 'WZ', 'weights'] * 0.471
    
    df_true_labels = df[df['true_labels'] == True]
    df_false_labels = df[df['true_labels'] == False]

    # Calculate efficiencies
    eff_signal = np.sum(df_true_labels['weights']) / np.sum(combined_data[combined_data['is_wh'] == True]['weights'])
    eff_background = np.sum(df_false_labels['weights']) / np.sum(combined_data[combined_data['is_wh'] == False]['weights'])
    print('Efficiency signal:', eff_signal)
    print('sqrt(Efficiency background):', np.sqrt(eff_background))

    plt.figure(figsize=(10, 8), dpi=600)

    # Histogram for Background
    n, bins = np.histogram(df_false_labels['m_H'], weights=df_false_labels['weights']*5, bins=8, range=(115, 135))
    bin_errors = np.sqrt(np.histogram(df_false_labels['m_H'], weights=(df_false_labels['weights']*5)**2, bins=8, range=(115, 135))[0])

    hep.histplot(n, bins, label='Background', histtype='fill', alpha=0.3, color='navy')
    hep.histplot(n, bins, yerr=bin_errors, histtype='step', lw=1, alpha=1, color='black')
    np.savetxt(base_path + 'background_ML.txt', n)
    np.savetxt(base_path + 'bins_ML.txt', bins)

    # Histogram for Signal
    scale_signal = 1
    n, bins = np.histogram(df_true_labels['m_H'], weights=df_true_labels['weights']*5*scale_signal, bins=8, range=(115, 135))
    bin_errors = np.sqrt(np.histogram(df_true_labels['m_H'], weights=(df_true_labels['weights']*5*scale_signal)**2, bins=8, range=(115, 135))[0])
    
    np.savetxt(base_path + 'signal_ML.txt', n)
    hep.histplot(n, bins, yerr=bin_errors, label=f'Signal x {scale_signal}', histtype='step', alpha=1, color='orangered')

    plt.xlabel(r'$m_{\mu_1 \mu_2}\ \mathrm{(GeV)}$')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathbf{CMS}\ \mathbf{Simulation}$', loc='left', pad=10, fontsize=24)
    plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=18)
    plt.xlim(115, 135)

    plt.tick_params(axis='both', pad=15)
    #plt.ylim(0, 6.5)

    plt.legend()
    plt.savefig(base_path + f'vh_dimuon_mass_cut{int(cut*10):02}to1.png', bbox_inches='tight')

    print('All done!')

if __name__ == "__main__":
    main()

