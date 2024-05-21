import torch
from torch.utils.data import DataLoader
from WZ_vs_WH import BinaryClassifier, NtupleDataclass

# Set a device to run the model on
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

def load_model(model_path, input_size):
    model = BinaryClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_and_print(model, data_loader, device):
    model.to(device)
    model.eval()
    true_labels = []
    predictions = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float()
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
    
    for true, pred in zip(true_labels, predictions):
        print(f"True: {true}, Predicted: {pred}")

# Example usage:
# Load the model
model_path = 'fold_4_model.pth'  # replace with the actual path
model = load_model(model_path, input_size=24)  # 24 is the number of input features

# Load data
csv_paths = [
    '/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
]

dataset = NtupleDataclass(csv_paths)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Predict and print
predict_and_print(model, data_loader, device)
