import torch
import torch.optim as optim
import torch.nn as nn
import icecream as ic
from torch.utils.data import DataLoader, TensorDataset
from dataclass import NtupleDataclass
from model import BinaryClassifier
from training import train_model, evaluate_model
from plotting import plot_training_log, plot_histogram, plot_roc_curve, plot_feature_importance_autograd, ROCPlotter
from utils import create_directory, save_log_data, get_input


def main():
    # Set a device to run the model on
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")

    # Default values
    default_model_name = 'test'
    default_model_path = '/work/ehettwer/HiggsMewMew/ML/projects/'
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

    # Log the training details
    save_log_data(create_path, model_name, batch_size, num_epochs, learning_rate, L2_regularisation)

    # Example file paths (replace with actual CSV file paths)
    csv_paths = [
    '/ceph/ehettwer/working_data/signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', 
    '/ceph/ehettwer/working_data/signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/signal_region/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    ]

    print('Sourcing the training data from the following CSV files:')
    for path in csv_paths:
        print(path)

    # Dataset and DataLoader
    dataset = NtupleDataclass(csv_paths, project_name=model_name, save_path=create_path, device=device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Test data
    feature_names, test_features, test_labels = dataset.get_test_data()
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Feature names:', feature_names)

    # Model, loss function, optimizer
    model = BinaryClassifier()
    # Perform log on CPU to avoid memory issues
    model.log_model_details(create_path)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    # initialize instance of roc plotter class
    roc_plotter = ROCPlotter(save_path=create_path)

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
            plot_training_log(log_data, epoch, save_path=create_path)
            plot_feature_importance_autograd(model, feature_names, test_features, device, epoch, save_path=create_path)

            roc_plotter.calculate_ROC(valid_output, valid_labels)
            roc_plotter.plot_roc_curve(epoch)
            roc_plotter.plot_auc()

            torch.save(model.state_dict(), create_path + f'{model_name}_epoch{epoch + 1}.pth')


if __name__ == "__main__":
    main()