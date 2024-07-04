from tkinter import font
from sympy import im
import torch
import torch.optim as optim
import torch.nn as nn
import icecream as ic
from torch.utils.data import DataLoader, TensorDataset
from dataclass import NtupleDataclass
from model import BinaryClassifier
from training import train_model, evaluate_model
from plotting import plot_training_log, plot_histogram, plot_feature_importance_autograd, ROCPlotter
from utils import create_directory, save_log_data, get_input
from tayloranalysis.model_extension import extend_model 
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def reduce(x: torch.Tensor):
    return torch.mean(x).cpu().detach().numpy()


def get_feature_combis(feature_list: list, combi_list: list):
    feature_combinations = []
    for combination in combi_list:
        feature_combi = tuple(feature_list[val] for val in combination)
        feature_combinations.append(feature_combi)
    return feature_combinations


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
    default_learning_rate = 0.0015
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
    #    '/ceph/ehettwer/working_data/full_sim/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    #    '/ceph/ehettwer/working_data/full_sim/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    csv_paths = [
    '/ceph/ehettwer/working_data/small_signal_region/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/small_signal_region/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106XZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/small_signal_region/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
    '/ceph/ehettwer/working_data/small_signal_region/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
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
    numbe_of_features = len(feature_names)
    print('Number of features:', numbe_of_features)

    # Model, loss function, optimizer
    WrappedModel = extend_model(BinaryClassifier)
    model = WrappedModel(numbe_of_features, 256, 128, 1)
    model.log_model_details(create_path)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_regularisation)

    # initialize instance of roc plotter class
    roc_plotter = ROCPlotter(save_path=create_path)

    # Training loop
    log_data = []

    tcs_training = []
    early_stop_counter = 0
    smallest_loss = 1000

    combinations = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,),(11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,)]  # 1st order taylor coefficients
    combinations += [i for i in itertools.permutations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 2)]  # 2nd order Taylor coefficients

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
        
        tc_dict = model.get_tc(
            "x",
            forward_kwargs={"x": test_features.to(device)},
            tc_idx_list=combinations,
            reduce_func=reduce
        )

        tcs_training.append(list(tc_dict.values()))
        labels = get_feature_combis(feature_names, combinations)
        labels = [",".join(label) for label in labels]

        plt.figure(figsize=(12, 8))
        plt.plot(tcs_training, label=labels)
        plt.xlabel("Epoch")
        plt.ylabel("Taylor Coefficient Value")
        plt.legend(fontsize='xx-small')
        plt.savefig(create_path + 'taylor_coefficients.png', bbox_inches="tight")
        plt.close()

        log_data.append(epoch_data)

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 1 == 0:
            plot_histogram(valid_output, valid_labels, epoch, save_path=create_path)
            plot_training_log(log_data, epoch, save_path=create_path)
            plot_feature_importance_autograd(model, feature_names, test_features, device, epoch, save_path=create_path)

            roc_plotter.calculate_ROC(valid_output, valid_labels)
            roc_plotter.plot_roc_curve(epoch)
            roc_plotter.plot_auc()

            torch.save(model.state_dict(), create_path + f'{model_name}_epoch{epoch + 1}.pth')

            # get a set of target taylor coefficients after training
            tc_dict = model.get_tc(
                "x",
                forward_kwargs={"x": test_features.to(device)},
                tc_idx_list=combinations,
                reduce_func=reduce
            )

            tc_dict_list = list(tc_dict.values())
            tc_dict_first_order = tc_dict_list[:len(feature_names)]
            tc_dict_second_order = tc_dict_list
            second_order_2D = np.array(tc_dict_second_order).reshape(-1, len(feature_names))

            labels_first_order = labels[:len(feature_names)]
            labels_second_order = labels

            
            print(tc_dict_first_order)
            print(len(tc_dict_first_order))
            print(tc_dict_second_order)
            print(len(tc_dict_second_order))
            print(labels_first_order)
            print(len(labels_first_order))
            print(labels_second_order)
            print(len(labels_second_order))
            
            
            # plot tcs after training
            plt.figure(figsize=(12, 8))
            plt.title("Taylor Coefficients after Training for given Features")
            plt.plot(labels_first_order, tc_dict_first_order, "+", color="black", markersize=10)
            plt.axhline(0, color='red', linestyle=':', linewidth=1)
            plt.xlabel("Taylor Coefficient")
            plt.xticks(rotation=90)
            plt.ylabel("Taylor Coefficient Value")
            plt.savefig(create_path + 'taylor_coefficients_after_training.png', bbox_inches="tight")
            plt.close()

            
            # create 2d matrix
            matrix_size = len(feature_names)
            rows = np.arange(matrix_size)
            cols = np.arange(matrix_size)

            matrix = np.array([[(i, j) for j in cols] for i in rows])
            print(matrix)

            plt.figure(figsize=(12, 8))
            sns.heatmap(second_order_2D, annot=False, fmt=".2f", cmap="coolwarm")
            plt.title("Second Order Taylor Coefficients after Training")
            plt.savefig(create_path + 'taylor_coefficients_second_order_after_training.png', bbox_inches="tight")
            

        # Early stopping
        if valid_loss < smallest_loss:
            smallest_loss = valid_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= 20:
            print('------------------------------------')
            print("ATTENTION: THE TRAINING HAS BEEN STOPPED EARLY DUE TO NO IMPROVEMENT IN VALIDATION LOSS.")
            print('please refer to the training log for more details')
            print('------------------------------------')
            break

if __name__ == "__main__":
    main()