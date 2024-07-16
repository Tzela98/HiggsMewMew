import numpy as np
import matplotlib.pyplot as plt
from sympy import plot
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import BinaryClassifier, BinaryClassifierCopy
import os


def load_model(model_class, model_path):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_validation_data(file_path, batch_size=32):
    data = pd.read_csv(file_path)
    # Assuming the last column is the label
    labels = data.iloc[:, -1].values
    features = data.iloc[:, :-1].values

    tensor_features = torch.tensor(features, dtype=torch.float32)
    tensor_labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(tensor_features, tensor_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, data.columns[:-1], data.columns[-1]


def perform_selection(model, dataloader, intervals, feature_columns, label_column, device='cpu'):
    interval_datasets = {interval: [] for interval in intervals}
    interval_labels = {interval: [] for interval in intervals}
    
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()
            
            for input_sample, prob, label in zip(inputs.cpu().numpy(), probabilities, labels):
                found_interval = False
                for i, (low, high) in enumerate(intervals):
                    if low <= prob <= high:  # Adjusted to include 'high' in the interval
                        interval_datasets[(low, high)].append(input_sample)
                        interval_labels[(low, high)].append(label)
                        found_interval = True
                        break
            # Handle case when 'high' is 1 and not found in any interval
            if not found_interval and high == 1:
                interval_datasets[(low, high)].append(input_sample)
                interval_labels[(low, high)].append(label)
    
    interval_dataframes = {}
    for interval in intervals:
        if interval_datasets[interval]:
            df = pd.DataFrame(interval_datasets[interval], columns=feature_columns)
            df[label_column] = interval_labels[interval]
            interval_dataframes[interval] = df
        else:
            interval_dataframes[interval] = pd.DataFrame(columns=list(feature_columns) + [label_column])
    
    return interval_dataframes


def plot_and_save_intervals_individual(interval_dataframes, output_dir, label_column='is_wh'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for interval, df in interval_dataframes.items():
        interval_str = f"{interval[0]}_{interval[1]}"
        
        for column in df.columns:
            print('plotting', column, 'in interval', interval_str)
            if column == label_column:
                continue
            
            plt.figure(figsize=(12, 8))
            df[column].plot(kind='hist', bins=30, histtype='step', alpha=0.7, )
            plt.title(f'{column} distribution in interval {interval}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            
            plot_filename = f"{column}_interval_{interval_str}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()


def plot_and_save_intervals_one_plot(interval_dataframes, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract feature columns
    first_df = interval_dataframes[next(iter(interval_dataframes))]
    feature_columns = list(first_df.columns)

    for column in feature_columns:
        print('plotting', column)
        plt.figure(figsize=(12, 8))
        
        for interval, df in interval_dataframes.items():
            interval_str = f"{interval[0]}_{interval[1]}"
            
            if column in df.columns:
                # Normalize histogram
                df[column].plot(kind='hist', bins=30, alpha=0.7, histtype='step', density=True, label=f'{interval[0]} to {interval[1]}')
        
        plt.title(f'{column} distribution across intervals')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        
        plot_filename = f"{column}_intervals_normalized.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()


def plot_and_save_intervals_signal_vs_background(interval_dataframes, output_dir, label_column='is_wh'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for interval, df in interval_dataframes.items():
        print('plotting interval', interval)
        interval_str = f"{interval[0]}_{interval[1]}"
        
        for column in df.columns:
            if column == label_column:
                continue
            
            plt.figure(figsize=(12, 8))
            
            # Plot true signal and background distributions
            plt.hist(df[df[label_column]==1][column], bins=30, histtype='step', density=True, alpha=0.7, label='Signal')
            plt.hist(df[df[label_column]==0][column], bins=30, histtype='step', density=True, alpha=0.7, label='Background')
            
            plt.title(f'{column} distribution in interval {interval_str}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            
            plot_filename = f"{column}_interval_{interval_str}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()






model_state_dict = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05_epoch90.pth'
validation_data_path = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05_test.csv'
intervals = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

model_class = BinaryClassifier

model = load_model(model_class, model_state_dict,)
validation_loader, feature_columns, label_column = load_validation_data(validation_data_path)
interval_datasets = perform_selection(model, validation_loader, intervals, feature_columns, label_column)

for interval, dataset in interval_datasets.items():
    print(f'Interval {interval}: {len(dataset)} samples')
    print(dataset.head())

plot_directory_individual = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05/interval_plots_individual'
plot_directory_one_plot = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05/interval_plots_one_plot'
plot_directory_sb = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_right_labels_limit_nmuons_optimal_parameters_DO05/interval_plots_sb'

plot_and_save_intervals_individual(interval_datasets, plot_directory_individual)
plot_and_save_intervals_one_plot(interval_datasets, plot_directory_one_plot)
plot_and_save_intervals_signal_vs_background(interval_datasets, plot_directory_sb)
