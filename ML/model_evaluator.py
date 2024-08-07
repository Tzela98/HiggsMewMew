from math import e
import numpy as np
import matplotlib.pyplot as plt
from sympy import plot
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from model import BinaryClassifier
from sklearn.metrics import roc_curve, auc
import pandas as pd


class ModelEvaluator:
    def __init__(self, model_class, model_path, device='cpu'):
        """
        Initialize the ModelEvaluator with the model class and model path.
        
        Args:
            model_class: The class of the model to be loaded.
            model_path: Path to the model state dictionary.
            device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model = self.load_model(model_class, model_path)
        self.device = device
        self.model.to(self.device)
        
    def load_model(self, model_class, model_path):
        """
        Load the model from the given path.

        Args:
            model_class: The class of the model to be loaded.
            model_path: Path to the model state dictionary.

        Returns:
            model: Loaded model.
        """
        model = model_class()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print("Model loaded successfully.")
        return model
    
    def load_validation_data(self, file_path, batch_size=32):
        """
        Load validation data from a CSV file.

        Args:
            file_path: Path to the CSV file containing validation data.
            batch_size: Batch size for DataLoader.

        Returns:
            dataloader: DataLoader for the validation data.
            feature_columns: List of feature column names.
            label_column: Name of the label column.
        """
        data = pd.read_csv(file_path)

        columns_to_drop = ['weights']
        weights = data['weights']
        data.drop(columns=columns_to_drop, inplace=True)
        
        label_column = 'is_wh'
        feature_columns = [col for col in data.columns if col != label_column]
        
        features = data[feature_columns].values
        labels = data[label_column].values
        
        # Convert to PyTorch tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print("Validation data loaded successfully.")
        return dataloader, feature_columns, label_column, weights
    
    def perform_selection(self, dataloader, intervals, feature_columns, label_column):
        """
        Perform selection of data based on model predictions and specified intervals.

        Args:
            dataloader: DataLoader for the validation data.
            intervals: List of intervals for selection.
            feature_columns: List of feature column names.
            label_column: Name of the label column.

        Returns:
            interval_dataframes: Dictionary of DataFrames for each interval.
        """
        interval_datasets = {interval: [] for interval in intervals}
        interval_labels = {interval: [] for interval in intervals}
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                labels = labels.cpu().numpy()
                
                for input_sample, prob, label in zip(inputs.cpu().numpy(), probabilities, labels):
                    found_interval = False
                    for i, (low, high) in enumerate(intervals):
                        if low <= prob <= high:
                            interval_datasets[(low, high)].append(input_sample)
                            interval_labels[(low, high)].append(label)
                            found_interval = True
                            break
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
        
        print("Data selection completed.")
        return interval_dataframes

    def plot_and_save_intervals_individual(self, interval_dataframes, output_dir, label_column='is_wh'):
        """
        Plot and save individual feature distributions for each interval.

        Args:
            interval_dataframes: Dictionary of DataFrames for each interval.
            output_dir: Directory to save the plots.
            label_column: Name of the label column.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for interval, df in interval_dataframes.items():
            interval_str = f"{interval[0]}_{interval[1]}"
            
            for column in df.columns:
                if column == label_column:
                    continue
                
                print(f'Plotting histogram for {column} in interval {interval_str}')

                plt.figure(figsize=(12, 8))
                df[column].plot(kind='hist', bins=30, histtype='step', alpha=0.7)
                plt.title(f'{column} distribution in interval {interval}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                
                plot_filename = f"{column}_interval_{interval_str}.png"
                plot_filepath = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_filepath)
                plt.close()

    def plot_and_save_intervals_one_plot(self, interval_dataframes, output_dir):
        """
        Plot and save feature distributions for all intervals in one plot.

        Args:
            interval_dataframes: Dictionary of DataFrames for each interval.
            output_dir: Directory to save the plots.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        first_df = interval_dataframes[next(iter(interval_dataframes))]
        feature_columns = list(first_df.columns)

        for column in feature_columns:
            plt.figure(figsize=(12, 8))
            print(f'Plotting histogram for {column} across intervals')
            
            for interval, df in interval_dataframes.items():
                interval_str = f"{interval[0]}_{interval[1]}"
                
                if column in df.columns:
                    df[column].plot(kind='hist', bins=30, alpha=0.7, histtype='step', density=True, label=f'{interval[0]} to {interval[1]}')
            
            plt.title(f'{column} distribution across intervals')
            plt.xlabel(column)
            plt.ylabel('Density')
            plt.legend()
            
            plot_filename = f"{column}_intervals_normalized.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath)
            plt.close()

    def plot_and_save_intervals_signal_vs_background(self, interval_dataframes, output_dir, label_column='is_wh'):
        """
        Plot and save feature distributions for signal vs. background in each interval.

        Args:
            interval_dataframes: Dictionary of DataFrames for each interval.
            output_dir: Directory to save the plots.
            label_column: Name of the label column.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for interval, df in interval_dataframes.items():
            interval_str = f"{interval[0]}_{interval[1]}"
            
            for column in df.columns:
                if column == label_column:
                    continue
                
                print(f'Plotting histogram for {column} in interval {interval_str}')
                plt.figure(figsize=(12, 8))
                
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
    
    def plot_roc_curve(self, dataloader, output_dir):
        """
        Calculate and plot the ROC curve.

        Args:
            dataloader: DataLoader for the validation data.
            output_dir: Directory where the ROC curve will be saved.
        """
        all_labels = []
        all_probs = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities)
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Add horizontal and vertical red lines at the specified coordinates
        TPR = 0.8031
        FPR = 0.3026
        plt.axhline(y=TPR, color='red', linestyle='--', lw=0.5)
        plt.axvline(x=FPR, color='red', linestyle='--', lw=0.5)
        
        # Mark the coordinates on the axes
        plt.text(0, TPR,f'{TPR}', color='red', va='bottom')
        plt.text(FPR, 0, f'{FPR}', color='red', ha='left')

        print(f"Saving ROC curve to {output_dir}")
        plt.savefig(output_dir + '/roc_curve.png', bbox_inches="tight")
        plt.close()


class simple_mass_cut:
    def __init__(self, file_path):
        """
        Initializes the class with the file path of the CSV file.
        
        :param file_path: str, path to the CSV file
        """
        self.file_path = file_path
        self.df = None

    def load_csv(self):
        """
        Loads the CSV file into a pandas DataFrame.
        """
        self.df = pd.read_csv(self.file_path)

    def filter_by_interval(self, column_name, lower_bound, upper_bound):
        """
        Filters the DataFrame to keep only the rows where the values in the specified column
        are within the given interval [lower_bound, upper_bound].

        :param column_name: str, name of the column to filter by
        :param lower_bound: float, lower bound of the interval
        :param upper_bound: float, upper bound of the interval
        :return: pandas DataFrame, filtered DataFrame
        """
        if self.df is not None:
            filtered_df = self.df[(self.df[column_name] >= lower_bound) & (self.df[column_name] <= upper_bound)]
            return filtered_df
        else:
            raise ValueError("Data frame is not loaded. Please load the CSV file first.")
        
    def plot_histograms(self, filtered_df, output_dir):
        """
        Plots histograms for every column except the last one in the filtered DataFrame.
        The last column contains truth information about the event being signal or background.

        :param filtered_df: pandas DataFrame, filtered DataFrame
        :param output_dir: str, directory to save the histograms
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if filtered_df is not None:
            # Exclude the last column
            columns_to_plot = filtered_df.columns[:-1]

            # Plot histograms for each column
            for column in columns_to_plot:
                print(f'Plotting histogram for {column}')
                plt.figure(figsize=(12, 8))
                filtered_df[column].hist(bins=30, histtype='step', alpha=0.7)
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')

                plot_filename = f'histogram_{column}.png'
                plot_filepath = os.path.join(output_dir, plot_filename)

                plt.savefig(plot_filepath)
                plt.close()
        else:
            raise ValueError("Filtered data frame is not provided.")




# Example usage
model_state_dict = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05_run2/WH_vs_WZ_corrected_optimal_DO05_run2_epoch120.pth'
validation_data_path = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05_run2/WH_vs_WZ_corrected_optimal_DO05_run2_test.csv'
intervals = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]

model_class = BinaryClassifier

evaluator = ModelEvaluator(model_class, model_state_dict)
validation_loader, feature_columns, label_column, weights = evaluator.load_validation_data(validation_data_path, batch_size=32)
interval_datasets = evaluator.perform_selection(validation_loader, intervals, feature_columns, label_column)

for interval, dataset in interval_datasets.items():
    print(f'Interval {interval}: {len(dataset)} samples')
    print(dataset.head())

path_head = '/work/ehettwer/HiggsMewMew/ML/projects/WH_vs_WZ_corrected_optimal_DO05_run2/'
plot_directory_individual = path_head + 'interval_plots_individual'
plot_directory_one_plot = path_head + 'interval_plots_one_plot'
plot_directory_sb = path_head + 'interval_plots_sb'
plot_directory_mass_cut = path_head + 'mass_cut_plots'
plot_directory_roc = path_head + 'roc_plots'

#evaluator.plot_and_save_intervals_individual(interval_datasets, plot_directory_individual)
#evaluator.plot_and_save_intervals_one_plot(interval_datasets, plot_directory_one_plot)
evaluator.plot_and_save_intervals_signal_vs_background(interval_datasets, plot_directory_sb)
evaluator.plot_roc_curve(validation_loader, plot_directory_roc)

#mass_cut = simple_mass_cut(validation_data_path)
#mass_cut.load_csv()
#mass_cut_df = mass_cut.filter_by_interval('m_H', 124, 126)
#mass_cut.plot_histograms(mass_cut_df, plot_directory_mass_cut)
