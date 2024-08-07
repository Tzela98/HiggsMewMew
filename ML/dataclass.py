import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# NtupleDataclass is a custom Dataset class that reads data from CSV files.
# It can be used to create a DataLoader for training and testing datasets.
# It also calculates class weights for imbalanced datasets.
# The class is designed to work with the BinaryClassifier model.
# The last column of the CSV file is assumed to be the label column.
# The class assumes that the data is preprocessed and ready for training.


class NtupleDataclass(Dataset):
    def __init__(self, csv_paths: list, project_name, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/', device='cuda', transform=None, test_size = 0.2):
        self.data_frames = [pd.read_csv(path) for path in csv_paths]
        self.transform = transform

        # Concatenate all data frames into one
        self.data_frame = pd.concat(self.data_frames, ignore_index=True)
        # Shuffle all data to mix different classes
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)

        # Drop unwanted columns
        columns_to_drop = ['Event', 'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'trg_sf']
        self.data_frame.drop(columns=columns_to_drop, inplace=True)

        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=42)
        
        # Store training and testing data frames
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        self.train_df.to_csv(f'{save_path}{project_name}_train.csv', index=False)
        self.test_df.to_csv(f'{save_path}{project_name}_test.csv', index=False)

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df.iloc[:, -1]),
            y=self.train_df.iloc[:, -1]
        )       
        
        self.pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # Select all columns except the last one as features
        features = self.train_df.iloc[idx, :-1].values.astype('float32')

        try:
            # Select the last column as the label
            label = self.train_df.iloc[idx, -1].astype('float32')
        except ValueError as e:
            raise Exception(f"Error converting label to float at index {idx}. Error: {e}")

        if self.transform:
            features = self.transform(features)
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return features, label

    def get_test_data(self):
        feature_names = self.test_df.columns[:-1]
        features = self.test_df.iloc[:, :-1].values.astype('float32')
        labels = self.test_df.iloc[:, -1].values.astype('float32')

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return feature_names, features, labels


class NtupleDataclass_Dev(Dataset):
    def __init__(self, csv_paths: list, project_name, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/', device='cuda', transform=None, test_size=0.2):
        self.data_frames = [pd.read_csv(path) for path in csv_paths]
        self.transform = transform

        # Concatenate all data frames into one
        self.data_frame = pd.concat(self.data_frames, ignore_index=True)
        
        # Separate the 'weights' column and store it
        self.weights = self.data_frame['weights']

        # Drop unwanted columns including 'weights'
        columns_to_drop = ['id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'trg_sf', 'weights', 'genWeight', 'Unnamed: 0', 'Unnamed: 0.1']
        self.data_frame.drop(columns=columns_to_drop, inplace=True)

        # Shuffle all data to mix different classes, keeping weights aligned
        self.data_frame['weights'] = self.weights
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.weights = self.data_frame['weights']
        self.data_frame.drop(columns=['weights'], inplace=True)

        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=42)
        
        # Store training and testing data frames
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        # Save the train and test data with weights for reference
        train_df_with_weights = train_df.copy()
        test_df_with_weights = test_df.copy()
        train_df_with_weights['weights'] = self.weights.loc[train_df_with_weights.index].values
        test_df_with_weights['weights'] = self.weights.loc[test_df_with_weights.index].values

        train_df_with_weights.to_csv(f'{save_path}{project_name}_train.csv', index=False)
        test_df_with_weights.to_csv(f'{save_path}{project_name}_test.csv', index=False)

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df.iloc[:, -1]),
            y=self.train_df.iloc[:, -1]
        )       
        
        self.pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        # Select all columns except the last one as features
        features = self.train_df.iloc[idx, :-1].values.astype('float32')

        try:
            # Select the last column as the label
            label = self.train_df.iloc[idx, -1].astype('float32')
        except ValueError as e:
            raise Exception(f"Error converting label to float at index {idx}. Error: {e}")

        if self.transform:
            features = self.transform(features)
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return features, label

    def get_test_data(self):
        feature_names = self.test_df.columns[:-1]
        features = self.test_df.iloc[:, :-1].values.astype('float32')
        labels = self.test_df.iloc[:, -1].values.astype('float32')

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return feature_names, features, labels


