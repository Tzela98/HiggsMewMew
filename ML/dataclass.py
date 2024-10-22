from cProfile import label
from turtle import back
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
        self.type = self.data_frame['type']
        #self.m_H = self.data_frame['m_H']

        # Drop unwanted columns including 'weights'
        columns_to_drop = ['type', 'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'trg_sf', 'weights', 'genWeight', 'Unnamed: 0', 'mvaTTH_1', 'mvaTTH_2', 'mvaTTH_3']
        self.data_frame.drop(columns=columns_to_drop, inplace=True)

        # Shuffle all data to mix different classes, keeping weights aligned
        self.data_frame['weights'] = self.weights
        self.data_frame['type'] = self.type
        #self.data_frame['m_H'] = self.m_H
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.weights = self.data_frame['weights']
        self.type = self.data_frame['type']
        #self.m_H = self.data_frame['m_H']
        self.data_frame.drop(columns=['weights'], inplace=True)
        self.data_frame.drop(columns=['type'], inplace=True)
        #self.data_frame.drop(columns=['m_H'], inplace=True)

        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=42)
        
        # Store training and testing data frames
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        # Save the train and test data with weights for reference
        train_df_with_weights = train_df.copy()
        test_df_with_weights = test_df.copy()
        train_df_with_weights['weights'] = self.weights.loc[train_df_with_weights.index].values
        test_df_with_weights['weights'] = self.weights.loc[test_df_with_weights.index].values
        train_df_with_weights['type'] = self.type.loc[train_df_with_weights.index].values
        test_df_with_weights['type'] = self.type.loc[test_df_with_weights.index].values
        #train_df_with_weights['m_H'] = self.m_H.loc[train_df_with_weights.index].values
        #test_df_with_weights['m_H'] = self.m_H.loc[test_df_with_weights.index].values

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


class NtupleDataclass_vbf(Dataset):
    def __init__(self, csv_paths: list, project_name, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/', device='cuda', transform=None, test_size=0.2):
        self.data_frames = [pd.read_csv(path) for path in csv_paths]
        self.transform = transform

        # Concatenate all data frames into one
        self.data_frame = pd.concat(self.data_frames, ignore_index=True)
        
        # Create 'signal' column based on 'is_vbf' and 'is_gluglu'
        self.data_frame['signal'] = ((self.data_frame['is_vbf'] == 1) | (self.data_frame['is_gluglu'] == 1)).astype(int)
        
        # Separate the 'weights' column and store it
        self.weights = self.data_frame['weights']
        self.m_vis = self.data_frame['m_vis']

        # Drop unwanted columns including 'weights'
        columns_to_drop = ['id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'trg_sf', 'weights', 'genWeight', 'Unnamed: 0', 'mvaTTH_1', 'mvaTTH_2', 
                           'is_ttbar', 'is_dyjets', 'is_vbf', 'is_gluglu', 'm_vis']
        self.data_frame.drop(columns=columns_to_drop, inplace=True)

        # Shuffle all data to mix different classes, keeping weights aligned
        self.data_frame['weights'] = self.weights
        self.data_frame['m_vis'] = self.m_vis
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.weights = self.data_frame['weights']
        self.m_vis = self.data_frame['m_vis']

        self.data_frame.drop(columns=['weights'], inplace=True)
        self.data_frame.drop(columns=['m_vis'], inplace=True)

        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=42)
        
        # Store training and testing data frames
        self.train_df = train_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        # Save the train and test data with weights for reference
        train_df_with_weights = train_df.copy()
        test_df_with_weights = test_df.copy()
        train_df_with_weights['weights'] = self.weights.loc[train_df_with_weights.index].values
        test_df_with_weights['weights'] = self.weights.loc[test_df_with_weights.index].values
        train_df_with_weights['m_vis'] = self.m_vis.loc[train_df_with_weights.index].values
        test_df_with_weights['m_vis'] = self.m_vis.loc[test_df_with_weights.index].values

        train_df_with_weights.to_csv(f'{save_path}{project_name}_train.csv', index=False)
        test_df_with_weights.to_csv(f'{save_path}{project_name}_test.csv', index=False)

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_df['signal']),
            y=self.train_df['signal']
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


class NewDataClass(Dataset):
    def __init__(self, csv_paths: list, project_name, save_path='/work/ehettwer/HiggsMewMew/ML/tmp/', device='cuda', transform=None, test_size=0.2):
        data_frames = [pd.read_csv(path) for path in csv_paths]
        data_frame = pd.concat(data_frames, ignore_index=True)
        self.transform = transform

        feature_names = ['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH', 'deltaR_12', 
                         'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H', 'q_1', 'q_2', 'q_3', 'pt_1', 'pt_2', 'pt_3', 
                         'nmuons', 'eta_1', 'eta_2', 'cosThetaStar12', 'cosThetaStar13', 'cosThetaStar23']
        self.genWeight = data_frame['weights']
        label_name = ['is_wh']
        background_type = data_frame['type']

        self.data_frame = data_frame[feature_names + background_type  + label_name]

        # Shuffle all data to mix different classes, keeping weights aligned
        self.data_frame['weights'] = self.weights
        self.data_frame = self.data_frame.sample(frac=1, random_state=42).reset_index(drop=True)
        self.weights = self.data_frame['weights']
        self.data_frame.drop(columns=['weights'], inplace=True)

        self.background_weight_dict = {
            'WZ': np.float32(212/288),
            'ZZ': np.float32(33/288),
            'DY': np.float32(32/288),
            'Top': np.float32(11/288),
        }
        
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


