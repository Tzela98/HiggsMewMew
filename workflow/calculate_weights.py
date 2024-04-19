# Import necessary libraries/modules
import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
from icecream import ic  # for debugging and logging
import os  # for file system operations
import filters as filters  # custom filters (not defined here)
import glob


def df_segmentation(df, variable, threshold=[]):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


def calculate_weights(datasets_info: str, dataset_base_path: str, lumi = 59.74 * 1000):
    info = pd.read_csv(datasets_info)
    for index, row in info.iterrows():
        dataset = pd.read_csv(dataset_base_path + 'unweighted/' + row['nicks'] + '.csv')
        id_iso_wgt = dataset['id_wgt_mu_1'] * dataset['iso_wgt_mu_1'] * dataset['id_wgt_mu_2'] * dataset['iso_wgt_mu_2']
        acceptance = dataset['genWeight']/(abs(dataset['genWeight']) * row['generator_weight'] * row['number_of_events'])
        weight = id_iso_wgt * acceptance * lumi * row['cross_section']
        dataset['weights'] = weight
        dataset.to_csv(dataset_base_path + 'backgrounds_weighted/' + row['nicks'] +'.csv', index=False)


def segmentation():
    info = pd.read_csv('workflow/csv_files/background_info.csv')
    regions_dict = {
        'ZRegion': [76, 106],
        'SideBand1': [110, 115],
        'SideBand2': [135, 150],
        'SignalRegion': [115, 135]
    }

    for index, row in info.iterrows():
        dataset = pd.read_csv('workflow/csv_files/backgrounds_weighted/' + row['nicks'] + '.csv')
        df_segmented_Z = df_segmentation(dataset, 'm_vis', [regions_dict['ZRegion']])
        df_segmented_Z[0].to_csv('workflow/csv_files/backgrounds_segmented/ZRegion/' + row['nicks'] + '_ZRegion.csv', index=False)


    for index, row in info.iterrows():
        dataset = pd.read_csv('workflow/csv_files/backgrounds_weighted/' + row['nicks'] + '.csv')
        df_segmented_SB1 = df_segmentation(dataset, 'm_vis', [regions_dict['SideBand1']])
        df_segmented_SB2 = df_segmentation(dataset, 'm_vis', [regions_dict['SideBand2']])
        df_Sideband = pd.concat([df_segmented_SB1[0], df_segmented_SB2[0]])
        df_Sideband.to_csv('workflow/csv_files/backgrounds_segmented/SidebandRegion/' + row['nicks'] + '_SidebandRegion.csv', index=False)

    
    for index, row in info.iterrows():
        dataset = pd.read_csv('workflow/csv_files/backgrounds_weighted/' + row['nicks'] + '.csv')
        df_segmented_SR = df_segmentation(dataset, 'm_vis', [regions_dict['SignalRegion']])
        df_segmented_SR[0].to_csv('workflow/csv_files/backgrounds_segmented/SignalRegion/' + row['nicks'] + '_SignalRegion.csv', index=False)


def main():
    #calculate_weights('workflow/csv_files/background_info.csv', '/work/ehettwer/HiggsMewMew/workflow/csv_files/')
    segmentation()


if __name__ == '__main__':
    main()