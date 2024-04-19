from operator import index
from turtle import back
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from icecream import ic
import os


def filter_for_jet_mass(df, threshold_mass):
    return df[df.mjj > threshold_mass]


def filter_pseudo_rapidity_separation(df, threshold_rapidity):
    return df[np.abs(df.jeta_1 - df.jeta_2) > threshold_rapidity]


def jet_selection(df, leading_jet_pt = 35, sub_leading_jet_pt = 25):
    if 'jpt_1' in df.columns and 'jpt_2' in df.columns:
        condition_1 = df['jpt_1'] > leading_jet_pt
        condition_2 = df['jpt_2'] > sub_leading_jet_pt

        filtered_df = df[condition_1 & condition_2]
        return filtered_df
    
    else: raise ValueError("Columns 'jpt_1' and 'jpt_2' are not in the DataFrame.")



def selection_pipeline(df, leading_jet_pt = 25, sub_leading_jet_pt = 35, threshold_mass = 400, threshold_rapidity = 2.5):
    return filter_pseudo_rapidity_separation(filter_for_jet_mass(jet_selection(df, leading_jet_pt, sub_leading_jet_pt), threshold_mass), threshold_rapidity)


def df_segmentation(df, variable, threshold = []):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


def plot_histograms(dataframes, bin_ranges_dict, variable):
    if len(dataframes) != len(bin_ranges_dict):
        raise ValueError("Anzahl der DataFrames und Bin-Ranges stimmt nicht überein.")

    num_subplots = len(dataframes)

    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    iterable_list = list(bin_ranges_dict.values())

    for i, (df, bins) in enumerate(zip(dataframes, iterable_list)):
        # Berechne die Position (Zeile, Spalte) im Subplot-Grid
        row = i // num_cols
        col = i % num_cols

        axs[row, col].hist(df[variable], bins=(bins[1] - bins[0]), alpha=0.7, label=f'DataFrame {i+1}')
        axs[row, col].set_title(f'Histogram for {variable} - DataFrame {i+1}')
        axs[row, col].set_xlabel(variable)
        axs[row, col].set_ylabel('Number of events')
        axs[row, col].set_yscale('log')
        axs[row, col].legend()

    # Entferne leere Subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()


def calculate_histograms(dataframes, bin_ranges_dict, variable):
    # Überprüfen, ob die Anzahl der DataFrames mit der Anzahl der Bin-Ranges übereinstimmt
    if len(dataframes) != len(bin_ranges_dict):
        raise ValueError("Anzahl der DataFrames und Bin-Ranges stimmt nicht überein.")

    # Initialisiere Listen für Histogrammzählungen und Bin-Kanten
    all_histogram_counts = []
    all_bin_edges = []

    # Iteration über DataFrames und Bin-Ranges
    iterable_list = list(bin_ranges_dict.values())

    for i, (df, bins) in enumerate(zip(dataframes, iterable_list)):
        # Berechne das Histogramm
        counts, bin_edges = np.histogram(df[variable], bins=(bins[1] - bins[0]))

        # Füge die Zählungen und Bin-Kanten zu den Listen hinzu
        all_histogram_counts.append(counts)
        all_bin_edges.append(bin_edges)

    return all_histogram_counts, all_bin_edges



def main():

    data_2018_vbf = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/Single_muon_data_2018.csv'))

    regions_dict = {
        'ZRegion': [76, 106],
        'SideBand1': [110, 115],
        'SideBand2': [135, 150],
        'SignalRegion': [115, 135]
    }

    data_2018_vbf_segmented = df_segmentation(data_2018_vbf, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    

    background_info = pd.read_csv('data_csv/background_info.csv', index_col='nicks')
    list_of_backgrounds = []
    
    for background_contribution in background_info.index:
        file_path = 'data_csv/' + background_contribution + '.csv'
        
        # Check if the file exists before reading it
        if os.path.exists(file_path):
            # Append the DataFrame to the list
            list_of_backgrounds.append(selection_pipeline(pd.read_csv(file_path)))
        else:
            print(f"File '{file_path}' not found.")
    
    backgrounds_dict = dict(zip(background_info.index, list_of_backgrounds))
    ic(backgrounds_dict)

    #for background_contribution in backgrounds_dict:

    plot_histograms(data_2018_vbf_segmented, regions_dict, 'm_vis')
    for background_contribution in backgrounds_dict:
        plot_histograms(df_segmentation(backgrounds_dict[background_contribution], 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']]), regions_dict, 'm_vis')
    

if __name__ == '__main__':
    main()