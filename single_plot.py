from calendar import c
from turtle import back, color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from icecream import ic


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


    n_mvis_Z, bins_mvis_Z = np.histogram(data_2018_vbf_segmented[0]['m_vis'], bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]))
    n_mvis_SB1, bins_mvis_SB1 = np.histogram(data_2018_vbf_segmented[1]['m_vis'], bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]))
    n_mvis_SB2, bins_mvis_SB2 = np.histogram(data_2018_vbf_segmented[2]['m_vis'], bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]))
    n_mvis_SR, bins_mvis_SR = np.histogram(data_2018_vbf_segmented[3]['m_vis'], bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]))

    fig = plt.figure(1, figsize=(12, 10))

    plt.plot((bins_mvis_Z[:-1] + (bins_mvis_Z[1] - bins_mvis_Z[0])/2), n_mvis_Z, marker='o', linestyle='None', color='black', markersize=2, label='Data CMS Run II - 2018')
    plt.plot((bins_mvis_SB1[:-1] + (bins_mvis_SB1[1] - bins_mvis_SB1[0])/2), n_mvis_SB1, marker='o', linestyle='None', color='black', markersize=2)
    plt.plot((bins_mvis_SB2[:-1] + (bins_mvis_SB2[1] - bins_mvis_SB2[0])/2), n_mvis_SB2, marker='o', linestyle='None', color='black', markersize=2)


    plt.xlabel(r'$m_{vis}$ / GeV')
    plt.ylabel('Events')
    plt.legend()
    plt.yscale('log')


    for background_contribution in backgrounds_dict:
        weights = backgrounds_dict[background_contribution]['genWeight']/np.abs(backgrounds_dict[background_contribution]['genWeight'])/(background_info.loc[background_contribution, 'generator_weight'] * background_info.loc[background_contribution, 'number_of_events'])* background_info.loc[background_contribution, 'cross_section'] * 10**(-12) * 59.7 * 10**(15)
        backgrounds_dict[background_contribution].insert(len(backgrounds_dict[background_contribution].columns), 'weights', weights)

    Z_region_backgrounds = []
    SB1_region_backgrounds = []
    SB2_region_backgrounds = []
    SR_region_backgrounds = []

    Z_weights = []
    SB1_weights = []
    SB2_weights = []
    SR_weights = []

    for background_contribution in backgrounds_dict:

        segmented_df = df_segmentation(backgrounds_dict[background_contribution], 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']]) 
        
        Z_region_backgrounds.append(list(segmented_df[0]['m_vis']))
        SB1_region_backgrounds.append(list(segmented_df[1]['m_vis']))
        SB2_region_backgrounds.append(list(segmented_df[2]['m_vis']))
        SR_region_backgrounds.append(list(segmented_df[3]['m_vis']))

        Z_weights.append(list(segmented_df[0]['weights']))
        SB1_weights.append(list(segmented_df[1]['weights']))
        SB2_weights.append(list(segmented_df[2]['weights']))
        SR_weights.append(list(segmented_df[3]['weights']))

    Z_region_backgrounds.sort(key=len)
    SB1_region_backgrounds.sort(key=len)
    SB2_region_backgrounds.sort(key=len)
    SR_region_backgrounds.sort(key=len)

    Z_weights.sort(key=len)
    SB1_weights.sort(key=len)
    SB2_weights.sort(key=len)
    SR_weights.sort(key=len)

    colors = sns.color_palette('flare', 12)
   
    plt.hist(sorted(Z_region_backgrounds, key=len), bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]), weights=sorted(Z_weights, key = len), stacked=True, histtype='step', fill=False, color=colors, label=background_info.index)
    plt.hist(sorted(SB1_region_backgrounds, key=len), bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]), weights=sorted(SB1_weights, key = len), stacked=True, histtype='step', fill=False, color=colors)
    plt.hist(sorted(SB2_region_backgrounds, key=len), bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]), weights=sorted(SB2_weights, key = len), stacked=True, histtype='step', fill=False, color=colors)
    plt.hist(sorted(SR_region_backgrounds, key=len), bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]), weights=sorted(SR_weights, key = len), stacked=True, histtype='step', fill=False, color=colors)
    
    plt.legend(prop={'size': 6})
    plt.show()
        
    
if __name__ == '__main__':
    main()