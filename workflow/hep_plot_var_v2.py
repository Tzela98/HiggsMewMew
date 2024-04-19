# Import necessary libraries/modules
from hmac import new  # for hashing
from tkinter import font  # for GUI font handling
import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
from icecream import ic  # for debugging and logging
import os  # for file system operations

from sympy import E  # Euler's number
import filters as filters  # custom filters (not defined here)

import mplhep as hep  # for plotting in HEP style
from torch import rand  # for random number generation

# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
# Place the CMS label at location 0 (top-left corner)
hep.cms.label(loc=0)


# Function for segmenting a DataFrame based on a variable and given thresholds
def df_segmentation(df, variable, threshold=[]):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


# Function to plot a variable
def calculate_histogram(df, variable, region, bins=30):
    n, bins = np.histogram(df[variable], bins=bins, weights=df['weights'])
    return n, bins
    

def plot_var(variable: str, bin_range: tuple, bin_number: int, region_index):
    background_info = pd.read_csv('workflow/csv_files/background_info.csv', index_col='nicks')

    ZRegion_backgrounds = []
    SidebandRegion_backgrounds = []
    SignalRegion_backgrounds = []

    for background_contribution in background_info.index:
        file_path_ZRegion = 'workflow/csv_files/backgrounds_segmented/ZRegion/' + background_contribution + '_ZRegion.csv'
        file_path_SideBand = 'workflow/csv_files/backgrounds_segmented/SidebandRegion/' + background_contribution + '_SidebandRegion.csv'
        file_path_SignalRegion = 'workflow/csv_files/backgrounds_segmented/SignalRegion/' + background_contribution + '_SignalRegion.csv'
        # Check if the file exists before reading it
        if os.path.exists(file_path_ZRegion):
            # Append the DataFrame to the list
            ZRegion_backgrounds.append(pd.read_csv(file_path_ZRegion))
        elif os.path.exists(file_path_SideBand):
            SidebandRegion_backgrounds.append(pd.read_csv(file_path))
        elif os.path.exists(file_path_SignalRegion):
            SignalRegion_backgrounds.append(pd.read_csv(file_path))
        else:
            print(f"File '{file_path_SideBand}' not found.")
    
    ZRegion_backgrounds_dict = dict(zip(background_info.index, ZRegion_backgrounds))
    SidebandRegion_backgrounds_dict = dict(zip(background_info.index, SidebandRegion_backgrounds))
    SignalRegion_backgrounds_dict = dict(zip(background_info.index, SignalRegion_backgrounds))

    list_of_top_contributions = ['ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                                 'ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                                 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                                 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X']

    list_of_DYJets_contributions_Z = ['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X']
    list_of_DYJets_contributions_rest = ['DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X']

    list_of_diboson_contributions = ['WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                                     'WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                                     'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X']

    list_of_ewk_contributions = ['EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole_RunIISummer20UL18NanoAODv9-106X']

    if region_index == 0:
        Top_Z_df = pd.concat([ZRegion_backgrounds_dict[contribution] for contribution in list_of_top_contributions], ignore_index=True)
        DYJets_Z_df = pd.concat([ZRegion_backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_Z], ignore_index=True)
        Diboson_Z_df = pd.concat([ZRegion_backgrounds_dict[contribution] for contribution in list_of_diboson_contributions], ignore_index=True)
        EWK_Z_df = pd.concat([ZRegion_backgrounds_dict[contribution] for contribution in list_of_ewk_contributions], ignore_index=True)

        backgrounds = [Top_Z_df[variable], DYJets_Z_df[variable], Diboson_Z_df[variable], EWK_Z_df[variable]]
        weights = [Top_Z_df['weights'], DYJets_Z_df['weights'], Diboson_Z_df['weights'], EWK_Z_df['weights']]

    elif region_index == 1:
        Top_SB_df = pd.concat([SidebandRegion_backgrounds_dict[contribution] for contribution in list_of_top_contributions], ignore_index=True)
        DYJets_SB_df = pd.concat([SidebandRegion_backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_rest], ignore_index=True)
        Diboson_SB_df = pd.concat([SidebandRegion_backgrounds_dict[contribution] for contribution in list_of_diboson_contributions], ignore_index=True)
        EWK_SB_df = pd.concat([SidebandRegion_backgrounds_dict[contribution] for contribution in list_of_ewk_contributions], ignore_index=True)

        backgrounds = [Top_SB_df[variable], DYJets_SB_df[variable], Diboson_SB_df[variable], EWK_SB_df[variable]]
        weights = [Top_SB_df['weights'], DYJets_SB_df['weights'], Diboson_SB_df['weights'], EWK_SB_df['weights']]

    elif region_index == 2:
        Top_SR_df = pd.concat([SignalRegion_backgrounds_dict[contribution] for contribution in list_of_top_contributions], ignore_index=True)
        DYJets_SR_df = pd.concat([SignalRegion_backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_rest], ignore_index=True)
        Diboson_SR_df = pd.concat([SignalRegion_backgrounds_dict[contribution] for contribution in list_of_diboson_contributions], ignore_index=True)
        EWK_SR_df = pd.concat([SignalRegion_backgrounds_dict[contribution] for contribution in list_of_ewk_contributions], ignore_index=True)

        backgrounds = [Top_SR_df[variable], DYJets_SR_df[variable], Diboson_SR_df[variable], EWK_SR_df[variable]]
        weights = [Top_SR_df['weights'], DYJets_SR_df['weights'], Diboson_SR_df['weights'], EWK_SR_df['weights']]

    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 9))
    fig1.subplots_adjust(hspace=0.05)

    hist_background_top, bin_edges = np.histogram(backgrounds[0], bins=bin_number, range=bin_range, weights=weights[0])
    hist_background_DYJets, _ = np.histogram(backgrounds[1], bins=bin_number, range=bin_range, weights=weights[1])
    hist_background_diboson, _ = np.histogram(backgrounds[2], bins=bin_number, range=bin_range, weights=weights[2])
    hist_background_EWK, _ = np.histogram(backgrounds[3], bins=bin_number, range=bin_range, weights=weights[3])

    # Plot stacked histograms of background contributions on the first subplot
    hep.histplot([hist_background_top, hist_background_diboson, hist_background_EWK, hist_background_DYJets], bins=bin_edges, stack=True, ax=ax1, label=['top', 'diboson', 'EWK', 'DYJets'])
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_xlim(bin_range)
    ax1.legend(fontsize='xx-small')
    plt.show()


def main():
    plot_var('pt_1', (0, 300), 30, 1)
    

if __name__ == '__main__':
    main()