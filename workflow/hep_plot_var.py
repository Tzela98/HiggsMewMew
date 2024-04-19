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
def plot_var(variable: str, bin_range: tuple, bin_number: int, region_index):
    # Read data and background MC ntuples into pandas dataframes
    data_2018_vbf = pd.read_csv('workflow/csv_files/single_muon_data_2018.csv')
    signal_sim_2018 = pd.read_csv('workflow/csv_files/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')
    background_info = pd.read_csv('workflow/csv_files/background_info.csv', index_col='nicks')

    list_of_backgrounds = []

    # Loop through each background contribution
    for background_contribution in background_info.index:
        file_path = 'workflow/csv_files/' + background_contribution + '.csv'

        # Check if the file exists before reading it
        if os.path.exists(file_path):
            # Append the DataFrame to the list
            list_of_backgrounds.append(pd.read_csv(file_path))
        else:
            print(f"File '{file_path}' not found.")

    # Create a dictionary of background contributions
    backgrounds_dict = dict(zip(background_info.index, list_of_backgrounds))

    # Define regions and their thresholds
    regions_dict = {
        'ZRegion': [76, 106],
        'SideBand1': [110, 115],
        'SideBand2': [135, 150],
        'SignalRegion': [115, 135]
    }

    regions_list = ['ZRegion', 'SideBand', 'SideBand', 'SignalRegion']

    # Segment the data based on predefined regions
    data_2018_vbf_segmented = df_segmentation(data_2018_vbf, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    
    list_of_signal_contributions = ['VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X']

    # Calculate weights for signal simulation

    for signal_contribution in list_of_signal_contributions:
        id_iso_wgt = signal_sim_2018['id_wgt_mu_1'] * signal_sim_2018['id_wgt_mu_2'] * signal_sim_2018['iso_wgt_mu_1'] * signal_sim_2018['iso_wgt_mu_2']
        weights = signal_sim_2018['genWeight'] / (np.abs(signal_sim_2018['genWeight']) * (pd.to_numeric(background_info.loc[signal_contribution, 'generator_weight']) * pd.to_numeric(background_info.loc[signal_contribution, 'number_of_events']))) * pd.to_numeric(background_info.loc[signal_contribution, 'cross_section']) * 10**(-12) * 59.7 * 10**(15) * id_iso_wgt
        signal_sim_2018.insert(len(signal_sim_2018.columns), 'weights', weights)
    
    signal_sim_segmented = df_segmentation(signal_sim_2018, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])

    # Calculate histograms and errors for data
    if region_index in [1, 2]:
        n_data, bins_data = np.histogram(pd.concat((data_2018_vbf_segmented[1][variable], data_2018_vbf_segmented[2][variable]), ignore_index=True), range=bin_range, bins=bin_number, density=False)
        n_sim, bins_sim = np.histogram(pd.concat((signal_sim_segmented[1][variable], signal_sim_segmented[2][variable]), ignore_index=True), range=bin_range, bins=bin_number, density=False)
    else:
        n_data, bins_data = np.histogram(data_2018_vbf_segmented[region_index][variable], range=bin_range, bins=bin_number, density=False)
        n_sim, bins_sim = np.histogram(signal_sim_segmented[region_index][variable], range=bin_range, bins=bin_number, density=False)

    n_error = np.sqrt(n_data)

    # Calculate weights for each background contribution
    for background_contribution in backgrounds_dict:
        id_iso_wgt = (backgrounds_dict[background_contribution]['id_wgt_mu_1'] * backgrounds_dict[background_contribution]['id_wgt_mu_2'] * backgrounds_dict[background_contribution]['iso_wgt_mu_1'] * backgrounds_dict[background_contribution]['iso_wgt_mu_2']) * backgrounds_dict[background_contribution]['trg_sf']
        weights = backgrounds_dict[background_contribution]['genWeight'] / (np.abs(backgrounds_dict[background_contribution]['genWeight']) * (pd.to_numeric(background_info.loc[background_contribution, 'generator_weight']) * pd.to_numeric(background_info.loc[background_contribution, 'number_of_events']))) * pd.to_numeric(background_info.loc[background_contribution, 'cross_section']) * 10**(-12) * 59.7 * 10**(15) * id_iso_wgt
        backgrounds_dict[background_contribution].insert(len(backgrounds_dict[background_contribution].columns), 'weights', weights)

    # Define lists of different background contributions
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



    colors = ['red', 'teal', 'forestgreen', 'orange', 'midnightblue']

    # Concatenate background contributions into DataFrames
    Top_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_top_contributions], ignore_index=True)
    DYJets_contribution_Z_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_Z], ignore_index=True)
    DYJets_contribution_rest_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_rest], ignore_index=True)
    Diboson_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_diboson_contributions], ignore_index=True)
    EWK_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_ewk_contributions], ignore_index=True)

    # Segment background contributions based on predefined regions
    Top_contribution_segmented = df_segmentation(Top_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    DYJets_contribution_Z_segmented = df_segmentation(DYJets_contribution_Z_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    DYJets_contribution_rest_segmented = df_segmentation(DYJets_contribution_rest_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    Diboson_contribution_segmented = df_segmentation(Diboson_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    EWK_contribution_segmented = df_segmentation(EWK_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])

    # Define backgrounds and weights based on the region index
    if region_index == 0:
        backgrounds = [Top_contribution_segmented[region_index][variable], Diboson_contribution_segmented[region_index][variable], DYJets_contribution_Z_segmented[region_index][variable], EWK_contribution_segmented[region_index][variable]]
        weights = [Top_contribution_segmented[region_index]['weights'], Diboson_contribution_segmented[region_index]['weights'], DYJets_contribution_Z_segmented[region_index]['weights'], EWK_contribution_segmented[region_index]['weights']]

    elif region_index in [1, 2]:

        comb_top_contributions_sb = pd.concat((Top_contribution_segmented[1][variable], Top_contribution_segmented[2][variable]), ignore_index=True)
        comb_top_weights_sb = pd.concat((Top_contribution_segmented[1]['weights'], Top_contribution_segmented[2]['weights']), ignore_index=True)

        comb_diboson_contributions_sb = pd.concat((Diboson_contribution_segmented[1][variable], Diboson_contribution_segmented[2][variable]), ignore_index=True)
        comb_diboson_weights_sb = pd.concat((Diboson_contribution_segmented[1]['weights'], Diboson_contribution_segmented[2]['weights']), ignore_index=True)

        comb_DYJets_contributions_sb = pd.concat((DYJets_contribution_rest_segmented[1][variable], DYJets_contribution_rest_segmented[2][variable]), ignore_index=True)
        comb_DYJets_weights_sb = pd.concat((DYJets_contribution_rest_segmented[1]['weights'], DYJets_contribution_rest_segmented[2]['weights']), ignore_index=True)
        
        comb_EWK_contributions_sb = pd.concat((EWK_contribution_segmented[1][variable], EWK_contribution_segmented[2][variable]), ignore_index=True)
        comb_EWK_weights_sb = pd.concat((EWK_contribution_segmented[1]['weights'], EWK_contribution_segmented[2]['weights']), ignore_index=True)

        backgrounds = [np.array(comb_top_contributions_sb), np.array(comb_diboson_contributions_sb), np.array(comb_DYJets_contributions_sb), np.array(comb_EWK_contributions_sb)]
        weights = [np.array(comb_top_weights_sb), np.array(comb_diboson_weights_sb), np.array(comb_DYJets_weights_sb), np.array(comb_EWK_weights_sb)]

    else:
         # Add your code here or remove the else clause if not needed
        backgrounds = [Top_contribution_segmented[region_index][variable], Diboson_contribution_segmented[region_index][variable], DYJets_contribution_rest_segmented[region_index][variable], EWK_contribution_segmented[region_index][variable]]
        weights = [Top_contribution_segmented[region_index]['weights'], Diboson_contribution_segmented[region_index]['weights'], DYJets_contribution_rest_segmented[region_index]['weights'], EWK_contribution_segmented[region_index]['weights']]

    # Calculate bin width and centers
    bin_width = bins_data[1] - bins_data[0]
    bin_centers = bins_data[:-1] + bin_width / 2

    # Create a figure with 2 subplots
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 9))
    fig1.subplots_adjust(hspace=0.05)

    # Plot data with error bars on the first subplot
    ax1.errorbar(bin_centers, n_data, yerr=n_error, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2, label='Data CMS Run II - 2018')

    # Calculate histograms for each background contribution
    hist_background_top, bin_edges = np.histogram(backgrounds[0], bins=bin_number, range=bin_range, weights=weights[0])
    hist_background_diboson, _ = np.histogram(backgrounds[1], bins=bin_number, range=bin_range, weights=weights[1])
    hist_background_DYJets, _ = np.histogram(backgrounds[2], bins=bin_number, range=bin_range, weights=weights[2])
    hist_background_EWK, _ = np.histogram(backgrounds[3], bins=bin_number, range=bin_range, weights=weights[3])

    # Plot stacked histograms of background contributions on the first subplot
    hep.histplot([n_sim, hist_background_top, hist_background_diboson, hist_background_EWK, hist_background_DYJets], bins=bin_edges, stack=True, ax=ax1, color=colors, label=['signal sim', 'top', 'diboson', 'EWK', 'DYJets'])
    
    # Calculate total Monte Carlo (MC) events
    n_MC = np.sum([hist_background_top, hist_background_diboson, hist_background_EWK, hist_background_DYJets], axis=0)

    # Customize the first subplot
    ax1.set_ylabel('Events')
    ax1.set_yscale('log')
    ax1.set_xlim(bin_range)
    ax1.set_title('CMS Run II - 2018 - ' + variable + ' in ' + regions_list[region_index])
    ax1.legend(fontsize='xx-small')

    # Calculate residuals (Data/MC) and their errors
    residuals = n_data / n_MC
    residuals_error = np.abs(n_data / n_MC) * np.sqrt((n_error / n_data) ** 2 + (np.sqrt(n_MC) / n_MC) ** 2)

    ax2.errorbar(bin_centers, residuals,  yerr=residuals_error, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)

    ax2.set_xlabel(variable + '/ GeV') 
    ax2.set_ylabel('Data/MC')
    ax2.axhline(y = 1, color = 'r', linestyle = '--', linewidth = 0.5)
    ax2.set_ylim(0.5, 1.5)

    #plt.savefig('plots/' + variable + '_2018_' + regions_list[region_index] + '_stacked.png')
    plt.show()

def main():
    
    list_of_variables = ['pt_1']
    range = (0, 300)

    for variable in list_of_variables:
        for i in [0, 1]:
            plot_var(variable, range, 30, i)
            print('plotted ' + variable + ' in region ' + str(i))

    

if __name__ == "__main__":
    main()
