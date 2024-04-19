from email.headerregistry import AddressHeader
import re
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from icecream import ic
import os

import mplhep as hep  # for plotting in HEP style

# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


def df_segmentation(df, variable, threshold = []):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


def main():

    data_2018_vbf = pd.read_csv('workflow/csv_files/single_muon_data_2018.csv')

    regions_dict = {
        'ZRegion': [76, 106],
        'SideBand1': [110, 115],
        'SideBand2': [135, 150],
        'SignalRegion': [115, 135]
    }

    data_2018_vbf_segmented = df_segmentation(data_2018_vbf, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    

    background_info = pd.read_csv('workflow/csv_files/background_info.csv', index_col='nicks')
    list_of_backgrounds = []
    
    for background_contribution in background_info.index:
        file_path = 'workflow/csv_files/' + background_contribution + '.csv'
        
        # Check if the file exists before reading it
        if os.path.exists(file_path):
            # Append the DataFrame to the list
            list_of_backgrounds.append(pd.read_csv(file_path))
        else:
            print(f"File '{file_path}' not found.")

    
    backgrounds_dict = dict(zip(background_info.index, list_of_backgrounds))
    
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 9))
    fig1.subplots_adjust(hspace=0.05)

    n_mvis_Z, bins_mvis_Z = np.histogram(data_2018_vbf_segmented[0]['m_vis'], bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]))
    n_mvis_SB1, bins_mvis_SB1 = np.histogram(data_2018_vbf_segmented[1]['m_vis'], bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]))
    n_mvis_SB2, bins_mvis_SB2 = np.histogram(data_2018_vbf_segmented[2]['m_vis'], bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]))
    n_mvis_SR, bins_mvis_SR = np.histogram(data_2018_vbf_segmented[3]['m_vis'], bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]))

    n_error_mvis_Z = np.sqrt(n_mvis_Z)
    n_error_mvis_SB1 = np.sqrt(n_mvis_SB1)
    n_error_mvis_SB2 = np.sqrt(n_mvis_SB2)
    n_error_mvis_SR = np.sqrt(n_mvis_SR)

    ax1.errorbar((bins_mvis_Z[:-1] + (bins_mvis_Z[1] - bins_mvis_Z[0])/2), n_mvis_Z, yerr=n_error_mvis_Z, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2, label='Data CMS Run II - 2018')
    ax1.errorbar((bins_mvis_SB1[:-1] + (bins_mvis_SB1[1] - bins_mvis_SB1[0])/2), n_mvis_SB1, yerr=n_error_mvis_SB1, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)
    ax1.errorbar((bins_mvis_SB2[:-1] + (bins_mvis_SB2[1] - bins_mvis_SB2[0])/2), n_mvis_SB2, yerr=n_error_mvis_SB2, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)

    for background_contribution in backgrounds_dict:
        id_iso_wgt = (backgrounds_dict[background_contribution]['id_wgt_mu_1'] * backgrounds_dict[background_contribution]['id_wgt_mu_2'] * backgrounds_dict[background_contribution]['iso_wgt_mu_1'] * backgrounds_dict[background_contribution]['iso_wgt_mu_2'])
        weights = (backgrounds_dict[background_contribution]['genWeight']/np.abs(backgrounds_dict[background_contribution]['genWeight'])/(pd.to_numeric(background_info.loc[background_contribution, 'generator_weight']) * pd.to_numeric(background_info.loc[background_contribution, 'number_of_events'])))* pd.to_numeric(background_info.loc[background_contribution, 'cross_section']) * 10**(-12) * 59.7 * 10**(15) * id_iso_wgt
        backgrounds_dict[background_contribution].insert(len(backgrounds_dict[background_contribution].columns), 'weights', weights)
    
    
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
    
    colors = ['teal', 'forestgreen', 'orange', 'midnightblue']

    Top_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_top_contributions], ignore_index=True)
    DYJets_contribution_Z_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_Z], ignore_index=True)
    DYJets_contribution_rest_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_rest], ignore_index=True)
    diboson_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_diboson_contributions], ignore_index=True)
    ewk_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_ewk_contributions], ignore_index=True)

    Top_contribution_segmented = df_segmentation(Top_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    DYJets_contribution_Z_segmented = df_segmentation(DYJets_contribution_Z_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    DYJets_contribution_rest_segmented = df_segmentation(DYJets_contribution_rest_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    diboson_contribution_segmented = df_segmentation(diboson_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    ewk_contribution_segmented = df_segmentation(ewk_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])

    Z_region_backgrounds = [Top_contribution_segmented[0]['m_vis'], diboson_contribution_segmented[0]['m_vis'], ewk_contribution_segmented[0]['m_vis'], DYJets_contribution_Z_segmented[0]['m_vis']]
    SB1_region_backgrounds = [Top_contribution_segmented[1]['m_vis'], diboson_contribution_segmented[1]['m_vis'], ewk_contribution_segmented[1]['m_vis'], DYJets_contribution_rest_segmented[1]['m_vis']]
    SB2_region_backgrounds = [Top_contribution_segmented[2]['m_vis'], diboson_contribution_segmented[2]['m_vis'], ewk_contribution_segmented[2]['m_vis'], DYJets_contribution_rest_segmented[2]['m_vis']]
    SR_region_backgrounds = [Top_contribution_segmented[3]['m_vis'], diboson_contribution_segmented[3]['m_vis'], ewk_contribution_segmented[3]['m_vis'], DYJets_contribution_rest_segmented[3]['m_vis']]

    Z_weights = [Top_contribution_segmented[0]['weights'], diboson_contribution_segmented[0]['weights'], ewk_contribution_segmented[0]['weights'], DYJets_contribution_Z_segmented[0]['weights']]
    SB1_weights = [Top_contribution_segmented[1]['weights'], diboson_contribution_segmented[1]['weights'], ewk_contribution_segmented[1]['weights'], DYJets_contribution_rest_segmented[1]['weights']]
    SB2_weights = [Top_contribution_segmented[2]['weights'], diboson_contribution_segmented[2]['weights'], ewk_contribution_segmented[2]['weights'], DYJets_contribution_rest_segmented[2]['weights']]
    SR_weights = [Top_contribution_segmented[3]['weights'], diboson_contribution_segmented[3]['weights'], ewk_contribution_segmented[3]['weights'], DYJets_contribution_rest_segmented[3]['weights']]

    n_MC_Z_top, bins_MC_Z_top = np.histogram(Z_region_backgrounds[0], bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]), weights=Z_weights[0])
    n_MC_Z_diboson, bins_MC_Z_diboson = np.histogram(Z_region_backgrounds[1], bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]), weights=Z_weights[1])
    n_MC_Z_ewk, bins_MC_Z_ewk = np.histogram(Z_region_backgrounds[2], bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]), weights=Z_weights[2])
    n_MC_Z_DYJets, bins_MC_Z_DYJets = np.histogram(Z_region_backgrounds[3], bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]), weights=Z_weights[3])
    
    hep.histplot([n_MC_Z_top, n_MC_Z_diboson, n_MC_Z_ewk, n_MC_Z_DYJets], bins_MC_Z_top, label=['top', 'diboson', 'ewk', 'DYJets'], stack=True, ax=ax1, color=colors, edges=True)
    
    n_MC_SB1_top, bins_MC_SB1_top = np.histogram(SB1_region_backgrounds[0], bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]), weights=SB1_weights[0])
    n_MC_SB1_diboson, bins_MC_SB1_diboson = np.histogram(SB1_region_backgrounds[1], bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]), weights=SB1_weights[1])
    n_MC_SB1_ewk, bins_MC_SB1_ewk = np.histogram(SB1_region_backgrounds[2], bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]), weights=SB1_weights[2])
    n_MC_SB1_DYJets, bins_MC_SB1_DYJets = np.histogram(SB1_region_backgrounds[3], bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]), weights=SB1_weights[3])

    hep.histplot([n_MC_SB1_top, n_MC_SB1_diboson, n_MC_SB1_ewk, n_MC_SB1_DYJets], bins_MC_SB1_top, stack=True, ax=ax1, color=colors, edges=True)
    
    n_MC_SB2_top, bins_MC_SB2_top = np.histogram(SB2_region_backgrounds[0], bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]), weights=SB2_weights[0])
    n_MC_SB2_diboson, bins_MC_SB2_diboson = np.histogram(SB2_region_backgrounds[1], bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]), weights=SB2_weights[1])
    n_MC_SB2_ewk, bins_MC_SB2_ewk = np.histogram(SB2_region_backgrounds[2], bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]), weights=SB2_weights[2])
    n_MC_SB2_DYJets, bins_MC_SB2_DYJets = np.histogram(SB2_region_backgrounds[3], bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]), weights=SB2_weights[3])

    hep.histplot([n_MC_SB2_top, n_MC_SB2_diboson, n_MC_SB2_ewk, n_MC_SB2_DYJets], bins_MC_SB2_top, stack=True, ax=ax1, color=colors, edges=True)

    n_MC_SR_top, bins_MC_SR_top = np.histogram(SR_region_backgrounds[0], bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]), weights=SR_weights[0])
    n_MC_SR_diboson, bins_MC_SR_diboson = np.histogram(SR_region_backgrounds[1], bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]), weights=SR_weights[1])
    n_MC_SR_ewk, bins_MC_SR_ewk = np.histogram(SR_region_backgrounds[2], bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]), weights=SR_weights[2])
    n_MC_SR_DYJets, bins_MC_SR_DYJets = np.histogram(SR_region_backgrounds[3], bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]), weights=SR_weights[3])

    hep.histplot([n_MC_SR_top, n_MC_SR_diboson, n_MC_SR_ewk, n_MC_SR_DYJets], bins_MC_SR_top, stack=True, ax=ax1, color=colors, edges=False)

    ax1.set_ylabel('Events')
    ax1.set_xlim(76, 150)
    #ax1.set_yscale('log')
    ax1.set_title('CMS Run II - 2018 Dimuon Mass and MC Backgrounds')
    ax1.legend(fontsize='xx-small')

    n_MC_Z = n_MC_Z_top + n_MC_Z_diboson + n_MC_Z_ewk + n_MC_Z_DYJets
    n_MC_SB1 = n_MC_SB1_top + n_MC_SB1_diboson + n_MC_SB1_ewk + n_MC_SB1_DYJets
    n_MC_SB2 = n_MC_SB2_top + n_MC_SB2_diboson + n_MC_SB2_ewk + n_MC_SB2_DYJets
    n_MC_SR = n_MC_SR_top + n_MC_SR_diboson + n_MC_SR_ewk + n_MC_SR_DYJets

    residuals_error_Z = np.abs(n_mvis_Z/n_MC_Z) * np.sqrt((n_error_mvis_Z/n_mvis_Z)**2 + (np.sqrt(n_MC_Z)/n_MC_Z)**2)
    residuals_error_SB1 = np.abs(n_mvis_SB1/n_MC_SB1) * np.sqrt((n_error_mvis_SB1/n_mvis_SB1)**2 + (np.sqrt(n_MC_SB1)/n_MC_SB1)**2)
    residuals_error_SB2 = np.abs(n_mvis_SB2/n_MC_SB2) * np.sqrt((n_error_mvis_SB2/n_mvis_SB2)**2 + (np.sqrt(n_MC_SB2)/n_MC_SB2)**2)

    ax2.errorbar((bins_mvis_Z[:-1] + (bins_mvis_Z[1] - bins_mvis_Z[0])/2), n_mvis_Z/n_MC_Z, yerr=residuals_error_Z,  marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)
    ax2.errorbar((bins_mvis_SB1[:-1] + (bins_mvis_SB1[1] - bins_mvis_SB1[0])/2), n_mvis_SB1/n_MC_SB1, yerr=residuals_error_SB1,  marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2) 
    ax2.errorbar((bins_mvis_SB2[:-1] + (bins_mvis_SB2[1] - bins_mvis_SB2[0])/2), n_mvis_SB2/n_MC_SB2, yerr=residuals_error_SB2,  marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)    
    
    #ic(n_mvis_SB2/np.sum(n_MC_SB2, axis=0))

    ax2.set_xlabel(r'$m_{vis}$ / GeV') 
    ax2.set_ylabel('Data / MC')
    ax2.axhline(y = 1, color = 'r', linestyle = '--', linewidth = 0.5)
    ax2.set_ylim(0.5, 1.5)

    #plt.savefig('plots/dimuon_mass_2018_v3.png')
    plt.show()

if __name__ == '__main__':
    main()