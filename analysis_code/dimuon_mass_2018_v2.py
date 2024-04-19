from email.headerregistry import AddressHeader
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from icecream import ic
import os

from pyparsing import line

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

    data_2018_vbf = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/Single_muon_data_2018_new.csv'))

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
        weights = (backgrounds_dict[background_contribution]['genWeight']/np.abs(backgrounds_dict[background_contribution]['genWeight'])/(pd.to_numeric(background_info.loc[background_contribution, 'generator_weight']) * pd.to_numeric(background_info.loc[background_contribution, 'number_of_events'])))* pd.to_numeric(background_info.loc[background_contribution, 'cross_section']) * 10**(-12) * 59.7 * 10**(15)
        backgrounds_dict[background_contribution].insert(len(backgrounds_dict[background_contribution].columns), 'weights', weights)
    
    ic(backgrounds_dict['DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020']['weights'])
    
    list_of_top_contributions = ['ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020', 'ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020', 
                                 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1', 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1', 
                                 'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020']

    list_of_DYJets_contributions_Z = ['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext2']
    
    list_of_DYJets_contributions_rest = ['DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020']

    list_of_other_contributions = ['EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18NanoAODv7-Nano02Apr2020', 'WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020', 
                                   'WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020', 'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1', 'ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020', 
                                   'ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1']
    
    colors = ['teal', 'forestgreen', 'midnightblue']

    Top_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_top_contributions], ignore_index=True)
    DYJets_contribution_Z_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_Z], ignore_index=True)
    DYJets_contribution_rest_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_DYJets_contributions_rest], ignore_index=True)
    Other_contribution_df = pd.concat([backgrounds_dict[contribution] for contribution in list_of_other_contributions], ignore_index=True)

    Top_contribution_segmented = df_segmentation(Top_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    DYJets_contribution_Z_segmented = df_segmentation(DYJets_contribution_Z_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    DYJets_contribution_rest_segmented = df_segmentation(DYJets_contribution_rest_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])
    Other_contribution_segmented = df_segmentation(Other_contribution_df, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])

    Z_region_backgrounds = [Top_contribution_segmented[0]['m_vis'], Other_contribution_segmented[0]['m_vis'], DYJets_contribution_Z_segmented[0]['m_vis']]
    SB1_region_backgrounds = [Top_contribution_segmented[1]['m_vis'], Other_contribution_segmented[1]['m_vis'], DYJets_contribution_rest_segmented[1]['m_vis']]
    SB2_region_backgrounds = [Top_contribution_segmented[2]['m_vis'], Other_contribution_segmented[2]['m_vis'], DYJets_contribution_rest_segmented[2]['m_vis']]
    SR_region_backgrounds = [Top_contribution_segmented[3]['m_vis'], Other_contribution_segmented[3]['m_vis'], DYJets_contribution_rest_segmented[3]['m_vis']]

    Z_weights = [Top_contribution_segmented[0]['weights'], Other_contribution_segmented[0]['weights'], DYJets_contribution_Z_segmented[0]['weights']]
    SB1_weights = [Top_contribution_segmented[1]['weights'], Other_contribution_segmented[1]['weights'], DYJets_contribution_rest_segmented[1]['weights']]
    SB2_weights = [Top_contribution_segmented[2]['weights'], Other_contribution_segmented[2]['weights'], DYJets_contribution_rest_segmented[2]['weights']]
    SR_weights = [Top_contribution_segmented[3]['weights'], Other_contribution_segmented[3]['weights'], DYJets_contribution_rest_segmented[3]['weights']]

    n_MC_Z, bins_MC_Z, patches_MC_Z = ax1.hist(Z_region_backgrounds, bins=(regions_dict['ZRegion'][1] - regions_dict['ZRegion'][0]), weights=Z_weights, histtype='step', stacked=True, color=colors, label=['Other', 'Top', 'DYJets'])
    n_MC_SB1, bins_MC_SB1, patches_MC_SB1 = ax1.hist(SB1_region_backgrounds, bins=(regions_dict['SideBand1'][1] - regions_dict['SideBand1'][0]), weights=SB1_weights, histtype='step', stacked=True, color=colors)
    n_MC_SB2, bins_MC_SB2, patches_MC_SB2 = ax1.hist(SB2_region_backgrounds, bins=(regions_dict['SideBand2'][1] - regions_dict['SideBand2'][0]), weights=SB2_weights, histtype='step', stacked=True, color=colors)
    n_MC_SR, bins_MC_SR, patches_MC_SR = ax1.hist(SR_region_backgrounds, bins=(regions_dict['SignalRegion'][1] - regions_dict['SignalRegion'][0]), weights=SR_weights, histtype='step', stacked=True, color=colors)

    ax1.set_ylabel('Events')
    ax1.set_xlim(76, 150)
    ax1.set_yscale('log')
    ax1.set_title('CMS Run II - 2018 Dimuon Mass Spectrum and MC Backgrounds')
    ax1.legend()

    residuals_error_Z = np.abs(n_mvis_Z/np.sum(n_MC_Z, axis=0)) * np.sqrt((n_error_mvis_Z/n_mvis_Z)**2 + (np.sqrt(np.sum(n_MC_Z, axis=0))/np.sum(n_MC_Z, axis=0))**2)
    residuals_error_SB1 = np.abs(n_mvis_SB1/np.sum(n_MC_SB1, axis=0)) * np.sqrt((n_error_mvis_SB1/n_mvis_SB1)**2 + (np.sqrt(np.sum(n_MC_SB1, axis=0))/np.sum(n_MC_SB1, axis=0))**2)
    residuals_error_SB2 = np.abs(n_mvis_SB2/np.sum(n_MC_SB2, axis=0)) * np.sqrt((n_error_mvis_SB2/n_mvis_SB2)**2 + (np.sqrt(np.sum(n_MC_SB2, axis=0))/np.sum(n_MC_SB2, axis=0))**2)

    ax2.errorbar((bins_mvis_Z[:-1] + (bins_mvis_Z[1] - bins_mvis_Z[0])/2), n_mvis_Z/np.sum(n_MC_Z, axis=0), yerr=residuals_error_Z,  marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)
    ax2.errorbar((bins_mvis_SB1[:-1] + (bins_mvis_SB1[1] - bins_mvis_SB1[0])/2), n_mvis_SB1/np.sum(n_MC_SB1, axis=0), yerr=residuals_error_SB1,  marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2) 
    ax2.errorbar((bins_mvis_SB2[:-1] + (bins_mvis_SB2[1] - bins_mvis_SB2[0])/2), n_mvis_SB2/np.sum(n_MC_SB2, axis=0), yerr=residuals_error_SB2,  marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)    
    
    ic(n_mvis_SB2/np.sum(n_MC_SB2, axis=0))

    ax2.set_xlabel(r'$m_{vis}$ / GeV') 
    ax2.set_ylabel('Data / MC')
    ax2.axhline(y = 1, color = 'r', linestyle = '--', linewidth = 0.5)
    ax2.set_ylim(0.5, 1.5)

    plt.savefig('plots/dimuon_mass_2018_v2.png')
    plt.show()

if __name__ == '__main__':
    main()