from hmac import new
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from icecream import ic
import os
import filters as filters

def df_segmentation(df, variable, threshold = []):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


def plot_single_object_variable(variable: str, bin_range: tuple, bin_number: int, region_index):

    data_2018_vbf = filters.selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/Single_muon_data_2018_new.csv'), 25, 35, 400, 2.5)
    background_info = pd.read_csv('data_csv/background_info.csv', index_col='nicks')

    list_of_backgrounds = []

    for background_contribution in background_info.index:
        file_path = 'data_csv/' + background_contribution + '.csv'
        
        # Check if the file exists before reading it
        if os.path.exists(file_path):
            # Append the DataFrame to the list
            list_of_backgrounds.append(filters.selection_pipeline(pd.read_csv(file_path), 25, 35, 400, 2.5))
        else:
            print(f"File '{file_path}' not found.")

    
    backgrounds_dict = dict(zip(background_info.index, list_of_backgrounds))

    regions_dict = {
        'ZRegion': [76, 106],
        'SideBand1': [110, 115],
        'SideBand2': [135, 150],
        'SignalRegion': [115, 135]
    }

    regions_list = ['ZRegion', 'SideBand1', 'SideBand2', 'SignalRegion']

    data_2018_vbf_segmented = df_segmentation(data_2018_vbf, 'm_vis', [regions_dict['ZRegion'], regions_dict['SideBand1'], regions_dict['SideBand2'], regions_dict['SignalRegion']])    
    
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 9))
    fig1.subplots_adjust(hspace=0.05)

    n_data, bins_data = np.histogram(data_2018_vbf_segmented[region_index][variable], range=bin_range, bins=bin_number, density=False)
    n_error = np.sqrt(n_data)
    ax1.errorbar((bins_data[:-1] + (bins_data[1] - bins_data[0])/2), n_data, yerr=n_error, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2, label='Data CMS Run II - 2018')

    for background_contribution in backgrounds_dict:
        weights = (backgrounds_dict[background_contribution]['genWeight']/np.abs(backgrounds_dict[background_contribution]['genWeight'])/(pd.to_numeric(background_info.loc[background_contribution, 'generator_weight']) * pd.to_numeric(background_info.loc[background_contribution, 'number_of_events'])))* pd.to_numeric(background_info.loc[background_contribution, 'cross_section']) * 10**(-12) * 59.7 * 10**(15)
        backgrounds_dict[background_contribution].insert(len(backgrounds_dict[background_contribution].columns), 'weights', weights)


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

    if region_index == 0:
        backgrounds = [Top_contribution_segmented[region_index][variable], Other_contribution_segmented[region_index][variable], DYJets_contribution_Z_segmented[region_index][variable]]
        weights = [Top_contribution_segmented[region_index]['weights'], Other_contribution_segmented[region_index]['weights'], DYJets_contribution_Z_segmented[region_index]['weights']]
    
    else:
        backgrounds = [Top_contribution_segmented[region_index][variable], Other_contribution_segmented[region_index][variable], DYJets_contribution_rest_segmented[region_index][variable]]
        weights = [Top_contribution_segmented[region_index]['weights'], Other_contribution_segmented[region_index]['weights'], DYJets_contribution_rest_segmented[region_index]['weights']]

    
    n_MC, bins_MC, patches_MC = ax1.hist(backgrounds, range=bin_range, bins=bin_number, weights=weights, histtype='step', stacked=True, color=colors, label=['Other', 'Top', 'DYJets'])

    ax1.set_ylabel('Events')
    #ax1.set_yscale('log')
    ax1.set_xlim(bin_range)
    ax1.set_title('CMS Run II - 2018 - ' + variable + ' in ' + regions_list[region_index])
    ax1.legend()

    residuals_error = np.abs(n_data/np.sum(n_MC, axis=0)) * np.sqrt((n_error/n_data)**2 + (np.sqrt(np.sum(n_MC, axis=0))/np.sum(n_MC, axis=0))**2)
    ax2.errorbar((bins_data[:-1] + (bins_data[1] - bins_data[0])/2), n_data/np.sum(n_MC, axis=0),  yerr=residuals_error, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2)
    
    ax2.set_xlabel(variable + '/ GeV') 
    ax2.axhline(y = 1, color = 'r', linestyle = '--', linewidth = 0.5)
    ax2.set_ylim(0.5, 1.5)

    #plt.savefig('plots/' + variable + '_2018_' + regions_list[region_index] + '.png')
    plt.show()

def main():
    
    list_of_variables = ['pt_1', 'pt_2', 'jpt_1', 'jpt_2', 'pt_vis']
    for variable in list_of_variables:
        plot_single_object_variable(variable, (0, 300), 30, 1)
        print('plotted ' + variable)

if __name__ == "__main__":
    main()
