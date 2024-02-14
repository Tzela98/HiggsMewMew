import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    data_2018_vbf = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/vbf_data_2018.csv'))
    DYJetsToLL_M_50 = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/DYJetsToLL_M-50_mc_genWeights.csv'))
    TTToSemiLeptonic = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/TTToSemiLeptonic_mc_2018.csv'))
    ST_t_channel_top_5f = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_top_5f_mc_2018.csv'))

    ST_t_channel_antitop_5f = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_antitop_5f_mc_2018.csv'), 400), 2.5)
    ST_tW_antitop_5f_inclusiveDecays = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_tW_antitop_5f_inclusiveDecays_mc_2018.csv'), 400), 2.5)
    ST_tW_top_5f_inclusiveDecays = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_tW_top_5f_inclusiveDecays_mc_2018.csv'), 400), 2.5)
    WWTo2L2Nu = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/WWTo2L2Nu_mc_2018.csv'), 400), 2.5)


    # Define orthogonal regions
    ZRegion = [76, 106]
    SideBand_1 = [110, 115]
    SideBand_2 = [135, 150]
    SignalRegion = [115, 135]

    data_2018_vbf_segmented = df_segmentation(data_2018_vbf, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])

    DYJetsToLL_M_50_segmented = df_segmentation(DYJetsToLL_M_50, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    TTToSemiLeptonic_segmented = df_segmentation(TTToSemiLeptonic, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    ST_t_channel_top_5f_segmented = df_segmentation(ST_t_channel_top_5f, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    
    ST_t_channel_antitop_5f_segmented = df_segmentation(ST_t_channel_antitop_5f, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    ST_tW_antitop_5f_inclusiveDecays_segmented = df_segmentation(ST_tW_antitop_5f_inclusiveDecays, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    ST_tW_top_5f_inclusiveDecays_segmented = df_segmentation(ST_tW_top_5f_inclusiveDecays, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    WWTo2L2Nu_segmented = df_segmentation(WWTo2L2Nu, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])

    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 9))
    fig1.subplots_adjust(hspace=0.05)

    n_mvis_Z, bins_mvis_Z = np.histogram(data_2018_vbf_segmented[0].m_vis, bins=(ZRegion[1] - ZRegion[0]), range=(ZRegion[0], ZRegion[1]))
    n_mvis_SB1, bins_mvis_SB1 = np.histogram(data_2018_vbf_segmented[1].m_vis, bins=(SideBand_1[1] - SideBand_1[0]), range=(SideBand_1[0], SideBand_1[1]))
    n_mvis_SB2, bins_mvis_SB2 = np.histogram(data_2018_vbf_segmented[2].m_vis, bins=(SideBand_2[1] - SideBand_2[0]), range=(SideBand_2[0], SideBand_2[1]))

    ax1.plot((bins_mvis_Z[:-1] + (bins_mvis_Z[1] - bins_mvis_Z[0])/2), n_mvis_Z, marker='o', linestyle='None', color='black', markersize=2, label='Data CMS Run II - 2018')
    ax1.plot((bins_mvis_SB1[:-1] + (bins_mvis_SB1[1] - bins_mvis_SB1[0])/2), n_mvis_SB1, marker='o', linestyle='None', color='black', markersize=2)
    ax1.plot((bins_mvis_SB2[:-1] + (bins_mvis_SB2[1] - bins_mvis_SB2[0])/2), n_mvis_SB2, marker='o', linestyle='None', color='black', markersize=2)

    # weights = 1/number total mc events * cross section * luminosity
    n_mc_generated_dy = 193119590
    n_mc_generated_TTT = 100790000
    n_mc_generated_ST_t_top = 5903676

    weights_DYJets_Z = (DYJetsToLL_M_50_segmented[0].genWeight/np.abs(DYJetsToLL_M_50_segmented[0].genWeight))/(0.677684 * n_mc_generated_dy) * 3.717 * 10**(8)
    weights_DYJets_SB1 = (DYJetsToLL_M_50_segmented[1].genWeight/np.abs(DYJetsToLL_M_50_segmented[1].genWeight))/(0.677684 * n_mc_generated_dy) * 3.717 * 10**(8)
    weights_DYJets_SB2 = (DYJetsToLL_M_50_segmented[2].genWeight/np.abs(DYJetsToLL_M_50_segmented[2].genWeight))/(0.677684 * n_mc_generated_dy) * 3.717 * 10**(8)
    weights_DYJets_Signal = (DYJetsToLL_M_50_segmented[3].genWeight/np.abs(DYJetsToLL_M_50_segmented[3].genWeight))/(0.677684 * n_mc_generated_dy) * 3.717 * 10**(8)
    # weights = cross section * luminosity * acceptance = 6225.4 pb * 59.7 (fb)^-1 = 3.717 * 10^8 * acceptance

    '''
    weights_TTT_Z = (TTToSemiLeptonic_segmented[0].genWeight/np.abs(TTToSemiLeptonic_segmented[0].genWeight))/(0.105 * n_mc_generated_TTT) * 2.141*10**(7)
    weights_TTT_SB1 = (TTToSemiLeptonic_segmented[1].genWeight/np.abs(TTToSemiLeptonic_segmented[1].genWeight))/(0.105 * n_mc_generated_TTT) * 2.141*10**(7)
    weights_TTT_SB2 = (TTToSemiLeptonic_segmented[2].genWeight/np.abs(TTToSemiLeptonic_segmented[2].genWeight))/(0.105 * n_mc_generated_TTT) * 2.141*10**(7)
    weights_TTT_Signal = (TTToSemiLeptonic_segmented[3].genWeight/np.abs(TTToSemiLeptonic_segmented[3].genWeight))/(0.105 * n_mc_generated_TTT) * 2.141*10**(7)
    '''
    weights_ST_t_channel_top_Z = (ST_t_channel_top_5f_segmented[0].genWeight/np.abs(ST_t_channel_top_5f_segmented[0].genWeight))/(0.9942740 * n_mc_generated_ST_t_top) * 8.12 * 10**(6)
    weights_ST_t_channel_top_SB1 = (ST_t_channel_top_5f_segmented[1].genWeight/np.abs(ST_t_channel_top_5f_segmented[1].genWeight))/(0.9942740 * n_mc_generated_ST_t_top) * 8.12 * 10**(6)
    weights_ST_t_channel_top_SB2 = (ST_t_channel_top_5f_segmented[2].genWeight/np.abs(ST_t_channel_top_5f_segmented[2].genWeight))/(0.9942740 * n_mc_generated_ST_t_top) * 8.12 * 10**(6)
    weights_ST_t_channel_top_Signal = (ST_t_channel_top_5f_segmented[3].genWeight/np.abs(ST_t_channel_top_5f_segmented[3].genWeight))/(0.9942740 * n_mc_generated_ST_t_top) * 8.12 * 10**(6)
    

    if np.shape(weights_DYJets_Z) != np.shape(DYJetsToLL_M_50_segmented[0].m_vis):
        raise ValueError("Weights and m_vis do not have the same shape.")
    

    n_mvis_mc_Z, bins_m_vis_mc_Z, patches_Z = ax1.hist(DYJetsToLL_M_50_segmented[0].m_vis, bins=(ZRegion[1] - ZRegion[0]), range=(ZRegion[0], ZRegion[1]), histtype='step', stacked=True, weights=weights_DYJets_Z, color='red', alpha=1, label='DYJetsToLL_M_50 MC')
    n_mvis_mc_SB1, bins_m_vis_mc_SB1, patches_SB1 = ax1.hist(DYJetsToLL_M_50_segmented[1].m_vis, bins=(SideBand_1[1] - SideBand_1[0]), range=(SideBand_1[0], SideBand_1[1]), histtype='step', stacked=True, weights=weights_DYJets_SB1, color='red', alpha=1)
    n_mvis_mc_SB2, bins_m_vis_mc_SB2, patches_SB2 = ax1.hist(DYJetsToLL_M_50_segmented[2].m_vis, bins=(SideBand_2[1] - SideBand_2[0]), range=(SideBand_2[0], SideBand_2[1]), histtype='step', stacked=True, weights=weights_DYJets_SB2, color='red', alpha=1)
    n_mvis_mc_Sig, bins_m_vis_mc_Sig, patches_Sig = ax1.hist(DYJetsToLL_M_50_segmented[3].m_vis, bins=(SignalRegion[1] - SignalRegion[0]), range=(SignalRegion[0], SignalRegion[1]), histtype='step', stacked=True, weights=weights_DYJets_Signal, color='red', alpha=1)
    
    n_mvis_mc_ST_t_top_Z, bins_m_vis_mc_ST_t_top_Z, patches_ST_t_top_Z = ax1.hist(ST_t_channel_top_5f_segmented[0].m_vis, bins=(ZRegion[1] - ZRegion[0]), range=(ZRegion[0], ZRegion[1]), histtype='step', stacked=True, weights=weights_ST_t_channel_top_Z, color='blue', alpha=1, label='ST_t_channel_top_5f MC')
    n_mvis_mc_ST_t_top_SB1, bins_m_vis_mc_ST_t_top_SB1, patches_ST_t_top_SB1 = ax1.hist(ST_t_channel_top_5f_segmented[1].m_vis, bins=(SideBand_1[1] - SideBand_1[0]), range=(SideBand_1[0], SideBand_1[1]), histtype='step', stacked=True, weights=weights_ST_t_channel_top_SB1, color='blue', alpha=1)
    n_mvis_mc_ST_t_top_SB2, bins_m_vis_mc_ST_t_top_SB2, patches_ST_t_top_SB2 = ax1.hist(ST_t_channel_top_5f_segmented[2].m_vis, bins=(SideBand_2[1] - SideBand_2[0]), range=(SideBand_2[0], SideBand_2[1]), histtype='step', stacked=True, weights=weights_ST_t_channel_top_SB2, color='blue', alpha=1)
    n_mvis_mc_ST_t_top_Sig, bins_m_vis_mc_ST_t_top_Sig, patches_ST_t_top_Sig = ax1.hist(ST_t_channel_top_5f_segmented[3].m_vis, bins=(SignalRegion[1] - SignalRegion[0]), range=(SignalRegion[0], SignalRegion[1]), histtype='step', stacked=True, weights=weights_ST_t_channel_top_Signal, color='blue', alpha=1)

    ax1.legend(loc='upper right')
    ax1.set_xlim([76, 150])
    ax1.set_yscale('log')
    ax1.set_ylabel('Events')

    ax2.plot((bins_mvis_Z[:-1] + (bins_mvis_Z[1] - bins_mvis_Z[0])/2), n_mvis_Z/n_mvis_mc_Z, marker='o', linestyle='None', color='black', markersize=2)
    ax2.plot((bins_mvis_SB1[:-1] + (bins_mvis_SB1[1] - bins_mvis_SB1[0])/2), n_mvis_SB1/n_mvis_mc_SB1, marker='o', linestyle='None', color='black', markersize=2) 
    ax2.plot((bins_mvis_SB2[:-1] + (bins_mvis_SB2[1] - bins_mvis_SB2[0])/2), n_mvis_SB2/n_mvis_mc_SB2, marker='o', linestyle='None', color='black', markersize=2)
    
    ax2.set_ylim([0.5, 1.5])
    ax2.set_xlabel('m_vis' + ' [GeV]')

    ic('Yields Z Region:')
    ic(len(n_mvis_Z), n_mvis_Z)
    ic(len(n_mvis_mc_Z), n_mvis_mc_Z)
    ic('Data: ', np.sum(n_mvis_Z))
    ic('DY: ', np.sum(n_mvis_mc_Z))

    fig1.savefig('plots/dimuon_mass.png')
    plt.show()

if __name__ == '__main__':
    main()
