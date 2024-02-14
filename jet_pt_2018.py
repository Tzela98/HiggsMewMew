from matplotlib import markers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import filters


def selection_pipeline(df, leading_jet_pt = 25, sub_leading_jet_pt = 35, threshold_mass = 400, threshold_rapidity = 2.5):
    return filters.pseudo_rapidity_separation(filters.dijet_mass_threshold(filters.jet_selection(df, leading_jet_pt, sub_leading_jet_pt), threshold_mass), threshold_rapidity)


def df_segmentation(df, variable, threshold = []):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented

def main():

    # Read Data and MC files and use vbf selection criteria not applied yet in KingMaker
    data_2018_vbf = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/vbf_data_2018.csv'))
    DYJetsToLL_M_50 = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/DYJetsToLL_M-50_mc_genWeights.csv'))
    TTToSemiLeptonic = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/TTToSemiLeptonic_mc_2018.csv'))
    ST_t_channel_top_5f = selection_pipeline(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_top_5f_mc_2018.csv'))

    ZRegion = [76, 110]
    SideBand_1 = [110, 115]
    SideBand_2 = [135, 150]
    SignalRegion = [115, 135]

    data_2018_vbf_segmented = df_segmentation(data_2018_vbf, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    DYJetsToLL_M_50_segmented = df_segmentation(DYJetsToLL_M_50, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    TTToSemiLeptonic_segmented = df_segmentation(TTToSemiLeptonic, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
    ST_t_channel_top_5f_segmented = df_segmentation(ST_t_channel_top_5f, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])

    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 9))
    fig1.subplots_adjust(hspace=0.05)

    n_jpt1_Signal, bins_jpt1_Signal = np.histogram(data_2018_vbf_segmented[3].jpt_2, bins=50, range=(0, 300))

    ax1.plot((bins_jpt1_Signal[:-1] + (bins_jpt1_Signal[1] - bins_jpt1_Signal[0])/2), n_jpt1_Signal, marker='o', linestyle='None', color='black', markersize=2, label='Data CMS Run II - 2018')

    # weights = 1/number total mc events * cross section * luminosity
    n_mc_generated_dy = 193119590
    n_mc_generated_ST_t_top = 5903676

    # weights = cross section * luminosity * acceptance
    weights_DYJets_Signal = (DYJetsToLL_M_50_segmented[3].genWeight/np.abs(DYJetsToLL_M_50_segmented[3].genWeight))/(0.677684 * n_mc_generated_dy) * 3.717 * 10**(8)
    weights_ST_t_channel_top_Signal = (ST_t_channel_top_5f_segmented[3].genWeight/np.abs(ST_t_channel_top_5f_segmented[3].genWeight))/(0.9942740 * n_mc_generated_ST_t_top) * 8.12 * 10**(6)
    
    
    n_DYJets_Signal, bins_DYJets_Signal, patches_DYJets_Signal = ax1.hist(DYJetsToLL_M_50_segmented[3].jpt_2, bins=50, range=(0, 300), weights=weights_DYJets_Signal, histtype='step', color='red', label='DYJetsToLL_M_50')
    n_ST_t_channel_top_Signal, bins_ST_t_channel_top__Signal, patches_ST_t_channel_top_Signal = ax1.hist(ST_t_channel_top_5f_segmented[3].jpt_2, bins=50, range=(0, 300), weights=weights_ST_t_channel_top_Signal, histtype='step', color='blue', label='ST_t_channel_top_5f')

    ax1.set_yscale('log')
    ax1.set_ylabel('Events')
    ax1.set_xlim(0, 300)
    ax1.legend()
    
    monte_carlo_Signal = n_DYJets_Signal + n_ST_t_channel_top_Signal
    ax2.plot((bins_jpt1_Signal[:-1] + (bins_jpt1_Signal[1] - bins_jpt1_Signal[0])/2), n_jpt1_Signal/monte_carlo_Signal, marker='o', linestyle='None', color='black', markersize=2)

    ax2.set_xlabel('Subleading Jet pT [GeV]')

    fig1.savefig('plots/signal_jpt_2.png')
    plt.show()

if __name__ == "__main__":
    main()