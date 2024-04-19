import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
from icecream import ic


def filter_for_jet_mass(df, threshold_value):
    return df[df.mjj > threshold_value]


def filter_pseudo_rapidity_separation(df, threshold_value):
    return df[np.abs(df.jeta_1 - df.jeta_2) > threshold_value]


def df_segmentation(df, variable, threshold = []):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


DYJetsToLL_M_50 = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/DYJetsToLL_M-50_mc_2018.csv'), 400), 2.5)
TTToSemiLeptonic = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/TTToSemiLeptonic_mc_2018.csv'), 400), 2.5)
ST_t_channel_top_5f = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_top_5f_mc_2018.csv'), 400), 2.5)
ST_t_channel_antitop_5f = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_antitop_5f_mc_2018.csv'), 400), 2.5)
ST_tW_antitop_5f_inclusiveDecays = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_tW_antitop_5f_inclusiveDecays_mc_2018.csv'), 400), 2.5)
ST_tW_top_5f_inclusiveDecays = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_tW_top_5f_inclusiveDecays_mc_2018.csv'), 400), 2.5)
WWTo2L2Nu = filter_pseudo_rapidity_separation(filter_for_jet_mass(pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/WWTo2L2Nu_mc_2018.csv'), 400), 2.5)

mc_2018_df = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/mc_2018.csv').transpose()
mc_2018_df.columns = mc_2018_df.iloc[0]
mc_2018_df = mc_2018_df[1:]

# Define orthogonal regions
ZRegion = [76, 106]
SideBand_1 = [110, 115]
SideBand_2 = [135, 150]
SignalRegion = [115, 135]

list_of_variables = ["pt_1", "eta_1", "phi_1", "pt_2", "eta_2", "phi_2", "jpt_1", "jeta_1", "jphi_1", "jpt_2", "jeta_2", "jphi_2", "m_vis", "mjj", "njets"]
list_of_samples = ['DYJetsToLL_M_50', 'TTToSemiLeptonic', 'ST_t_channel_top_5f', 'ST_t_channel_antitop_5f', 'ST_tW_antitop_5f_inclusiveDecays',
                   'ST_tW_top_5f_inclusiveDecays', 'WWTo2L2Nu']

DYJetsToLL_M_50_segmented = df_segmentation(DYJetsToLL_M_50, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
TTToSemiLeptonic_segmented = df_segmentation(TTToSemiLeptonic, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
ST_t_channel_top_5f_segmented = df_segmentation(ST_t_channel_top_5f, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
ST_t_channel_antitop_5f_segmented = df_segmentation(ST_t_channel_antitop_5f, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
ST_tW_antitop_5f_inclusiveDecays_segmented = df_segmentation(ST_tW_antitop_5f_inclusiveDecays, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
ST_tW_top_5f_inclusiveDecays_segmented = df_segmentation(ST_tW_top_5f_inclusiveDecays, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])
WWTo2L2Nu_segmented = df_segmentation(WWTo2L2Nu, 'm_vis', [ZRegion, SideBand_1, SideBand_2, SignalRegion])

datasamples = ak.Array([DYJetsToLL_M_50, TTToSemiLeptonic, ST_t_channel_top_5f, ST_t_channel_antitop_5f, ST_tW_antitop_5f_inclusiveDecays, 
                        ST_tW_top_5f_inclusiveDecays, WWTo2L2Nu])
length_datasamples = [len(DYJetsToLL_M_50), len(TTToSemiLeptonic), len(ST_t_channel_top_5f), len(ST_t_channel_antitop_5f), 
                               len(ST_tW_antitop_5f_inclusiveDecays), len(ST_tW_top_5f_inclusiveDecays), len(WWTo2L2Nu)]

mc_2018_df = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/mc_2018.csv').transpose()
mc_2018_df.columns = mc_2018_df.iloc[0]
mc_2018_df = mc_2018_df[1:]

print(mc_2018_df.head(n=7))
weight = 1/mc_2018_df['total_events']
weight_DYJetsToLL_M_50 = [weight[0]]*len(DYJetsToLL_M_50)
weight_TTToSemiLeptonic = [weight[1]]*len(TTToSemiLeptonic)
weight_ST_t_channel_top_5f = [weight[2]]*len(ST_t_channel_top_5f)
weight_ST_t_channel_antitop_5f = [weight[3]]*len(ST_t_channel_antitop_5f)
weight_ST_tW_antitop_5f_inclusiveDecays = [weight[4]]*len(ST_tW_antitop_5f_inclusiveDecays)
weight_ST_tW_top_5f_inclusiveDecays = [weight[5]]*len(ST_tW_top_5f_inclusiveDecays)
weight_WWTo2L2Nu = [weight[6]]*len(WWTo2L2Nu)

# Plotting
list_of_colors = ['purple', 'blue', 'midnightblue', 'darkgreen', 'turquoise', 'orange', 'red']

fig1, (ax1, ax2) = plt.subplots(1, 2)

for sample in list_of_samples:
    ax1.hist(eval(sample + '_segmented')[0].m_vis, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
    ax1.hist(eval(sample + '_segmented')[1].m_vis, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
    ax1.hist(eval(sample + '_segmented')[2].m_vis, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
    ax1.hist(eval(sample + '_segmented')[3].m_vis, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
ax1.set_yscale('log')

for sample in list_of_samples:
    ax2.hist(eval(sample + '_segmented')[0].mjj, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
    ax2.hist(eval(sample + '_segmented')[1].mjj, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
    ax2.hist(eval(sample + '_segmented')[2].mjj, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
    ax2.hist(eval(sample + '_segmented')[3].mjj, bins=20, histtype='step', stacked=True, color=list_of_colors[list_of_samples.index(sample)])
ax2.set_yscale('log')

plt.show()
