import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak


def filter_for_jet_mass(df, threshold_value):
    return df[df.mjj > threshold_value]


def filter_pseudo_rapidity_separation(df, threshold_value):
    return df[np.abs(df.jeta_1 - df.jeta_2) > threshold_value]


def region_cut(df, varible, threshold = [0, 10000]):
    return df[(df.variable > threshold[0]) & (df.variable < threshold[1])]


DYJetsToLL_M_50 = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/DYJetsToLL_M-50_mc_2018.csv')
TTToSemiLeptonic = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/TTToSemiLeptonic_mc_2018.csv')
ST_t_channel_top_5f = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_top_5f_mc_2018.csv')
ST_t_channel_antitop_5f = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_t-channel_antitop_5f_mc_2018.csv')
ST_tW_antitop_5f_inclusiveDecays = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_tW_antitop_5f_inclusiveDecays_mc_2018.csv')
ST_tW_top_5f_inclusiveDecays = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/ST_tW_top_5f_inclusiveDecays_mc_2018.csv')
WWTo2L2Nu = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/WWTo2L2Nu_mc_2018.csv')

mc_2018_df = pd.read_csv('/work/ehettwer/HiggsMewMew/data_csv/mc_2018.csv').transpose()
mc_2018_df.columns = mc_2018_df.iloc[0]
mc_2018_df = mc_2018_df[1:]

DYJetsToLL_M_50 = filter_pseudo_rapidity_separation(filter_for_jet_mass(DYJetsToLL_M_50, 400), 2.5)
TTToSemiLeptonic = filter_pseudo_rapidity_separation(filter_for_jet_mass(TTToSemiLeptonic, 400), 2.5)
ST_t_channel_top_5f = filter_pseudo_rapidity_separation(filter_for_jet_mass(ST_t_channel_top_5f, 400), 2.5)
ST_t_channel_antitop_5f = filter_pseudo_rapidity_separation(filter_for_jet_mass(ST_t_channel_antitop_5f, 400), 2.5)
ST_tW_antitop_5f_inclusiveDecays = filter_pseudo_rapidity_separation(filter_for_jet_mass(ST_tW_antitop_5f_inclusiveDecays, 400), 2.5)
ST_tW_top_5f_inclusiveDecays = filter_pseudo_rapidity_separation(filter_for_jet_mass(ST_tW_top_5f_inclusiveDecays, 400), 2.5)
WWTo2L2Nu = filter_pseudo_rapidity_separation(filter_for_jet_mass(WWTo2L2Nu, 400), 2.5)

datasamples = ak.Array([DYJetsToLL_M_50, TTToSemiLeptonic, ST_t_channel_top_5f, ST_t_channel_antitop_5f, ST_tW_antitop_5f_inclusiveDecays, 
                        ST_tW_top_5f_inclusiveDecays, WWTo2L2Nu])
length_datasamples = [len(DYJetsToLL_M_50), len(TTToSemiLeptonic), len(ST_t_channel_top_5f), len(ST_t_channel_antitop_5f), 
                               len(ST_tW_antitop_5f_inclusiveDecays), len(ST_tW_top_5f_inclusiveDecays), len(WWTo2L2Nu)]


# two muon/jet variables
m_vis_complete = [DYJetsToLL_M_50.m_vis, TTToSemiLeptonic.m_vis, ST_t_channel_top_5f.m_vis, ST_t_channel_antitop_5f.m_vis, 
                  ST_tW_antitop_5f_inclusiveDecays.m_vis, ST_tW_top_5f_inclusiveDecays.m_vis, WWTo2L2Nu.m_vis]
mjj_complete = [DYJetsToLL_M_50.mjj, TTToSemiLeptonic.mjj, ST_t_channel_top_5f.mjj, ST_t_channel_antitop_5f.mjj,
                ST_tW_antitop_5f_inclusiveDecays.mjj, ST_tW_top_5f_inclusiveDecays.mjj, WWTo2L2Nu.mjj]


# muon variables
pt_1_complete = [DYJetsToLL_M_50.pt_1, TTToSemiLeptonic.pt_1, ST_t_channel_top_5f.pt_1, ST_t_channel_antitop_5f.pt_1, 
                 ST_tW_antitop_5f_inclusiveDecays.pt_1, ST_tW_top_5f_inclusiveDecays.pt_1, WWTo2L2Nu.pt_1]
pt_2_complete = [DYJetsToLL_M_50.pt_2, TTToSemiLeptonic.pt_2, ST_t_channel_top_5f.pt_2, ST_t_channel_antitop_5f.pt_2, 
                 ST_tW_antitop_5f_inclusiveDecays.pt_2, ST_tW_top_5f_inclusiveDecays.pt_2, WWTo2L2Nu.pt_2]
eta_1_complete = [DYJetsToLL_M_50.eta_1, TTToSemiLeptonic.eta_1, ST_t_channel_top_5f.eta_1, ST_t_channel_antitop_5f.eta_1, 
                  ST_tW_antitop_5f_inclusiveDecays.eta_1, ST_tW_top_5f_inclusiveDecays.eta_1, WWTo2L2Nu.eta_1]
eta_2_complete = [DYJetsToLL_M_50.eta_2, TTToSemiLeptonic.eta_2, ST_t_channel_top_5f.eta_2, ST_t_channel_antitop_5f.eta_2, 
                  ST_tW_antitop_5f_inclusiveDecays.eta_2, ST_tW_top_5f_inclusiveDecays.eta_2, WWTo2L2Nu.eta_2]
phi_1_complete = [DYJetsToLL_M_50.phi_1, TTToSemiLeptonic.phi_1, ST_t_channel_top_5f.phi_1, ST_t_channel_antitop_5f.phi_1, 
                  ST_tW_antitop_5f_inclusiveDecays.phi_1, ST_tW_top_5f_inclusiveDecays.phi_1, WWTo2L2Nu.phi_1]
phi_2_complete = [DYJetsToLL_M_50.phi_2, TTToSemiLeptonic.phi_2, ST_t_channel_top_5f.phi_2, ST_t_channel_antitop_5f.phi_2, 
                  ST_tW_antitop_5f_inclusiveDecays.phi_2, ST_tW_top_5f_inclusiveDecays.phi_2, WWTo2L2Nu.phi_2]

# jet variables
jpt_1_complete = [DYJetsToLL_M_50.jpt_1, TTToSemiLeptonic.jpt_1, ST_t_channel_top_5f.jpt_1, ST_t_channel_antitop_5f.jpt_1, 
                    ST_tW_antitop_5f_inclusiveDecays.jpt_1, ST_tW_top_5f_inclusiveDecays.jpt_1, WWTo2L2Nu.jpt_1]
jpt_2_complete = [DYJetsToLL_M_50.jpt_2, TTToSemiLeptonic.jpt_2, ST_t_channel_top_5f.jpt_2, ST_t_channel_antitop_5f.jpt_2,
                    ST_tW_antitop_5f_inclusiveDecays.jpt_2, ST_tW_top_5f_inclusiveDecays.jpt_2, WWTo2L2Nu.jpt_2]
jeta_1_complete = [DYJetsToLL_M_50.jeta_1, TTToSemiLeptonic.jeta_1, ST_t_channel_top_5f.jeta_1, ST_t_channel_antitop_5f.jeta_1,
                    ST_tW_antitop_5f_inclusiveDecays.jeta_1, ST_tW_top_5f_inclusiveDecays.jeta_1, WWTo2L2Nu.jeta_1]
jeta_2_complete = [DYJetsToLL_M_50.jeta_2, TTToSemiLeptonic.jeta_2, ST_t_channel_top_5f.jeta_2, ST_t_channel_antitop_5f.jeta_2,
                    ST_tW_antitop_5f_inclusiveDecays.jeta_2, ST_tW_top_5f_inclusiveDecays.jeta_2, WWTo2L2Nu.jeta_2]
jphi_1_complete = [DYJetsToLL_M_50.jphi_1, TTToSemiLeptonic.jphi_1, ST_t_channel_top_5f.jphi_1, ST_t_channel_antitop_5f.jphi_1,
                    ST_tW_antitop_5f_inclusiveDecays.jphi_1, ST_tW_top_5f_inclusiveDecays.jphi_1, WWTo2L2Nu.jphi_1]
jphi_2_complete = [DYJetsToLL_M_50.jphi_2, TTToSemiLeptonic.jphi_2, ST_t_channel_top_5f.jphi_2, ST_t_channel_antitop_5f.jphi_2,
                    ST_tW_antitop_5f_inclusiveDecays.jphi_2, ST_tW_top_5f_inclusiveDecays.jphi_2, WWTo2L2Nu.jphi_2]

weights = np.array(mc_2018_df.weights)
weight = []
for i in range(len(length_datasamples)):
    weight.append([weights[i]] * length_datasamples[i])

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(m_vis_complete, bins=50, range=(70, 150), weights = weight, histtype='step', stacked=True)
ax1.set_title('m_vis')
ax1.set_yscale('log')
ax2.hist(mjj_complete, bins=50, range=(0,1500), weights = weight, histtype='step', stacked=True)
ax2.set_title('mjj')
ax2.set_yscale('log')

fig1.savefig('plots/m_vis_mjj.png')

fig2, axs2 = plt.subplots(2, 3)

axs2[0, 0].hist(pt_1_complete, bins=40, range=(0, 200), weights = weight, histtype='step', stacked=True)
axs2[0, 0].set_title('pt_1')
axs2[0, 0].set_yscale('log')
axs2[1, 0].hist(pt_2_complete, bins=40, range=(0, 200), weights = weight, histtype='step', stacked=True)
axs2[1, 0].set_title('pt_2')
axs2[1, 0].set_yscale('log')
axs2[0, 1].hist(eta_1_complete, bins=40, weights = weight, histtype='step', stacked=True)
axs2[0, 1].set_title('eta_1')
axs2[1, 1].hist(eta_2_complete, bins=40, weights = weight, histtype='step', stacked=True)
axs2[1, 1].set_title('eta_2')
axs2[0, 2].hist(phi_1_complete, bins=40, weights = weight, range=(-3.5, 3.5), histtype='step', stacked=True)
axs2[0, 2].set_title('phi_1')
axs2[1, 2].hist(phi_2_complete, bins=40, weights = weight, range=(-3.5, 3.5), histtype='step', stacked=True)
axs2[1, 2].set_title('phi_2')

fig2.savefig('plots/pt_eta_phi.png')

fig3, axs3 = plt.subplots(2, 3)

axs3[0, 0].hist(jpt_1_complete, bins=40, weights = weight, range=(0, 400), histtype='step', stacked=True)
axs3[0, 0].set_title('jpt_1')
axs3[0, 0].set_yscale('log')
axs3[1, 0].hist(jpt_2_complete, bins=40, weights = weight, range=(0, 200), histtype='step', stacked=True)
axs3[1, 0].set_title('jpt_2')
axs3[1, 0].set_yscale('log')
axs3[0, 1].hist(jeta_1_complete, bins=40, weights = weight, histtype='step', stacked=True)
axs3[0, 1].set_title('jeta_1')
axs3[1, 1].hist(jeta_2_complete, bins=40, weights = weight, histtype='step', stacked=True)
axs3[1, 1].set_title('jeta_2')
axs3[0, 2].hist(jphi_1_complete, bins=40, weights = weight, range=(-3.5, 3.5), histtype='step', stacked=True)
axs3[0, 2].set_title('jphi_1')
axs3[1, 2].hist(jphi_2_complete, bins=40, weights = weight, range=(-3.5, 3.5), histtype='step', stacked=True)
axs3[1, 2].set_title('jphi_2')

fig3.savefig('plots/jpt_jeta_jphi.png')

plt.show()
