import glob
import uproot
import pandas as pd
import numpy as np
from icecream import ic
import filters as filters



def open_to_dataframe(dataset):
    events = uproot.open(dataset)
    dataframe = events['ntuple'].arrays(["pt_1", "eta_1", "phi_1", "pt_2", "eta_2", "phi_2", "jpt_1", "jeta_1",
                                         "jphi_1", "jpt_2", "jeta_2", "jphi_2", "m_vis", "mjj", "pt_dijet", "pt_vis", 
                                         "njets", "puweight", "id_wgt_mu_1", "id_wgt_mu_2", "iso_wgt_mu_1", "iso_wgt_mu_2", "genWeight", "trg_single_mu24",
                                         'trg_sf'], library="pd")
    if dataframe.empty:
        raise ValueError("Dataset is empty!")
    return dataframe


def combined_dataframes(data: list):
    all_events = pd.DataFrame()
    for dataset in data:
        all_events = pd.concat([all_events, open_to_dataframe(dataset)])
    return all_events


def open_multiple_paths(paths: list):
    all_paths = []
    for path in paths:
        all_paths = all_paths + glob.glob(path, recursive=True)
    return sorted(all_paths)


DYJetsToLL_M_50_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/DYJetsToLL_M-50*/vbf/*.root'])
DYJetsToLL_M_100_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/DYJetsToLL_M-100*/vbf/*.root'])

EWK_LLJJ_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/EWK_LLJJ*/vbf/*.root'])

ST_t_top_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/ST_t-channel_top_5f*/vbf/*.root'])
ST_t_antitop_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/ST_t-channel_antitop_5f*/vbf/*.root'])
TTTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/TTTo2L2Nu*/vbf/*.root'])
TTToSemiLeptonic_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/TTToSemiLeptonic*/vbf/*.root'])

WWTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/WWTo2L2Nu*/vbf/*.root'])
WZTo3LNu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/WZTo3LNu*/vbf/*.root'])
ZZTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/UL_MC_Backgrounds/CROWNRun/2018/ZZTo2L2Nu*/vbf/*.root'])

signal_sim_path = open_multiple_paths(['/ceph/ehettwer/ntuples/signal_samples_mc_2018/CROWNRun/2018/VBFHToMuMu_M125*/vbf/*.root'])



dataset_nicks = ['DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole_RunIISummer20UL18NanoAODv9-106X',
                    'ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X']


data_paths = [DYJetsToLL_M_100_path, DYJetsToLL_M_50_path, EWK_LLJJ_path, ST_t_antitop_path, ST_t_top_path, TTTo2L2Nu_path, 
              TTToSemiLeptonic_path, WWTo2L2Nu_path, WZTo3LNu_path, ZZTo2L2Nu_path, signal_sim_path]

ic(data_paths)

# Loop through datasets and paths, combine DataFrames, do some more selection and save to CSV

for dataset_nick, data_path in zip(dataset_nicks, data_paths):
    # leading_jet_pt = 25, sub_leading_jet_pt = 35, threshold_mass = 400, threshold_rapidity = 2.5
    # trg_single_mu24 == 1 is the trigger variable
    df_combined = filters.selection_pipeline_trg(combined_dataframes(data_path), 25, 35, 400, 2.5)
    df_combined.to_csv(f'workflow/csv_files/{dataset_nick}.csv')
    print(f"Saved {dataset_nick}.csv")







