import glob
from sympy import N
from torch import empty
import uproot
import pandas as pd
import numpy as np
from icecream import ic
import filters as filters



def open_to_dataframe(dataset):
    try:
        root_file = uproot.open(dataset)
        tree = root_file['ntuple']
        dataframe = tree.arrays(['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH',
                                 'deltaR_12', 'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H',
                                 'pt_1', 'pt_2', 'pt_3', 'trg_sf', 'id_wgt_mu_1', 'id_wgt_mu_2', 
                                 'iso_wgt_mu_1', 'iso_wgt_mu_2'], library="pd")
        return dataframe
    except KeyError as e:
        print(f"KeyError: {e} occurred while reading {dataset}. Skipping this file.")
        return None


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


DYJetsToLL_M_50_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/DYJetsToLL_M-50*/wh_mmm/*.root'])
DYJetsToLL_M_100_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/DYJetsToLL_M-100*/wh_mmm/*.root'])

EWK_LLJJ_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/EWK_LLJJ*/wh_mmm/*.root'])

ST_t_top_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/ST_t-channel_top_5f*/wh_mmm/*.root'])
ST_t_antitop_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/ST_t-channel_antitop_5f*/wh_mmm/*.root'])
TTTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/TTTo2L2Nu*/wh_mmm/*.root'])
TTToSemiLeptonic_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/TTToSemiLeptonic*/wh_mmm/*.root'])

WWTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/WWTo2L2Nu*/wh_mmm/*.root'])
WZTo3LNu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/WZTo3LNu*/wh_mmm/*.root'])
ZZTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/ZZTo2L2Nu*/wh_mmm/*.root'])
ZZTo4L_path = open_multiple_paths(['/ceph/ehettwer/ntuples/WH_full/CROWNRun/2018/ZZTo4L*/wh_mmm/*.root'])

#signal_sim_path = open_multiple_paths(['/ceph/ehettwer/ntuples/signal_samples_mc_2018/CROWNRun/2018/VBFHToMuMu_M125*/vbf/*.root'])



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
                    'ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X']


data_paths = [DYJetsToLL_M_100_path, DYJetsToLL_M_50_path, EWK_LLJJ_path, ST_t_antitop_path, ST_t_top_path, TTTo2L2Nu_path, 
              TTToSemiLeptonic_path, WWTo2L2Nu_path, WZTo3LNu_path, ZZTo2L2Nu_path]

ic(data_paths)

# Loop through datasets and paths, combine DataFrames, do some more selection and save to CSV

for dataset_nick, data_path in zip(dataset_nicks, data_paths):
    # leading_jet_pt = 25, sub_leading_jet_pt = 35, threshold_mass = 400, threshold_rapidity = 2.5
    # trg_single_mu24 == 1 is the trigger variable
    df_combined = combined_dataframes(data_path)
    df_combined.to_csv(f'/ceph/ehettwer/working_data/{dataset_nick}.csv')
    print(f"Saved {dataset_nick}.csv")







