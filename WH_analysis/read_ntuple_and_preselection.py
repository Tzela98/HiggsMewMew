import glob
from sympy import N
from torch import empty
import uproot
import pandas as pd
import numpy as np
from icecream import ic
from tqdm import tqdm
import filters as filters


def open_to_dataframe(dataset):
    try:
        root_file = uproot.open(dataset)
        tree = root_file['ntuple']
        dataframe = tree.arrays(['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH',
                                 'deltaR_12', 'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H', 'q_1', 'q_2', 'q_3',
                                 'pt_1', 'pt_2', 'pt_3', 'nmuons', 'eta_1', 'eta_2', 'cosThetaStar12', 'cosThetaStar13', 'cosThetaStar23',
                                 'trg_sf', 'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'genWeight',
                                 'is_wh'], library="pd")
        return dataframe
    except KeyError as e:
        print(f"KeyError: {e} occurred while reading {dataset}. Skipping this file.")
        return None


def combined_dataframes(data: list):
    all_events = pd.DataFrame()
    for dataset in tqdm(data, desc="Combining DataFrames"):
        df = open_to_dataframe(dataset)
        if df is not None:
            all_events = pd.concat([all_events, df])
    return all_events


def open_multiple_paths(paths: list):
    all_paths = []
    for path in paths:
        all_paths = all_paths + glob.glob(path, recursive=True)
    return sorted(all_paths)


def df_segmentation(df, variable, threshold: tuple):
    df_segmented = (df[(df[variable] > threshold[0]) & (df[variable] < threshold[1])])
    return df_segmented


def filter_muons_from_csv(file_path):
    df = pd.read_csv(file_path)
    filtered_df = df[df['nmuons'] <= 3]
    return filtered_df


DYJetsToLL_M_50_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/DYJetsToLL_M-50*/*/*.root'])
DYJetsToLL_M_100_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/DYJetsToLL_M-100*/*/*.root'])

EWK_LLJJ_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/EWK_LLJJ*/*/*.root'])

ST_t_top_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/ST_t-channel_top_5f*/*/*.root'])
ST_t_antitop_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/ST_t-channel_antitop_5f*/*/*.root'])
TTTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/TTTo2L2Nu*/*/*.root'])
TTToSemiLeptonic_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/TTToSemiLeptonic*/*/*.root'])

WWTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/WWTo2L2Nu*/*/*.root'])
WZTo3LNu_mllmin_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/WZTo3LNu_mllmin0p1*/*/*.root'])
WZTo3LNu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/WZTo3LNu_TuneCP5*/*/*.root'])
ZZTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/ZZTo2L2Nu*/*/*.root'])
ZZTo4L_path = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/ZZTo4L*/*/*.root'])

signal_sim_path1 = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/WminusHToMuMu*/*/*.root'])
signal_sim_path2 = open_multiple_paths(['/ceph/ehettwer/ntuples/full_training_samples/CROWNRun/2018/WplusHToMuMu*/*/*.root'])



dataset_nicks = ['DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole_RunIISummer20UL18NanoAODv9-106X',
                    'ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X']


data_paths = [DYJetsToLL_M_100_path, DYJetsToLL_M_50_path, EWK_LLJJ_path, ST_t_antitop_path, ST_t_top_path, TTTo2L2Nu_path, 
              TTToSemiLeptonic_path, WWTo2L2Nu_path, WZTo3LNu_mllmin_path ,WZTo3LNu_path, ZZTo2L2Nu_path, ZZTo4L_path, 
              signal_sim_path1, signal_sim_path2]

ic(data_paths)

# Loop through datasets and paths, combine DataFrames, do some more selection and save to CSV

for dataset_nick, data_path in zip(dataset_nicks, data_paths):
    # leading_jet_pt = 25, sub_leading_jet_pt = 35, threshold_mass = 400, threshold_rapidity = 2.5
    # trg_single_mu24 == 1 is the trigger variable
    print('-----------------------------------')
    df_combined = combined_dataframes(data_path)
    print('dataset_nick:', dataset_nick)
    print('length of df_combined:', len(df_combined))
    df_signal_region = df_segmentation(df_combined, 'm_H', (115, 135))
    print('number of working events: ', len(df_signal_region))
    df_signal_region.to_csv(f'/work/ehettwer/HiggsMewMew/data/including_genWeight/{dataset_nick}.csv')
    print(f"Saved {dataset_nick}.csv")







