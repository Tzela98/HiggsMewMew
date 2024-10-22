import glob
from torch import empty
import uproot
import pandas as pd
import numpy as np
from icecream import ic
from tqdm import tqdm
import filters as filters
import os


def open_data_to_dataframe(dataset):
    try:
        root_file = uproot.open(dataset)
        tree = root_file['ntuple']
        dataframe = tree.arrays(['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH',
                                 'deltaR_12', 'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H', 'q_1', 'q_2', 'q_3',
                                 'pt_1', 'pt_2', 'pt_3', 'nmuons', 'eta_1', 'eta_2', 'cosThetaStar12', 'cosThetaStar13', 'cosThetaStar23',
                                 'mvaTTH_1', 'mvaTTH_2', 'mvaTTH_3'], library="pd")
        return dataframe
    except KeyError as e:
        print(f"KeyError: {e} occurred while reading {dataset}. Skipping this file.")
        return None


def open_MC_to_dataframe(dataset):
    try:
        root_file = uproot.open(dataset)
        tree = root_file['ntuple']
        dataframe = tree.arrays(['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH',
                                 'deltaR_12', 'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H', 'q_1', 'q_2', 'q_3',
                                 'pt_1', 'pt_2', 'pt_3', 'nmuons', 'eta_1', 'eta_2', 'cosThetaStar12', 'cosThetaStar13', 'cosThetaStar23',
                                 'trg_sf', 'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'genWeight',
                                 'mvaTTH_1', 'mvaTTH_2', 'mvaTTH_3', 'is_wh'], library="pd")
        return dataframe
    except KeyError as e:
        print(f"KeyError: {e} occurred while reading {dataset}. Skipping this file.")
        return None
    

def combined_dataframes_MC(data: list):
    all_events = pd.DataFrame()
    for dataset in tqdm(data, desc="Combining DataFrames"):
        df = open_MC_to_dataframe(dataset)
        if df is not None:
            all_events = pd.concat([all_events, df])
    return all_events


def combined_dataframes_data(data: list):
    all_events = pd.DataFrame()
    for dataset in tqdm(data, desc="Combining DataFrames"):
        df = open_data_to_dataframe(dataset)
        if df is not None:
            all_events = pd.concat([all_events, df])
    return all_events


def open_multiple_paths(paths: list):
    all_paths = []
    for path in paths:
        all_paths = all_paths + glob.glob(path, recursive=True)
    return sorted(all_paths)


def df_segmentation(df, variable, threshold: tuple):
    if df is None:
        print('Dataframe is empty. Skipping segmentation...')
        return None
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame. Skipping segmentation...")
        return None
    df_segmented = df[(df[variable] > threshold[0]) & (df[variable] < threshold[1])]
    return df_segmented


def mva_score_cut(df, mva_score_1, mva_score_2, mva_score_3, threshold):
    if df is None:
        print('Dataframe is empty. Skipping MVA score cut...')
        return None
    df_mva_score = (df[(df[mva_score_1] > threshold) & (df[mva_score_2] > threshold) & (df[mva_score_3] > threshold)])
    return df_mva_score
    


DYJetsToLL_M_50_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/DYJetsToLL_M-50*/*/*.root'])
DYJetsToLL_M_100_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/DYJetsToLL_M-100*/*/*.root'])

TTTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/TTTo2L2Nu*/*/*.root'])
TTToSemiLeptonic_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/TTToSemiLeptonic*/*/*.root'])

WZTo3LNu_mllmin_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/WZTo3LNu_mllmin0p1*/*/*.root'])
WZTo3LNu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/WZTo3LNu_TuneCP5*/*/*.root'])
ZZTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/ZZTo2L2Nu*/*/*.root'])
ZZTo4L_path = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/ZZTo4L*/*/*.root'])

Wminus = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/WminusHToMuMu*/*/*.root'])
Wplus = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/WplusHToMuMu*/*/*.root'])



MC_nicks = ['DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
                    'WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X']


MC_paths = [DYJetsToLL_M_100_path, DYJetsToLL_M_50_path, TTTo2L2Nu_path, 
              TTToSemiLeptonic_path, WZTo3LNu_mllmin_path ,WZTo3LNu_path, ZZTo2L2Nu_path, ZZTo4L_path, 
              Wminus, Wplus]

ic(MC_paths)

save_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print('Directory does not exist. Creating directory...')
    print('Created directory:', save_path)


for MC_nick, MC_path in zip(MC_nicks, MC_paths):
    print('-----------------------------------')
    df_combined = combined_dataframes_MC(MC_path)
    print('dataset_nick:', MC_nick)
    print('number of events before mass cut', len(df_combined))
    df_signal_region = df_segmentation(df_combined, 'm_H', (110, 150))
    df_signal_region = mva_score_cut(df_signal_region, 'mvaTTH_1', 'mvaTTH_2', 'mvaTTH_3', 0.4)
    print('number of events after mass cut: ', len(df_signal_region))
    df_signal_region.to_csv(save_path + f'{MC_nick}.csv')
    print(f"Saved {MC_nick}.csv")


signal_paths = open_multiple_paths(['/ceph/ehettwer/ntuples/final_run2/SingleMuon*/*/*.root',])
ic(signal_paths)

data = combined_dataframes_data(signal_paths)
data_mass_cut = df_segmentation(data, 'm_H', (110, 150))
data_mass_cut = mva_score_cut(data_mass_cut, 'mvaTTH_1', 'mvaTTH_2', 'mvaTTH_3', 0.4)
print('number of events after mass cut: ', len(data_mass_cut))

data_mass_cut.to_csv(save_path + 'SingleMuonData.csv')