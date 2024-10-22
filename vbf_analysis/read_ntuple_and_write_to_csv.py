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
        dataframe = tree.arrays(['eta_1', 'eta_2', 'jeta_1', 'jeta_2', 'jphi_1', 'jphi_2', 'jpt_1', 'jpt_2', 'm_vis', 'mjj', 'njets',
                                 'nmuons', 'phi_1', 'phi_2', 'pt_1', 'pt_2', 'pt_dijet', 'pt_vis',
                                 'mvaTTH_1', 'mvaTTH_2'], library="pd")
        return dataframe
    except KeyError as e:
        print(f"KeyError: {e} occurred while reading {dataset}. Skipping this file.")
        return None


def open_MC_to_dataframe(dataset):
    try:
        root_file = uproot.open(dataset)
        tree = root_file['ntuple']
        dataframe = tree.arrays(['eta_1', 'eta_2', 'jeta_1', 'jeta_2', 'jphi_1', 'jphi_2', 'jpt_1', 'jpt_2', 'm_vis', 'mjj', 'njets',
                                 'nmuons', 'phi_1', 'phi_2', 'pt_1', 'pt_2', 'pt_dijet', 'pt_vis',
                                 'trg_sf', 'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 'genWeight',
                                 'mvaTTH_1', 'mvaTTH_2', 'is_vbf', 'is_dyjets', 'is_ttbar'], library="pd")
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


def mva_score_cut(df, mva_score_1, mva_score_2, threshold):
    if df is None:
        print('Dataframe is empty. Skipping MVA score cut...')
        return None
    df_mva_score = (df[(df[mva_score_1] > threshold) & (df[mva_score_2] > threshold)])
    return df_mva_score
    


DYJetsToLL_M_50_path = open_multiple_paths(['/ceph/ehettwer/ntuples/vbf_all_samples/DYJetsToLL_M-50*/*/*.root'])
DYJetsToLL_M_100_path = open_multiple_paths(['/ceph/ehettwer/ntuples/vbf_all_samples/DYJetsToLL_M-100*/*/*.root'])
top_path =  open_multiple_paths(['/ceph/ehettwer/ntuples/vbf_all_samples/TTTo2L2Nu*/*/*.root'])

vbf = open_multiple_paths(['/ceph/ehettwer/ntuples/vbf_all_samples/VBF*/*/*.root'])
gluglu = open_multiple_paths(['/ceph/ehettwer/ntuples/vbf_all_samples/GluGlu*/*/*.root'])



MC_nicks = ['DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
            'DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X',
            'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
            'VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
            'GluGluHToMuMu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X']


MC_paths = [DYJetsToLL_M_100_path, DYJetsToLL_M_50_path, top_path, vbf, gluglu]

ic(MC_paths)

save_path = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_tight_cut/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print('Directory does not exist. Creating directory...')
    print('Created directory:', save_path)

mass_cut = (115, 135)

for MC_nick, MC_path in zip(MC_nicks, MC_paths):
    print('-----------------------------------')
    df_combined = combined_dataframes_MC(MC_path)
    print('dataset_nick:', MC_nick)
    print('number of events before mass cut', len(df_combined))
    df_signal_region = df_segmentation(df_combined, 'm_vis', mass_cut)
    df_signal_region = mva_score_cut(df_signal_region, 'mvaTTH_1', 'mvaTTH_2', 0.4)
    print('number of events after mass cut: ', len(df_signal_region))

    if MC_nick == 'GluGluHToMuMu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X':
        df_signal_region['is_gluglu'] = 1

    df_signal_region.to_csv(save_path + f'{MC_nick}.csv')
    print(f"Saved {MC_nick}.csv")


signal_paths = open_multiple_paths(['/ceph/ehettwer/ntuples/vbf_all_samples/SingleMuon*/*/*.root',])
ic(signal_paths)

data = combined_dataframes_data(signal_paths)
data_mass_cut = df_segmentation(data, 'm_vis', mass_cut)
data_mass_cut = mva_score_cut(data_mass_cut, 'mvaTTH_1', 'mvaTTH_2', 0.4)
print('number of events after mass cut: ', len(data_mass_cut))

data_mass_cut.to_csv(save_path + 'SingleMuonData.csv')