import glob
import uproot
import pandas as pd
import numpy as np
from icecream import ic
import filters as filter


def open_to_dataframe(dataset):
    events = uproot.open(dataset)
    dataframe = events['ntuple'].arrays(["pt_1", "eta_1", "phi_1", "pt_2", "eta_2", "phi_2", "jpt_1", "jeta_1",
                                         "jphi_1", "jpt_2", "jeta_2", "jphi_2", "m_vis", "mjj", "pt_dijet", "pt_vis", 
                                         "njets", "puweight", "id_wgt_mu_1", "id_wgt_mu_2", "iso_wgt_mu_1", "iso_wgt_mu_2", "genWeight"], library="pd")
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


data_path = open_multiple_paths(['/ceph/ehettwer/ntuples/single_muon_data_2018_new/CROWNRun/2018/*/vbf/*.root'])

DYJetsToLL_M_50_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/DYJets*/vbf/*.root'])
DYJetsToLL_M105To160_path = open_multiple_paths(['/ceph/ehettwer/ntuples/dyjets_105_160_new/CROWNRun/2018/DY*/vbf/*.root'])

EWK_LLJJ_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/EWK_LLJJ*/vbf/*.root'])

top_ST_t_channel_antitop_5f_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/top_ST_t-channel_antitop*/vbf/*.root'])
top_ST_t_channel_top_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/top_ST_t-channel_top*/vbf/*.root'])
top_ST_tW_antitop_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/top_ST_tW_antitop*/vbf/*.root'])
top_ST_tW_top_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/top_ST_tW_top*/vbf/*.root'])
top_TTToSemiLeptonic_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/top_TTToSemiLeptonic*/vbf/*.root']) 

diboson_WWTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/diboson_WWTo2L2Nu*/vbf/*.root'])
diboson_WZTo2L2Q_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/diboson_WZTo2L2Q*/vbf/*.root'])
diboson_ZZTo2L2Nu_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/diboson_ZZTo2L2Nu*/vbf/*.root'])
diboson_ZZTo2L2Q_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/diboson_ZZTo2L2Q*/vbf/*.root'])
diboson_ZZTo4L_path = open_multiple_paths(['/ceph/ehettwer/ntuples/backgrounds_2018/CROWNRun/2018/diboson_ZZTo4L*/vbf/*.root'])


dataset_nicks = ['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext2',
                        'DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1',
                        'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1',
                        'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1',
                        'ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020',
                        'ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext1']


data_paths = [DYJetsToLL_M_50_path, DYJetsToLL_M105To160_path, EWK_LLJJ_path, top_ST_t_channel_antitop_5f_path,
              top_ST_t_channel_top_path, top_ST_tW_antitop_path, top_ST_tW_top_path,
              top_TTToSemiLeptonic_path, diboson_WWTo2L2Nu_path, diboson_WZTo2L2Q_path,
              diboson_ZZTo2L2Nu_path, diboson_ZZTo2L2Q_path, diboson_ZZTo4L_path]

# Loop through datasets and paths, combine DataFrames, and save to CSV
for dataset_nick, data_path in zip(dataset_nicks, data_paths):
    df_combined = filter.selection_pipeline(combined_dataframes(data_path), 25, 35, 400, 2.5)
    df_combined.to_csv(f'data_csv/data_preselected/{dataset_nick}.csv')
    print(f"Saved {dataset_nick}.csv")







