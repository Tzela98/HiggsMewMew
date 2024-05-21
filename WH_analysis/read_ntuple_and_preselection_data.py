import glob
import uproot
import pandas as pd
import numpy as np
import filters as filters


def open_to_dataframe(dataset):
    events = uproot.open(dataset)
    dataframe = events['ntuple'].arrays(['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH',
                                        'deltaR_12', 'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H',
                                        'pt_1', 'pt_2', 'pt_3', 'nmuons', 'eta_1', 'eta_2',
                                        'trg_sf', 'id_wgt_mu_1', 'id_wgt_mu_2', 'iso_wgt_mu_1', 'iso_wgt_mu_2', 
                                        'is_wh'], library="pd")
    if dataframe.empty:
        raise ValueError("Dataset is empty.")
    return dataframe


def combined_dataframes(data):
    all_events = pd.DataFrame()
    for dataset in data:
        all_events = pd.concat([all_events, open_to_dataframe(dataset)])
    return all_events


def open_multiple_paths(paths: list):
    all_paths = []
    for path in paths:
        all_paths = all_paths + glob.glob(path, recursive=True)
    return sorted(all_paths)


def df_segmentation(df, variable, threshold: tuple):
    df_segmented = (df[(df[variable] > threshold[0]) & (df[variable] < threshold[1])])
    return df_segmented


def main():

    path = ["/ceph/ehettwer/ntuples/WH_Background_Sim/CROWNRun/2018/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X/*/*.root"]
    all_paths = open_multiple_paths(path)

    all_data = combined_dataframes(all_paths)
    all_data_signal_region = df_segmentation(all_data, 'm_H', (115, 135))
    print('length of the dataset:')
    print(len(all_data_signal_region))
    all_data_signal_region.to_csv('/ceph/ehettwer/working_data/test_statistics/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', index=False)

if __name__ == '__main__':
    main()

