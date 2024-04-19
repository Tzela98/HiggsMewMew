import glob
import uproot
import pandas as pd
import numpy as np
import filters as filters


def open_to_dataframe(dataset):
    events = uproot.open(dataset)
    dataframe = events['ntuple'].arrays(["pt_1", "eta_1", "phi_1", "pt_2", "eta_2", "phi_2", "jpt_1", "jeta_1",
                                         "jphi_1", "jpt_2", "jeta_2", "jphi_2", "m_vis", "mjj", "pt_dijet", "pt_vis", 
                                         "njets", "trg_single_mu24"], 
                                         library="pd")
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


def main():

    path = ["/ceph/ehettwer/ntuples/single_muon_data_2018_trgvar/CROWNRun/2018/*/vbf/*.root"]
    all_paths = open_multiple_paths(path)

    all_data_2018 = filters.selection_pipeline(combined_dataframes(all_paths), 25, 35, 400, 2.5)
    all_data_2018.to_csv('workflow/csv_files/no_trg_selection/single_muon_data_2018.csv')

if __name__ == '__main__':
    main()


