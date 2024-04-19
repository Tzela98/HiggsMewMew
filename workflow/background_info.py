from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utility
import glob
from icecream import ic



def main():

    # Define a dict containing the newly calculated generator weights and luminosity. 
    # A generator weight equal to 1 means that the generator weight is not yet calculated.
    # Structure of the dict: {dataset: [weight, luminosity]}

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

    weight_data_dict = {
        '/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [0.6499288910928841, 247.8],
        '/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM': [0.6729741358064714, 6529.0],
        '/EWK_LLJJ_MLL-50_MJJ-120_TuneCP5_13TeV-madgraph-pythia8_dipole/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [1, 1.719],
        '/ST_t-channel_antitop_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [0.9945629173086801, 71.74],
        '/ST_t-channel_top_5f_InclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [1, 119.7	],
        '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [0.9919484915378955, 88.29],
        '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [0.9921590909090909, 365.34],
        '/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM': [0.996268761256754, 10.48],
        '/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM': [0.6600782199229979, 5.213],
        '/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [0.9978920895690334, 0.6008],
        '/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': [0.0008228, 0.998673]
    }

    df = pd.DataFrame.from_dict(weight_data_dict, orient='index', columns=['generator_weight', 'cross_section'])
    
    df['nicks'] = dataset_nicks
    
    log_path = '/work/ehettwer/KingMaker/data/logs/UL_MC_Backgrounds/Output'
    number_of_events = []
    
    for dataset in df.nicks:
        path_list = utility.open_multiple_paths([log_path + '/' + dataset + '/*.txt'])
        number_of_events.append(utility.parse_txt_files(path_list))

    ic(number_of_events)
    number_of_events_filtered = [list(filter(None, row)) for row in number_of_events]

    sum_of_events = [sum(number_of_events) for number_of_events in number_of_events_filtered]
    df['number_of_events'] = sum_of_events
    df = df.rename_axis('path').reset_index()
    df.set_index('nicks', inplace=True)

    df.to_csv('workflow/csv_files/background_info.csv')

if __name__ == '__main__':
    main()