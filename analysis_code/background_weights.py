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

    weight_data_dict = {
        '/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext2-v1/NANOAODSIM': [0.6776847030381536, 6225.4],
        '/DYJetsToLL_M-105To160_TuneCP5_PSweights_13TeV-amcatnloFXFX-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [0.6539836983160123, 47.12],
        '/EWK_LLJJ_MLL_105-160_SM_5f_LO_TuneCH3_13TeV-madgraph-herwig7_corrected/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [0.998510425351675, 0.078],
        '/ST_t-channel_antitop_5f_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [1, 35.9],
        '/ST_t-channel_top_5f_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [0.9942740760163667, 35.9],
        '/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM': [1, 80.95],
        '/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM': [0.9954065430297979, 136.02],
        '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [1, 358.57],
        '/WWTo2L2Nu_NNPDF31_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [0.9961806441634767, 12.178],
        '/WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [1, 6.321],
        '/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM': [0.9987058907737456, 0.601],
        '/ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21-v1/NANOAODSIM': [0.6385277251074166, 3.696],
        '/ZZTo4L_TuneCP5_13TeV_powheg_pythia8/RunIIAutumn18NanoAODv7-Nano02Apr2020_102X_upgrade2018_realistic_v21_ext1-v1/NANOAODSIM': [0.9898865453893182, 1.325]
    }

    df = pd.DataFrame.from_dict(weight_data_dict, orient='index', columns=['generator_weight', 'cross_section'])
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
    df['nicks'] = dataset_nicks
    
    log_path = '/work/ehettwer/KingMaker/data/logs/backgrounds_2018_new/Output'
    number_of_events = []
    
    for dataset in df.nicks:
        path_list = utility.open_multiple_paths([log_path + '/' + dataset + '/*.txt'])
        number_of_events.append(utility.parse_txt_files(path_list))
    
    sum_of_events = [sum(number_of_events) for number_of_events in number_of_events]
    df['number_of_events'] = sum_of_events
    df = df.rename_axis('path').reset_index()
    df.set_index('nicks', inplace=True)

    df.to_csv('data_csv/background_info.csv')

if __name__ == '__main__':
    main()