import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
import os

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


def calculate_weights(dataset, cross_section, generator_weight, number_of_events, lumi=59.74 * 1000):
    id_iso_wgt = dataset['id_wgt_mu_1'] * dataset['iso_wgt_mu_1'] * dataset['id_wgt_mu_2'] * dataset['iso_wgt_mu_2']
    acceptance = dataset['genWeight'] / (abs(dataset['genWeight']) * generator_weight * number_of_events)
    weight = id_iso_wgt * acceptance * lumi * cross_section
    
    # Insert the new column 'weights' before the last column
    last_col_idx = len(dataset.columns) - 1
    dataset.insert(last_col_idx, 'weights', weight)
    
    return dataset


def main():
    base_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_tight_cut/'

    Wplus_nick = 'WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    Wminus_nick = 'WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    WZTo3LNu_TuneCP5_nick = 'WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    WZTo3LNu_mllmin0p1_nick = 'WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    ZZTo4L_nick = 'ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    DYJetsToLL_nick = 'DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    TTTo2L2Nu_nick ='TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    Wplus = pd.read_csv(base_path + Wplus_nick)
    Wplus['type'] = 'signal'
    Wminus = pd.read_csv(base_path + Wminus_nick)
    Wminus['type'] = 'signal'
    WZTo3LNu_TuneCP5 = pd.read_csv(base_path + WZTo3LNu_TuneCP5_nick)
    WZTo3LNu_TuneCP5['type'] = 'WZ'
    WZTo3LNu_mllmin0p1 = pd.read_csv(base_path + WZTo3LNu_mllmin0p1_nick)
    WZTo3LNu_mllmin0p1['type'] = 'WZ'
    ZZTo4L = pd.read_csv(base_path + ZZTo4L_nick)
    ZZTo4L['type'] = 'ZZ'
    DYJetsToLL = pd.read_csv(base_path + DYJetsToLL_nick)
    DYJetsToLL['type'] = 'DY'
    TTTo2L2Nu = pd.read_csv(base_path + TTTo2L2Nu_nick)
    TTTo2L2Nu['type'] = 'Top'

    cross_section_Wplus = 0.0001858
    generator_weight_Wplus = 0.9447331670822943
    number_of_events_Wplus = 599000


    cross_section_Wminus = 0.0001164
    generator_weight_WminusHToMuMu = 0.5412
    number_of_events_Wminus = 600000

    cross_section_WZTo3LNu_mllmin0p1 = 62.78
    generator_weight_WZTo3LNu_mllmin0p1 = 0.9253359049173302
    number_of_events_WZTo3LNu_mllmin0p1 = 89270000

    cross_section_WZTo3LNu_TuneCP5_ = 5.213
    generator_weight_WZTo3LNu_TuneCP5 = 0.6600782199229979
    number_of_events_WZTo3LNu_TuneCP5 = 9821283

    cross_section_ZZTo4L = 1.325 
    generator_weight_ZZTo4L = 0.9898860798438346
    number_of_events_ZZTo4L = 98488000

    cross_section_DYJetsToLL = 247.8
    generator_weight_DYJetsToLL = 0.6499288910928841
    number_of_events_DYJetsToLL = 219326954

    cross_section_TTTo2L2Nu = 88.51
    generator_weight_TTTo2L2Nu = 1
    number_of_events_TTTo2L2Nu = 145020000


    lumi = 59.74 * 1000

    save_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_tight_cut_weights_normed/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Directory does not exist. Creating directory...')
        print('Created directory:', save_path)

    Wplus = calculate_weights(Wplus, cross_section_Wplus, generator_weight_Wplus, number_of_events_Wplus, lumi)
    Wminus = calculate_weights(Wminus, cross_section_Wminus, generator_weight_WminusHToMuMu, number_of_events_Wminus, lumi)
    WZTo3LNu_TuneCP5 = calculate_weights(WZTo3LNu_TuneCP5, cross_section_WZTo3LNu_TuneCP5_, generator_weight_WZTo3LNu_TuneCP5, number_of_events_WZTo3LNu_TuneCP5, lumi)
    WZTo3LNu_mllmin0p1 = calculate_weights(WZTo3LNu_mllmin0p1, cross_section_WZTo3LNu_mllmin0p1, generator_weight_WZTo3LNu_mllmin0p1, number_of_events_WZTo3LNu_mllmin0p1, lumi)
    ZZTo4L = calculate_weights(ZZTo4L, cross_section_ZZTo4L, generator_weight_ZZTo4L, number_of_events_ZZTo4L, lumi)
    DYJetsToLL = calculate_weights(DYJetsToLL, cross_section_DYJetsToLL, generator_weight_DYJetsToLL, number_of_events_DYJetsToLL, lumi)
    TTTo2L2Nu = calculate_weights(TTTo2L2Nu, cross_section_TTTo2L2Nu, generator_weight_TTTo2L2Nu, number_of_events_TTTo2L2Nu, lumi)

    norm_to = (len(WZTo3LNu_TuneCP5) + len(WZTo3LNu_mllmin0p1))/212 * 288
    print('Normalisation factor:', norm_to)

    background_weight_dict = {
            'WZ': np.float32(212/288),
            'ZZ': np.float32(33/288),
            'DY': np.float32(32/288),
            'Top': np.float32(11/288),
        }

    Wplus.to_csv(save_path + Wplus_nick, index=False)
    Wminus.to_csv(save_path + Wminus_nick, index=False)
    WZTo3LNu_TuneCP5.to_csv(save_path + WZTo3LNu_TuneCP5_nick, index=False)
    WZTo3LNu_mllmin0p1.to_csv(save_path + WZTo3LNu_mllmin0p1_nick, index=False)
    ZZTo4L.to_csv(save_path + ZZTo4L_nick, index=False)
    DYJetsToLL.to_csv(save_path + DYJetsToLL_nick, index=False)
    TTTo2L2Nu.to_csv(save_path + TTTo2L2Nu_nick, index=False)

    print('Done!')


if __name__ == '__main__':
    main()