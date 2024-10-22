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
    base_path = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_tight_cut/'

    vbf_nick = 'VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    gluglu_nick = 'GluGluHToMuMu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    DYJetsToLL_nick = 'DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    top_nick = 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    vbf = pd.read_csv(base_path + vbf_nick)
    gluglu = pd.read_csv(base_path + gluglu_nick)
    DYJetsToLL = pd.read_csv(base_path + DYJetsToLL_nick)
    top = pd.read_csv(base_path + top_nick)

    cross_section_vbf = 0.0008228
    generator_weight_vbf = 0.9986
    number_of_events_vbf = 2000000


    cross_section_gluglu = 0.01057
    generator_weight_gluglu = 0.9914
    number_of_events_gluglu = 999000

    cross_section_DYJetsToLL = 247.8
    generator_weight_DYJetsToLL = 0.6499288910928841
    number_of_events_DYJetsToLL = 219326954

    cross_section_top =  88.51
    generator_weight_top = 1.0
    number_of_events_top =  145020000

    lumi = 59.74 * 1000

    save_path = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_tight_cut_weights/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Directory does not exist. Creating directory...')
        print('Created directory:', save_path)

    vbf = calculate_weights(vbf, cross_section_vbf, generator_weight_vbf, number_of_events_vbf, lumi)
    gluglu = calculate_weights(gluglu, cross_section_gluglu, generator_weight_gluglu, number_of_events_gluglu, lumi)
    DYJetsToLL = calculate_weights(DYJetsToLL, cross_section_DYJetsToLL, generator_weight_DYJetsToLL, number_of_events_DYJetsToLL, lumi)
    top = calculate_weights(top, cross_section_top, generator_weight_top, number_of_events_top, lumi)

    vbf.to_csv(save_path + vbf_nick, index=False)
    gluglu.to_csv(save_path + gluglu_nick, index=False)
    DYJetsToLL.to_csv(save_path + DYJetsToLL_nick, index=False)
    top.to_csv(save_path + top_nick, index=False)

    print('Done!')


if __name__ == '__main__':
    main()