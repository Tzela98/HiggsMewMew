from hmac import new  # for hashing
from tkinter import font  # for GUI font handling
import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
from icecream import ic  # for debugging and logging
import os  # for file system operations

from sympy import E  # Euler's number
import filters as filters  # custom filters (not defined here)

import mplhep as hep  # for plotting in HEP style
from torch import rand  # for random number generation

# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
# Place the CMS label at location 0 (top-left corner)
hep.cms.label(loc=0)


# Function for segmenting a DataFrame based on a variable and given thresholds
def df_segmentation(df, variable, threshold=[]):
    df_segmented = []
    for region in range(len(threshold)):
        df_segmented.append(df[(df[variable] > threshold[region][0]) & (df[variable] < threshold[region][1])])
    return df_segmented


def eventwise_weight():
    # Set the plotting style to CMS style
    hep.style.use(hep.style.CMS)
    # Place the CMS label at location 0 (top-left corner)
    hep.cms.label(loc=0)

    background_info = pd.read_csv('workflow/csv_files/background_info.csv', index_col='nicks')
    df_Z_region = pd.read_csv('workflow/csv_files/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')
    df_rest_region = pd.read_csv('workflow/csv_files/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')
    df_data = pd.read_csv('workflow/csv_files/single_muon_data_2018.csv')
    
    generator_weight_Z = 0.6729741358064714
    number_of_events_Z = 195510810

    sum_genweights_Z = generator_weight_Z * number_of_events_Z * df_Z_region['genWeight'].abs()


    ic(generator_weight_Z, df_Z_region['genWeight'])
    ic(sum_genweights_Z)


    acceptance_Z = df_Z_region['genWeight'] / sum_genweights_Z
    ic(acceptance_Z)

    luminosity_Z = 59.7  # in fb^-1
    cross_section_Z = 6529 * 1000 # in pb

    generator_weight_sb = 0.6499288910928841
    number_of_events_sb = 219326954

    sum_genweights_sb = generator_weight_sb * number_of_events_sb * df_rest_region['genWeight'].abs()
    acceptance_sb = df_rest_region['genWeight'] / sum_genweights_sb
    ic(acceptance_sb)

    luminosity_sb = 59.7 # in fb^-1
    cross_section_sb = 247.8 * 1000 # in pb

    weight_Z = acceptance_Z * luminosity_Z * cross_section_Z
    weight_sb = acceptance_sb * luminosity_sb * cross_section_sb

    plt.hist(df_rest_region['m_vis'], weights=weight_sb, bins=80, range=(70, 150), histtype='step', label='Sideband')
    plt.hist(df_Z_region['m_vis'], weights=weight_Z, bins=80, range=(70, 150), histtype='step', label='Z region')
    plt.hist(df_data['m_vis'], bins=80, range=(70, 150), histtype='step', label='Data')

    plt.yscale('log')
    plt.legend()
    plt.show()
    

def global_weight():
     
    hep.style.use(hep.style.CMS)
    # Place the CMS label at location 0 (top-left corner)
    hep.cms.label(loc=0)

    background_info = pd.read_csv('workflow/csv_files/background_info.csv', index_col='nicks')
    df_Z_region = pd.read_csv('workflow/csv_files/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')
    df_rest_region = pd.read_csv('workflow/csv_files/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')
    df_data = pd.read_csv('workflow/csv_files/single_muon_data_2018.csv')   

    plt.hist(df_data['m_vis'], bins=80, range=(70, 150), histtype='step', label='Data')

    luminosity = 59.7  # in fb^-1
    cross_section_Z = 6529 / 1000 # in pb
    cross_section_sb = 247.8 / 1000 # in pb
    number_of_events_Z = 195510810
    number_of_events_sb = 219326954

    sf_Z = (3.898 * 10**8) / number_of_events_Z
    sf_sb = (1.475 * 10**7) / number_of_events_sb

    print(sf_Z, sf_sb)

    n_Z, bins_Z = np.histogram(df_Z_region['m_vis'], bins=80, range=(70, 150))
    n_sb, bins_sb = np.histogram(df_rest_region['m_vis'], bins=80, range=(70, 150))

    hep.histplot(n_Z * sf_Z, bins_Z, label='Z region', histtype='step')
    hep.histplot(n_sb * sf_sb, bins_sb, label='Sideband', histtype='step')

    plt.legend()
    plt.yscale('log')
    plt.show()

def main():
    eventwise_weight()
    global_weight()


if __name__ == '__main__':
    main()