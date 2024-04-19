import matplotlib.pyplot as plt  # for plotting
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
from icecream import ic  # for debugging and logging
import filters as filters  # custom filters (not defined here)

import mplhep as hep  # for plotting in HEP style
from torch import rand  # for random number generation

# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


# Function for segmenting a DataFrame based on a variable and given thresholds
def main():
    background_info = pd.read_csv('workflow/csv_files/background_info.csv', index_col='nicks')
    df_Z_region = pd.read_csv('workflow/csv_files/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')
    df_data = pd.read_csv('workflow/csv_files/single_muon_data_2018.csv')

    id_iso_wgt = df_Z_region['id_wgt_mu_1'] * df_Z_region['iso_wgt_mu_1'] * df_Z_region['id_wgt_mu_2'] * df_Z_region['iso_wgt_mu_2']
    n_id_iso, bins_id_iso = np.histogram(id_iso_wgt, bins=50)
    n_pu, bins_pu = np.histogram(df_Z_region['puweight'], bins=50)

    hep.histplot(n_pu, bins_pu, label='Pileup weights')
    hep.histplot(n_id_iso, bins_id_iso, label='ID and ISO weights')

    plt.xlim(0, 2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()