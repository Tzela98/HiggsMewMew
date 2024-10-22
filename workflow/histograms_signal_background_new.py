from turtle import back, color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
import icecream as ic

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


def df_segmentation(df, variable, threshold: tuple):
    if df is None:
        print('Dataframe is empty. Skipping segmentation...')
        return None
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame. Skipping segmentation...")
        return None
    df_segmented = df[(df[variable] > threshold[0]) & (df[variable] < threshold[1])]
    return df_segmented


def histogram_data_signal_background(data, signal, background_top, background_DY, background_ZZ, background_WZ, variable, number_bins, hist_range, save_path=None):
    signal_weights = signal['weights']
    background_WZ_weights = background_WZ['weights']
    background_ZZ_weights = background_ZZ['weights']
    background_DY_weights = background_DY['weights']
    background_top_weights = background_top['weights']
    
    data = data[variable]
    signal = signal[variable]
    background_WZ = background_WZ[variable]
    background_ZZ = background_ZZ[variable]
    background_DY = background_DY[variable]
    background_top = background_top[variable]

    # Create the histograms
    hist_data, bins = np.histogram(data, bins=number_bins, range=hist_range)
    np.savetxt('workflow/histogram_data.txt', hist_data)
    np.savetxt('workflow/bins.txt', bins)

    hist_signal, _ = np.histogram(signal, bins=bins, weights=signal_weights, range=hist_range)
    np.savetxt('workflow/histogram_signal.txt', hist_signal)

    hist_background_WZ, _ = np.histogram(background_WZ, bins=bins, weights=background_WZ_weights)
    hist_background_ZZ, _ = np.histogram(background_ZZ, bins=bins, weights=background_ZZ_weights)
    hist_background_DY, _ = np.histogram(background_DY, bins=bins, weights=background_DY_weights)
    hist_background_top, _ = np.histogram(background_top, bins=bins, weights=background_top_weights)

    print('yields WZ:', np.sum(hist_background_WZ))
    print('yields ZZ:', np.sum(hist_background_ZZ))
    print('yields DY:', np.sum(hist_background_DY))
    print('yields top:', np.sum(hist_background_top))
    print('yields all:', np.sum(hist_background_WZ) + np.sum(hist_background_ZZ) + np.sum(hist_background_DY) + np.sum(hist_background_top))

    combined_background_histogram = hist_background_WZ + hist_background_ZZ + hist_background_DY + hist_background_top
    np.savetxt('workflow/combined_background_histogram.txt', combined_background_histogram)

    # Create the stacked histogram
    plt.figure(figsize=(10, 8))
    hep.histplot(hist_data, bins=bins, yerr=True, label='Data', histtype='errorbar', color='black')
    hep.histplot([hist_background_top, hist_background_DY, hist_background_ZZ, hist_background_WZ], bins=bins, label=['Top', 'DY', 'ZZ', 'WZ'], color=['darkgreen', 'darkorange', 'cyan', 'limegreen'], histtype='fill', alpha=0.7, stack=True)
    hep.histplot([hist_background_top, hist_background_DY, hist_background_ZZ, hist_background_WZ], bins=bins, histtype='step', lw=0.7, color='black', stack=True)
    hep.histplot(hist_signal, bins=bins, label='Signal x 20', color='red', histtype='step', alpha=1, lw=1.5)

    plt.xlabel('Dimuon mass in [GeV]')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)
    plt.xlim(110, 150)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    # Define the file paths of the CSV files
    background_WZTo3LNu_mllmin0p1_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_WZTo3LNu_TuneCP5_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    backgound_ZZTo4L_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_DYJets = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_TTTo2L2Nu = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    signal_Wplus_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    signal_Wminus_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    data_csv_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut/SingleMuonData.csv'
    data = df_segmentation(pd.read_csv(data_csv_path), 'm_H', (110, 150))
    
    # Load the data from the CSV files
    background_WZTo3LNu_mllmin0p1 = pd.read_csv(background_WZTo3LNu_mllmin0p1_path)
    background_WZTo3LNu_TuneCP5 = df_segmentation(pd.read_csv(background_WZTo3LNu_TuneCP5_path), 'm_H', (110, 150))
    background_ZZTo4L = df_segmentation(pd.read_csv(backgound_ZZTo4L_path), 'm_H', (110, 150))
    background_DYJets = df_segmentation(pd.read_csv(background_DYJets), 'm_H', (110, 150))
    background_TTTo2L2Nu = df_segmentation(pd.read_csv(background_TTTo2L2Nu), 'm_H', (110, 150))

    signal_Wplus = pd.read_csv(signal_Wplus_path)
    signal_Wminus = pd.read_csv(signal_Wminus_path)

    signal = df_segmentation(pd.concat([signal_Wplus, signal_Wminus]), 'm_H', (110, 150))
    print(signal)

    histogram_data_signal_background(data, signal, background_TTTo2L2Nu, background_DYJets, background_ZZTo4L,  background_WZTo3LNu_TuneCP5, 'm_H', number_bins=8, hist_range=(110, 150), save_path='workflow/histogram_wide_norm.png')
    print('Histogram saved as workflow/histogram.png')

if __name__ == '__main__':
    main()
