from turtle import back, color
from cycler import V
from networkx import difference
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplhep as hep
import glob
import icecream as ic
from sympy import comp
import scipy.optimize as opt

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


def chi_squared(background_scale, hist_data, hist_background_DY, hist_background_top, bins):
    # Scale the background histograms
    scaled_background_DY = hist_background_DY * background_scale
    scaled_background_top = hist_background_top * background_scale
    
    # Combine scaled backgrounds
    combined_background_histogram = scaled_background_DY + scaled_background_top
    
    # Calculate the chi-squared
    chi2 = np.sum(((hist_data - combined_background_histogram) ** 2) / (hist_data + 1e-10))  # Adding small value to avoid division by zero
    return chi2


def df_segmentation(df, variable, threshold: tuple):
    if df is None:
        print('Dataframe is empty. Skipping segmentation...')
        return None
    if variable not in df.columns:
        print(f"Variable '{variable}' not found in DataFrame. Skipping segmentation...")
        return None
    df_segmented = df[(df[variable] > threshold[0]) & (df[variable] < threshold[1])]
    return df_segmented


def histogram_data_signal_background(data, signal, background_top, background_DY, background_ZZ, background_WZ, variable, number_bins, hist_range, scale_signal, save_path=None):
    signal_weights = signal['weights'] * scale_signal
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

    print('yield of WZ:', np.sum(hist_background_WZ))
    print('yield of ZZ:', np.sum(hist_background_ZZ))
    print('yield of DY:', np.sum(hist_background_DY))
    print('yield of top:', np.sum(hist_background_top))
    print('yield all backgrounds:', np.sum(hist_background_WZ) + np.sum(hist_background_ZZ) + np.sum(hist_background_DY) + np.sum(hist_background_top))

    combined_background_histogram = hist_background_WZ + hist_background_ZZ + hist_background_DY + hist_background_top
    np.savetxt('workflow/histogram_background.txt', combined_background_histogram)

    # Create the stacked histogram
    plt.figure(figsize=(10, 8), dpi=600)
    hep.histplot(hist_data, bins=bins, yerr=True, label='Data', histtype='errorbar', color='black')
    hep.histplot([hist_background_top, hist_background_DY, hist_background_ZZ, hist_background_WZ], bins=bins, label=['Top', 'DY', 'ZZ', 'WZ'], color=['darkgreen', 'darkorange', 'cyan', 'limegreen'], histtype='fill', alpha=0.7, stack=True)
    hep.histplot([hist_background_top, hist_background_DY, hist_background_ZZ, hist_background_WZ], bins=bins, histtype='step', lw=0.7, color='black', stack=True)
    hep.histplot(hist_signal, bins=bins, label='WH x 20', color='red', histtype='step', alpha=1, lw=1.5)

    plt.xlabel(r'$m_{\mu_1 \mu_2}\ \mathrm{(GeV)}$')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=18)
    plt.xlim(hist_range)
    plt.tick_params(axis='both', pad=15)

    
    # Get handles and labels, then reverse them
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print('saved histogram to', save_path)


def histogram_data_signal_background_vbf_chi2(data, vbf_signal, gluglu_signal, background_DY, background_top, variable, number_bins, hist_range, scale_signal, save_path=None):
    vbf_signal_weights = vbf_signal['weights'] * scale_signal
    gluglu_signal_weights = gluglu_signal['weights'] * scale_signal
    background_DY_weights = background_DY['weights']
    background_top_weights = background_top['weights']
    
    data = data[variable]
    vbf_signal = vbf_signal[variable]
    gluglu_signal = gluglu_signal[variable]
    background_DY = background_DY[variable]
    background_top = background_top[variable]

    # Create the histograms
    hist_data, bins = np.histogram(data, bins=number_bins, range=hist_range)
    np.savetxt('vbf_analysis/histogram_data_vbf.txt', hist_data)
    np.savetxt('vbf_analysis/bins_vbf.txt', bins)

    hist_vbf_signal, _ = np.histogram(vbf_signal, bins=bins, weights=vbf_signal_weights, range=hist_range) 
    hist_gluglu_signal, _ = np.histogram(gluglu_signal, bins=bins, weights=gluglu_signal_weights, range=hist_range) 
    np.savetxt('vbf_analysis/histogram_signal_vbf.txt', hist_vbf_signal + hist_gluglu_signal)

    hist_background_DY, _ = np.histogram(background_DY, bins=bins, weights=background_DY_weights)
    hist_background_top, _ = np.histogram(background_top, bins=bins, weights=background_top_weights)

    print('yield of DY:', np.sum(hist_background_DY))
    print('yield of top:', np.sum(hist_background_top))
    print('yield all backgrounds:', np.sum(hist_background_DY) + np.sum(hist_background_top))

    # Initial guess for background_scale
    initial_guess = 1.0
    
    # Perform the optimization
    result = opt.minimize(chi_squared, initial_guess, args=(hist_data, hist_background_DY, hist_background_top, bins))
    optimal_background_scale = result.x[0]

    print(f'Optimal background scale: {optimal_background_scale}')

    # Scale the background histograms with the optimal factor
    scaled_background_DY = hist_background_DY * optimal_background_scale
    scaled_background_top = hist_background_top * optimal_background_scale
    combined_background_histogram = scaled_background_DY + scaled_background_top

    np.savetxt('vbf_analysis/histogram_background_vbf.txt', combined_background_histogram)

    # Create the stacked histogram
    plt.figure(figsize=(10, 8), dpi=600)
    hep.histplot(hist_data, bins=bins, yerr=True, label='Data', histtype='errorbar', color='black')
    hep.histplot([scaled_background_top, scaled_background_DY], bins=bins, label=['Top', 'DY'], color=['darkgreen', 'darkorange'], histtype='fill', alpha=0.7, stack=True)
    hep.histplot([scaled_background_top, scaled_background_DY], bins=bins, histtype='step', lw=0.7, color='black', stack=True)
    hep.histplot([hist_vbf_signal, hist_gluglu_signal], bins=bins, label=[f'VBF x {scale_signal}', f'ggH x {scale_signal}'], color=['red', 'gold'], histtype='step', alpha=1, lw=1.5, stack=True)

    plt.xlabel(r'$m_{\mu_1 \mu_2}\ \mathrm{(GeV)}$')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=18)
    plt.xlim(hist_range)
    plt.tick_params(axis='both', pad=15)
    
    # Get handles and labels, then reverse them
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print('saved histogram to', save_path)



def histogram_data_signal_background_vbf(data, vbf_signal, gluglu_signal, background_DY, background_top, variable, number_bins, hist_range, scale_signal, save_path=None):
    vbf_signal_weights = vbf_signal['weights'] * scale_signal
    gluglu_signal_weights = gluglu_signal['weights'] * scale_signal
    background_DY_weights = background_DY['weights'] 
    background_top_weights = background_top['weights']
    
    data = data[variable]
    vbf_signal = vbf_signal[variable]
    gluglu_signal = gluglu_signal[variable]
    background_DY = background_DY[variable]
    background_top = background_top[variable]

    # Create the histograms
    hist_data, bins = np.histogram(data, bins=number_bins, range=hist_range)
    np.savetxt('vbf_analysis/histogram_data_vbf.txt', hist_data)
    np.savetxt('vbf_analysis/bins_vbf.txt', bins)

    hist_vbf_signal, _ = np.histogram(vbf_signal, bins=bins, weights=vbf_signal_weights, range=hist_range) 
    hist_gluglu_signal, _ = np.histogram(gluglu_signal, bins=bins, weights=gluglu_signal_weights, range=hist_range) 
    np.savetxt('vbf_analysis/histogram_signal_vbf.txt', hist_vbf_signal + hist_gluglu_signal)

    hist_background_DY, _ = np.histogram(background_DY, bins=bins, weights=background_DY_weights)
    hist_background_top, _ = np.histogram(background_top, bins=bins, weights=background_top_weights)

    print('yield of DY:', np.sum(hist_background_DY))
    print('yield of top:', np.sum(hist_background_top))
    print('yield all backgrounds:', np.sum(hist_background_DY) + np.sum(hist_background_top))

    combined_background_histogram = hist_background_DY + hist_background_top
    np.savetxt('vbf_analysis/histogram_background_vbf.txt', combined_background_histogram)

    # Create the stacked histogram
    plt.figure(figsize=(10, 8), dpi=600)
    hep.histplot(hist_data, bins=bins, yerr=True, label='Data', histtype='errorbar', color='black')
    hep.histplot([hist_background_top, hist_background_DY], bins=bins, label=['Top', 'DY'], color=['darkgreen', 'darkorange'], histtype='fill', alpha=0.7, stack=True)
    hep.histplot([hist_background_top, hist_background_DY], bins=bins, histtype='step', lw=0.7, color='black', stack=True)
    hep.histplot([hist_vbf_signal, hist_gluglu_signal], bins=bins, label=[f'VBF Signal x {scale_signal}', f'ggH Signal x {scale_signal}'], color=['red', 'gold'], histtype='step', alpha=1, lw=1.5, stack=True)

    plt.xlabel(r'$m_{\mu_1 \mu_2}\ \mathrm{(GeV)}$')
    plt.ylabel('Events/Bin')
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'59.7 fb$^{-1}$ at 13 TeV (2018)', loc='right', pad=10, fontsize=18)
    plt.xlim(hist_range)
    plt.tick_params(axis='both', pad=15)
    
    # Get handles and labels, then reverse them
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print('saved histogram to', save_path)



def histogram_sb_ratio(signal, background, variable, number_bins, hist_range, save_base_path):
    signal = signal[variable]
    background = background[variable]

    signal_hist, bins = np.histogram(signal, bins=number_bins, range=hist_range)
    signal_hist_err = np.sqrt(signal_hist)
    background_hist, _ = np.histogram(background, bins=number_bins, range=hist_range)
    background_hist_err = np.sqrt(background_hist)

    signal_hist_norm = signal_hist / np.sum(signal_hist)
    signal_hist_norm_err = signal_hist_err / np.sum(signal_hist)
    background_hist_norm = background_hist / np.sum(background_hist)
    background_hist_norm_err = background_hist_err / np.sum(background_hist)

    ratio = signal_hist_norm / background_hist_norm
    ratio_err = ratio * np.sqrt((signal_hist_norm_err / signal_hist_norm) ** 2 + (background_hist_norm_err / background_hist_norm) ** 2)

    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)

    # Histogram of Signal and Background
    ax1 = fig.add_subplot(gs[0])
    hep.histplot(background_hist_norm, bins=bins, yerr=background_hist_norm_err, label='Background', histtype='step', ax=ax1)
    hep.histplot(signal_hist_norm, bins=bins, yerr=signal_hist_norm_err, label='Signal', histtype='step', ax=ax1)
    ax1.set_ylabel('Events/Bin')
    ax1.set_xlim(hist_range)
    ax1.legend()
    ax1.set_title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    ax1.set_title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)

    plt.setp(ax1.get_xticklabels(), visible=False)

    # Ratio of Signal and Background
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    hep.histplot(ratio, bins=bins, yerr=ratio_err, label='Signal/Background', histtype='errorbar', ax=ax2, color='orangered', ecolor='black', capsize=2, markersize=4)
    #ax2.errorbar(bins[:-1], ratio, yerr=ratio_err, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2, label='Signal/Background')
    ax2.set_ylim(0.5, 1.5)
    ax2.set_xlabel(f'{variable}')
    ax2.set_ylabel('Ratio')
    ax2.axhline(y=1, color='black', linestyle='--', lw=1)

    save_path = save_base_path + f'{variable}_signal_background_ratio.png'
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print('saved histogram to', save_path)

    plt.close()


def histogram_sb_ratio_dev(signal, background_top, background_DY, background_ZZ, background_WZ, variable, number_bins, hist_range, save_base_path, xlabel):
    background_WZ_weights = background_WZ['weights']
    background_ZZ_weights = background_ZZ['weights']
    background_DY_weights = background_DY['weights']
    background_top_weights = background_top['weights']
    
    signal = signal[variable]
    background_WZ = background_WZ[variable]
    background_ZZ = background_ZZ[variable]
    background_DY = background_DY[variable]
    background_top = background_top[variable]

    hist_signal, bins = np.histogram(signal, bins=number_bins, range=hist_range)

    hist_background_WZ, _ = np.histogram(background_WZ, bins=number_bins, weights=background_WZ_weights, range=hist_range)
    hist_background_ZZ, _ = np.histogram(background_ZZ, bins=number_bins, weights=background_ZZ_weights, range=hist_range)
    hist_background_DY, _ = np.histogram(background_DY, bins=number_bins, weights=background_DY_weights, range=hist_range)
    hist_background_top, _ = np.histogram(background_top, bins=number_bins, weights=background_top_weights, range=hist_range)

    combined_backgrounds = hist_background_WZ + hist_background_ZZ + hist_background_DY + hist_background_top

    # Calculate errors and clip any negative values to zero before taking sqrt
    err_background_WZ = np.sqrt(np.clip(np.histogram(background_WZ, bins=number_bins, weights=background_WZ_weights**2, range=hist_range)[0], 0, None))
    err_background_ZZ = np.sqrt(np.clip(np.histogram(background_ZZ, bins=number_bins, weights=background_ZZ_weights**2, range=hist_range)[0], 0, None))
    err_background_DY = np.sqrt(np.clip(np.histogram(background_DY, bins=number_bins, weights=background_DY_weights**2, range=hist_range)[0], 0, None))
    err_background_top = np.sqrt(np.clip(np.histogram(background_top, bins=number_bins, weights=background_top_weights**2, range=hist_range)[0], 0, None))

    combined_background_errors = np.sqrt(err_background_WZ**2 + err_background_ZZ**2 + err_background_DY**2 + err_background_top**2)

    # Normalize histograms, ensuring the sum is not zero
    signal_sum = np.sum(hist_signal)
    background_sum = np.sum(combined_backgrounds)
    
    signal_hist_norm = np.divide(hist_signal, signal_sum, out=np.zeros_like(hist_signal, dtype=float), where=signal_sum!=0)
    signal_hist_norm_err = np.divide(np.sqrt(hist_signal), signal_sum, out=np.zeros_like(hist_signal, dtype=float), where=signal_sum!=0)
    
    background_hist_norm = np.divide(combined_backgrounds, background_sum, out=np.zeros_like(combined_backgrounds, dtype=float), where=background_sum!=0)
    background_hist_norm_err = np.divide(combined_background_errors, background_sum, out=np.zeros_like(combined_background_errors, dtype=float), where=background_sum!=0)

    # Calculate the ratio and its error, safely handling divisions
    ratio = np.divide(signal_hist_norm, background_hist_norm, out=np.zeros_like(signal_hist_norm, dtype=float), where=background_hist_norm!=0)
    difference = abs(np.subtract(signal_hist_norm, background_hist_norm))
    ratio_err = ratio * np.sqrt(np.divide(signal_hist_norm_err**2, signal_hist_norm**2, out=np.zeros_like(signal_hist_norm_err, dtype=float), where=signal_hist_norm!=0) +
                                np.divide(background_hist_norm_err**2, background_hist_norm**2, out=np.zeros_like(background_hist_norm_err, dtype=float), where=background_hist_norm!=0))
    difference_err = np.sqrt(np.divide(signal_hist_norm_err**2, signal_hist_norm**2, out=np.zeros_like(signal_hist_norm_err, dtype=float), where=signal_hist_norm!=0) +
                             np.divide(background_hist_norm_err**2, background_hist_norm**2, out=np.zeros_like(background_hist_norm_err, dtype=float), where=background_hist_norm!=0))

    total_difference = np.sum(difference)
    
    # Clip any negative errors to zero to avoid issues with plotting
    ratio_err = np.clip(ratio_err, 0, None)

    plt.figure(figsize=(10, 8))

    # Histogram of Signal and Background
    hep.histplot(background_hist_norm, bins=bins, yerr=background_hist_norm_err, label='Background', histtype='step', color='navy')
    hep.histplot(signal_hist_norm, bins=bins, yerr=signal_hist_norm_err, label='Signal', histtype='step',  color='orangered')
    plt.ylabel('a. u.')
    plt.xlabel(xlabel)
    plt.xlim(hist_range)
    plt.legend()
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)

    save_path = save_base_path + f'{variable}_signal_background.png'
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print('saved histogram to', save_path)

    plt.close()


def compare_two_backgrounds(background1, background2, variable, number_bins, hist_range, save_base_path):
    # Extract the variable data
    background1 = background1[variable]
    background2 = background2[variable]

    # Calculate histograms and errors
    hist_background1, bins = np.histogram(background1, bins=number_bins, range=hist_range)
    hist_background2, _ = np.histogram(background2, bins=number_bins, range=hist_range)

    err_background1 = np.sqrt(hist_background1)
    err_background2 = np.sqrt(hist_background2)

    # Normalize histograms, ensuring the sum is not zero
    hist_background1_sum = hist_background1.sum()
    hist_background2_sum = hist_background2.sum()
    
    background1_hist_norm = np.divide(hist_background1, hist_background1_sum, out=np.zeros_like(hist_background1, dtype=float), where=hist_background1_sum!=0)
    background1_hist_norm_err = np.divide(err_background1, hist_background1_sum, out=np.zeros_like(err_background1, dtype=float), where=hist_background1_sum!=0)

    background2_hist_norm = np.divide(hist_background2, hist_background2_sum, out=np.zeros_like(hist_background2, dtype=float), where=hist_background2_sum!=0)
    background2_hist_norm_err = np.divide(err_background2, hist_background2_sum, out=np.zeros_like(err_background2, dtype=float), where=hist_background2_sum!=0)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    hep.histplot(background1_hist_norm, bins=bins, yerr=background1_hist_norm_err, label='Background 1', histtype='step', color='navy')
    hep.histplot(background2_hist_norm, bins=bins, yerr=background2_hist_norm_err, label='Background 2', histtype='step', color='orangered')

    plt.ylabel('a. u.')
    plt.xlim(hist_range)
    plt.legend()
    plt.title(r'$\mathit{Private\ work}\ \mathrm{\mathbf{CMS\ Simulation}}$', loc='left', pad=10, fontsize=24)
    plt.title(r'${13\ TeV\ (2018)}$', loc='right', pad=10, fontsize=24)

    # Save the plot
    save_path = save_base_path + f'{variable}_background_comparison.png'
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print('Saved histogram to', save_path)

    plt.close()




def main():
    # Define the file paths of the CSV files
    background_WZTo3LNu_mllmin0p1_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_WZTo3LNu_TuneCP5_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    backgound_ZZTo4L_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_DYJets_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_TTTo2L2Nu_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    signal_Wplus_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    signal_Wminus_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut_weights/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    vbf_background_DYJets_path = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_loose_cut_weights/DYJetsToLL_M-100to200_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    vbf_background_top_path = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_loose_cut_weights/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    vbf_signal = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_loose_cut_weights/VBFHToMuMu_M125_TuneCP5_withDipoleRecoil_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    gluglu_signal = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_loose_cut_weights/GluGluHToMuMu_M-125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    data_csv_path = '/work/ehettwer/HiggsMewMew/data/ntuples_final_final_loose_cut/SingleMuonData.csv'
    vbf_data_path = '/work/ehettwer/HiggsMewMew/data/vbf_ntuples_loose_cut/SingleMuonData.csv' 

    mass_cut = (110, 150)

    data = df_segmentation(pd.read_csv(data_csv_path), 'm_H', mass_cut)
    vbf_data = df_segmentation(pd.read_csv(vbf_data_path), 'm_vis', mass_cut)
    
    # Load the data from the CSV files
    background_WZTo3LNu_mllmin0p1 = pd.read_csv(background_WZTo3LNu_mllmin0p1_path)
    background_WZTo3LNu_TuneCP5 = df_segmentation(pd.read_csv(background_WZTo3LNu_TuneCP5_path), 'm_H', mass_cut)
    background_ZZTo4L = df_segmentation(pd.read_csv(backgound_ZZTo4L_path), 'm_H', mass_cut)
    background_DYJets = df_segmentation(pd.read_csv(background_DYJets_path), 'm_H', mass_cut)
    background_TTTo2L2Nu = df_segmentation(pd.read_csv(background_TTTo2L2Nu_path), 'm_H', mass_cut)

    signal_Wplus = pd.read_csv(signal_Wplus_path)
    signal_Wminus = pd.read_csv(signal_Wminus_path)

    vbf_background_DYJets = df_segmentation(pd.read_csv(vbf_background_DYJets_path), 'm_vis', mass_cut)
    vbf_background_top = df_segmentation(pd.read_csv(vbf_background_top_path), 'm_vis', mass_cut)
    vbf_signal = df_segmentation(pd.read_csv(vbf_signal), 'm_vis', mass_cut)
    gluglu_signal = df_segmentation(pd.read_csv(gluglu_signal), 'm_vis', mass_cut)

    signal_combined = df_segmentation(pd.concat([signal_Wplus, signal_Wminus]), 'm_H', mass_cut)
    background_combined = pd.concat([background_WZTo3LNu_TuneCP5, background_ZZTo4L, background_DYJets, background_TTTo2L2Nu])

    histogram_data_signal_background(data, signal_combined, background_TTTo2L2Nu, background_DYJets, background_ZZTo4L,  background_WZTo3LNu_TuneCP5, 'm_H', number_bins=8, hist_range=mass_cut, scale_signal=20, save_path='thesis_plots/plots_vh/histogram_wh.png')
    histogram_data_signal_background_vbf_chi2(vbf_data, vbf_signal, gluglu_signal, vbf_background_DYJets, vbf_background_top, 'm_vis', number_bins=8, hist_range=mass_cut,  scale_signal=20, save_path='thesis_plots/plots_vbf/histogram_vbf.png')
    #histogram_sb_ratio_dev(signal_combined, background_TTTo2L2Nu, background_DYJets, background_ZZTo4L,  background_WZTo3LNu_TuneCP5, 'pt_H', number_bins=30, hist_range=(0, 250), save_base_path='thesis_plots/Plots/')


    list_of_variables = ['deltaEta_13', 'deltaEta_23', 'deltaEta_WH', 'deltaPhi_12', 'deltaPhi_13', 'deltaPhi_WH', 'deltaR_12', 'deltaR_13', 'deltaR_23', 'eta_H', 'm_H', 'phi_H', 'pt_H', 'q_1', 'q_2', 'q_3',
                         'pt_1', 'pt_2', 'pt_3', 'nmuons', 'eta_1', 'eta_2', 'cosThetaStar12', 'cosThetaStar13', 'cosThetaStar23']
    list_of_plot_ranges = [(0, 4), (0, 4), (0, 4), (-3, 3), (-3, 3), (-3, 3), (0.5, 5), (0.5, 5), (0.5, 5), (-4, 4), (115, 135), (-3, 3), (0, 250), (-1, 1), (-1, 1), (-1, 1), 
                           (20, 250), (20, 250), (20, 250), (0, 10), (-2.2, 2.2), (-2.2, 2.2), (-1, 1), (-1, 1), (-1, 1)]
    list_of_xlabels = [
    r'$\Delta\eta_{13}$', r'$\Delta\eta_{23}$', r'$\Delta\eta_{H;3}$', 
    r'$\Delta\phi_{12}$', r'$\Delta\phi_{13}$', r'$\Delta\phi_{H;3}$', 
    r'$\Delta R_{12}$', r'$\Delta R_{13}$', r'$\Delta R_{23}$', 
    r'$\eta_H$', r'$m_H$', r'$\phi_H$', r'$p_T^H$', 
    r'$q_1$', r'$q_2$', r'$q_3$', 
    r'$(p_T)_1$', r'$(p_T)_2$', r'$(p_T)_3$', 
    r'$n_{\mu}$', r'$\eta_1$', r'$\eta_2$', 
    r'$\cos\theta^*_{12}$', r'$\cos\theta^*_{13}$', r'$\cos\theta^*_{23}$'
    ]

    list_of_vbf_variables = ['eta_1', 'eta_2', 'jeta_1', 'jeta_2', 'jphi_1', 'jphi_2', 'jpt_1', 'jpt_2', 'm_vis', 'mjj', 'njets', 'nmuons', 'phi_1', 'phi_2', 'pt_1', 'pt_2', 'pt_dijet', 'pt_vis']
    list_of_vbf_plot_ranges = [(-2.2, 2.2), (-2.2, 2.2), (-5, 5), (-5, 5), (-3.5, 3.5), (-3.5, 3.5), (20, 250), (20, 250), (115, 135), (400, 2500), (2, 5), (0, 10), (-3.5, 3.5), (-3.5, 3.5), (20, 250), (20, 250), (0, 250), (0, 250)]
    list_of_vbf_xlabels = [
    r'$\eta_1$', r'$\eta_2$', r'$\eta_{j1}$', r'$\eta_{j2}$', r'$\phi_{j1}$', r'$\phi_{j2}$', r'$(p_T)_{j1}$', r'$(p_T)_{j2}$',
    r'$m_{\mu_1 \mu_2}$', r'$m_{jj}$', r'$n_{\mathrm{jets}}$', r'$n_{\mu}$', r'$\phi_1$', r'$\phi_2$', r'$(p_T)_1$', r'$(p_T)_2$', r'$(p_T)_{\mathrm{dijet}}$', r'$(p_T)_{\mathrm{vis}}$'
    ]

    #for variable, plot_range, xlabel in zip(list_of_variables, list_of_plot_ranges, list_of_xlabels):
    #    compare_two_backgrounds(background_WZTo3LNu_TuneCP5, background_ZZTo4L, variable, number_bins=30, hist_range=plot_range, save_base_path='thesis_plots/Plots/')
    #    histogram_sb_ratio_dev(signal_combined, background_TTTo2L2Nu, background_DYJets, background_ZZTo4L,  background_WZTo3LNu_TuneCP5, variable, number_bins=30, hist_range=plot_range, save_base_path='thesis_plots/plots_vh/', xlabel=xlabel)
    #    print(f'Finished plotting {variable}')

    #for variable, plot_range, xlabel in zip(list_of_vbf_variables, list_of_vbf_plot_ranges, list_of_vbf_xlabels):
    #    histogram_sb_ratio_dev(vbf_signal, vbf_background_top, vbf_background_DYJets, vbf_background_top, vbf_background_DYJets, variable, number_bins=30, hist_range=plot_range, save_base_path='thesis_plots/plots_vbf/', xlabel=xlabel)
    #    print(f'Finished plotting {variable}')

if __name__ == '__main__':
    main()
