from sre_constants import RANGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)

# This Script is used to plot the ratio of two variables from two different datasets
# The datasets are concatenated and the ratio is calculated for each bin
# The ratio is plotted with error bars


def concatenate_csv(files):
    # Create an empty list to store DataFrames
    data_frames = []
    
    # Iterate over the list of files
    for file in files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Append the DataFrame to the list
        data_frames.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(data_frames, ignore_index=True)
    return concatenated_df


def plot_ratio(signal, background, output_folder, variable, range):
    # Calculate histograms
    hist1, bins1 = np.histogram(signal, bins=30, range=range)
    hist2, bins2 = np.histogram(background, bins=30, range=range)

    hist1_err = np.sqrt(hist1)
    hist2_err = np.sqrt(hist2)

    # Normalize histograms for plotting
    hist1_sum = np.sum(hist1)
    hist2_sum = np.sum(hist2)

    hist1_norm = hist1 / hist1_sum
    hist2_norm = hist2 / hist2_sum

    hist1_norm_err = hist1_err / hist1_sum
    hist2_norm_err = hist2_err / hist2_sum

    # Calculate ratio and its error
    ratio = hist1_norm / hist2_norm
    ratio_err = ratio * np.sqrt((hist1_norm_err / hist1_norm) ** 2 + (hist2_norm_err / hist2_norm) ** 2)

    # Create subplots
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)

    # Histogram subplot
    ax0 = fig.add_subplot(gs[0])
    hep.histplot(hist1_norm, bins1, label='Signal', histtype='step', ax=ax0)
    hep.histplot(hist2_norm, bins2, label='Background', histtype='step', ax=ax0)

    ax0.set_ylabel('Normalized Counts')
    ax0.legend()
    ax0.set_title(f'Histograms of {variable}')

    # Ratio subplot
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.errorbar(bins1[:-1], ratio, yerr=ratio_err, marker='o', linestyle='None', color='black', ecolor='#333333', capsize=2, markersize=2)
    ax1.axhline(y=1, color='black', linestyle='--')
    ax1.set_xlabel(variable)
    ax1.set_ylabel('Ratio')
    ax1.set_ylim(0.5, 1.5)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0)

    save_path = os.path.join(output_folder, f'ratio_{variable}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # Return the histograms and bins for ROC calculation
    return hist1_norm, hist2_norm, bins1


def calculate_roc(signal, background, bins, selection_criteria):
    # Calculate TP, FP, FN, TN
    TP = np.sum(signal[(bins[:-1] >= selection_criteria[0]) & (bins[:-1] <= selection_criteria[1])])
    FP = np.sum(background[(bins[:-1] >= selection_criteria[0]) & (bins[:-1] <= selection_criteria[1])])
    FN = np.sum(signal) - TP
    TN = np.sum(background) - FP

    # Calculate TPR and FPR
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR, FPR


def main():
    # Read the CSV file into a pandas DataFrame
    signal_path = ['/work/ehettwer/HiggsMewMew/data/bug_fix/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
                   '/work/ehettwer/HiggsMewMew/data/bug_fix/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv']
    background_path = ['/work/ehettwer/HiggsMewMew/data/bug_fix/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
                       '/work/ehettwer/HiggsMewMew/data/bug_fix/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv']

    df1 = concatenate_csv(signal_path)
    df2 = concatenate_csv(background_path)

    variables = ['m_H']
    output_folder = 'Thesis_plots/individual_variables'
    os.makedirs(output_folder, exist_ok=True)


    for variable in variables:
        signal = df1[variable]
        background = df2[variable]

        signal_hist, background_hist, bins = plot_ratio(signal, background, output_folder, variable, range=(115, 135))
        TPR, FPR = calculate_roc(signal_hist, background_hist, bins, selection_criteria=(122, 128))
        print(f'TPR: {TPR:.4f}, FPR: {FPR:.4f}')
        print(f'Ratio plot for {variable} saved successfully!')


if __name__ == '__main__':
    main()