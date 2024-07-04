import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


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


def plot_ratio(signal, background, output_folder, variable):
    hist1, bins1 = np.histogram(signal, bins=30)
    hist2, bins2 = np.histogram(background, bins=30)

    hist1_err = np.sqrt(hist1)
    hist2_err = np.sqrt(hist2)

    # Normalize histograms
    hist1_sum = np.sum(hist1)
    hist2_sum = np.sum(hist2)

    hist1 = hist1 / hist1_sum
    hist2 = hist2 / hist2_sum

    hist1_err = hist1_err / hist1_sum
    hist2_err = hist2_err / hist2_sum

    # Calculate ratio and its error
    ratio = hist1 / hist2
    ratio_err = ratio * np.sqrt((hist1_err / hist1) ** 2 + (hist2_err / hist2) ** 2)


    plt.figure(figsize=(8, 6))
    plt.errorbar(bins1[:-1], ratio, yerr=ratio_err, marker='o', linestyle='None', color='red', ecolor='black', capsize=2, markersize=2, label='Signal/Background')
    plt.ylim(0.5, 1.5)
    plt.xlabel('Bins')
    plt.ylabel('Ratio')
    plt.legend()

    # draw red dottet line at ratio = 1
    plt.axhline(y=1, color='r', linestyle='--')

    plt.title(f'signal/background {signal.name}')

    save_path = os.path.join(output_folder, f'ratio_{variable}.png')
    plt.savefig(save_path, bbox_inches='tight')


def return_oppostite_chagrge(df, q1, q2):
    filtered_df = df[df[q1] * df[q2] < 0]
    return filtered_df


def return_same_charge(df, q1, q2):
    filtered_df = df[df[q1] * df[q2] > 0]
    return filtered_df


def main():
    # Read the CSV file into a pandas DataFrame
    signal_path = ['/ceph/ehettwer/working_data/inclusive_charge/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv',
                   '/ceph/ehettwer/working_data/inclusive_charge/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv']
    background_path = ['/ceph/ehettwer/working_data/inclusive_charge/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv']

    df1 = concatenate_csv(signal_path)
    df2 = concatenate_csv(background_path)

    variables = ['m_H', 'pt_H', 'phi_H', 'eta_H']
    output_folder = 'WH_analysis/ratio_plots'
    os.makedirs(output_folder, exist_ok=True)


    for variable in variables:
        signal = df1[variable]
        background = df2[variable]

        plot_ratio(signal, background, output_folder, variable)
        print(f'Ratio plot for {variable} saved successfully!')


if __name__ == '__main__':
    main()