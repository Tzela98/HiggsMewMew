import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import os

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


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

    save_path = os.path.join(output_folder, f'charge_separated_{variable}.png')
    plt.savefig(save_path)


def return_oppostite_chagrge(df, q1, q2):
    filtered_df = df[df[q1] * df[q2] < 0]
    return filtered_df


def return_same_charge(df, q1, q2):
    filtered_df = df[df[q1] * df[q2] > 0]
    return filtered_df


def main():
    # Read the CSV file into a pandas DataFrame
    file_path1 = '/ceph/ehettwer/working_data/inclusive_charge/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    file_path2 = '/ceph/ehettwer/working_data/inclusive_charge/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    charge_filtered_df1 = return_oppostite_chagrge(df1, 'q_1', 'q_3')
    charge_filtered_df2 = return_oppostite_chagrge(df2, 'q_1', 'q_3')

    variables = ['cosThetaStar13', 'cosThetaStar23']
    output_folder = 'WH_analysis/ratio_plots'
    os.makedirs(output_folder, exist_ok=True)


    for variable in variables:
        signal = charge_filtered_df1[variable]
        background = charge_filtered_df2[variable]

        plot_ratio(signal, background, output_folder, variable)
        print(f'Ratio plot for {variable} saved successfully!')


if __name__ == '__main__':
    main()