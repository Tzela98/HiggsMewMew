import signal
from turtle import back
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob
from icecream import ic



hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)


def parse_log_files(file_paths):
    # Initialize an empty list to store the results
    results = []

    # Iterate through the list of file paths
    for file_path in file_paths:
        # Initialize variables to store the result
        result = None

        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

            # Check if the line before the last line ends with "job exit code : 0"
            if lines[-2].strip() == "job exit code : 0":
                # Iterate through lines to find the first instance of "all="
                for line in lines:
                    if "all=" in line:
                        # Extract the substring after "all="
                        start_index = line.find("all=")
                        result_str = line[start_index + 4:].strip()

                        # Find the end of the integer part
                        end_index = 0
                        for char in result_str:
                            if not char.isdigit():
                                break
                            end_index += 1

                        # Extract the integer part
                        result_str = result_str[:end_index]

                        # Convert the extracted string to an integer
                        try:
                            result = int(result_str)
                        except ValueError:
                            # Handle the case where the conversion to integer fails
                            print("Error: Unable to convert '{}' to an integer.".format(result_str))
                        break  # Stop searching after the first instance

        # Append the result to the list of results
        results.append(result)

    return np.sum(results)


def load_data(file_paths):
    # Create an empty list to store DataFrames
    data_frames = []
    
    # Iterate over the list of files
    for file in file_paths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        # Append the DataFrame to the list
        data_frames.append(df)
    
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(data_frames, ignore_index=True)
    return concatenated_df


def open_multiple_paths(paths: list):
    all_paths = []
    for path in paths:
        all_paths = all_paths + glob.glob(path, recursive=True)
    return sorted(all_paths)


def calculate_weights(dataset, cross_section, generator_weight, number_of_events, lumi = 59.74 * 1000):
    id_iso_wgt = dataset['id_wgt_mu_1'] * dataset['iso_wgt_mu_1'] * dataset['id_wgt_mu_2'] * dataset['iso_wgt_mu_2']
    acceptance = dataset['genWeight']/(abs(dataset['genWeight']) * generator_weight * number_of_events)
    weight = id_iso_wgt * acceptance * lumi * cross_section
    dataset['weights'] = weight

    return dataset


def histogram_dataset(dataset, variable, weights, plotname):
    plt.figure(figsize=(10, 8))
    n, bins, patches = plt.hist(dataset[variable], bins=10, range=(115, 135), weights=dataset[weights], histtype='step', label=plotname)

    plt.xlabel('dimuon mass [GeV]')
    plt.xlim(115, 135)
    plt.ylabel('Events')
    plt.legend()

    plt.savefig('workflow/plots/' + plotname + '.png', bbox_inches='tight')
    plt.close()

    print(f"Saved plot to workflow/plots/{plotname}.png")


def histogram_signal_background(signal, background, variable, weights, plotname):
    plt.figure(figsize=(10, 8))
    n, bins, patches = plt.hist(background[variable], bins=10, range=(115, 135), weights=background[weights], histtype='step', label='background')
    n, bins, patches = plt.hist(signal[variable], bins=10, range=(115, 135), weights=signal[weights], histtype='step', label='signal')

    plt.xlabel('dimuon mass [GeV]')
    plt.xlim(115, 135)
    plt.ylabel('Events')
    plt.legend()

    plt.savefig('workflow/plots/' + plotname + '.png', bbox_inches='tight')
    plt.close()

    print(f"Saved plot to workflow/plots/{plotname}.png")


def main():
    # Define the file paths of the CSV files
    background_csv_path1 = '/work/ehettwer/HiggsMewMew/data/including_genWeight/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    background_csv_path2 = '/work/ehettwer/HiggsMewMew/data/including_genWeight/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'


    signal_csv_path1 = '/work/ehettwer/HiggsMewMew/data/including_genWeight/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    signal_csv_path2 = '/work/ehettwer/HiggsMewMew/data/including_genWeight/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv'

    background_log_paths = [
    '/work/ehettwer/KingMaker/data/logs/full_training_samples/Output/WZTo3LNu_mllmin0p1_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
    '/work/ehettwer/KingMaker/data/logs/full_training_samples/Output/WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv9-106X'
    ]

    signal_log_paths = [
    '/work/ehettwer/KingMaker/data/logs/full_sim_samples2/Output/WplusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X',
    '/work/ehettwer/KingMaker/data/logs/full_sim_samples2/Output/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X'
    ]

    
    # Load the data from the CSV files
    background1 = pd.read_csv(background_csv_path1)
    background2 = pd.read_csv(background_csv_path2)

    signal1 = pd.read_csv(signal_csv_path1)
    signal2 = pd.read_csv(signal_csv_path2)

    number_of_events_background1 = parse_log_files(open_multiple_paths([background_log_paths[0] + '/*.txt']))
    number_of_events_background2 = parse_log_files(open_multiple_paths([background_log_paths[1] + '/*.txt']))

    number_of_events_signal1 = parse_log_files(open_multiple_paths([signal_log_paths[0] + '/*.txt']))
    number_of_events_signal2 = parse_log_files(open_multiple_paths([signal_log_paths[1] + '/*.txt']))

    cross_section_background1 = 62.78
    cross_section_background2 = 5.257

    cross_section_signal1 = 0.000867
    cross_section_signal2 = 0.0005412

    generator_weight_background1 = 1-2*0.0368
    generator_weight_background2 = 1-2*0.169

    generator_weight_signal1 = 1-2*0.0276
    generator_weight_signal2 = 1-2*0.0274

    lumi = 59.74 * 1000

    background1 = calculate_weights(background1, cross_section_background1, generator_weight_background1, number_of_events_background1 + number_of_events_background2, lumi)
    background2 = calculate_weights(background2, cross_section_background2, generator_weight_background2, number_of_events_background1 + number_of_events_background2, lumi)

    combined_background = pd.concat([background1, background2], ignore_index=True)

    signal1 = calculate_weights(signal1, cross_section_signal1, generator_weight_signal1, number_of_events_signal1 + number_of_events_signal2, lumi)
    signal2 = calculate_weights(signal2, cross_section_signal2, generator_weight_signal2, number_of_events_signal1 + number_of_events_signal2, lumi)

    combined_signal = pd.concat([signal1, signal2], ignore_index=True)

    histogram_dataset(combined_background, 'm_H', 'weights', 'background')
    histogram_dataset(combined_signal, 'm_H', 'weights', 'signal')

    histogram_signal_background(combined_signal, combined_background, 'm_H', 'weights', 'signal_and_background')



if __name__ == '__main__':
    main()