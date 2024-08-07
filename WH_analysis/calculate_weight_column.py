from email import generator
from networkx import number_attracting_components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import glob

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


def calculate_weights(dataset, cross_section, generator_weight, number_of_events, lumi=59.74 * 1000):
    id_iso_wgt = dataset['id_wgt_mu_1'] * dataset['iso_wgt_mu_1'] * dataset['id_wgt_mu_2'] * dataset['iso_wgt_mu_2']
    acceptance = dataset['genWeight'] / (abs(dataset['genWeight']) * generator_weight * number_of_events)
    weight = id_iso_wgt * acceptance * lumi * cross_section
    
    # Insert the new column 'weights' before the last column
    last_col_idx = len(dataset.columns) - 1
    dataset.insert(last_col_idx, 'weights', weight)
    
    return dataset


def main():
    data_path = '/work/ehettwer/HiggsMewMew/data/including_genWeight/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv'
    log_path = '/work/ehettwer/KingMaker/data/logs/full_training_samples/Output/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X'

    data = pd.read_csv(data_path)

    number_of_events = parse_log_files(open_multiple_paths([log_path + '/*.txt']))
    cross_section = 1.325
    generator_weight = 1 - 2*0.004985

    lumi = 59.74 * 1000

    data = calculate_weights(data, cross_section, generator_weight, number_of_events, lumi)
    data.to_csv('/work/ehettwer/HiggsMewMew/ZZTo4L_TuneCP5_13TeV_powheg_pythia8_RunIISummer20UL18NanoAODv9-106X.csv', index=False)

    print('Done!')


if __name__ == '__main__':
    main()