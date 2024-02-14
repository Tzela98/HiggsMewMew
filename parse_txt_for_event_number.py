from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time


# Define function to parse a log file for the event size processed by an hpcondor job
# The function takes the path to the log file (str) as an argument and return the event size (int)
def parse_txt_file(file_path):
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

    return result


def fraction_failed_jobs(paths):
    total_files = len(paths)
    successful_searches = 0

    for file_path in paths:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

            # Check if the line before last line ends with "job exit code : 0"
            if lines[-2].strip() == "job exit code : 0":
                successful_searches += 1

    # Calculate the fraction
    fraction_found = successful_searches / total_files if total_files > 0 else 0
    fraction_failed = 1 - fraction_found
    return fraction_failed


def open_multiple_paths(paths: list):
    all_paths = []
    for path in paths:
        all_paths = all_paths + glob.glob(path, recursive=True)
    return sorted(all_paths)


if __name__ == '__main__':
    # Define the path to the directory containing the log files
    file_path = '/work/ehettwer/KingMaker/data/logs/DYJetsToLL_inclusive_genweights/Output/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIIAutumn18NanoAODv7-Nano02Apr2020_ext2/*.txt'
    list_of_paths = open_multiple_paths([file_path])

    # Initialize a list to store the event numbers
    number_of_events = []
    for log_file in list_of_paths:
        print(log_file)
        events = parse_txt_file(log_file)
        if events is not None:
            number_of_events.append(events)

    failed_jobs = fraction_failed_jobs(list_of_paths)
    sum_of_all_events = sum(number_of_events)

    print('the sum of all events is:', sum_of_all_events)
    print('the fraction of failed jobs is:', failed_jobs)

