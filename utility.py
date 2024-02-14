from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import csv
import os


# Define function to parse a log file for the event size processed by an hpcondor job
# The function takes the path to the log file (str) as an argument and return the event size (int)
def parse_txt_files(file_paths):
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

    return results


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


def total_lines_in_csv(files):
    total_lines = 0

    for file_path in tqdm(files):
        try:
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                total_lines += sum(1 for _ in reader)
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    return total_lines

