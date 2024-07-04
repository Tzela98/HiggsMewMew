import csv
import pandas as pd


def count_true_in_column(file_path, column_name):
    """
    Count the number of times the value True appears in the specified column of the CSV file.

    :param file_path: str, path to the CSV file
    :param column_name: str, the name of the column to search for True values
    :return: int, count of True values in the specified column
    """
    true_count = 0

    # Open the CSV file
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Iterate through each row in the CSV
        for row in reader:
            # Check if the value in the specified column is True
            if row[column_name].strip().lower() == 'true':
                true_count += 1

    return true_count


def set_last_column_to_true(input_csv: str, output_csv: str) -> None:
    """
    Reads a CSV file, sets every value in the last column to True, and writes the result to a new CSV file.
    
    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Get the name of the last column
    last_column = df.columns[-1]
    
    # Set all values in the last column to True
    df[last_column] = True
    
    # Write the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)


set_last_column_to_true('/work/ehettwer/HiggsMewMew/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv', '/work/ehettwer/HiggsMewMew/WminusHToMuMu_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X.csv')

# Example usage:
# Assuming the CSV file has a column named 'status' and is located at 'data.csv'
file_path = 'ML/projects/WH_vs_WZ_optimised_parameters_mediumL2_inclusive_mH/WH_vs_WZ_optimised_parameters_mediumL2_inclusive_mH_test.csv'
column_name = 'is_wh'
print(f"The value True appears {count_true_in_column(file_path, column_name)} times in the column '{column_name}'.")

