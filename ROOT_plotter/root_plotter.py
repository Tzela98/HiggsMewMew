import sys
import uproot
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def vars_in_file(root_file_path):
    root_file = uproot.open(root_file_path)
    tree = root_file['ntuple']
    variables = tree.keys()
    return variables

def read_root_file(root_file_path, variable):
    events = uproot.open(root_file_path)
    dataframe = events['ntuple'].arrays(variable, library="pd")
    if len(dataframe) == 0:
        raise ValueError(f"Dataset is empty for variable {variable} in file {root_file_path}")
    return dataframe

def remove_cutoff(df, variable, percent):
    threshold = df[variable].quantile(1 - percent / 100)
    filtered_df = df[df[variable] <= threshold]
    return filtered_df

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <root_files_directory>")
        sys.exit(1)

    # Extract the directory path containing root files from command-line arguments
    root_files_dir = sys.argv[1]

    # Ensure the directory exists
    if not os.path.isdir(root_files_dir):
        print(f"Directory {root_files_dir} does not exist!")
        sys.exit(1)

    # List all root files in the directory
    root_files = [os.path.join(root_files_dir, f) for f in os.listdir(root_files_dir) if f.endswith('.root')]

    if not root_files:
        print(f"No root files found in directory {root_files_dir}")
        sys.exit(1)

    # Determine the list of variables from the first root file
    var_list = vars_in_file(root_files[0])

    for var in var_list:
        # Initialize an empty DataFrame to store aggregated data for the current variable
        aggregated_df = pd.DataFrame()

        for root_file in root_files:
            try:
                df = read_root_file(root_file, var)
                aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading variable {var} from file {root_file}: {e}")

        if aggregated_df.empty:
            print(f"No data found for variable {var} across all files.")
            continue

        if aggregated_df[var].dtype in [np.float32, np.float64, np.int32, np.int64]:
            aggregated_df = remove_cutoff(aggregated_df, var, 0.5)

            df_mean = aggregated_df[var].mean()
            df_std = aggregated_df[var].std()
            df_yield = aggregated_df[var].count()

            print(f'Plotting histogram of variable: {var} of type: {aggregated_df[var].dtype}')

            # Plot the histogram
            plt.figure()

            plt.hist(aggregated_df[var], bins=100, label=var, color='teal', histtype='step')
            plt.xlabel(var)
            plt.ylabel("Events")
            
            plt.text(0.75, 0.95, 'Mean: {:.2f}'.format(df_mean), transform=plt.gca().transAxes)
            plt.text(0.75, 0.9, 'Std.: {:.2f}'.format(df_std), transform=plt.gca().transAxes)
            plt.text(0.75, 0.85, 'Events: {:.0f}'.format(df_yield), transform=plt.gca().transAxes)
            
            plt.title("Histogram of " + var)
            
            # Save the plot to the current directory
            thisdir = os.path.dirname(os.path.realpath(__file__))
            plt.savefig(os.path.join(thisdir, var + "_histogram.png"))
            plt.close()

        else:
            print(f'Skipping variable: {var} of type: {aggregated_df[var].dtype}')

if __name__ == "__main__":
    main()
