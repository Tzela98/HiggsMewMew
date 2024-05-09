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
    print("Variables in the file: ", variables)
    return variables

def read_root_file(root_file_path, variable):
    events = uproot.open(root_file_path)
    dataframe = events['ntuple'].arrays(variable, library="pd")
    if len(dataframe) == 0:
        raise ValueError("Dataset is empty!")
    return dataframe

def remove_cutoff(df, variable, percent):
    threshold = df[variable].quantile(1 - percent / 100)
    filtered_df = df[df[variable] <= threshold]
    return filtered_df

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <root_file_path> <variable_name>")
        sys.exit(1)

    # Extract root file path and variable name from command-line arguments
    root_file_path = sys.argv[1]

    var_list = vars_in_file(root_file_path)

    for var in var_list:
        df = read_root_file(root_file_path, var)

        if df[var].dtype in [np.float32, np.float64, np.int32, np.int64]:

            df = remove_cutoff(df, var, 0.5)

            df_mean = df[var].mean()
            df_std = df[var].std()
            df_yield = df[var].count()

            print('plotting histogram of variable: ', var, 'of type: ', df[var].dtype)

            # Plot the histogram
            plt.figure()

            plt.hist(df[var], bins=100, label=var, color='teal', histtype='step')
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
            print('Skipping variable: ', var, ' of type: ', df[var].dtype)

if __name__ == "__main__":
    main()