import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep  # for plotting in HEP style

# Set the plotting style to CMS style
hep.style.use(hep.style.CMS)
# Place the CMS label at location 0 (top-left corner)
hep.cms.label(loc=0)

def draw_histograms(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate over each file
    for file in files:
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            # Extract the name before the first underscore in the CSV file name
            plot_name = file.split('_')[0]
            
            # Read the CSV file into a pandas DataFrame
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # Create a folder to save histograms
            output_folder = 'WH_analysis/tmp_plots'
            os.makedirs(output_folder, exist_ok=True)

            # Iterate over each column in the DataFrame
            for column in df.columns:
                # Check if the column contains numerical data
                if df[column].dtype in ['int64', 'float64']:
                    print(f'Plotting histogram for {column} in {plot_name}')
                    hist, bins = np.histogram(df[column], bins=30)
                    plt.figure(figsize=(8, 6))
                    hep.histplot(hist, bins, yerr=False, label='Data')
                    plt.xlabel(column)
                    plt.ylabel('Events')
                    # Save histogram plot with desired name
                    plt.savefig(os.path.join(output_folder, f'{plot_name}_{column}_histogram.png'))
                    plt.close()

# Provide the path to the folder containing CSV files here
folder_path = '/ceph/ehettwer/working_data/signal_region'
draw_histograms(folder_path)
