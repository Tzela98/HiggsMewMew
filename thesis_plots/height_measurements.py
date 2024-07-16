import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats
import mplhep as hep
from sympy import plot

hep.style.use(hep.style.CMS)
hep.cms.label(loc=0)

def plot_height_histogram(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert height from inches to cm
    df['Height_cm'] = df['Height'] * 2.54
    
    # Sample sizes to consider
    sample_sizes = [50, 500, len(df)]
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(22, 8))
    
    # Plotting for each sample size
    for i, size in enumerate(sample_sizes):
        # Take a random sample of the specified size
        if size < len(df):
            df_sample = df.sample(n=size, random_state=42)
        else:
            df_sample = df
        
        # Plot histogram of height in cm
        axs[i].hist(df_sample['Height_cm'], bins=30, density=True, histtype='step', linewidth=1.5)
        
        # Fit a normal distribution to the data
        mean, std_dev = stats.norm.fit(df_sample['Height_cm'])
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean, std_dev)
        
        # Plot the fitted normal distribution
        axs[i].plot(x, p, linewidth=1.5)
        
        # Add text with mean and std dev
        textstr = '\n'.join((
            r'$\mu=%.2f$ cm' % (mean, ),
            r'$\sigma=%.2f$ cm' % (std_dev, )))
        props = dict(boxstyle='round', alpha=0.5)
        axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
        
        axs[i].set_xlabel('Height (cm)')
        axs[i].set_ylabel('Density')
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('thesis_plots/Plots/height_distribution_subplots.png')



def generate_height_data():
    # Set parameters for the normal distribution
    average_height = 175  # average height in cm
    std_dev = 10  # standard deviation in cm

    # Define sample sizes
    sample_sizes = [30, 100, 10000]

    # Generate random height data for different sample sizes
    samples = [np.random.normal(average_height, std_dev, size) for size in sample_sizes]

    # Generate x values for the normal distribution curve
    x = np.linspace(average_height - 4*std_dev, average_height + 4*std_dev, 1000)
    # Calculate the y values for the normal distribution curve
    y = norm.pdf(x, average_height, std_dev)

    # Plot histograms for each sample size with the normal distribution curve
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax, sample, size in zip(axes, samples, sample_sizes):
        ax.hist(sample, bins=30, density=True, histtype='step', label='Sample Data')
        ax.plot(x, y, lw=1.5, label='Normal Distribution')
        ax.set_title(f'Sample Size: {size}')
        ax.set_xlabel('Height (cm)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize='xx-small', loc='upper right')

    # Show plot
    plt.tight_layout()
    plt.savefig('thesis_plots/Plots/height_distribution_sample_size.png')



plot_height_histogram('/work/ehettwer/HiggsMewMew/thesis_plots/SOCR-HeightWeight.csv')