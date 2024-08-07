from tkinter import font
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
    sample_sizes = [30, 250, len(df)]
    
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
        axs[i].hist(df_sample['Height_cm'], bins=(i+1)*5, density=True, histtype='step', linewidth=1.5, label='Histogram')
        
        # Fit a normal distribution using MLE
        mean_mle, std_dev_mle = stats.norm.fit(df_sample['Height_cm'])
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p_mle = stats.norm.pdf(x, mean_mle, std_dev_mle)
        
        # Plot the MLE normal distribution in red
        axs[i].plot(x, p_mle, 'r-', linewidth=1.5, label='MLE Fit')
        
        # Calculate and plot the normal distribution using sample mean and variance (point estimates)
        mean_point = df_sample['Height_cm'].mean()
        std_dev_point = df_sample['Height_cm'].std()
        p_point = stats.norm.pdf(x, mean_point, std_dev_point)
        
        # Plot the point estimate normal distribution in blue
        axs[i].plot(x, p_point, 'b--', linewidth=1.5, label='Point Estimate')
        
        # Add text with mean and std dev for MLE
        textstr = '\n'.join((
            r'$\mu_{MLE}=%.2f$ cm' % (mean_mle, ),
            r'$\sigma_{MLE}=%.2f$ cm' % (std_dev_mle, ),
            r'$\mu_{point}=%.2f$ cm' % (mean_point, ),
            r'$\sigma_{point}=%.2f$ cm' % (std_dev_point, )))
        props = dict(boxstyle='round', alpha=0.5)
        axs[i].text(0.05, 0.95, textstr, transform=axs[i].transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
        
        axs[i].set_xlabel('Height (cm)')
        axs[i].set_ylabel('Density')
        axs[i].grid(True)
        axs[i].legend(fontsize='xx-small')
    
    plt.tight_layout()
    plt.savefig('thesis_plots/Plots/height_distribution_subplots.png')
    print('Saved plot to thesis_plots/Plots/height_distribution_subplots.png')


def calculate_confidence_intervals(data):
    # Calculate the sample mean and standard deviation
    sample_mean = np.mean(data)
    sample_std_dev = np.std(data, ddof=1)

    t_critical = stats.t.ppf(q = 0.975, df=24)  # Get the t-critical value*

    # Calculate the standard error of the mean
    standard_error = sample_std_dev / np.sqrt(len(data))
    margin_of_error = t_critical * standard_error

    # Calculate the 95% confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return sample_mean, standard_error, lower_bound, upper_bound


def plot_mean_confidence_intervals(csv_file):
    data = pd.read_csv(csv_file)
    data['Height_cm'] = data['Height'] * 2.54
    data = data.sample(frac=1, random_state=42)

    true_mean = data['Height_cm'].mean()

    # Draw 100 datasets of 30 without double counting
    sample_means = []
    sample_std_errors = []
    lower_bounds = []
    upper_bounds = []
    intervals_not_touching_true_mean = 0

    number_of_samples = 200
    for i in range(number_of_samples):
        sample = data.sample(n=30, replace=False, random_state=i)
        sample_mean, standard_error, lower_bound, upper_bound = calculate_confidence_intervals(sample['Height_cm'])
        sample_means.append(sample_mean)
        sample_std_errors.append(standard_error)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

        # Check if the confidence interval does not touch the true mean
        if upper_bound < true_mean or lower_bound > true_mean:
            intervals_not_touching_true_mean += 1

    # Calculate the percentage of intervals not touching the true mean
    percentage_not_touching_true_mean = (intervals_not_touching_true_mean / 100) * 100
    print(f"Percentage of confidence intervals not touching the true mean: {percentage_not_touching_true_mean:.2f}%")

    # Plot the sample means with 95% confidence intervals
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    axs[0].errorbar(range(1, number_of_samples + 1), sample_means, yerr=[np.array(sample_means) - np.array(lower_bounds), np.array(upper_bounds) - np.array(sample_means)], fmt='o', markersize=3, label='Sample Mean')
    axs[0].axhline(y=true_mean, color='r', linestyle='--', label='True Mean')
    axs[0].set_xlabel('Sample Number')
    axs[0].set_ylabel('Height (cm)')
    axs[0].legend()

    # Plot histogram of sample means
    mean_of_sample_means = np.mean(sample_means)
    std_dev_of_sample_means = np.std(sample_means)
    two_sigma_lower = mean_of_sample_means - 2 * std_dev_of_sample_means
    two_sigma_upper = mean_of_sample_means + 2 * std_dev_of_sample_means

    axs[1].hist(sample_means, bins=5, alpha=1, histtype='step')
    axs[1].axvline(mean_of_sample_means, color='r', linestyle='--', label='Mean of Sample Means', linewidth=1)
    axs[1].axvline(true_mean, color='orange', linestyle='--', label='True Mean', linewidth=1)
    axs[1].axvline(two_sigma_lower, color='g', linestyle='--', label='2-Sigma Lower Bound', linewidth=1)
    axs[1].axvline(two_sigma_upper, color='g', linestyle='--', label='2-Sigma Upper Bound', linewidth=1)
    axs[1].set_xlabel('Sample Mean')
    axs[1].set_ylabel('Frequency')
    axs[1].legend(fontsize='x-small')

    plt.tight_layout()
    plt.savefig('thesis_plots/Plots/height_confidence_intervals.png')
    plt.close()

    print('Saved plot to thesis_plots/Plots/height_confidence_intervals.png')




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
plot_mean_confidence_intervals('/work/ehettwer/HiggsMewMew/thesis_plots/SOCR-HeightWeight.csv')