import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (norm, iqr, skewnorm, cauchy, logistic,anderson,
    gumbel_r,lognorm, t,johnsonsu, gennorm, kstest, expon, gaussian_kde)
from uv_analysis import dip_test, fit_normal, fit_skewnormal, bootstrap_skewnormal_ci, fit_gmm
from matplotlib import cm
from sklearn.mixture import GaussianMixture
import matplotlib as mpl

def plot_histogram(data, code, year, direction, save_path=None, ax=None):
    """
    Plot a histogram with customizable options and Freedman-Diaconis rule for bin width.
    
    Args:
    data: Array-like dataset to plot the histogram for.
    title: Title of the plot (default is a descriptive title).
    xlabel: Label for the x-axis (default is 'ln(Unit Price)').
    ylabel: Label for the y-axis (default is 'Frequency').
    alpha: Transparency level for the bars (default is 0.7).
    grid: Whether to display gridlines (default is True).
    save_path: If provided, saves the plot to the specified path (default is None).
    ax: If provided, uses the given Axes object to plot on it. Otherwise, creates a new figure.
    """
    mpl.rcParams['pdf.fonttype'] = 42                                         # Set rcParams to ensure editable text in the PDF
    # Step 1: Calculate IQR and bin width using Freedman-Diaconis Rule
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)  # Step 2: Compute bin edges
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure
    
    # Step 3: Plot histogram
    ax.hist(data, bins=bin_edges, edgecolor='black', alpha=0.7)
    text_d = 'imports' if direction == 'm' else 'exports'
    ax.set_title(f"Histogram of unit values for HS {code} {text_d} in {year}")
    ax.set_xlabel("ln(Unit Price)")
    ax.set_ylabel("Counts")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_gmm_bic(bic_values, max_components=50, save_path=None, ax=None):
    """
    Plot the BIC values for different numbers of components in GMM.
    
    Args:
    bic_values: List of BIC values for different components.
    max_components: The maximum number of components considered (default is 50).
    title: The title of the plot (default is a descriptive title).
    color: Color of the plot line (default is 'blue').
    save_path: If provided, saves the plot to the specified path (default is None).
    ax: If provided, uses the given Axes object to plot on it. Otherwise, creates a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no ax is passed
        
    x_values = range(2, len(bic_values) + 2)
    ax.plot(x_values, bic_values, marker='o', linestyle='-', color='blue')
    ax.set_title("BIC for Different Numbers of Components in GMM")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("BIC Value")
    ax.grid(True)
    
    best_component_index = np.argmin(bic_values)
    ax.axvline(x=2 + best_component_index, color='red', linestyle='--', label=f"Best Component: {2 + best_component_index}")
    ax.legend()
    ax.set_xticks(x_values)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_skn(data, code, year, direction, ci_median=None,ci_mode=None, save_path=None, ax=None):
    
    mpl.rcParams['pdf.fonttype'] = 42                                         # Set rcParams to ensure editable text in the PDF
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no ax is passed
    
    x = np.linspace(min(data), max(data), 1000)
    
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)  # Step 2: Compute bin edges
   
    ax.hist(data, bins=bin_edges, density=True, alpha=0.5, label="Data", color="gray")
    
    re = fit_skewnormal(data, pt=None)
    if ci_median is None or ci_mode is None:
        ci_median, ci_mode = bootstrap_skewnormal_ci(data, n_bootstraps=1000)
    
    ax.plot(x, skewnorm.pdf(x, re['a_skew'], re['loc_skew'], re['scale_skew']), lw=2, label='Skew-normal fit')

    # Highlight mean, median, and mode
    a,b,c,d = re['mean_skew'],re['median_skew'],re['mode_skew'],re['ci95_mean']
    ax.axvline(a, color='orange', linestyle='--', label=f'Mean: {a:.3f} (log),\n{np.exp(a):.3f} USD/kg, 95% CI: ({np.exp(d[0]):.3f}, {np.exp(d[1]):.3f})')
    ax.axvline(b, color='purple', linestyle='--', label=f'Median: {b:.3f} (log),\n{np.exp(b):.3f} USD/kg, 95% CI: ({np.exp(ci_median[0]):.3f}, {np.exp(ci_median[1]):.3f})')
    ax.axvline(c, color='green', linestyle='--', label=f'Mode: {c:.3f} (log),\n{np.exp(c):.3f} USD/kg, 95% CI: ({np.exp(ci_mode[0]):.3f}, {np.exp(ci_mode[1]):.3f})')
    
    text_d = 'imports' if direction == 'm' else 'exports'
    ax.set_title(f"Unit values in a skew-normal distribution for HS {code} {text_d} in {year}")
    ax.set_xlabel("ln(Unit Price)")
    ax.set_ylabel("Density")
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_n(data, code, year, direction, save_path=None, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no ax is passed
    
    x = np.linspace(min(data), max(data), 1000)
    
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)  # Step 2: Compute bin edges
   
    ax.hist(data, bins=bin_edges, density=True, alpha=0.5, label="Data", color="gray")
    
    re = fit_normal(data, pt=None)
    
    ax.plot(x, norm.pdf(x, re['mu_norm'], re['sigma_norm']), lw=2, label='Normal fit')

    # Highlight mean, median, and mode
    a,b = re['mu_norm'],re['ci_mean_norm']
    ax.axvline(a, color='orange', linestyle='--', label=f'Mean: {a:.3f} (log)),\n{np.exp(a):.3f} USD/kg, 95% CI: ({np.exp(b[0]):.3f}, {np.exp(b[1]):.3f})')
    
    text_d = 'imports' if direction == 'm' else 'exports'
    ax.set_title(f"Unit values in a normal distribution for HS {code} {text_d} in {year}")
    ax.set_xlabel("ln(Unit Price)")
    ax.set_ylabel("Density")
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

