import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (norm, iqr, skewnorm, cauchy, logistic,anderson,
    gumbel_r,lognorm, t,johnsonsu, gennorm, kstest, expon, gaussian_kde)
from uv_analysis import dip_test, fit_normal, fit_skewnormal, bootstrap_skewnormal_ci, fit_gmm
from matplotlib import cm
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from uv_analysis import dip_test, fit_normal, fit_skewnormal, bootstrap_skewnormal_ci, find_gmm, fit_gmm, fit_gmm2,fit_gmm3


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


def plot_gdpcountry(df, code, year, direction, save_path=None, ax=None):
    """
    Generate a scatter plot showing the involvement of countries in unit values,
    with countries ordered by GDP per capita.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'ln_uv' (log-transformed unit values),
                       'ln_gdp' (log-transformed GDP per capita),
                       and 'partnerISO' (country ISO3 codes).
    code (str): HS code for labeling the plot.
    year (int): Year for labeling the plot.
    direction (str): "import" or "export" to indicate trade direction.
    save_path (str, optional): If provided, saves the plot to this path.
    ax (matplotlib.axes.Axes, optional): If provided, plots on this axis.

    Returns:
    None (Displays or saves the plot)
    """
    #df = df2
    
    # Order countries by GDP per capita
    country_order = df.groupby('partnerISO')['ln_gdp'].mean().sort_values().index
    country_position = {country: i for i, country in enumerate(country_order)}

    # Assign positions for plotting
    df['y_position'] = df['partnerISO'].map(country_position)

    # Create figure if no ax is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 30))

    # Create scatter plot
    ax.scatter(df['ln_uv'], df['y_position'], alpha=0.5, s=10)

    # Format y-axis with country names
    ax.set_yticks(range(len(country_order)))
    ax.set_yticklabels(country_order)

    # Set labels and title
    text_d = "import" if direction.lower() == "import" else "export"
    ax.set_xlabel("Log Unit Value")
    ax.set_ylabel("Countries Ordered by GDP per Capita")
    ax.set_title(f"Unit values in a normal distribution for HS {code} {text_d} in {year}")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save or show the figure
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_gmm1d_country(df, selected_countries, code, year, direction, save_path=None):
    """
    Generate overlapping 1D GMM density plots for selected representative countries using separate axes.

    Parameters:
    df (pd.DataFrame): Data containing 'ln_uv' and 'partnerISO'.
    selected_countries (list): Ordered list of selected countries.
    code (str): HS code for the commodity.
    year (int): Year of analysis.
    direction (str): "import" or "export".
    save_path (str, optional): Path to save the figure.

    Returns:
    None (Displays or saves the plot).
    """
    num_countries = len(selected_countries)
    fig, axes = plt.subplots(nrows=num_countries, ncols=1, figsize=(10, num_countries * 1.5), sharex=True)

    # Ensure axes is always iterable (even if there's only one country)
    if num_countries == 1:
        axes = [axes]

    # Determine global min and max for ln_uv to ensure comparability
    global_min = df[df['partnerISO'].isin(selected_countries)]['ln_uv'].min()
    global_max = df[df['partnerISO'].isin(selected_countries)]['ln_uv'].max()

    for i, country in enumerate(selected_countries):
        subset = df[df['partnerISO'] == country]
        gdp_per_capital = df[df['partnerISO'] == country]['gdp'].mean()
        netWgt = df[df['partnerISO'] == country]['netWgt'].sum()
        ax = axes[i]  # Assign subplot to the current country

        if not subset.empty:
            # Get the best number of GMM components dynamically
            best_component, _ = find_gmm(subset[['ln_uv']], max_components=50, 
                                         convergence_threshold=5, threshold=0.2)

            # Run GMM fitting for this country and pass the corresponding subplot
            fit_gmm(subset, ['ln_uv'], best_component, code, year, direction, plot=True, ax=ax)

            # Add country label inside its subplot
            ax.text(0, 0.4, f'{country}\nTrade volume:{netWgt:.2e} kg\nGDP per capita: {gdp_per_capital:.2e} USD)', fontsize=12, verticalalignment='center', transform=ax.transAxes)
            # Remove individual y-ticks for cleaner ridge appearance
            ax.set_yticks([])

            # Set x-axis range based on global min and max for comparability
            ax.set_xlim(global_min, global_max)
            
            # Remove titles, x and y labels for individual subplots
            ax.set_title("")  # Empty title
            ax.set_xlabel("")  # Empty xlabel
            ax.set_ylabel("")  # Empty ylabel
            
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)


    # Labels and title (only for the bottom subplot)
    axes[-1].set_xlabel("ln (Unit value)")
    
    # Common Y label
    fig.text(0.01, 0.98, "Density", va='top', rotation='vertical', fontsize=12)

    fig.suptitle(f"Unit value distribution of the representative exporter(s) for HS {code} traded in {year}", fontsize=14)

    # Save or show plot
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()