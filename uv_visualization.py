import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (norm, iqr, skewnorm, cauchy, logistic,anderson,
    gumbel_r,lognorm, t,gennorm, johnsonsu, gennorm, kstest, expon, gaussian_kde,logistic)
from matplotlib import cm
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from uv_analysis import fit_all_unimodal_models, fit_logistic,dip_test, fit_normal, fit_skewnormal, fit_studentt, fit_gennorm, fit_johnsonsu, bootstrap_parametric_ci, find_gmm_components, fit_gmm, ensure_cov_matrix,fit_gmm2_flexible,fit_gmm3

def plot_histogram(data, code, year, flow, unit_label="USD/kg", save_path=None, ax=None):
    """
    Plot a histogram with customizable options and Freedman-Diaconis rule for bin width.
    
    Args:
        data: Array-like dataset to plot the histogram for.
        code: HS code for title annotation.
        year: Year of the data for title annotation.
        flow: 'm' for import or 'x' for export.
        unit_label: Label for the x-axis unit (e.g., 'USD/kg' or 'USD/u' etc.).
        save_path: Path to save the figure. If None, it will display instead.
        ax: Optional matplotlib Axes object to plot on.
    """
    mpl.rcParams['pdf.fonttype'] = 42                                         # Set rcParams to ensure editable text in the PDF
    data = np.asarray(data)  # Ensure it's a NumPy array for slicing speed
    data = data[~np.isnan(data)]  # Drop NaNs if any

    # Efficient bin computation using Freedman-Diaconis rule
    q75, q25 = np.percentile(data, [75, 25])
    iqrg = q75 - q25
    n = len(data)

    if iqrg == 0 or n < 2:
        bin_edges = 10  # fallback to 10 bins if not enough spread
    else:
        bin_width = 2 * iqrg / (n ** (1 / 3))
        bin_count = int(np.ceil((data.max() - data.min()) / bin_width))
        bin_edges = bin_count if bin_count > 0 else 10
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))  # Create a new figure
    
    # Step 3: Plot histogram
    ax.hist(data, bins=bin_edges, edgecolor='black', alpha=0.6)
    text_d = 'imports' if flow == 'm' else 'exports'
    ax.set_title(f"Histogram of unit values for HS {code} {text_d} in {year}")
    ax.set_xlabel(f"ln(Unit Price) [{unit_label}]")
    ax.set_ylabel("Counts")
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Save or show the plot
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if ax is None:
            plt.close()   
    else:
        plt.tight_layout()
        plt.show()
        

def plot_dist(data, code, year, flow,
              dist=None, best_fit=None, fit_result=None, all_results=None,
              ci_mean=None, ci_median=None, ci_mode=None, ci_var=None,
              save_path=None, ax=None):
    """
    Unified plot function for any supported unimodal distribution.
    Can plot a single fit or compare all fits with AIC/BIC.

    Args:
        data (array-like): The log-transformed unit values.
        code (str): HS code for labeling.
        year (int): Year for labeling.
        flow (str): 'm' for import, 'x' for export.
        dist (str): Distribution name (used when comparing single fit).
        best_fit (str): Best-fit distribution name for highlighting.
        fit_result (dict): Fit result for the selected distribution.
        all_results (dict): All fit results (for multi-fit plot).
        ci_mean (tuple): CI for mean.
        ci_median (tuple): CI for median.
        ci_mode (tuple): CI for mode.
        ci_variance (tuple): CI for parametric variance.
        save_path (str): Optional file path to save the plot.
        ax (matplotlib axis): Optional axis for subplotting.
    """

    mpl.rcParams['pdf.fonttype'] = 42
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(min(data), max(data), 1000)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)

    ax.hist(data, bins=bin_edges, density=True, alpha=0.4, label="Data", color="gray")
    text_d = 'imports' if flow == 'm' else 'exports'

    colors = {
        'normal': '#e41a1c',
        'skewnorm': '#377eb8',
        'studentt': '#4daf4a',
        'gennorm': '#984ea3',
        'johnsonsu': '#ff7f00',
        'logistic': 'brown'
    }

    # Compare all distributions
    if not dist and all_results:
        for name, res in all_results.items():
            color = colors.get(name, None)
            if name == 'normal':
                y = norm.pdf(x, res['mu_norm'], res['sigma_norm'])
                label = f"Normal (AIC={res['aic_norm']:.1f}, BIC={res['bic_norm']:.1f})"
            elif name == 'skewnorm':
                y = skewnorm.pdf(x, res['a_skew'], res['loc_skew'], res['scale_skew'])
                label = f"Skewnorm (AIC={res['aic_skewnorm']:.1f}, BIC={res['bic_skewnorm']:.1f})"
            elif name == 'studentt':
                y = t.pdf(x, res['df_t'], res['loc_t'], res['scale_t'])
                label = f"Student-t (AIC={res['aic_t']:.1f}, BIC={res['bic_t']:.1f})"
            elif name == 'gennorm':
                y = gennorm.pdf(x, res['beta_gn'], res['loc_gn'], res['scale_gn'])
                label = f"GenNorm (AIC={res['aic_gn']:.1f}, BIC={res['bic_gn']:.1f})"
            elif name == 'johnsonsu':
                y = johnsonsu.pdf(x, res['a_jsu'], res['b_jsu'], res['loc_jsu'], res['scale_jsu'])
                label = f"Johnson SU (AIC={res['aic_jsu']:.1f}, BIC={res['bic_jsu']:.1f})"
            elif name == 'logistic':
                y = logistic.pdf(x, res['loc_log'], res['scale_log'])
                label = f"Logistic (AIC={res['aic_logistic']:.1f}, BIC={res['bic_logistic']:.1f})"
            else:
                continue

            lw = 2.5 if name == best_fit else 1.5
            ax.plot(x, y, label=label, lw=lw, alpha=0.9 if name == best_fit else 0.6, color=color)

        ax.set_title(f"Distribution fits (AIC/BIC) for HS {code} {text_d} in {year}")

        # Highlight vertical lines for best-fit statistics
        if fit_result and ci_mean and ci_median and ci_mode and ci_var:
            mean_key = [k for k in fit_result if k.startswith('mean')][0]
            mean = fit_result[mean_key]
            median = fit_result[[k for k in fit_result if 'median' in k][0]]
            mode = fit_result[[k for k in fit_result if 'mode' in k][0]]
            var_key = [k for k in fit_result if k.startswith('variance')][0]
            var = fit_result[var_key]
            sample_var_key = [k for k in fit_result if k.startswith('sample_variance')][0]
            sample_var = fit_result[sample_var_key]
        

            ax.axvline(mean, color='black', linestyle='--',
                       label=f'Mean: {mean:.3f} ({np.exp(mean):.3f} USD/kg)\n95%CI: ({np.exp(ci_mean[0]):.3f}, {np.exp(ci_mean[1]):.3f})')
            ax.axvline(median, color='black', linestyle=':', 
                       label=f'Median: {median:.3f} ({np.exp(median):.3f} USD/kg)\n95%CI: ({np.exp(ci_median[0]):.3f}, {np.exp(ci_median[1]):.3f})')
            ax.axvline(mode, color='black', linestyle=(0, (3, 1, 1, 1, 1, 1)),
                   label=f'Mode (log): {mode:.3f} ({np.exp(mode):.3f} USD/kg)\n95%CI: ({np.exp(ci_mode[0]):.3f}, {np.exp(ci_mode[1]):.3f})')
            ax.text(0.75, 0.35,
                    f"Variance: {var:.4f}\n95%CI: ({ci_var[0]:.4f}, {ci_var[1]:.4f})\n"
                    f"Sample variance: {sample_var:.4f}",
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=10 )

    else:
        # Plot single fit distribution
        if dist == 'normal' and fit_result:
            y = norm.pdf(x, fit_result['mu_norm'], fit_result['sigma_norm'])
        elif dist == 'skewnorm' and fit_result:
            y = skewnorm.pdf(x, fit_result['a_skew'], fit_result['loc_skew'], fit_result['scale_skew'])
        elif dist == 'studentt' and fit_result:
            y = t.pdf(x, fit_result['df_t'], fit_result['loc_t'], fit_result['scale_t'])
        elif dist == 'gennorm' and fit_result:
            y = gennorm.pdf(x, fit_result['beta_gn'], fit_result['loc_gn'], fit_result['scale_gn'])
        elif dist == 'johnsonsu' and fit_result:
            y = johnsonsu.pdf(x, fit_result['a_jsu'], fit_result['b_jsu'], fit_result['loc_jsu'], fit_result['scale_jsu'])
        elif dist == 'logistic' and fit_result:
            y = logistic.pdf(x, fit_result['loc_log'], fit_result['scale_log'])
        else:
            raise ValueError("Unsupported distribution or missing fit_result.")

        ax.plot(x, y, lw=2, label=f"{dist.capitalize()} fit")
        ax.set_title(f"Unit values in a {dist.capitalize()} distribution for HS {code} {text_d} in {year}")

        # Plot vertical lines if CI provided
        if fit_result and ci_mean and ci_median and ci_mode and ci_var:
            mean_key = [k for k in fit_result if k.startswith('loc') or k.startswith('mean')][0]
            mean = fit_result[mean_key]
            median = fit_result[[k for k in fit_result if 'median' in k][0]]
            mode = fit_result[[k for k in fit_result if 'mode' in k][0]]
            var_key = [k for k in fit_result if k.startswith('variance')][0]
            var = fit_result[var_key]
            sample_var_key = [k for k in fit_result if k.startswith('sample_variance')][0]
            sample_var = fit_result[sample_var_key]

            ax.axvline(mean, color='black', linestyle='--',
                       label=f'Mean: {mean:.3f} ({np.exp(mean):.3f} USD/kg)\n95%CI: ({np.exp(ci_mean[0]):.3f}, {np.exp(ci_mean[1]):.3f})')
            ax.axvline(median, color='black', linestyle=':', 
                       label=f'Median: {median:.3f} ({np.exp(median):.3f} USD/kg)\n95%CI: ({np.exp(ci_median[0]):.3f}, {np.exp(ci_median[1]):.3f})')
            ax.axvline(mode, color='black', linestyle=(0, (3, 1, 1, 1, 1, 1)),
                   label=f'Mode (log): {mode:.3f} ({np.exp(mode):.3f} USD/kg)\n95%CI: ({np.exp(ci_mode[0]):.3f}, {np.exp(ci_mode[1]):.3f})')
            ax.text(0.75, 0.35,
                    f"Variance: {var:.4f}\n95%CI: ({ci_var[0]:.4f}, {ci_var[1]:.4f})\n"
                    f"Sample variance: {sample_var:.4f}",
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=10 )

    ax.set_xlabel("ln(Unit Price)")
    ax.set_ylabel("Density")
    ax.legend(loc='best')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
def plot_gdpcountry(df, code, year, flow, save_path=None, ax=None):
    """
    Generate a scatter plot showing the involvement of countries in unit values,
    with countries ordered by GDP per capita.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'ln_uv' (log-transformed unit values),
                       'ln_gdp' (log-transformed GDP per capita),
                       and 'partnerISO' (country ISO3 codes).
    code (str): HS code for labeling the plot.
    year (int): Year for labeling the plot.
    flow (str): "import" or "export" to indicate trade flow.
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
    text_d = "import" if flow.lower() == "import" else "export"
    ax.set_xlabel("Log Unit Value")
    ax.set_ylabel("Countries Ordered by GDP per Capita")
    ax.set_title(f"Unit values in a normal distribution for HS {code} {text_d} in {year}")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save or show the figure
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_gmm1d_country(df, selected_countries, code, year, flow, save_path=None):
    """
    Generate overlapping 1D GMM density plots for selected representative countries using separate axes.

    Parameters:
    df (pd.DataFrame): Data containing 'ln_uv' and 'partnerISO'.
    selected_countries (list): Ordered list of selected countries.
    code (str): HS code for the commodity.
    year (int): Year of analysis.
    flow (str): "import" or "export".
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
            fit_gmm(subset, ['ln_uv'], best_component, code, year, flow, plot=True, ax=ax)

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