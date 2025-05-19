import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (norm, iqr, skewnorm, cauchy, logistic,anderson,
    gumbel_r,lognorm, t,gennorm, johnsonsu, gennorm, kstest, expon, gaussian_kde,logistic)
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from uv_analysis import bootstrap_parametric_ci, find_gmm_components, fit_gmm, ensure_cov_matrix,fit_gmm2_flexible
from uv_config import load_config


config = load_config()

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
        unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
        save_path = os.path.join(config["dirs"]["figures"], 
                            f"hist_{code}_{year}_{flow}_{unit_suffix}.png")
        plt.savefig(save_path, dpi=300)
        if ax is None:
            plt.close()   
    else:
        plt.tight_layout()
        plt.show()
        

def plot_dist(
    data,
    code,
    year,
    flow,
    unit_label="USD/kg",
    dist=None,
    best_fit_name=None,
    report_best_fit_uni=None,
    report_all_uni_fit=None,
    raw_params_dict=None,
    ci=None,
    save_path=None,
    ax=None):

    colors = {
        "norm": "#66c2a5", "skewnorm": "#fc8d62", "t": "#8da0cb",
        "gennorm": "#e78ac3", "johnsonsu": "#a6d854", "logistic": "#ffd92f"
    }

    mpl.rcParams['pdf.fonttype'] = 42
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    x = np.linspace(min(data), max(data), 1000)
    hist_output = ax.hist(data, bins='fd', density=True, alpha=0.4, color="gray")
    hist_patch = hist_output[2][0]  # Get the first rectangle patch from the bar container
    handles_dist = [hist_patch]
    labels_dist = ["Histogram"]
    text_d = 'imports' if flow == 'm' else 'exports'


    if not dist and report_all_uni_fit and raw_params_dict:
        for name in set(k.split('_')[0] for k in raw_params_dict if k.endswith('_params')):
            try:
                dist_obj = globals()[name]
                params = raw_params_dict[f"{name}_params"]
                y = dist_obj.pdf(x, *params)
                aic = report_all_uni_fit.get(f"{name}_aic", None)
                bic = report_all_uni_fit.get(f"{name}_bic", None)
                label = f"{name.capitalize()}"
                if aic is not None and bic is not None:
                    label += f" (AIC={aic:.1f}, BIC={bic:.1f})"
                line = ax.plot(x, y, label=label, lw=1.5, alpha=0.9, color=colors.get(name))[0]
                handles_dist.append(line)
                labels_dist.append(label)
                if name == best_fit_name:
                    fitted_color = line.get_color()
            except (KeyError, ValueError):
                continue
        ax.set_title(f"Distribution fits of unit values ({unit_label}) for HS {code} {text_d} in {year}")

    elif dist and report_best_fit_uni and raw_params_dict and f"{dist}_params" in raw_params_dict:
        try:
            dist_obj = globals()[dist]
            params = raw_params_dict[f"{dist}_params"]
            y = dist_obj.pdf(x, *params)
            line = ax.plot(x, y, lw=2, label=f"{dist.capitalize()} fit", color=colors.get(dist))[0]
            fitted_color = line.get_color()
            ax.set_title(f"Unit values ({unit_label}) in a {dist.capitalize()} distribution for HS {code} {text_d} in {year}")
        except (KeyError, ValueError):
            pass

    handles_stats, labels_stats = [], []
    if report_best_fit_uni and best_fit_name and ci:
        mean = report_best_fit_uni[f"{best_fit_name}_mean"]
        median = report_best_fit_uni[f"{best_fit_name}_median"]
        mode = report_best_fit_uni[f"{best_fit_name}_mode"]
        var = report_best_fit_uni[f"{best_fit_name}_variance"]
        sample_var = report_best_fit_uni[f"{best_fit_name}_sample_variance"]
        skew = report_best_fit_uni.get(f"{best_fit_name}_skew", None)
        kurt = report_best_fit_uni.get(f"{best_fit_name}_kurtosis", None)

        dummy = Line2D([], [], linestyle="none")
        extra_labels = [
            f"Best fit (AIC/BIC): {best_fit_name.capitalize()}"
        ]

        handles_stats.append(dummy)
        labels_stats.append(extra_labels[0])  # only the AIC/BIC line

        # Now the vertical lines (mean, median, mode)
        handles_stats.append(ax.axvline(mean, color=fitted_color, linestyle='--'))
        labels_stats.append(
            f'Mean: {mean:.3f} ({np.exp(mean):.3f} {unit_label})\n'
            f'95% CI: ({np.exp(ci["ci_mean_lower"]):.3f}, {np.exp(ci["ci_mean_upper"]):.3f})'
        )

        handles_stats.append(ax.axvline(median, color=fitted_color, linestyle=':'))
        labels_stats.append(
            f'Median: {median:.3f} ({np.exp(median):.3f} {unit_label})\n'
            f'95% CI: ({np.exp(ci["ci_median_lower"]):.3f}, {np.exp(ci["ci_median_upper"]):.3f})'
        )

        handles_stats.append(ax.axvline(mode, color=fitted_color, linestyle=(0, (3, 1, 1, 1, 1, 1))))
        labels_stats.append(
            f'Mode: {mode:.3f} ({np.exp(mode):.3f} {unit_label})\n'
            f'95% CI: ({np.exp(ci["ci_mode_lower"]):.3f}, {np.exp(ci["ci_mode_upper"]):.3f})'
        )

        # Now add the other stats
        extra_labels = [
            f"Variance: {var:.3f} (95% CI: {ci['ci_variance_lower']:.3f}–{ci['ci_variance_upper']:.3f})",
            f"Sample variance: {sample_var:.3f}"
        ]
        if skew is not None:
            extra_labels.append(f"Skewness: {skew:.3f}")
        if kurt is not None:
            extra_labels.append(f"Kurtosis: {kurt:.3f}")

        handles_stats.extend([dummy] * len(extra_labels))
        labels_stats.extend(extra_labels)

    xlabel = f"ln(Unit Value) [{unit_label}]" 
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    if handles_dist:
        leg1 = ax.legend(handles_dist, labels_dist, loc="upper left", 
                    fontsize=8, title="Candidate distributions")
        ax.add_artist(leg1)
    if handles_stats:
        ax.legend(handles_stats, labels_stats, loc="upper right", 
                   fontsize=8, title="Best-fit parameters")

    plt.tight_layout()
    if save_path:
        unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
        save_path = os.path.join(config["dirs"]["figures"], 
           f"fit_{code}_{year}_{flow}_{unit_suffix}_{best_fit_name}.png")
        plt.savefig(save_path, dpi=300)
        if ax is None:
            plt.close()
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