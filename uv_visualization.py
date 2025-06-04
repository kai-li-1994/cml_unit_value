import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from uv_config import load_config
from scipy.stats import norm, skewnorm, t, gennorm, johnsonsu, logistic
config = load_config()

def plot_histogram(data, code, year, flow, unit_label="USD/kg", save_path=None, ax=None, file_format="png"):
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
        file_format: File format for saving the figure (e.g., 'png', 'pdf', 'svg').
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
    ax.hist(data, bins=bin_edges, color='lightgray', edgecolor = 'white')
    text_d = 'imports' if flow == 'm' else 'exports'
    ax.set_title(f"Histogram of unit values ({unit_label}) for HS {code} {text_d} in {year}")
    ax.set_xlabel(f"ln(Unit Price) [{unit_label}]")
    ax.set_ylabel("Counts")
    ax.text(0.75, 0.9, f"Sample size: {len(data):,}", transform=ax.transAxes)
    #ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Save or show the plot
    if save_path:
        plt.tight_layout()
        unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
        save_path = os.path.join(config["dirs"]["figures"], 
                            f"hist_{code}_{year}_{flow}_{unit_suffix}.{file_format}")
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
    ax=None,
    file_format="png"):

    colors = {
        "norm": "#66c2a5", "skewnorm": "#fc8d62", "t": "#8da0cb",
        "gennorm": "#e78ac3", "johnsonsu": "#a6d854", "logistic": "#ffd92f"
    }

    mpl.rcParams['pdf.fonttype'] = 42
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    fitted_color = "black"  # default fallback color

    x = np.linspace(min(data), max(data), 1000)
    hist_output = ax.hist(data, bins='fd', density=True, alpha=0.4, color="gray")
    hist_patch = hist_output[2][0]  # Get the first rectangle patch from the bar container
    handles_dist = [hist_patch]
    labels_dist = [f"Histogram (sample size: {len(data):,})"]
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
                    fontsize=8, title="Candidate distributions", frameon=False)
        ax.add_artist(leg1)
    if handles_stats:
        ax.legend(handles_stats, labels_stats, loc="upper right", 
                   fontsize=8, title="Best-fit parameters", frameon=False)

    plt.tight_layout()
    if save_path:
        unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
        save_path = os.path.join(config["dirs"]["figures"], 
           f"fit_{code}_{year}_{flow}_{unit_suffix}_{best_fit_name}.{file_format}")
        plt.savefig(save_path, dpi=300)
        if ax is None:
            plt.close()
    else:
        plt.show()
        
