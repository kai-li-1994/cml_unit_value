import numpy as np
import pandas as pd
from scipy.stats import (
    norm,
    iqr,
    skewnorm,
    cauchy,
    logistic,
    anderson,
    gumbel_r,
    lognorm,
    t,
    gennorm,
    johnsonsu,
    gennorm,
    kstest,
    expon,
    gaussian_kde,
)
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from diptest import diptest
import time
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from sklearn.metrics import silhouette_score
import subprocess
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed
from uv_config import load_config
import os

config = load_config()

def bootstrapped_kde_test(data, num_bootstrap=1000):
    bandwidth = 0.9 * np.std(data) * len(data) ** (-1 / 5)
    data = np.asarray(data)  # Ensures data is a numpy array
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        data.reshape(-1, 1)
    )

    x_grid = np.linspace(data.min(), data.max(), 1000)
    density = np.exp(kde.score_samples(x_grid[:, None]))
    real_peaks, _ = find_peaks(density)
    real_peak_count = len(real_peaks)

    bootstrap_peak_counts = []
    for _ in range(num_bootstrap):
        bootstrap_sample = np.random.normal(
            np.mean(data), np.std(data), size=len(data)
        )
        kde_bootstrap = KernelDensity(
            kernel="gaussian", bandwidth=bandwidth
        ).fit(bootstrap_sample[:, None])
        density_bootstrap = np.exp(
            kde_bootstrap.score_samples(x_grid[:, None])
        )
        peaks_bootstrap, _ = find_peaks(density_bootstrap)
        bootstrap_peak_counts.append(len(peaks_bootstrap))

    p_value = np.mean(np.array(bootstrap_peak_counts) >= real_peak_count)

    return real_peak_count, p_value, bootstrap_peak_counts



def modality_test(df, r_script_path="uv_modality_test.R", mod0=1, col="ln_uv", methods=None, cap_size=25000, logger = None):
    """
    Run R-based modality tests on a univariate column from a pandas DataFrame.

    Parameters:
        df : pandas.DataFrame
            The input data containing the column to test.
        r_script_path : str
            Path to the R script that runs the modality test.
        mod0 : int
            Null hypothesis: number of modes to test against.
        col : str
            Column name in the DataFrame to test.
        methods : list of str, optional
            List of modality test methods to run (e.g. ["ACR", "SI"]).
            If None, defaults to all available methods in the R script.
        cap_size : int, optional (default = 25000)
            Maximum number of observations used as input when running 
            bootstrapped or kernel-based modality tests.
            This cap improves computational efficiency for large datasets 
            (e.g., those exceeding 100,000 records) while preserving enough 
            statistical resolution for reliable modality classification.

    Returns:
        dict : A summary dictionary with modality test results, including p-values,
               decisions for each method, and the overall modality decision.
    """
    full_df = df[col].dropna()
    original_n = len(full_df)
    

    # Decide if capping is needed
    boot_methods = {"SI", "HY", "CH", "ACR"}
    cap_needed = methods is None or any(m in boot_methods for m in methods)
    
    if cap_needed and original_n > cap_size:
        sampled = full_df.sample(cap_size, random_state=42)
        was_capped = True
    else:
        sampled = full_df
        was_capped = False

    values = sampled.to_numpy()
    csv_data = "\n".join(f"{v}" for v in values)

    # Compose command
    cmd = [config["rscript_exec"], r_script_path, str(mod0)]
    if methods:
        cmd += methods

    try:
        process = subprocess.run(
            cmd,
            input=csv_data,
            text=True,
            capture_output=True,
            check=True
        )
        pvals_df = pd.read_csv(StringIO(process.stdout))

        # Decision logic
        reject = pvals_df["P_Value"] < 0.05
        pvals_df["Decision"] = reject.map(lambda x: "reject" if x else "fail to reject")
        n_reject = reject.sum()
        final_decision = "multimodal" if n_reject >= (len(pvals_df) // 2 + 1) else "unimodal"

        # Summary report
        report_modality = {
            "t_method": "majority",
            "t_modality_decision": final_decision,
            "t_modality_votes": n_reject,
            "t_sample_capped": was_capped,
            "t_sample_used": len(sampled),
            "t_sample_original": original_n
        }

        for _, row in pvals_df.iterrows():
            method_name = row["Method"]
            report_modality[f"t_{method_name}_P"] = row["P_Value"]
            report_modality[f"t_{method_name}_decision"] = row["Decision"]

        logger.info("📈 Modality Test Summary:")
        logger.info(f"- Column tested: {col}")
        logger.info(f"- Original sample size: {original_n}")
        if was_capped:
            logger.warning(f"❗ Sample capped at {cap_size} for efficiency.")
        else:
            logger.info("✅ Full sample used (no capping applied).")
        logger.info(f"- Final decision: {final_decision} ({n_reject} reject out of {len(pvals_df)})")

        return report_modality, report_modality["t_modality_decision"]

    except subprocess.CalledProcessError as e:
        logger.error("❌ R script for modality test failed.")
        logger.error(f"STDERR:\n{e.stderr}")
        report_modality = {
            "t_name": "modality_test",
            "t_modality_error": str(e),
            "t_modality_decision": "error",
            "t_sample_capped": was_capped,
            "t_sample_used": len(sampled),
            "t_sample_original": original_n
        }
        
        return report_modality, "error"
    

def _estimate_mode(dist, args, data):
    """Estimate mode as the peak of the PDF over the data range."""
    result = minimize_scalar(
        lambda x: -dist.pdf(x, *args),
        bounds=(min(data), max(data)),
        method="bounded",
    )
    return result.x

def _fit_distribution(dist_obj, data, logger=None):
    """
    Fit a given scipy.stats distribution object to the data and return a dictionary
    of results with semantically named keys, plus the original params tuple.

    Args:
        dist_obj: A scipy.stats distribution object (e.g., scipy.stats.norm).
        data: 1D array-like of data points to fit.
        logger: Optional logger for info output.

    Returns:
        dist_name (str), result (dict), params (tuple): Name, result dict, and raw parameter tuple.
    """
    dist_name = dist_obj.name
    params = dist_obj.fit(data)
    loglik = np.sum(dist_obj.logpdf(data, *params))
    mean, var, skew, kurt = dist_obj.stats(*params, moments="mvsk")
    sample_var = np.var(data, ddof=1)
    median = dist_obj.median(*params)
    mode = _estimate_mode(dist_obj, params, data)
    n = len(data)

    aic = 2 * len(params) - 2 * loglik
    bic = len(params) * np.log(n) - 2 * loglik

    # Semantic parameter naming for reporting
    param_name_map = {
        "norm":      ["loc", "scale"],
        "skewnorm":  ["a", "loc", "scale"],
        "t":         ["df", "loc", "scale"],
        "gennorm":   ["beta", "loc", "scale"],
        "johnsonsu": ["a", "b", "loc", "scale"],
        "logistic":  ["loc", "scale"]
    }

    param_names = param_name_map.get(dist_name, [f"param{i+1}" for i in range(len(params))])
    param_dict = {f"{dist_name}_{name}": val for name, val in zip(param_names, params)}

    result = {
        **param_dict,
        f"{dist_name}_loglik": loglik,
        f"{dist_name}_mean": mean,
        f"{dist_name}_median": median,
        f"{dist_name}_mode": mode,
        f"{dist_name}_variance": var,
        f"{dist_name}_sample_variance": sample_var,
        f"{dist_name}_skew": skew,
        f"{dist_name}_kurtosis": kurt,
        f"{dist_name}_aic": aic,
        f"{dist_name}_bic": bic,
    }

    if logger:
        logger.info(f"Fitted {dist_name.capitalize()} Distribution")
        for k, v in result.items():
            logger.info(f"{k}: {v:.4f}" if isinstance(v, (float, int)) else f"{k}: {v}")

    return dist_name, result, params  # Note: returning raw params separately

def fit_all_unimodal_models(data, logger=None):
    """
    Fit all candidate unimodal distributions and select the best one based on AIC and BIC.

    Args:
        data (array-like): Input data to fit.
        logger (logging.Logger, optional): Logger for info output.

    Returns:
        tuple: (best_fit_name, report_best_fit_uni, report_all_uni_fit, raw_params_dict)
    """
    from scipy.stats import norm, skewnorm, t, gennorm, johnsonsu, logistic
    model_objs = [norm, skewnorm, t, gennorm, johnsonsu, logistic]

    report_all_uni_fit = {}
    raw_params_dict = {}
    scores = {}

    for dist_obj in model_objs:
        dist_name, result, params = _fit_distribution(dist_obj, data, logger=logger)
        report_all_uni_fit.update(result)
        raw_params_dict[f"{dist_name}_params"] = params
        scores[dist_name] = (result[f"{dist_name}_aic"], result[f"{dist_name}_bic"])

    best_fit_name = min(scores.items(), key=lambda x: (x[1][0], x[1][1]))[0]
    report_best_fit_uni = {
        k: v for k, v in report_all_uni_fit.items() if k.startswith(best_fit_name)
    }

    report_all_uni_fit["best_fit_name"] = best_fit_name

    if logger:
        logger.info(f"Best fit based on AIC/BIC: {best_fit_name.capitalize()}")

    return best_fit_name, report_best_fit_uni, report_all_uni_fit, raw_params_dict

def bootstrap_parametric_ci(
    data, dist="skewnorm", n_bootstraps=1000, confidence=0.95, logger=None
):
    """
    Bootstrap confidence intervals for mean, median, mode, and variance
    of a parametric distribution.

    Args:
        data (array-like): Input data to estimate the confidence intervals.
        dist (str): Distribution name: 'normal', 'skewnorm', 'studentt',
        'gennorm', 'johnsonsu', 'logistic'.
        n_bootstraps (int): Number of bootstrap samples.
        confidence (float): Confidence level (default 0.95).

    Returns:
        tuple: CI for mean, median, mode, and variance.
    """
    n = len(data)
    alpha = 1 - confidence
    boot_means, boot_medians, boot_modes, boot_vars = [], [], [], []

    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)

        if dist == "normal":
            mu, sigma = norm.fit(sample)
            mean, var = norm.stats(mu, sigma, moments="mv")
            median = mu
            mode = mu

        elif dist == "skewnorm":
            a, loc, scale = skewnorm.fit(sample)
            mean, var = skewnorm.stats(a, loc=loc, scale=scale, moments="mv")
            median = skewnorm.median(a, loc=loc, scale=scale)
            mode = _estimate_mode(skewnorm, (a, loc, scale), sample)

        elif dist == "studentt":
            df_, loc_, scale_ = t.fit(sample)
            mean, var = t.stats(df_, loc_, scale_, moments="mv")
            median = t.median(df_, loc_, scale_)
            mode = _estimate_mode(t, (df_, loc_, scale_), sample)

        elif dist == "gennorm":
            beta_, loc_, scale_ = gennorm.fit(sample)
            mean, var = gennorm.stats(beta_, loc_, scale_, moments="mv")
            median = gennorm.median(beta_, loc_, scale_)
            mode = _estimate_mode(gennorm, (beta_, loc_, scale_), sample)

        elif dist == "johnsonsu":
            a_, b_, loc_, scale_ = johnsonsu.fit(sample)
            mean, var = johnsonsu.stats(a_, b_, loc_, scale_, moments="mv")
            median = johnsonsu.median(a_, b_, loc_, scale_)
            mode = _estimate_mode(johnsonsu, (a_, b_, loc_, scale_), sample)

        elif dist == "logistic":
            loc_, scale_ = logistic.fit(sample)
            mean, var = logistic.stats(loc_, scale_, moments="mv")
            median = logistic.median(loc_, scale_)
            mode = _estimate_mode(logistic, (loc_, scale_), sample)

        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        boot_means.append(mean)
        boot_medians.append(median)
        boot_modes.append(mode)
        boot_vars.append(var)

    # === CI Computation ===
    def _ci_bounds(arr):
        return (
            np.percentile(arr, 100 * alpha / 2),
            np.percentile(arr, 100 * (1 - alpha / 2))
        )

    ci_mean = _ci_bounds(boot_means)
    ci_median = _ci_bounds(boot_medians)
    ci_mode = _ci_bounds(boot_modes)
    ci_var = _ci_bounds(boot_vars)

    if logger:
        logger.info(f"{confidence:.0%} CI for Mean: {ci_mean[0]:.3f} – {ci_mean[1]:.3f} (log), {np.exp(ci_mean[0]):.3f} – {np.exp(ci_mean[1]):.3f} USD/kg")
        logger.info(f"{confidence:.0%} CI for Median: {ci_median[0]:.3f} – {ci_median[1]:.3f} (log), {np.exp(ci_median[0]):.3f} – {np.exp(ci_median[1]):.3f} USD/kg")
        logger.info(f"{confidence:.0%} CI for Mode: {ci_mode[0]:.3f} – {ci_mode[1]:.3f} (log), {np.exp(ci_mode[0]):.3f} – {np.exp(ci_mode[1]):.3f} USD/kg")
        logger.info(f"{confidence:.0%} CI for Variance: {ci_var[0]:.3f} – {ci_var[1]:.3f}")

    return {
    "ci_mean_lower": ci_mean[0],
    "ci_mean_upper": ci_mean[1],
    "ci_median_lower": ci_median[0],
    "ci_median_upper": ci_median[1],
    "ci_mode_lower": ci_mode[0],
    "ci_mode_upper": ci_mode[1],
    "ci_variance_lower": ci_var[0],
    "ci_variance_upper": ci_var[1]
}

def find_gmm_components(
    data,
    code,
    year,
    flow,
    unit_label="USD/kg",
    max_components=50,
    convergence_threshold=5,
    reg_covar=1e-3,
    threshold=0.2,
    n_init=10,
    plot=False,
    ax=None,
    save_path=None
):
    """
    Efficient selection of GMM components using BIC and slope-based adjustment.
    
    Select Optimal Number of GMM Components Using BIC and Tangent-Based Analysis
    This method identifies the most suitable number of Gaussian Mixture Model 
    (GMM) components for a given dataset using the Bayesian Information 
    Criterion (BIC), enhanced by slope-based (tangent) heuristics.

    BIC balances model fit and complexity:
        BIC = k * log(n) - 2 * log(L)
    where:
        - k: number of model parameters,
        - n: number of data points,
        - L: maximum likelihood of the model.

    While BIC penalizes complexity, it may not always yield the simplest 
    sufficient model. Its curve often shows:
    - **L-shape**: steep early improvement followed by plateauing,
    - **Tick-shape**: continued improvement followed by sharp worsening (overfitting).

    This method analyzes the BIC curve's shape by comparing the slope between 
    each candidate (k) and the global minimum (k_best):
        
        Slope_k = (BIC_k - BIC_best) / |k - k_best|

    Alternative candidates are flagged if their slope is significantly smaller 
    (e.g., < 20%) than the steepest slope nearby:
    - For L-shape: compares slope before the minimum,
    - For tick-shape: compares slope after the minimum.

    This allows identifying simpler yet statistically competitive models.

    Parameters:
        data : array-like
            Input 1D or 2D data array.
        max_components : int
            Maximum number of GMM components to evaluate (starting from 2).
        convergence_threshold : int
            Stop early if BIC-optimal value doesn't change for this many iterations.
        reg_covar : float
            Regularization term for GMM covariance matrices.
        threshold : float
            Slope ratio threshold for L-shape and tick-shape adjustments.
        n_init : int
            Number of initializations for GMM (robustness to local optima).
        plot : bool
            Whether to show the BIC plot.
        ax : matplotlib Axes
            Optional matplotlib axes object to draw the plot.
        save_path : str or None
            Optional path to save the plot as an image file.

    Returns:
        optimal_k : int
            Selected number of components.
        bic_values : list
            List of BIC values for each model.
        report : dict
            Diagnostic info and selection metadata.
    """
    # Convert to numpy array if it's a DataFrame
    if hasattr(data, "to_numpy"):
        data = data.to_numpy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples = data.shape[0]
    bic_values = np.empty(max_components - 1)  # store BIC for each k from 2 to max_components
    stable_counter = 0  # count how long the best BIC stays unchanged
    prev_best = None

    # Fit GMMs from 2 to max_components
    for i, n_components in enumerate(range(2, max_components + 1)):
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=42,
            reg_covar=reg_covar,
            n_init=n_init
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        bic_values[i] = bic

        # Track if the best number of components is changing
        current_best = int(np.argmin(bic_values[: i + 1])) + 2  # +2 since index 0 corresponds to 2 components
        if prev_best is None:
            prev_best = current_best
        elif current_best == prev_best:
            stable_counter += 1
        else:
            stable_counter = 0
        prev_best = current_best

        # Early stopping if best BIC is stable
        if stable_counter >= convergence_threshold:
            bic_values = bic_values[: i + 1]
            break

    # Find best k by BIC minimum
    best_k = int(np.argmin(bic_values)) + 2
    best_bic = np.min(bic_values)

    # Compute slope (tangent) to detect L-/tick-shaped elbow patterns
    tangents = [
        (bic - best_bic) / abs(idx - (best_k - 2)) if idx != (best_k - 2) else 0
        for idx, bic in enumerate(bic_values)
    ]

    # Analyze for early cut-off points via slope changes
    best_idx = best_k - 2
    l_adj = None
    tick_adj = None

    if best_idx >= 2:
        max_pre = max(tangents[:best_idx])           # largest pre-peak slope
        max_post = max(tangents[best_idx + 1:])      # largest post-peak slope

        for i in range(best_idx):
            if tangents[i] < threshold * max_pre:
                l_adj = i + 2  # candidate from L-shape detection
                break
        for i in range(best_idx):
            if tangents[i] < threshold * max_post:
                tick_adj = i + 2  # candidate from tick-shape detection
                break

    # Select smallest among BIC-best and adjustments
    candidates = [best_k]
    notes = {}
    if l_adj and l_adj < best_k:
        candidates.append(l_adj)
        notes[l_adj] = "L-shape adjustment"
    if tick_adj and tick_adj < best_k:
        candidates.append(tick_adj)
        notes[tick_adj] = "Tick-shape adjustment"

    optimal_k = min(candidates)

    print(f"\U0001F4C8 GMM selection: BIC-best={best_k}, L-adjust={l_adj}, Tick-adjust={tick_adj}, Selected={optimal_k}")

    # Optional plot
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = np.arange(2, 2 + len(bic_values))
        ax.plot(x_vals, bic_values, marker='o', label="BIC")
        ax.axvline(best_k, linestyle="--", color="red", label=f"Best BIC: {best_k}")
        for k in notes:
            ax.axvline(k, linestyle=":", label=f"{notes[k]}: {k}")
        ax.set_xticks(x_vals)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("BIC")
        text_d = 'imports' if flow == 'm' else 'exports'
        ax.set_title(f"GMM component selection based on BIC for unit values ({unit_label}) of HS {code} {text_d} in {year}")
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.tight_layout()
            unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
            save_path = os.path.join(config["dirs"]["figures"], 
                 f"cs_{code}_{year}_{flow}_{unit_suffix}.png")
            plt.savefig(save_path, dpi=300)
            if ax is None:
                plt.close()   
        else:
            plt.tight_layout()
            plt.show()
            

    # Return diagnostics and report
    report_gmm_cselect = {
        "cs_n_samples": n_samples,
        "cs_optimal_components": optimal_k,
        "cs_bic_best_components": best_k,
        "cs_bic_at_optimal_components": round(bic_values[optimal_k - 2],4),
        "cs_bic_at_best_components": round(bic_values[best_k - 2],4),
        "cs_l_shape_adjustment": l_adj,
        "cs_tick_shape_adjustment": tick_adj,
        "cs_converged_early": stable_counter >= convergence_threshold,
        "cs_threshold": threshold, 
        "cs_n_init": n_init,
        "cs_reg_covar": reg_covar      
    }

    return optimal_k, bic_values.tolist(), report_gmm_cselect

def fit_gmm(
    df,
    columns,
    best_component,
    code,
    year,
    flow,
    plot=True,
    save_path=None,
    ax=None,
    n_init=10,
    reg_covar = 1e-3,
    unit_label="USD/kg",
):
    """
    Fits a Gaussian Mixture Model (GMM) to log unit value data, computes
    cluster statistics, optionally plots the distribution, and returns results.

    Parameters:
    - df: DataFrame with the data.
    - columns: List with one column name (e.g. ['ln_uv']).
    - best_component: Number of GMM components.
    - code, year, flow: Metadata for labeling.
    - cc: If True, return country composition for each cluster.
    - plot: If True, plot GMM fit and histogram.
    - save_path: File path to save plot.
    - ax: Matplotlib axis (optional).
    - n_init: Number of GMM initializations.

    Returns:
    - Dictionary with GMM statistics and metadata.
    """
    # === Data Extraction ===
    if len(columns) != 1:
        raise ValueError(f"Expected exactly one column for 1D GMM, but got {len(columns)}: {columns}")
    
    data = df[columns[0]].values.reshape(-1, 1) # turn 1D array into a 2D column vector, required by scikit-learn's GMM implementation
    
    # === GMM Initialization and Fitting ===
    gmm = GaussianMixture(
        n_components=best_component,
        random_state=42,
        n_init=n_init,
        reg_covar=reg_covar
    )
    gmm.fit(data)
    
    # ==== Extracting GMM Parameters ===
    means = gmm.means_.flatten()
    proportions = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()

    # ==== Confidence intervals ===
    N = len(data)
    standard_errors = np.sqrt(covariances) / np.sqrt(N * proportions)
    z_alpha_half = norm.ppf(0.975)
    lower_bound = means - z_alpha_half * standard_errors
    upper_bound = means + z_alpha_half * standard_errors

    # ==== Sorting by Cluster Size (Proportion) ===
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]

    # ==== Country composition (NumPy version) ===
    sorted_country_proportions = None
    component_labels = gmm.predict(data) # assign each data point to its most likely cluster using the fitted GMM.
    partner_codes = df["partnerISO"].values
    sorted_country_proportions = {}

    for new_i, old_i in enumerate(sorted_indices):
        mask = component_labels == old_i # creates a boolean mask
        selected = partner_codes[mask] # select countries with this masked component
        if len(selected) > 0:
            unique, counts = np.unique(selected, return_counts=True)
            proportions = counts / counts.sum()
            sorted_pairs = sorted(zip(unique, proportions), key=lambda x: x[1], reverse=True)
            sorted_country_proportions[new_i] = sorted_pairs
        else:
            sorted_country_proportions[new_i] = {}
        
    # ==== Plot for the total and component GMM PDF ===
    if plot:
        plt.rcParams["pdf.fonttype"] = 42
        own_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            own_figure = True

        range_uv = data.max() - data.min()
        margin = 0.1 * range_uv
        x_min, x_max = data.min() - margin, data.max() + margin # 5% padding avoid cutting off GMM tails

        x_vals = np.linspace(x_min, x_max, 1000)

        cmap = colormaps.get_cmap("tab20")
        colors = [cmap(i / best_component) for i in range(best_component)]

        pdf = np.exp(gmm.score_samples(x_vals.reshape(-1, 1))) # pdf for total GMM-fitted curve

        for i in range(best_component):
            mu = sorted_means[i]
            sigma = np.sqrt(sorted_covariances[i])
            a = sorted_proportions[i]
            component_pdf = a * norm.pdf(x_vals, loc=mu, scale=sigma)

            if sorted_country_proportions:
                top_countries = sorted_country_proportions[i]
                countries_str = ", ".join(
                    [f"{k}: {v*100:.0f}%" for k, v in top_countries[:3]]
                )
                label = f"Cluster ({i+1}) {a*100:.0f}%: {np.exp(mu):.2f} {unit_label}\n({countries_str})"
            else:
                label = f"Cluster ({i+1}) {a*100:.0f}%: {np.exp(mu):.2f} {unit_label}"

            ax.plot(x_vals, component_pdf, label=label, color=colors[i])

            # Annotate cluster number
            ax.text(
                mu,
                component_pdf.max() / 2,
                f"({i+1})",
                fontsize=8,
                ha="center",
                va="center"
            )

        # Histogram
        bin_edges = np.histogram_bin_edges(data.flatten(), bins='fd') # decide bin width useing Freeman-Diaconis rule
        ax.hist(data.flatten(), bins=bin_edges, density=True, alpha=0.4, label="Histogram", color="gray")

        # Overall GMM
        ax.plot(x_vals, pdf, label="Overall GMM", color="black", linewidth=2)
        
        # === Evenly split all legend entries into upper left and upper right ===
        handles_all, labels_all = ax.get_legend_handles_labels()
        legend_entries = list(zip(handles_all, labels_all))

        total = len(legend_entries)
        mid = total // 2
        left_entries = legend_entries[:mid]
        right_entries = legend_entries[mid:]

        if left_entries:
            leg1 = ax.legend(*zip(*left_entries), loc="upper left", frameon=False)
            ax.add_artist(leg1)
        if right_entries:
            leg2 = ax.legend(*zip(*right_entries), loc="upper right", frameon=False)
        
        text_d = 'imports' if flow == 'm' else 'exports'
        ax.set_title(f"GMM fit of unit values ({unit_label}) for HS {code} {text_d} in {year}")
        xlabel = f"ln(Unit Value) [{unit_label}]" 
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
            
        if save_path:
            plt.tight_layout()
            unit_suffix = unit_label.split("/")[-1]  # e.g., "kg" from "USD/kg"
            save_path = os.path.join(config["dirs"]["figures"], 
                 f"fit_{code}_{year}_{flow}_{unit_suffix}_gmm_1d.png")
            plt.savefig(save_path, dpi=300)
            if ax is None:
                plt.close()   
        else:
            plt.tight_layout()
            plt.show()
            
    # === Init report with general info ===
    gmm_1d_report = {
        "gmm_components": int(best_component),
        "gmm_samples": int(N),
        "gmm_reg_covar": float(reg_covar),
        "gmm_n_init": int(n_init),
    }

    # === Unified pass per component ===
    for i in range(best_component):
        mu = float(sorted_means[i])
        p = float(sorted_proportions[i])
        cov = float(sorted_covariances[i])
        lo = float(sorted_lower_bound[i])
        hi = float(sorted_upper_bound[i])

        gmm_1d_report[f"gmm_c{i+1}_mean"] = round(mu, 4)
        gmm_1d_report[f"gmm_c{i+1}_proportion"] = round(p, 4)
        gmm_1d_report[f"gmm_c{i+1}_covariance"] = round(cov, 4)
        gmm_1d_report[f"gmm_c{i+1}_mean_ci_lower"] = round(lo, 4)
        gmm_1d_report[f"gmm_c{i+1}_mean_ci_upper"] = round(hi, 4)

        if sorted_country_proportions:
            top_items = sorted_country_proportions.get(i, [])[:3]
            for j, (country, share) in enumerate(top_items):
                gmm_1d_report[f"gmm_c{i+1}_country{j+1}"] = f"{country} ({round(share * 100, 2)}%)"

    return gmm_1d_report


