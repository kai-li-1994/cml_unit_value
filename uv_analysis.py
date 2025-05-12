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


def dip_test(data):
    """Return Hartigan's dip statistic and the p-value for unimodality test.
    Args:
       data (array-like): A one-dimensional array or pandas Series containing
                           the data to be tested.
    Returns:
        No.
    """
    dip_statistic, p_value = diptest(data)
    if p_value < 0.05:
        return False, p_value  # False indicates multimodal
    else:
        return True, p_value  # True indicates unimodal


def modality_test(df, r_script_path="uv_modality_test.R", mod0=1, col="ln_uv", methods=None, cap_size=25000):
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

    # Decide if capping is needed based on methods
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

    # Path to the Rscript executable (not the .R script)
    rscript_exec = "C:/Users/lik6/AppData/Local/Programs/R/R-4.5.0/bin/x64/Rscript.exe"
    #rscript_exec = "C:/Program Files/R/R-4.4.1/bin/x64/Rscript.exe"

    # Compose command
    cmd = [rscript_exec, r_script_path, str(mod0)]
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

        # Determine decision per method
        reject = pvals_df["P_Value"] < 0.05
        pvals_df["Decision"] = reject.map(lambda x: "reject" if x else "fail to reject")
        n_reject = reject.sum()
        
        final_decision = "multimodal" if n_reject >= (len(pvals_df) // 2 + 1) else "unimodal"

        # Compose output dictionary
        report_modality = {
            "step": "modality_test",
            "method": "majority",
            "modality_decision": final_decision,
            "modality_votes": n_reject,
            "modality_sample_capped": was_capped,
            "modality_used_n": len(sampled),
            "modality_original_n": original_n
        }

        for _, row in pvals_df.iterrows():
            method_name = row["Method"]
            report_modality[f"{method_name}_P"] = row["P_Value"]
            report_modality[f"{method_name}_Decision"] = row["Decision"]

        print("📈 Modality Test Summary"
             f"\n- Column tested: {col}"
             f"\n- Original sample size: {original_n}")
        if was_capped:
            print(f"❗ Sample capped at {cap_size} for efficiency.")
        else:
            print("✅ Full sample used (no capping applied).")
        print(f"\n- Final decision: {final_decision} ({n_reject} reject out of {len(pvals_df)})\n")

        return report_modality

    except subprocess.CalledProcessError as e:
        print("R script failed.\nSTDERR:\n", e.stderr)
        return {
            "step": "modality_test",
            "modality_error": str(e),
            "modality_sample_capped": was_capped,
            "modality_used_n": len(sampled),
            "modality_original_n": original_n
        }
    

def estimate_mode(dist, args, data):
    """Estimate mode as the peak of the PDF over the data range."""
    result = minimize_scalar(
        lambda x: -dist.pdf(x, *args),
        bounds=(min(data), max(data)),
        method="bounded",
    )
    return result.x


def fit_normal(data, pt=False):
    """
    Fit a normal distribution to the data and return the parameters.
    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics,
        and log-likelihood.
    """
    mu, sigma = norm.fit(data)
    loglik = np.sum(norm.logpdf(data, mu, sigma))
    mean, var, skew, kurt = norm.stats(mu, sigma, moments="mvsk")
    sample_var = np.var(
        data, ddof=1
    )  # sample variance (ddof=1 for unbiased estimate)
    median = mu
    mode = mu
    n = len(data)
    aic = 2 * 2 - 2 * loglik
    bic = 2 * np.log(n) - 2 * loglik

    results = {
        "mu_norm": mu,
        "sigma_norm": sigma,
        "log_likelihood_norm": loglik,
        "mean_norm": mean,
        "median_norm": median,
        "mode_norm": mode,
        "variance_norm": var,
        "sample_variance_norm": sample_var,
        "skew_norm": skew,
        "kurtosis_norm": kurt,
        "aic_norm": aic,
        "bic_norm": bic,
    }

    if pt:
        print("Fitted Normal Distribution:")
        for k, v in results.items():
            print(
                f"{k}: {v:.3f}"
                if isinstance(v, (float, int))
                else f"{k}: {v}"
            )

    return results


def fit_skewnormal(data, pt=False):
    """
    Fit a skew-normal distribution to the data and return the parameters.
    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics,
        and log-likelihood.
    """
    a, loc, scale = skewnorm.fit(data)
    loglik = np.sum(skewnorm.logpdf(data, a, loc, scale))
    mean, var, skew, kurt = skewnorm.stats(
        a, loc=loc, scale=scale, moments="mvsk"
    )
    sample_var = np.var(
        data, ddof=1
    )  # sample variance (ddof=1 for unbiased estimate)
    median = skewnorm.median(a, loc=loc, scale=scale)
    mode = estimate_mode(skewnorm, (a, loc, scale), data)
    n = len(data)
    aic = 2 * 3 - 2 * loglik
    bic = 3 * np.log(n) - 2 * loglik

    results = {
        "a_skew": a,
        "loc_skew": loc,
        "scale_skew": scale,
        "log_likelihood_skewnorm": loglik,
        "mean_skew": mean,
        "median_skew": median,
        "mode_skew": mode,
        "variance_skew": var,
        "sample_variance_skew": sample_var,
        "skew_skew": skew,
        "kurtosis_skew": kurt,
        "aic_skewnorm": aic,
        "bic_skewnorm": bic,
    }

    if pt:
        print("Fitted Skew-Normal Distribution:")
        for k, v in results.items():
            print(
                f"{k}: {v:.3f}"
                if isinstance(v, (float, int))
                else f"{k}: {v}"
            )

    return results


def fit_studentt(data, pt=False):
    """
    Fit a Student's t-distribution to the data and return the parameters.
    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics, 
              and log-likelihood.
    """
    df, loc, scale = t.fit(data)
    loglik = np.sum(t.logpdf(data, df, loc, scale))
    mean, var, skew, kurt = t.stats(df, loc, scale, moments="mvsk")
    sample_var = np.var(data, ddof=1)
    median = t.median(df, loc, scale)
    mode = estimate_mode(t, (df, loc, scale), data)
    n = len(data)
    aic = 2 * 3 - 2 * loglik
    bic = 3 * np.log(n) - 2 * loglik

    results = {
        "df_t": df,
        "loc_t": loc,
        "scale_t": scale,
        "log_likelihood_t": loglik,
        "mean_t": mean,
        "median_t": median,
        "mode_t": mode,
        "variance_t": var,
        "sample_variance_t": sample_var,
        "skew_t": skew,
        "kurtosis_t": kurt,
        "aic_t": aic,
        "bic_t": bic,
    }

    if pt:
        print("Fitted Student-t Distribution:")
        for k, v in results.items():
            print(
                f"{k}: {v:.3f}"
                if isinstance(v, (float, int))
                else f"{k}: {v}"
            )

    return results


def fit_gennorm(data, pt=False):
    """
    Fit a generalized normal (exponential power) distribution to the data and 
    return the parameters.
    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics, 
             and log-likelihood.
    """
    beta, loc, scale = gennorm.fit(data)
    loglik = np.sum(gennorm.logpdf(data, beta, loc, scale))
    mean, var, skew, kurt = gennorm.stats(beta, loc, scale, moments="mvsk")
    sample_var = np.var(data, ddof=1)
    median = gennorm.median(beta, loc, scale)
    mode = estimate_mode(gennorm, (beta, loc, scale), data)
    n = len(data)
    aic = 2 * 3 - 2 * loglik
    bic = 3 * np.log(n) - 2 * loglik

    results = {
        "beta_gn": beta,
        "loc_gn": loc,
        "scale_gn": scale,
        "log_likelihood_gn": loglik,
        "mean_gn": mean,
        "median_gn": median,
        "mode_gn": mode,
        "variance_gn": var,
        "sample_variance_gn": sample_var,
        "skew_gn": skew,
        "kurtosis_gn": kurt,
        "aic_gn": aic,
        "bic_gn": bic,
    }

    if pt:
        print("Fitted Generalized Normal Distribution:")
        for k, v in results.items():
            print(
                f"{k}: {v:.3f}"
                if isinstance(v, (float, int))
                else f"{k}: {v}"
            )

    return results


def fit_johnsonsu(data, pt=False):
    """
    Fit a Johnson SU distribution to the data and return the parameters.
    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics, and 
              log-likelihood.
    """
    a, b, loc, scale = johnsonsu.fit(data)
    loglik = np.sum(johnsonsu.logpdf(data, a, b, loc, scale))
    mean, var, skew, kurt = johnsonsu.stats(a, b, loc, scale, moments="mvsk")
    sample_var = np.var(data, ddof=1)
    median = johnsonsu.median(a, b, loc, scale)
    mode = estimate_mode(johnsonsu, (a, b, loc, scale), data)
    n = len(data)
    aic = 2 * 4 - 2 * loglik
    bic = 4 * np.log(n) - 2 * loglik

    results = {
        "a_jsu": a,
        "b_jsu": b,
        "loc_jsu": loc,
        "scale_jsu": scale,
        "log_likelihood_jsu": loglik,
        "mean_jsu": mean,
        "median_jsu": median,
        "mode_jsu": mode,
        "variance_jsu": var,
        "sample_variance_jsu": sample_var,
        "skew_jsu": skew,
        "kurtosis_jsu": kurt,
        "aic_jsu": aic,
        "bic_jsu": bic,
    }

    if pt:
        print("Fitted Johnson SU Distribution:")
        for k, v in results.items():
            print(
                f"{k}: {v:.3f}"
                if isinstance(v, (float, int))
                else f"{k}: {v}"
            )

    return results


def fit_logistic(data, pt=False):
    """
    Fit a logistic distribution to the data and return the parameters.
    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics, 
              and log-likelihood.
    """
    loc, scale = logistic.fit(data)
    loglik = np.sum(logistic.logpdf(data, loc, scale))
    mean, var, skew, kurt = logistic.stats(loc, scale, moments="mvsk")
    sample_var = np.var(data, ddof=1)
    median = logistic.median(loc, scale)
    mode = estimate_mode(logistic, (loc, scale), data)
    n = len(data)
    aic = 2 * 2 - 2 * loglik
    bic = 2 * np.log(n) - 2 * loglik

    results = {
        "loc_log": loc,
        "scale_log": scale,
        "log_likelihood_logistic": loglik,
        "mean_logistic": mean,
        "median_logistic": median,
        "mode_logistic": mode,
        "variance_logistic": var,
        "sample_variance_logistic": sample_var,
        "skew_logistic": skew,
        "kurtosis_logistic": kurt,
        "aic_logistic": aic,
        "bic_logistic": bic,
    }

    if pt:
        print("Fitted Logistic Distribution:")
        for k, v in results.items():
            print(
                f"{k}: {v:.3f}"
                if isinstance(v, (float, int))
                else f"{k}: {v}"
            )

    return results


def bootstrap_parametric_ci(
    data, dist="skewnorm", n_bootstraps=1000, confidence=0.95
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
            mode = estimate_mode(skewnorm, (a, loc, scale), sample)

        elif dist == "studentt":
            df_, loc_, scale_ = t.fit(sample)
            mean, var = t.stats(df_, loc_, scale_, moments="mv")
            median = t.median(df_, loc_, scale_)
            mode = estimate_mode(t, (df_, loc_, scale_), sample)

        elif dist == "gennorm":
            beta_, loc_, scale_ = gennorm.fit(sample)
            mean, var = gennorm.stats(beta_, loc_, scale_, moments="mv")
            median = gennorm.median(beta_, loc_, scale_)
            mode = estimate_mode(gennorm, (beta_, loc_, scale_), sample)

        elif dist == "johnsonsu":
            a_, b_, loc_, scale_ = johnsonsu.fit(sample)
            mean, var = johnsonsu.stats(a_, b_, loc_, scale_, moments="mv")
            median = johnsonsu.median(a_, b_, loc_, scale_)
            mode = estimate_mode(johnsonsu, (a_, b_, loc_, scale_), sample)

        elif dist == "logistic":
            loc_, scale_ = logistic.fit(sample)
            mean, var = logistic.stats(loc_, scale_, moments="mv")
            median = logistic.median(loc_, scale_)
            mode = estimate_mode(logistic, (loc_, scale_), sample)

        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        boot_means.append(mean)
        boot_medians.append(median)
        boot_modes.append(mode)
        boot_vars.append(var)

    # Compute confidence intervals after bootstrapping
    ci_mean = (
        np.percentile(boot_means, 100 * alpha / 2),
        np.percentile(boot_means, 100 * (1 - alpha / 2)),
    )
    ci_median = (
        np.percentile(boot_medians, 100 * alpha / 2),
        np.percentile(boot_medians, 100 * (1 - alpha / 2)),
    )
    ci_mode = (
        np.percentile(boot_modes, 100 * alpha / 2),
        np.percentile(boot_modes, 100 * (1 - alpha / 2)),
    )
    ci_var = (
        np.percentile(boot_vars, 100 * alpha / 2),
        np.percentile(boot_vars, 100 * (1 - alpha / 2)),
    )

    print(
        f"{confidence*100:.0f}% CI for Mean:\n{ci_mean[0]:.3f} – {ci_mean[1]:.3f} (log), "
        f"{np.exp(ci_mean[0]):.3f} – {np.exp(ci_mean[1]):.3f} USD/kg"
    )
    print(
        f"{confidence*100:.0f}% CI for Median:\n{ci_median[0]:.3f} – {ci_median[1]:.3f} (log), "
        f"{np.exp(ci_median[0]):.3f} – {np.exp(ci_median[1]):.3f} USD/kg"
    )
    print(
        f"{confidence*100:.0f}% CI for Mode:\n{ci_mode[0]:.3f} – {ci_mode[1]:.3f} (log), "
        f"{np.exp(ci_mode[0]):.3f} – {np.exp(ci_mode[1]):.3f} USD/kg"
    )
    print(
        f"{confidence*100:.0f}% CI for Variance:\n{ci_var[0]:.3f} – {ci_var[1]:.3f}"
    )

    return ci_mean, ci_median, ci_mode, ci_var


def fit_all_unimodal_models(data, pt=False):
    """
    Fit all candidate unimodal distributions and select the best one based 
    on AIC and BIC.

    Args:
        data (array-like): Input data to fit.
        pt (bool): If True, prints the results for each fitted distribution.

    Returns:
        tuple: (best_fit_name, best_fit_result_dict, all_results_dict)
    """
    model_funcs = {
        "normal": fit_normal,
        "skewnorm": fit_skewnormal,
        "studentt": fit_studentt,
        "gennorm": fit_gennorm,
        "johnsonsu": fit_johnsonsu,
        "logistic": fit_logistic,
    }

    results = {}
    scores = {}

    for name, func in model_funcs.items():
        if pt:
            print("=" * 50)
            print(f"Fitting {name.capitalize()} Distribution")
        result = func(data, pt=pt)
        results[name] = result

        # Find keys for AIC and BIC
        aic_key = [k for k in result if k.startswith("aic")][0]
        bic_key = [k for k in result if k.startswith("bic")][0]
        scores[name] = (result[aic_key], result[bic_key])

    # Select best fit by AIC, then BIC if tie
    best_fit_name = min(scores.items(), key=lambda x: (x[1][0], x[1][1]))[0]
    best_result = results[best_fit_name]

    if pt:
        print("=" * 50)
        print(f"Best fit based on AIC/BIC: {best_fit_name.capitalize()}")

    return best_fit_name, best_result, results


def find_gmm_components(
    data,
    max_components=50,
    convergence_threshold=5,
    reg_covar=1e-3,
    threshold=0.2,
    plot=True,
    ax=None,
    save_path=None,
):
    """
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

    Parameters
    ----------
    data : array-like (1D or 2D)
        Input data for GMM fitting (e.g., log-transformed unit values).
    max_components : int, default=50
        Maximum number of components to consider.
    convergence_threshold : int, default=5
        Number of stable iterations (same best BIC) before early stopping.
    reg_covar : float, default=1e-3
        Regularization term for GMM covariance matrices.
    threshold : float, default=0.2
        Slope ratio threshold for flagging alternative candidates.
    plot : bool, default=True
        Whether to display a BIC curve plot.
    ax : matplotlib.axes.Axes or None
        Matplotlib axes object to draw the plot on, or None to create a new figure.
    save_path : str or None
        If provided, saves the figure to the specified path.

    Returns
    -------
    optimal_component : int
        Selected number of GMM components based on BIC and shape heuristics.
    bic_values : list of float
        BIC scores for each component count (starting from 2).


    """

    # Convert DataFrame to numpy array and ensure it's 2D
    if hasattr(data, "to_numpy"):
        data = data.to_numpy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    bic_values = []
    stable_counter = 0
    prev_best = None

    # Loop over number of components starting from 2
    for n_components in range(2, max_components + 1):
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=41,
            reg_covar=reg_covar,
            n_init=10,
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        bic_values.append(bic)

        # Track stability in BIC minimum to allow early stopping
        current_best = int(np.argmin(bic_values)) + 2
        if prev_best is None:
            prev_best = current_best
        elif current_best == prev_best:
            stable_counter += 1
        else:
            stable_counter = 0
        prev_best = current_best

        if stable_counter >= convergence_threshold:
            break

    best_component = int(np.argmin(bic_values)) + 2
    global_best_bic = bic_values[best_component - 2]

    # Compute tangent slopes between each point and the global minimum
    tangents = []
    for i, bic in enumerate(bic_values):
        steps = abs(i - (best_component - 2))
        slope = (bic - global_best_bic) / steps if steps != 0 else 0
        tangents.append(slope)

    # Detect L-shape and tick-shape patterns by comparing slopes
    def detect_curve_shapes(tangents, best_component, threshold):
        pre_min_tangents = tangents[: best_component - 2]
        post_min_tangents = tangents[best_component - 1 :]

        l_shape_adjustment = None
        tick_shape_adjustment = None

        if pre_min_tangents:
            max_pre_slope = max(pre_min_tangents)
            for idx, slope in enumerate(pre_min_tangents):
                if (
                    l_shape_adjustment is None
                    and slope < threshold * max_pre_slope
                ):
                    l_shape_adjustment = idx + 2  # Early saturation (L-shape)

            if post_min_tangents:
                max_post_slope = max(post_min_tangents)
                for idx, slope in enumerate(pre_min_tangents):
                    if (
                        tick_shape_adjustment is None
                        and slope < threshold * max_post_slope
                    ):
                        tick_shape_adjustment = (
                            idx + 2
                        )  # Overfitting after best (tick-shape)

        return l_shape_adjustment, tick_shape_adjustment

    # Check for curve shape candidates
    l_shape_adjustment, tick_shape_adjustment = detect_curve_shapes(
        tangents, best_component, threshold
    )

    # Final candidate selection based on minimal component count
    candidates = [best_component]
    local_minima = {}
    if l_shape_adjustment:
        candidates.append(l_shape_adjustment)
        local_minima[
            l_shape_adjustment
        ] = "L-shape pattern (early saturation)"
    if tick_shape_adjustment:
        candidates.append(tick_shape_adjustment)
        local_minima[
            tick_shape_adjustment
        ] = "Tick-shape pattern (post-minimum overfitting)"

    optimal_component = min(candidates)

    # Reporting
    print(f"Global best component (min BIC): {best_component}")
    print(
        f"L-shape candidate (early saturation): {l_shape_adjustment or 'None'}"
    )
    print(
        f"Tick-shape candidate (post-minimum overfitting): {tick_shape_adjustment or 'None'}"
    )

    # Plotting results
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = np.arange(2, 2 + len(bic_values))
        ax.plot(
            x_vals,
            bic_values,
            marker="o",
            linestyle="-",
            color="blue",
            label="BIC",
        )
        ax.axvline(
            best_component,
            color="red",
            linestyle="--",
            label=f"Global Min: {best_component}",
        )
        for idx, label in local_minima.items():
            ax.axvline(
                idx, linestyle=":", linewidth=2, label=f"{label}: {idx}"
            )
        ax.set_xticks(x_vals)
        ax.set_title(
            "BIC for Different Numbers of Components in GMM (starting from 2)"
        )
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("BIC Value")
        ax.grid(True)
        ax.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    return optimal_component, bic_values

def find_gmm_components2(
    data,
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
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("BIC")
        ax.set_title("GMM Component Selection via BIC")
        ax.legend()
        ax.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    # Return diagnostics and report
    report_gmm_cselect = {
        "step": "gmm_component_selection",
        "n_samples": n_samples,
        "optimal_k": optimal_k,
        "bic_best_k": best_k,
        "bic_at_optimal_k": bic_values[optimal_k - 2],
        "bic_at_best_k": bic_values[best_k - 2],
        "l_shape_adjustment": l_adj,
        "tick_shape_adjustment": tick_adj,
        "converged_early": stable_counter >= convergence_threshold,
        "threshold": threshold, 
        "n_init": n_init,
        "reg_covar": reg_covar      
    }

    return optimal_k, bic_values.tolist(), report_gmm_cselect

def fit_gmm(
    df,
    columns,
    best_component,
    code,
    year,
    flow,
    cc=None,
    plot=False,
    save_path=None,
    ax=None,
    n_init=10,
    reg_covar = 1e-3
):
    """
    Fits a Gaussian Mixture Model (GMM) to a dataset, prints statistics, 
    and optionally plots results.

    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to fit GMM on (should be ['ln_uv'] in this case).
    - best_component: Number of GMM components to fit.
    - code: HS code for the commodity.
    - year: Year of analysis.
    - flow: 'm' for imports, 'x' for exports.
    - cc: If True, includes country composition in output.
    - plot: If True, generates a plot.
    - save_path: File path to save the plot if needed.
    - ax: Matplotlib axis object (for subplot usage). If None, a new figure 
          will be created.

    Returns:
    - Dictionary containing sorted means, proportions, and covariances.
    """
    
    data = df[columns].to_numpy()
    gmm = GaussianMixture(
        n_components=best_component, 
        random_state=42, 
        n_init=n_init,
        reg_covar=reg_covar
    )
    
    gmm.fit(data)

    means = gmm.means_.flatten()
    proportions = gmm.weights_.flatten()
    covariances = gmm.covariances_.flatten()

    # Confidence intervals
    N = len(data)
    standard_errors = np.sqrt(covariances) / np.sqrt(N * proportions)
    z_alpha_half = norm.ppf(0.975)
    lower_bound = means - z_alpha_half * standard_errors
    upper_bound = means + z_alpha_half * standard_errors

    # Sort by proportions (descending)
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]

    # Country composition in clusters
    if cc:
        component_labels = gmm.predict(data)
        country_proportions = {}
        for i in range(best_component):
            country_counts = df.loc[
                np.where(component_labels == i)[0], "partnerISO"
            ].value_counts(normalize=True)
            country_proportions[i] = country_counts

        sorted_country_proportions = {
            i: pd.Series(country_proportions[i]).sort_values(ascending=False)
            for i in range(best_component)
        }

    # **Plot GMM Components**
    if plot:
        plt.rcParams["pdf.fonttype"] = 42

        # **Check if ax is None (no subplot assigned)**
        own_figure = False
        if ax is None:
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )  # Create new figure only if no subplot exists
            own_figure = True  # We created our own figure

        x_vals = np.linspace(data.min(), data.max(), 1000).reshape(-1)
        colors = cm.get_cmap("tab20", best_component)

        pdf = np.exp(gmm.score_samples(x_vals.reshape(-1, 1)))

        for i in range(best_component):
            component_pdf = (
                sorted_proportions[i]
                * np.exp(
                    -0.5
                    * ((x_vals - sorted_means[i]) ** 2)
                    / sorted_covariances[i]
                )
                / np.sqrt(2 * np.pi * sorted_covariances[i])
            )

            a, b = sorted_proportions[i], sorted_means[i]

            if cc:
                top_countries = sorted_country_proportions[i].head(3)
                countries_str = ", ".join(
                    [
                        f"{country}: {proportion * 100:.0f}%"
                        for country, proportion in top_countries.items()
                    ]
                )
                ax.plot(
                    x_vals,
                    component_pdf,
                    label=f"({i + 1}) {a*100:.0f}%: {np.exp(b):.2f} USD/kg ({countries_str})",
                    color=colors(i),
                )
            else:
                ax.plot(
                    x_vals,
                    component_pdf,
                    label=f"({i + 1}) {a*100:.0f}%: {np.exp(b):.2f} USD/kg",
                    color=colors(i),
                )

            # Add cluster number on top of each peak
            mean_x_position = sorted_means[i]
            mean_y_position = np.max(component_pdf) / 2
            ax.text(
                mean_x_position,
                mean_y_position,
                f"({i + 1})",
                fontsize=8,
                ha="center",
                va="center",
            )

        # Plot histogram
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2 * iqr / len(data) ** (1 / 3)
        bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)
        ax.hist(
            data,
            bins=bin_edges,
            density=True,
            alpha=0.5,
            label="Data",
            color="gray",
        )

        # Plot overall GMM PDF
        ax.plot(x_vals, pdf, label="Overall GMM", color="black", linewidth=2)

        # Add legend and labels
        if ax is None:
            ax.legend(
                ncol=2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.1),
                fancybox=True,
                shadow=True,
            )
        else:
            ax.legend(ncol=2, loc="upper right")

        ax.set_title(
            f"Unit values in a GMM distribution for HS {code} {flow} in {year}"
        )
        ax.set_xlabel("Log Unit Value")
        ax.set_ylabel("Density")

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300)
        if own_figure:
            plt.show()

    return {
        "means": sorted_means,
        "proportions": sorted_proportions,
        "covariances": sorted_covariances,
    }

def fit_gmm12(
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
    reg_covar = 1e-3
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
                label = f"({i+1}) {a*100:.0f}%: {np.exp(mu):.2f} USD/kg ({countries_str})"
            else:
                label = f"({i+1}) {a*100:.0f}%: {np.exp(mu):.2f} USD/kg"

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
        ax.hist(data.flatten(), bins=bin_edges, density=True, alpha=0.5, label="Data", color="gray")

        # Overall GMM
        ax.plot(x_vals, pdf, label="Overall GMM", color="black", linewidth=2)
        ax.legend(ncol=2, loc="upper right")
        ax.set_title(f"Unit values in a GMM distribution for HS {code} {flow} in {year}")
        ax.set_xlabel("Log Unit Value")
        ax.set_ylabel("Density")

        if save_path:
            plt.savefig(save_path, dpi=300)
        if own_figure:
            plt.show()
            
    # ==== Final report ====
    gmm_1d_report = {
        "gmm_step": "gmm_fit_1d",
        "gmm_components": int(best_component),
        "gmm_samples": int(N),
        "gmm_reg_covar": float(reg_covar),
        "gmm_n_init": int(n_init),
        "gmm_means": f"({', '.join(f'{v:.4f}' for v in sorted_means)})",
        "gmm_proportions": f"({', '.join(f'{v:.4f}' for v in sorted_proportions)})",
        "gmm_covariances": f"({', '.join(f'{v:.4f}' for v in sorted_covariances)})",
        "gmm_means_95ci": f"({'; '.join(f'({lo:.4f}, {hi:.4f})' for lo, hi in zip(sorted_lower_bound, sorted_upper_bound))})"
    }

    # Add country summaries
    if sorted_country_proportions:
        for i, country_share_list in sorted_country_proportions.items():
            top_items = country_share_list[:3]
            others_items = country_share_list[3:]

            top_strs = [
                f"{country} ({share * 100:.2f}%)"
                for country, share in top_items
            ]
            if others_items:
                others_share = sum(v for _, v in others_items)
                top_strs.append(f"Others ({others_share * 100:.2f}%)")

            gmm_1d_report[f"gmm_c{i+1}_country_shares"] = ", ".join(top_strs)

    return gmm_1d_report

def re_minmax_log(scaled_values, scaler, feature_index):
    """
    Reverse the Min-Max scaling and log transformation in one step.
    Args:
    - scaled_values (array): Scaled values to reverse.
    - scaler (MinMaxScaler object): The MinMaxScaler used during scaling.
    - feature_index (int): The index of the feature to reverse.

    Returns:
    - Original values before scaling and log transformation.
    """
    # Step 1: Reverse Min-Max scaling
    feature_min = scaler.data_min_[
        feature_index
    ]  # Original min value for the feature
    feature_range = scaler.data_range_[
        feature_index
    ]  # Original range for the feature
    scaled_back = feature_min + (scaled_values * feature_range)

    # Step 2: Reverse the log transformation
    original_values = np.exp(scaled_back)  # Reverse log1p -> expm1

    return original_values


def fit_gmm2(
    data,
    components,
    code,
    year,
    flow,
    plot="2D",
    save_path=None,
    ax=None,
):
    """
    Fits a Gaussian Mixture Model (GMM) to a dataset with two features and 
    provides statistics along with an optional 2D plot.

    Parameters:
    - data: ndarray or DataFrame with two columns (ln_Unit_Price and ln_netWgt).
    - components: Number of GMM components to fit.
    - code: HS code for the commodity.
    - year: Year of analysis.
    - flow: 'm' for imports, 'x' for exports.
    - plot: If '2D', generates a 2D contour plot of GMM clusters.
    - save_path: Path to save the plot (if provided).
    - ax: Axes object for the plot (optional).

    Returns:
    - Dictionary containing means, proportions, covariances, 
      and confidence intervals.
    """
    data = data.to_numpy()

    scaler = MinMaxScaler()  # StandardScaler()
    data = scaler.fit_transform(
        data
    )  # z-score normalization, with the mean and the standard deviation at 0 and 1, repectively.

    # Fit the GMM model
    gmm = GaussianMixture(n_components=components, random_state=42, n_init=10)
    gmm.fit(data)

    # Extract statistics from the fitted GMM
    means = gmm.means_
    proportions = gmm.weights_
    covariances = gmm.covariances_

    # Calculate standard errors for the means
    N = len(data)
    standard_errors = np.sqrt(
        np.array([np.diag(cov) for cov in covariances])
    ) / np.sqrt(N * proportions[:, np.newaxis])

    # 95% confidence intervals
    z_alpha_half = norm.ppf(0.975)
    lower_bound = means - z_alpha_half * standard_errors
    upper_bound = means + z_alpha_half * standard_errors

    # Sort components by proportions
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]

    # Text for import/export
    text_d = "imports" if flow == "m" else "exports"

    print(
        f"In {year}, the unit values for HS {code} {text_d} are represented by {components} clusters."
    )

    # Report means
    mean_values = [
        f"({np.exp(mean[0]):.3f}, {np.exp(mean[1]):.3f})"
        for mean in sorted_means
    ]
    print(f"Means (ln_Unit_Price, ln_netWgt): {', '.join(mean_values)}")

    # Report confidence intervals
    ci_values = [
        f"[({np.exp(lower[0]):.3f}, {np.exp(lower[1]):.3f}), ({np.exp(upper[0]):.3f}, {np.exp(upper[1]):.3f})]"
        for lower, upper in zip(sorted_lower_bound, sorted_upper_bound)
    ]
    print(f"95% Confidence Intervals: {', '.join(ci_values)}")

    # Report proportions
    proportions_values = [f"{prop*100:.2f}%" for prop in sorted_proportions]
    print(f"Proportions: {', '.join(proportions_values)}")

    # Plot 2D GMM
    if plot == "2D":
        # Define grid limits
        x_min, x_max = min(data[:, 0]), max(data[:, 0])
        y_min, y_max = min(data[:, 1]), max(data[:, 1])
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid_data = np.vstack([xx.ravel(), yy.ravel()]).T

        # Generate probabilities for the grid
        ##gmm_probs = np.exp(gmm.score_samples(grid_data)).reshape(xx.shape)
        colors = cm.get_cmap("tab20", components)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.2,
            label="Data",
            color="gray",
            s=10,
        )

        ##ax.contourf(xx, yy, gmm_probs, levels=20, cmap="Blues", alpha=0.5)

        # Plot contours for each component
        for i in range(components):
            diff = grid_data - sorted_means[i]
            cov_inv = np.linalg.inv(
                sorted_covariances[i]
            )  # Inverse of covariance matrix
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            component_prob = (
                sorted_proportions[i]
                * np.exp(exponent)
                / (2 * np.pi * np.sqrt(np.linalg.det(sorted_covariances[i])))
            )
            component_prob = component_prob.reshape(xx.shape)
            ax.contour(
                xx,
                yy,
                component_prob,
                levels=5,
                colors=[colors(i)],
                alpha=0.9,
                linewidths=0.8,
            )

        # Example data
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Update the tick labels with original values
        ori_x = [re_minmax_log(tick, scaler, 0) for tick in xticks]
        ori_y = [re_minmax_log(tick, scaler, 1) for tick in xticks]

        ori_x = [round(label, 2) for label in ori_x]
        ori_y = [round(label, 2) for label in ori_y]
        ori_y = [
            f"{label:.2e}" if label >= 1000 else round(label, 2)
            for label in ori_y
        ]
        lb_x = [f"{tick} ({label})" for tick, label in zip(xticks, ori_x)]
        lb_y = [f"{tick}\n({label})" for tick, label in zip(xticks, ori_y)]

        ax.set_xticks(xticks)  # Set the tick positions (scaled values)
        ax.set_yticks(xticks)  # Set the tick positions (scaled values)
        ax.set_xticklabels(
            lb_x, fontsize=10
        )  # Set the tick labels with original data
        ax.set_yticklabels(
            lb_y, fontsize=10
        )  # Set the tick labels with original data

        # Ensure equal aspect ratio
        ax.set_aspect("equal", adjustable="box")

        # Add labels, title, and legend
        ax.set_title(
            f"GMM Clustering for HS {code} {text_d} in {year}", fontsize=14
        )
        ax.set_xlabel("ln_Unit_Price", fontsize=12)
        ax.set_ylabel("ln_netWgt", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        # Add the custom legend entry for the scaled and original data pair
        handles.append(plt.Line2D([0], [0], color="white", lw=2))
        labels.append("0.4(0.24): Log-Min-Max scaled data (original data)")

        # Add the legend back to the plot with the custom line and original ones
        ax.legend(handles, labels, loc="upper right", fontsize=10)

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    # Return statistics as a dictionary
    return {
        "means": sorted_means,
        "proportions": sorted_proportions,
        "covariances": sorted_covariances,
        "confidence_intervals": {
            "lower": sorted_lower_bound,
            "upper": sorted_upper_bound,
        },
    }


def ensure_cov_matrix(cov, cov_type):
    """Convert diagonal or spherical covariance to full matrix if needed."""
    if cov_type == "diag":
        return np.diag(cov)
    elif cov_type == "spherical":
        return np.eye(2) * cov  # assumes 2D feature
    return cov


def fit_gmm2_flexible(
    data,
    components,
    code,
    year,
    flow,
    plot="2D",
    save_path=None,
    ax=None,
    covariance_type="full",
):
    data = data.to_numpy()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    gmm = GaussianMixture(
        n_components=components,
        covariance_type=covariance_type,
        random_state=42,
        reg_covar=1e-6,
        n_init=10,
    )
    gmm.fit(data)

    means = gmm.means_
    proportions = gmm.weights_
    covariances = gmm.covariances_

    N = len(data)

    # Handle standard error computation
    if covariance_type == "full":
        standard_errors = np.sqrt(
            np.array([np.diag(cov) for cov in covariances])
        ) / np.sqrt(N * proportions[:, np.newaxis])
    elif covariance_type == "diag":
        standard_errors = np.sqrt(covariances) / np.sqrt(
            N * proportions[:, np.newaxis]
        )
    elif covariance_type == "spherical":
        standard_errors = np.sqrt(covariances)[:, np.newaxis] / np.sqrt(
            N * proportions[:, np.newaxis]
        )
    else:
        raise NotImplementedError(
            f"Covariance type {covariance_type} is not supported for standard error computation."
        )

    z_alpha_half = norm.ppf(0.975)
    lower_bound = means - z_alpha_half * standard_errors
    upper_bound = means + z_alpha_half * standard_errors

    sorted_indices = np.argsort(proportions)[::-1]
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]

    text_d = "imports" if flow == "m" else "exports"
    print(
        f"In {year}, the unit values for HS {code} {text_d} are represented by {components} clusters."
    )

    mean_values = [
        f"({np.exp(mean[0]):.3f}, {np.exp(mean[1]):.3f})"
        for mean in sorted_means
    ]
    print(f"Means (ln_Unit_Price, ln_netWgt): {', '.join(mean_values)}")

    ci_values = [
        f"[({np.exp(lower[0]):.3f}, {np.exp(lower[1]):.3f}), ({np.exp(upper[0]):.3f}, {np.exp(upper[1]):.3f})]"
        for lower, upper in zip(sorted_lower_bound, sorted_upper_bound)
    ]
    print(f"95% Confidence Intervals: {', '.join(ci_values)}")

    proportions_values = [f"{prop*100:.2f}%" for prop in sorted_proportions]
    print(f"Proportions: {', '.join(proportions_values)}")

    if plot == "2D":
        x_min, x_max = min(data[:, 0]), max(data[:, 0])
        y_min, y_max = min(data[:, 1]), max(data[:, 1])
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )
        grid_data = np.vstack([xx.ravel(), yy.ravel()]).T

        colors = cm.get_cmap("tab20", components)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.2,
            label="Data",
            color="gray",
            s=10,
        )

        for i in range(components):
            cov_matrix = ensure_cov_matrix(
                sorted_covariances[i], covariance_type
            )
            cov_inv = np.linalg.inv(cov_matrix)
            diff = grid_data - sorted_means[i]
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            det_cov = np.linalg.det(cov_matrix)
            component_prob = (
                sorted_proportions[i]
                * np.exp(exponent)
                / (2 * np.pi * np.sqrt(det_cov))
            )
            component_prob = component_prob.reshape(xx.shape)
            ax.contour(
                xx,
                yy,
                component_prob,
                levels=5,
                colors=[colors(i)],
                alpha=0.9,
                linewidths=0.8,
            )

        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ori_x = [
            np.exp(
                scaler.data_min_[0]
                + tick * (scaler.data_max_[0] - scaler.data_min_[0])
            )
            for tick in xticks
        ]
        ori_y = [
            np.exp(
                scaler.data_min_[1]
                + tick * (scaler.data_max_[1] - scaler.data_min_[1])
            )
            for tick in xticks
        ]

        ori_x = [round(label, 2) for label in ori_x]
        ori_y = [
            f"{label:.2e}" if label >= 1000 else round(label, 2)
            for label in ori_y
        ]
        lb_x = [f"{tick} ({label})" for tick, label in zip(xticks, ori_x)]
        lb_y = [f"{tick}\n({label})" for tick, label in zip(xticks, ori_y)]

        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        ax.set_xticklabels(lb_x, fontsize=10)
        ax.set_yticklabels(lb_y, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"GMM Clustering for HS {code} {text_d} in {year}", fontsize=14
        )
        ax.set_xlabel("ln_Unit_Price", fontsize=12)
        ax.set_ylabel("ln_netWgt", fontsize=12)
        ax.legend(fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

    return {
        "means": sorted_means,
        "proportions": sorted_proportions,
        "covariances": sorted_covariances,
        "confidence_intervals": {
            "lower": sorted_lower_bound,
            "upper": sorted_upper_bound,
        },
    }


def fit_gmm3(df, columns, components, code, year, flow, save_path=None):
    """
    Fits a Gaussian Mixture Model (GMM) to a dataset with three features and provides
    statistics along with an optional 3D plot.

    Parameters:
    - data: ndarray or DataFrame with three columns (ln_Unit_Price, ln_netWgt, 
            Country_Grouped_Unit_Value).
    - components: Number of GMM components to fit.
    - code: HS code for the commodity.
    - year: Year of analysis.
    - flow: 'm' for imports, 'x' for exports.
    - save_path: Path to save the plot (if provided).

    Returns:
    - Dictionary containing means, proportions, and covariances.
    """
    data = df[columns].to_numpy()  # Extract numerical data

    scaler = MinMaxScaler()  # Normalize data (Min-Max scaling)
    data = scaler.fit_transform(data)

    gmm = GaussianMixture(
        n_components=components, random_state=42, reg_covar=1e-3
    )  # Fit the GMM model
    gmm.fit(data)

    means = gmm.means_  # Extract statistics from the fitted GMM
    proportions = gmm.weights_
    covariances = gmm.covariances_

    sorted_indices = np.argsort(proportions)[
        ::-1
    ]  # Sort components by proportions (largest clusters first)
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]

    # Assign cluster labels to each data point
    df["cluster"] = gmm.predict(data)

    # Identify the most representative country in each cluster based on total net weight
    representative_countries = {}

    for i in range(components):
        cluster_subset = df[
            df["cluster"] == i
        ]  # Select data points in cluster i

        if not cluster_subset.empty:
            # Sum net weight per country in this cluster
            country_weight_sums = cluster_subset.groupby("partnerISO")[
                "ln_netWgt"
            ].sum()

            # Select the country with the highest net weight contribution
            representative_countries[i] = country_weight_sums.idxmax()

    # Convert dictionary to a unique sorted list of countries
    unique_countries = list(set(representative_countries.values()))
    country_gdp_order = (
        df[df["partnerISO"].isin(unique_countries)]
        .groupby("partnerISO")["ln_gdp"]
        .mean()
        .sort_values()
        .index.tolist()
    )

    text_d = (
        "imports" if flow == "m" else "exports"
    )  # Text for import/export
    print(
        f"In {year}, the unit values for HS {code} {text_d} are represented by {components} clusters."
    )

    fig = plt.figure(figsize=(12, 8))  # 3D Visualization
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.get_cmap("tab20", components)

    ax.scatter(
        data[:, 0], data[:, 1], data[:, 2], alpha=0.1, color="gray", s=5
    )  # Plot each data point with transparency for clarity

    # Plot each cluster mean and draw 3D ellipsoids
    for i in range(components):
        color = colors(i)
        ax.scatter(
            sorted_means[i, 0],
            sorted_means[i, 1],
            sorted_means[i, 2],
            color=color,
            marker="o",
            s=200,
            edgecolors="black",
            label=f"Cluster {i+1}",
        )

        # Compute 3D covariance ellipsoid
        eigenvalues, eigenvectors = np.linalg.eigh(sorted_covariances[i])
        radii = (
            np.sqrt(eigenvalues) * 2
        )  # 2 standard deviations for visualization

        # Generate points on a unit sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # # Scale and rotate the unit sphere to match the covariance ellipse
        # ellipsoid = np.stack([x, y, z], axis=-1)
        # ellipsoid = ellipsoid @ np.diag(radii)  # Scale
        # ellipsoid = ellipsoid @ eigenvectors.T  # Rotate
        # ellipsoid += sorted_means[i]  # Translate to mean
        ellipsoid = (
            np.stack([x, y, z], axis=-1) @ np.diag(radii) @ eigenvectors.T
            + sorted_means[i]
        )

        # Plot the ellipsoid as a mesh
        ax.plot_wireframe(
            ellipsoid[:, :, 0],
            ellipsoid[:, :, 1],
            ellipsoid[:, :, 2],
            color=color,
            alpha=0.3,
        )

    ax.set_xlabel("ln_Unit_Price")
    ax.set_ylabel("ln_netWgt")
    ax.set_zlabel("Country_Grouped_Unit_Value")
    ax.set_title(f"3D GMM Clustering for HS {code} {text_d} in {year}")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return {
        "means": sorted_means,
        "proportions": sorted_proportions,
        "covariances": sorted_covariances,
        "representative countries": representative_countries,
    }, country_gdp_order