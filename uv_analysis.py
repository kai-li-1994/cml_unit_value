import numpy as np
import pandas as pd
from scipy.stats import (norm, iqr, skewnorm, cauchy, logistic,anderson,
    gumbel_r,lognorm, t,johnsonsu, gennorm, kstest, expon, gaussian_kde)
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from diptest import diptest
import time
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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

def fit_normal(data,  pt = None):
    """
    Fit a normal distribution to the data and return the parameters.
    Args:
        data (array-like): Input data to fit.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics, 
        and log-likelihood.
    """
    mu_norm, sigma_norm = norm.fit(data)
    log_likelihood_norm = np.sum(norm.logpdf(data, mu_norm, sigma_norm))
    z_alpha_half = 1.96  # 95% confidence
    ci_mean_norm = (mu_norm - z_alpha_half * sigma_norm / np.sqrt(len(data)),
                    mu_norm + z_alpha_half * sigma_norm / np.sqrt(len(data)))
    
    aic_norm = 2 * 2 - 2 * log_likelihood_norm  # 2 parameters: mu and sigma
    bic_norm = np.log(len(data)) * 2 - 2 * log_likelihood_norm                 

    results = {
        'mu_norm': mu_norm,
        'sigma_norm': sigma_norm,
        'log_likelihood_norm': log_likelihood_norm,
        'ci_mean_norm': ci_mean_norm,
        'aic_norm': aic_norm,
        'bic_norm': bic_norm  
    }
    
    if pt:
        print("Fitted Normal Distribution Parameters and Statistics:")
        for key, value in results.items():
            if isinstance(value, tuple):
                print(f"{key}: ({', '.join(f'{v:.3f}' for v in value)})")
            elif isinstance(value, (float, int)):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
     
    return results


def fit_skewnormal(data, pt=None):
    """
    Fit a skew-normal distribution to the data and return multiple parameters.
    Args:
        data (array-like): Input data to fit.
    Returns:
        dict: A dictionary containing the fitted parameters, statistics, and log-likelihood.
    """

    a_skew, loc_skew, scale_skew = skewnorm.fit(data)
    log_likelihood_skewnorm = np.sum(skewnorm.logpdf(data, a_skew, loc=loc_skew, scale=scale_skew))
    mean_skew, variance_skew = skewnorm.stats(a_skew, loc=loc_skew, scale=scale_skew, moments='mv')
    median_skew = skewnorm.median(a_skew, loc=loc_skew, scale=scale_skew)
    
    mode_skew = minimize_scalar(lambda x: -skewnorm.pdf(x, a_skew, loc_skew, scale_skew),
                                      bounds=(min(data), max(data)), method='bounded').x
    
    standard_error_mean = np.sqrt(variance_skew /len(data))
    z_alpha_half = 1.96  # 95% confidence
    ci95_mean = (mean_skew - z_alpha_half * standard_error_mean, 
               mean_skew + z_alpha_half * standard_error_mean)
    
    aic_skewnorm = 2 * 3 - 2 * log_likelihood_skewnorm  # 3 parameters: a, loc, scale
    bic_skewnorm = np.log(len(data)) * 3 - 2 * log_likelihood_skewnorm
    
    
    results = {
        'a_skew': a_skew,
        'loc_skew': loc_skew,
        'scale_skew': scale_skew,
        'log_likelihood_skewnorm': log_likelihood_skewnorm,
        'mean_skew': mean_skew,
        'median_skew': median_skew,
        'mode_skew': mode_skew,
        'ci95_mean': ci95_mean,
        'variance_skew': variance_skew,  
        'aic_skewnorm': aic_skewnorm, 
        'bic_skewnorm': bic_skewnorm, 
    }
    if pt:
        print("Fitted Skew-Normal Distribution Parameters and Statistics:")   
        for key, value in results.items():
            if isinstance(value, tuple):
                print(f"{key}: ({', '.join(f'{v:.3f}' for v in value)})")
            elif isinstance(value, (float, int)):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
               
    return results

def bootstrap_skewnormal_ci(data, n_bootstraps=1000):
    """
    Calculate bootstrap confidence intervals for the median and mode of the 
    skew-normal distribution.
    Args:
        data (array-like): Input data to estimate the confidence intervals.
        n_bootstraps (int): Number of bootstrap samples to generate (default is 1000).
    Returns:
        tuple: A tuple containing the 95% confidence intervals for the median and mode.
    """
    start_time = time.time()
    n = len(data)
    boot_medians, boot_modes = [], []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        a_sample, loc_sample, scale_sample = skewnorm.fit(sample)
        boot_medians.append(skewnorm.median(a_sample, loc=loc_sample, 
                                            scale=scale_sample))
        boot_modes.append(minimize_scalar(lambda x: -skewnorm.pdf(x, a_sample, 
        loc_sample, scale_sample), bounds=(min(sample), max(sample)),
                                          method='bounded').x)
    
    ci_median = (np.percentile(boot_medians, 2.5), np.percentile(boot_medians,
                                                                 97.5))
    ci_mode = (np.percentile(boot_modes, 2.5), np.percentile(boot_modes, 
                                                             97.5))
    elapsed_time = time.time() - start_time
    exp = (f"Median 95% CI: ({ci_median[0]:.3f}, {ci_median[1]:.3f})"
           f"Mode 95% CI: ({ci_mode[0]:.3f}, {ci_mode[1]:.3f}).\n"
           f"Processing time: {elapsed_time:.2f} seconds.")
    print(exp)
    return ci_median, ci_mode

def find_gmm(data, max_components=50, convergence_threshold=5, threshold=0.2):
    """
    Fit a GMM to the input data and find the best number of components
    (clusters) by minimizing the BIC score and then detect the turning point
    in case of L-shaped BIC curve, with dynamic slope threshold.
    
    Args:
    data: A Pandas DataFrame with the selected features (any number of columns).
    max_components: The maximum number of components to evaluate (default is 50).
    convergence_threshold: The number of consecutive iterations where the 
    optimal component count must remain stable before stopping (default is 5).
    stability_fraction: Fraction of the BIC value used to determine slope 
    stability (20% by default).
    
    Returns:
    int: The optimal number of components (clusters).
    list: The BIC values for each component count evaluated.
    02212025: fix the issue of l (740400, 2018, 2023) shape and tick shape (740311, 2018) bic values
    also 740311, 2018 does not work with reg_covar=1e-3 but -6. 391590 2023 works -3 but not -6
    """
    # Convert DataFrame to NumPy array (handles any number of features dynamically)
    data = data.to_numpy()
   
    # To store BIC values
    bic_values = []
    stable_iterations = 0
    prev_best_component = None

    for num_components in range(2, max_components + 1):
        gmm = GaussianMixture(n_components=num_components, random_state=42,reg_covar=1e-3)   # First fit the GMM with default regularization (no regularization)
        gmm.fit(data)                                                         # GMM in scikit-learn expects a 2D array with shape (n_samples, n_features), while in this case the data is univariate.  
        
        bic_values.append(gmm.bic(data))
        
        # Identify the best number of components so far
        best_component = 2 + np.argmin(bic_values)
        
        if num_components == 2:
            prev_best_component = best_component
            continue
        
        # Check for stable best component
        if best_component == prev_best_component:
            stable_iterations += 1
        else:
            stable_iterations = 0
        
        prev_best_component = best_component
        
        if stable_iterations >= convergence_threshold:
          #print(f"Stable best component count: {best_component}")
          break
      
    # Step 1: Find the best component using BIC
    best_component = np.argmin(bic_values) + 2  # Find the best component via BIC
    
    global_best_bic = bic_values[best_component - 2]
    
    # Step 2: Calculate the height difference from the global best
    height_diff = [bic - global_best_bic for bic in bic_values]

    # Step 3: Calculate the tangent for each component
    tangents = []
    for i in range(len(bic_values)):
        if i<best_component-2:
            steps = best_component - 2 - i  # Steps from component i to the global minimum
        else:
            steps = i- (best_component - 2)
        tangent = height_diff[i] / steps if steps !=0 else 0# Calculate the tangent (slope)
        tangents.append(tangent)

    # Step 4: Check for the component with the largest tangent before the global minimum
    lst_bf, lst_af= tangents[:best_component-2], tangents[best_component-1:]
    max_tangent = max(lst_bf[:-1]) # Largest tangent before the global minimum and excude the one before the global min

    # Step 5: Look for components before the global minimum with a tangent that is 20% smaller
    local_optimal_component = None
    for i in range(len(lst_bf)):
        if tangents[i] < (max_tangent * threshold):  # If tangent is 25% smaller than the largest tangent
            local_optimal_component = i + 2  # Update with the smaller component
            print(f"Best component selected due to L shape: {local_optimal_component}")
            break

    # Step 6: Handle tick shape BIC curve
    # Check if the BIC values after the global minimum are stable or increasing
    avg_af = np.mean(lst_af)  # Calculate average slope after global min

    # Step 7: Check if any component before the global min has a BIC value lower than 20% of the average tangent
    for i in range(len(lst_bf)):
        if tangents[i] < avg_af * threshold:
            local_optimal_component = i + 2 
            print(f"Best component selected due to tick shape: {local_optimal_component}")
            break

        
    if local_optimal_component is None or abs(local_optimal_component - best_component) <=2:
        print(f"Best component selected: {best_component}")
        return best_component, bic_values
    else:
        print(f"Best component selected: {local_optimal_component}")
        return local_optimal_component, bic_values
        
            
    # # Calculate slopes (differences between consecutive BIC values)
    # bic_diffs = np.diff(bic_values)  # Calculate the difference between consecutive BICs
    # slopes = np.divide(bic_diffs, bic_values[:-1]) # Calculate the slope between consecutive points

    # # Step 1: Find the best component using BIC
    # best_component = np.argmin(bic_values) + 2  # Find the best component via BIC

    # # # Step 2: Check for L-shape and find turning point with dynamic slope threshold
    # turning_point = None
    # for i in range(1, best_component):  # Only check within the range from 1 to best_component
    #     # 1) Check for turning point (negative to positive slope)
    #     if slopes[i - 1] < 0 and slopes[i] > 0:
    #         # 2) Calculate the average of all previous slopes before i-1
    #         if i - 1 > 0:  # Ensure there are previous slopes to calculate the average
    #             average_slope = np.mean(np.abs(slopes[:i-1]))   # Average of slopes before the turning point
    #         else:
    #             average_slope = 0  # Default to 0 if there are no previous slopes
    #         # 3) Check if slopes are small relative to this average slope
    #         if all(abs(slopes[j]) < stability_fraction * abs(average_slope) 
    #                for j in range(i, best_component)):
    #             turning_point = i + 2  # Add 1 to map to actual component number
    #             break

    # # Print the detected turning point and best component
    # if turning_point is None or abs(turning_point - best_component) <=2:      #
    #     print(f"No L-shaped BIC values detecgted. Best component is {best_component}.")
    #     return best_component, bic_values
    # else:
    #     print(f"Turning point of L-shaped BIC values detected at component {turning_point}.")
    #     return turning_point, bic_values

def fit_gmm(df, columns, best_component, code, year, direction, plot = None, save_path=None, ax=None): 

    gmm = GaussianMixture(n_components=best_component, random_state=42)

    data = df[columns].to_numpy()
    gmm.fit(data)
    
    covariances = gmm.covariances_.flatten()                                  # You can also check other properties of the GMM, such as covariances:

    is_stick_like = np.any(covariances < 1e-3)  # Check if there are any stick-like clusters (very small variance)

    if is_stick_like:
        
        print("Stick-like cluster detected. Using regularization (1e-3).")
        gmm = GaussianMixture(n_components=best_component, random_state=42, 
                              reg_covar=1e-3)                                 # If stick-like clusters are detected, apply higher regularization
        gmm.fit(data)
    else:
        print("Fitting without regularization.")
        
    means = gmm.means_.flatten()                                              # Re-extract the updated means, proportions, and variances after fitting with the selected reg_covar
    proportions = gmm.weights_.flatten()                                      # Extract the proportions (weights) of each component in the data  
    covariances = gmm.covariances_.flatten()                                  # You can also check other properties of the GMM, such as covariances:

    N = len(data)                                                             # Calculate the number of data points (N) in your dataset

    standard_errors = np.sqrt(covariances) / np.sqrt(N * proportions)         # Compute the standard error for each component's mean

    z_alpha_half = norm.ppf(0.975)                                            # Calculate the 95% confidence interval for each meanFor 95% confidence, z_alpha/2 is approx. 1.96
    lower_bound = means.flatten() - z_alpha_half * standard_errors
    upper_bound = means.flatten() + z_alpha_half * standard_errors

    sorted_indices = np.argsort(proportions)[::-1]                            # Sort the means, proportions, and covariances based on the proportions (weights)
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]
    sorted_lower_bound = lower_bound[sorted_indices]
    sorted_upper_bound = upper_bound[sorted_indices]
    
    component_labels = gmm.predict(data)  # This gives the component assignment for each data point
    # Now calculate the proportion of each country in each component (peak)
    country_proportions = {}
    for i in range(best_component):
        country_counts = df.loc[np.where(component_labels == i)[0], 
                                 'partnerISO'].value_counts(normalize=True)
        country_proportions[i] = country_counts
    
    sorted_country_proportions = {}
    for i in range(best_component):
        sorted_country_proportions[i] = pd.Series(country_proportions[i]).sort_values(ascending=False)

    text_d = 'imports' if direction == 'm' else 'exports'
    # General template output
    print(f"In {year}, the unit values for HS {code} {text_d}\n"
          f"are represented by {best_component} clusters,")

    # Print means
    mean_values = [f"{np.exp(mean):.3f}" for mean in sorted_means]  # Exponentiate back to normal scale

    chunk_size = 7  # Adjust the chunk size as needed
    mean_values_chunks = [", ".join(mean_values[i:i+chunk_size]) 
                          for i in range(0, len(mean_values), chunk_size)]
    
    print("with their mean unit prices at")
    
    for chunk in mean_values_chunks:
     print(chunk)  # Print each chunk on a new line


    # Print 95% confidence intervals
    ci_values = [f"({np.exp(lower):.3f}, {np.exp(upper):.3f})"
         for lower, upper in zip(sorted_lower_bound, sorted_upper_bound)]

    chunk_size = 3  # Adjust the chunk size as needed
    ci_values_chunks = [", ".join(ci_values[i:i+chunk_size]) 
                          for i in range(0, len(ci_values), chunk_size)]
    
    print("with 95% confidence intervals ranging from")
    
    for chunk in ci_values_chunks:
     print(chunk)  # Print each chunk on a new line
     
    # # Print proportions (percentages)
    # proportions_values = [f"{proportion*100:.2f}%" 
    #                       for proportion in sorted_proportions]
    # chunk_size = 7  # Adjust the chunk size as needed
    # proportions_values_chunks = [", ".join(proportions_values[i:i+chunk_size]) 
    #               for i in range(0, len(proportions_values), chunk_size)]
    
    # print("These components account for")
    # for chunk in proportions_values_chunks:
    #  print(chunk)  # Print each chunk on a new line
     
    # # Print country composition for each component in the legend
    # for i in range(best_component):
    #     print(f"Component {i + 1} country composition:")
    #     if i in sorted_country_proportions:
    #         top_countries = sorted_country_proportions[i].head(3)  # Get the top 3 countries
    #         for country, proportion in top_countries.items():
    #             print(f"  {country}: {proportion * 100:.2f}%")
    #     print("\n")
 
    # Print proportions (percentages) and country composition in a single line
    proportions_values = [
        f"{proportion*100:.2f}% ({', '.join([f'{country}: {proportion * 100:.2f}%' for country, proportion in sorted_country_proportions[i].head(3).items()])})"
        for i, proportion in enumerate(sorted_proportions)
    ]

    chunk_size = 1  # Adjust the chunk size as needed
    proportions_values_chunks = [
        ", ".join(proportions_values[i:i+chunk_size]) 
        for i in range(0, len(proportions_values), chunk_size)
    ]

    print("These components account for:")
    for chunk in proportions_values_chunks:
        print(chunk)  # Print each chunk on a new line
       
   
    if plot:
        
       x = np.linspace(data.min(), data.max() , 1000).reshape(-1, 1)  # Grid of values for the column
    
       colors = cm.get_cmap('tab20', best_component)  # Adjust color map for best_component

       gmm = GaussianMixture(n_components=best_component, random_state=42, reg_covar=1e-3)
       
       gmm.fit(data)  # Make sure data is passed in a column-vector form
       
       pdf = np.exp(gmm.score_samples(x))

       # Plot each GMM component, using sorted values
       for i in range(best_component):
           component_pdf = sorted_proportions[i] * np.exp(
               -0.5 * ((x - sorted_means[i]) ** 2) / sorted_covariances[i]
           ) / np.sqrt(2 * np.pi * sorted_covariances[i])
           if ax is None:
               fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure if no ax is passed
               
           # Get top 3 countries for the current component
           top_countries = sorted_country_proportions[i].head(3)  # Get the top 3 countries
           countries_str = ", ".join([f"{country}: {proportion * 100:.0f}%" for country, proportion in top_countries.items()])
           
           a,b = sorted_proportions[i],sorted_means[i]
           ax.plot(x, component_pdf, label=f"({i + 1}) {a*100:.0f}%: {np.exp(b):.2f} USD/kg ({countries_str})", color=colors(i))
           
           # Add number in the middle of the component curve
           mean_x_position = sorted_means[i]  # Position to place the number
           mean_y_position = np.max(component_pdf) / 2  # Center vertically
           ax.text(mean_x_position, mean_y_position, f"({i + 1})", fontsize=8, ha='center', va='center')
       
       iqr = np.percentile(data, 75) - np.percentile(data, 25)
       bin_width = 2 * iqr / len(data) ** (1 / 3)
       bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)  # Step 2: Compute bin edges
       # Plot the histogram of data
       ax.hist(data, bins=bin_edges, density=True, alpha=0.5, label="Data", color="gray")

       # Plot the overall GMM PDF
       ax.plot(x, pdf, label="Overall GMM", color="black", linewidth=2)

       # Add legend and labels
       ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
       ax.set_title(f"Unit values in a GMM distribution for HS {code} {text_d} in {year}")
       ax.set_xlabel("Data Value")
       ax.set_ylabel("Density")
       
       if save_path:
           plt.savefig(save_path, bbox_inches="tight")
       else:
           plt.show()
           
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
    feature_min = scaler.data_min_[feature_index]  # Original min value for the feature
    feature_range = scaler.data_range_[feature_index]  # Original range for the feature
    scaled_back = feature_min + (scaled_values * feature_range)
    
    # Step 2: Reverse the log transformation
    original_values = np.exp(scaled_back)  # Reverse log1p -> expm1
    
    return original_values

def fit_gmm2(data, components, code, year, direction, plot='2D', save_path=None, ax=None):
    """
    Fits a Gaussian Mixture Model (GMM) to a dataset with two features and provides
    statistics along with an optional 2D plot.

    Parameters:
    - data: ndarray or DataFrame with two columns (ln_Unit_Price and ln_netWgt).
    - components: Number of GMM components to fit.
    - code: HS code for the commodity.
    - year: Year of analysis.
    - direction: 'm' for imports, 'x' for exports.
    - plot: If '2D', generates a 2D contour plot of GMM clusters.
    - save_path: Path to save the plot (if provided).
    - ax: Axes object for the plot (optional).

    Returns:
    - Dictionary containing means, proportions, covariances, and confidence intervals.
    """
    data = data.to_numpy()
    
    scaler = MinMaxScaler()#StandardScaler()
    data = scaler.fit_transform(data)  # z-score normalization, with the mean and the standard deviation at 0 and 1, repectively. 


    # Fit the GMM model
    gmm = GaussianMixture(n_components=components, random_state=42, reg_covar=1e-3)
    gmm.fit(data) 

    # Extract statistics from the fitted GMM
    means = gmm.means_
    proportions = gmm.weights_
    covariances = gmm.covariances_

    # Calculate standard errors for the means
    N = len(data)
    standard_errors = np.sqrt(np.array([np.diag(cov) for cov in covariances])) / np.sqrt(N * proportions[:, np.newaxis])

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
    text_d = 'imports' if direction == 'm' else 'exports'

    print(f"In {year}, the unit values for HS {code} {text_d} are represented by {components} clusters.")
    
    # Report means
    mean_values = [f"({np.exp(mean[0]):.3f}, {np.exp(mean[1]):.3f})" for mean in sorted_means]
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
    if plot == '2D':
        # Define grid limits
        x_min, x_max = min(data[:, 0]), max(data[:, 0])
        y_min, y_max = min(data[:, 1]), max(data[:, 1])
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_data = np.vstack([xx.ravel(), yy.ravel()]).T

        # Generate probabilities for the grid
        ##gmm_probs = np.exp(gmm.score_samples(grid_data)).reshape(xx.shape)
        colors = cm.get_cmap('tab20', components)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data points
        ax.scatter(data[:, 0], data[:, 1], alpha=0.2, label="Data", color="gray", s=10)

        ##ax.contourf(xx, yy, gmm_probs, levels=20, cmap="Blues", alpha=0.5)

        # Plot contours for each component
        for i in range(components):
            diff = grid_data - sorted_means[i]
            cov_inv = np.linalg.inv(sorted_covariances[i])  # Inverse of covariance matrix
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1) 
            component_prob = sorted_proportions[i] * np.exp(exponent) / (
                2 * np.pi * np.sqrt(np.linalg.det(sorted_covariances[i]))
            )
            component_prob = component_prob.reshape(xx.shape)
            ax.contour(xx, yy, component_prob, levels=5, colors=[colors(i)], alpha=0.9, linewidths=0.8)
            
        # Example data
        xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # Update the tick labels with original values
        ori_x = [re_minmax_log(tick, scaler, 0) for tick in xticks]
        ori_y = [re_minmax_log(tick, scaler, 1) for tick in xticks]
        
        ori_x = [round(label, 2) for label in ori_x]
        ori_y = [round(label, 2) for label in ori_y]
        ori_y = [f"{label:.2e}" if label >= 1000 else round(label, 2) for label in ori_y]
        lb_x= [f"{tick} ({label})" for tick, label in zip(xticks, ori_x)]
        lb_y= [f"{tick}\n({label})" for tick, label in zip(xticks, ori_y)]
        
        ax.set_xticks(xticks)  # Set the tick positions (scaled values)
        ax.set_yticks(xticks)  # Set the tick positions (scaled values)
        ax.set_xticklabels(lb_x, fontsize=10)  # Set the tick labels with original data
        ax.set_yticklabels(lb_y, fontsize=10)  # Set the tick labels with original data
        



        # Ensure equal aspect ratio
        ax.set_aspect('equal', adjustable='box')


        # Add labels, title, and legend
        ax.set_title(f"GMM Clustering for HS {code} {text_d} in {year}", fontsize=14)
        ax.set_xlabel("ln_Unit_Price", fontsize=12)
        ax.set_ylabel("ln_netWgt", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        # Add the custom legend entry for the scaled and original data pair
        handles.append(plt.Line2D([0], [0], color='white', lw=2))
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
            "upper": sorted_upper_bound
        }
    }

def fit_gmm3(data, components, code, year, direction, save_path=None):
    """
    Fits a Gaussian Mixture Model (GMM) to a dataset with three features and provides
    statistics along with an optional 3D plot.

    Parameters:
    - data: ndarray or DataFrame with three columns (ln_Unit_Price, ln_netWgt, Country_Grouped_Unit_Value).
    - components: Number of GMM components to fit.
    - code: HS code for the commodity.
    - year: Year of analysis.
    - direction: 'm' for imports, 'x' for exports.
    - save_path: Path to save the plot (if provided).

    Returns:
    - Dictionary containing means, proportions, and covariances.
    """
    data = data.to_numpy()
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)


    # Fit the GMM model
    gmm = GaussianMixture(n_components=components, random_state=42, reg_covar=1e-3)
    gmm.fit(data)

    # Extract statistics from the fitted GMM
    means = gmm.means_
    proportions = gmm.weights_
    covariances = gmm.covariances_

    # Sort components by proportions
    sorted_indices = np.argsort(proportions)[::-1]
    sorted_means = means[sorted_indices]
    sorted_proportions = proportions[sorted_indices]
    sorted_covariances = covariances[sorted_indices]

    # Text for import/export
    text_d = 'imports' if direction == 'm' else 'exports'
    print(f"In {year}, the unit values for HS {code} {text_d} are represented by {components} clusters.")
    
    # Optimized 3D Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.get_cmap('tab20', components)

    # Plot each data point with transparency for clarity
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.1, color='gray', s=5)
    
    # Plot each cluster mean and draw 3D ellipsoids
    for i in range(components):
        color = colors(i)
        ax.scatter(sorted_means[i, 0], sorted_means[i, 1], sorted_means[i, 2], 
                   color=color, marker='o', s=200, edgecolors='black', label=f'Cluster {i+1}')
        
        # Compute 3D covariance ellipsoid
        eigenvalues, eigenvectors = np.linalg.eigh(sorted_covariances[i])
        radii = np.sqrt(eigenvalues) * 2  # 2 standard deviations for visualization
        
        # Generate points on a unit sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        # Scale and rotate the unit sphere to match the covariance ellipse
        ellipsoid = np.stack([x, y, z], axis=-1)
        ellipsoid = ellipsoid @ np.diag(radii)  # Scale
        ellipsoid = ellipsoid @ eigenvectors.T  # Rotate
        ellipsoid += sorted_means[i]  # Translate to mean
        
        # Plot the ellipsoid as a mesh
        ax.plot_wireframe(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2], color=color, alpha=0.3)
    
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
        "covariances": sorted_covariances
    }
            

   
    

