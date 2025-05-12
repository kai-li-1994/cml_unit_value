"""
UVPicker - Unit Value Analysis Tool
Copyright (c) 2025 Kai Li
Licensed under LGPL v3.0 – See LICENSE file for details.
Funding Acknowledgment:
- European Horizon Project (No. 101060142) "RESOURCE – REgional project development aSsistance fOr the Uptake of an aRagonese Circular Economy"
- Financial support from CML, Leiden University, for full access to the UN Comtrade database
"""
import time
from uv_preparation import (
    clean_trade_tariff,
    load_config,
    extract_trade_tariff,
    extract_trade,
    clean_trade,
    detect_outliers,
)
from uv_analysis import (
    fit_all_unimodal_models,
    fit_logistic,
    dip_test,
    modality_test,
    fit_normal,
    fit_skewnormal,
    fit_studentt,
    fit_gennorm,
    fit_johnsonsu,
    bootstrap_parametric_ci,
    find_gmm_components,
    fit_gmm,
    fit_gmm12,
    ensure_cov_matrix,
    fit_gmm2_flexible,
    fit_gmm3,
    find_gmm_components2
)
from uv_visualization import (
    plot_histogram,
    plot_dist,
    plot_gdpcountry,
    plot_gmm1d_country,
)

from uv_logger_setup import logger_setup, logger_time_info

config = load_config()
logger = logger_setup()
# subscription_key = "4a624b220f67400c9a6ef19b1890f1f9"
# path = 'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_2023_numpy'
# code = '870322'
# year = '2018'
# flow = 'm'

def cmltrade_uv(path, code, year, flow):
    zero_time = time.time()  # Starting the total analysis timer
    print(f"Starting analysis for HS code {code} in year {year}...\n")

    # Step 1: clean trade
    logger.info(f"Trade data cleaning (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    df_uv, df_q, report_cleaning = clean_trade_tariff(path, code, year, flow, config)
    logger_time_info("Trade data cleaning", start_time)

    # Step 2: Detect outliers
    logger.info(f"Outlier detection (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    df_filtered, df_outliers, report_outlier = detect_outliers(
                         df_uv, ["ln_uv"], label="Kg-based UV")
    if df_q is not None and not df_q.empty:
        df_q_filtered, df_q_outliers, report_q_outlier = detect_outliers(
                       df_q, ["ln_uv_q"], label="Non-kg-based UV")
    logger_time_info("Outlier detection", start_time)
    
    # Step 3: Histogram
    logger.info(f"Outlier Detection (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    plot_histogram(
        df_filtered["ln_uv"],
        code,
        year,
        flow,
        save_path="740311_m_2018.pdf",
        ax=None,
    )
    if df_uv is not None and not df_q.empty:
        plot_histogram(
            df_q_filtered["ln_uv_q"],
            code,
            year,
            flow,
            save_path="740311_m_2018.pdf",
            ax=None,
        )
        logger_time_info("Histogram plotting", start_time)
    
    # Step 4: Modality test
    logger_section_header("Modality Test")
    print("Running modality test on unit values...")
    start_time = time.time()
    report_modality = modality_test(df_filtered)
    if df_q is not None and not df_q.empty:
        report_q_modality = modality_test(df_q_filtered, col="ln_uv_q")
    else:
        print("⚠️ Skipping quantity-weighted modality test: df1_q is None or empty.")          
    logger_time_info("modality test", start_time)
    
    # Step 5: Distribution fit
    if is_unimodal:
        print_section_header("Test for Best Unimodal Distribution")
        start_time = time.time()
        # Fit all unimodal distribution candidates and evaluate AIC/BIC
        best_fit, fit_result, all_results = fit_all_unimodal_models(
            df2["ln_uv"], pt=True
        )
        print(
            f"- {best_fit.capitalize()} distribution fits best based on AIC and BIC."
        )
        logger_time_info("Test for best unimodal distribution", start_time)

        print_section_header(
            "Bootstrapping CI for mean, median, mode, and variance values"
        )
        start_time = time.time()
        print(f"Bootstrapping for {best_fit} in 1000 iterations ...")
        ci_mean, ci_median, ci_mode, ci_var = bootstrap_parametric_ci(
            df2["ln_uv"], dist=best_fit, n_bootstraps=1000
        )
        logger_time_info("Bootstrapping CI", start_time)

        print_section_header("Plotting")
        start_time = time.time()
        plot_dist(
            data=df2["ln_uv"],
            code=code,
            year=year,
            flow=flow,
            dist=None,
            best_fit=best_fit,
            fit_result=fit_result,
            all_results=all_results,
            ci_mean=ci_mean,
            ci_median=ci_median,
            ci_mode=ci_mode,
            ci_var=ci_var,
            save_path=f"{code}_{flow}_{year}_distribution_fit.pdf",
            ax=None,
        )

        logger_time_info("Plot", start_time)

    else:
        # Step 5: Distribution fit
        logger_section_header(
            "Fitting a 1D Gaussian Mixture Model (GMM) on unit values"
        )
        start_time = time.time()

        components, bic_values = find_gmm_components(
            df_filtered[["ln_uv"]],
            max_components=50,
            convergence_threshold=5,
            reg_covar=1e-3,
            threshold=0.2,
            plot=True,
            ax=None,
            save_path=None,
        )
        logger_time_info("Fitting a 1D GMM on unit values", start_time)
        
        start_time = time.time()
        optimal_k, bic_values, report_gmm_cselect = find_gmm_components2(
                             df_filtered[["ln_uv"]],
                             max_components=50,
                             convergence_threshold=5,
                             reg_covar=1e-3,
                             threshold=0.2,
                             n_init=10,
                             plot=True,
                             ax=None,
                             save_path=None)
        logger_time_info("Fitting a 1D GMM on unit values", start_time)
    
        fit_gmm(
            df_filtered,
            ["ln_uv"],
            optimal_k,
            code,
            year,
            flow,
            cc=True,
            plot=True,
            ax=None,
        )
        logger_time_info("Fitting a 1D GMM on unit values", start_time)
        
        start_time = time.time()
        gmm_1d_report = fit_gmm12(
            df_filtered,
            ["ln_uv"],
            optimal_k,
            code,
            year,
            flow,
            plot=True,
            save_path=None,
            ax=None,
            n_init=10,
            reg_covar = 1e-3
        )

        logger_section_header("Fitting a 2D GMM on unit values and trade volume")
        
        start_time = time.time()
        components, bic_values = find_gmm_components(
            df2[["ln_uv", "ln_netWgt"]],
            max_components=50,
            convergence_threshold=5,
        )
        fit_gmm2_flexible(
            df2[["ln_uv", "ln_netWgt"]],
            components,
            code,
            year,
            flow,
            plot="2D",
            save_path=None,
            ax=None,
            covariance_type="full",
        )
        logger_time_info(
            "Fitting a 2D GMM on unit values and trade volume", start_time
        )

        components, bic_values = find_gmm_components(
            df2[["ln_uv", "ln_gdp"]],
            max_components=50,
            convergence_threshold=5,
        )
        fit_gmm2(
            df2[["ln_uv", "ln_gdp"]],
            components,
            code,
            year,
            flow,
            plot="2D",
            save_path=None,
            ax=None,
        )

        logger_section_header(
            "Fitting a 3D GMM on unit values, trade volume, and GDP per capita"
        )
        components, bic_values = find_gmm_components(
            df2[["ln_uv", "ln_netWgt", "ln_gdp"]],
            max_components=50,
            convergence_threshold=5,
        )
        a, b = fit_gmm3(
            df2,
            ["ln_uv", "ln_netWgt", "ln_gdp"],
            components,
            code,
            year,
            flow,
            save_path=None,
        )
        logger_time_info(
            "Fitting a 3D GMM on unit values, trade volume, and GDP per capita",
            start_time,
        )
        plot_gmm1d_country(
            df2, b, code, year, flow, save_path="country_gmm.svg"
        )

        # Extra plot on the countries' involvement

    total_time = time.time() - zero_time
    print(f"Analysis complete. Total time: {total_time:.2f} seconds.")
