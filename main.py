"""
UVPicker - Unit Value Analysis Tool
Copyright (c) 2025 Kai Li
Licensed under LGPL v3.0 – See LICENSE file for details.
Funding Acknowledgment:
- European Horizon Project (No. 101060142) "RESOURCE – REgional project development aSsistance fOr the Uptake of an aRagonese Circular Economy"
- Financial support from CML, Leiden University, for full access to the UN Comtrade database
test text
"""

import time

from uv_logger import logger_setup, logger_time

from uv_config import load_config

from uv_preparation import clean_trade, detect_outliers

from uv_analysis import (
    fit_all_unimodal_models,
    modality_test,
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


# subscription_key = "4a624b220f67400c9a6ef19b1890f1f9"
#path = 'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_2023_numpy'
code = '850511'
year = '2023'
flow = 'm'

config = load_config(input_dir='C:/Users/lik6/Data/ComtradeTariffline/merge/')

logger = logger_setup(code = code, year =year,flow =flow,log_dir=config["dirs"]["logs"])


def cmltrade_uv(code, year, flow):
    zero_time = time.time()  # Starting the total analysis timer
    print(f"Starting analysis for HS code {code} in year {year}...\n")

    # === Step 1: Clean trade data ===
    logger.info(f"Clean trade data (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    df_uv, df_q, report_clean, non_kg_unit = clean_trade(code, year, flow, config, logger)
    logger_time("Clean trade data", start_time, logger)

    # === Step 2: Detect outliers ===
    logger.info(f"Outlier detection (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    df_filtered, df_outliers, report_outlier = detect_outliers(
                         df_uv, ["ln_uv"],  code, year, flow, logger, label="Kg-based UV")
    if df_q is not None and not df_q.empty:
        df_q_filtered, df_q_outliers, report_q_outlier = detect_outliers(
                       df_q, ["ln_uv_q"],  code, year, flow, logger, label="Non-kg-based UV")

    logger_time("Detect outliers", start_time, logger)
    
    # === Step 3: Histogram ===
    logger.info(f"Histogram (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    plot_histogram(
        df_filtered["ln_uv"],
        code, 
        year, 
        flow,
        unit_label="USD/kg", 
        save_path=True
    )
    if df_q is not None and not df_q.empty:
        plot_histogram(
            df_q_filtered["ln_uv_q"],
            code, 
            year, 
            flow,
            unit_label= non_kg_unit, 
            save_path=True
        )
    logger_time("Histogram", start_time, logger)
    
    # === Step 4: Modality test === 
    logger.info(f"Modality Test (HS {code}, {year}, {flow.upper()})")
    print("Running modality test on unit values...")
    start_time = time.time()
    report_modality, modality_decision = modality_test(df_filtered, logger = logger)
    if df_q is not None and not df_q.empty:
        report_q_modality, modality_q_decision = modality_test(df_q_filtered, col="ln_uv_q", logger = logger)
    else:
        logger.info("⚠️ Skipping quantity-weighted modality test: df1_q is None or empty.")          
    logger_time("Modality test", start_time, logger)
    
    # Step 5: Distribution fit
    if modality_decision == "unimodal":
        logger.info(f"Unimodal distribution fitting for kg-based UV ({year}, {flow.upper()})")
        start_time = time.time()
        best_fit_name, report_best_fit_uni, report_all_uni_fit, raw_params_dict = fit_all_unimodal_models(df_filtered["ln_uv"], logger = logger)
        logger.info(f"- {best_fit_name.capitalize()} distribution fits best based on AIC and BIC.")
        logger_time("Unimodal fit (kg-based)", start_time, logger)
        
        start_time = time.time()
        logger.info(f"Bootstrapping CI (kg-based) ({year}, {flow.upper()})")
        report_ci_uni = bootstrap_parametric_ci(df_filtered["ln_uv"], dist=best_fit_name, n_bootstraps=1000)
        logger_time("Bootstrapping CI (kg-based)", start_time, logger)
        
        logger.info(f"Unimodal distribution fit plot for kg-based UV ({year}, {flow.upper()})")
        start_time = time.time()
        plot_dist(
            df_filtered["ln_uv"], 
            code, 
            year, 
            flow, 
            unit_label="USD/kg",
            dist=None, 
            best_fit_name=best_fit_name, 
            report_best_fit_uni=report_best_fit_uni, 
            report_all_uni_fit=report_all_uni_fit, 
            raw_params_dict=raw_params_dict,
            ci=report_ci_uni,
            save_path=True, 
            ax=None)
        logger_time("Unimodal distribution fit plot for kg-based UV", start_time, logger)
    else:
        logger.info("Fitting GMM on kg-based UV (multimodal)")
        start_time = time.time()
        optimal_k, bic_values, report_gmm_cselect = find_gmm_components2(df_filtered[["ln_uv"]])
        gmm_1d_report = fit_gmm12(
            df_filtered,
            ["ln_uv"],
            optimal_k,
            code,
            year,
            flow,
            plot=True,
            save_path=f"{code}_{flow}_{year}_uvkg_gmm_fit.pdf",
            n_init=10,
            reg_covar=1e-3
        )
        logger_time("Fitting GMM on kg-based UV (multimodal)", start_time, logger)
        
    if df_q is not None and not df_q.empty:
        if modality_q_decision == "unimodal":
            logger.info("Unimodal distribution fitting for non-kg UV")
            start_time = time.time()
            best_fit_q, fit_result_q, all_results_q = fit_all_unimodal_models(
                                           df_q_filtered["ln_uv_q"], pt=True)
            logger.info(f"- {best_fit_q.capitalize()} distribution fits best (non-kg).")
            logger_time("Unimodal distribution fitting for non-kg UV", start_time, logger)
            
            start_time = time.time()
            logger.info("Bootstrapping CI (non-kg-based)")
            ci_mean_q, ci_median_q, ci_mode_q, ci_var_q = bootstrap_parametric_ci(df_q_filtered["ln_uv_q"], dist=best_fit_q, n_bootstraps=1000)
            plot_dist(
                data=df_q_filtered["ln_uv_q"],
                code=code,
                year=year,
                flow=flow,
                dist=None,
                best_fit=best_fit_q,
                fit_result=fit_result_q,
                all_results=all_results_q,
                ci_mean=ci_mean_q,
                ci_median=ci_median_q,
                ci_mode=ci_mode_q,
                ci_var=ci_var_q,
                save_path=f"{code}_{flow}_{year}_uvq_distribution_fit.pdf"
            )
            logger_time("Bootstrapping CI (non-kg-based)", start_time, logger)
        else:
            start_time = time.time()
            logger.info("Fitting GMM on non-kg-based UV (multimodal)")
            optimal_k_q, bic_values_q, report_gmm_q = find_gmm_components2(df_q_filtered[["ln_uv_q"]])
            gmm_1d_report_q = fit_gmm12(
                df_q_filtered,
                ["ln_uv_q"],
                optimal_k_q,
                code,
                year,
                flow,
                plot=True,
                save_path=f"{code}_{flow}_{year}_uvq_gmm_fit.pdf",
                n_init=10,
                reg_covar=1e-3
            )
            logger_time("Fitting GMM on non-kg-based UV (multimodal)", start_time, logger)
    
        
    if modality_decision:
        logger.info(f"Distribution fit (HS {code}, {year}, {flow.upper()})")
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
