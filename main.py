"""
UVPicker - Unit Value Analysis Tool
Copyright (c) 2025 Kai Li
Licensed under LGPL v3.0 – See LICENSE file for details.
Funding Acknowledgment:
- European Horizon Project (No. 101060142) "RESOURCE – REgional project development aSsistance fOr the Uptake of an aRagonese Circular Economy"
- Financial support from CML, Leiden University, for full access to the UN Comtrade database
"""

import time
from pathlib import Path
from uv_logger import logger_setup, logger_time

from uv_config import load_config, prefix_dict_keys, save_report_dict

from uv_preparation import clean_trade, detect_outliers

from uv_analysis import (
    modality_test,
    modality_test2,
    fit_all_unimodal_models,
    bootstrap_parametric_ci,
    find_gmm_components,
    fit_gmm,
)

from uv_visualization import plot_histogram, plot_dist

# subscription_key = "4a624b220f67400c9a6ef19b1890f1f9"
# path = 'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_2023_numpy'
code = "370110"
year = "2013"
flow = "m"


def cmltrade_uv(code, year, flow):
    
    # === Load config (including folder paths, mappings, thresholds)
    config = load_config(
        input_dir=Path(r"C:\Users\lik6\OneDrive - Universiteit Leiden\PlasticTradeFlow\tradeflow\cml_trade\data_uncomtrade")
    )

    # === Setup logger for this specific HS code, year, and flow
    logger = logger_setup(
        code=code, year=year, flow=flow, log_dir=config["dirs"]["logs"]
    )
    zero_time = time.time()  # Starting the total analysis timer
    print(f"Starting analysis for HS code {code} in year {year}...\n")
    # %% Step 1: Clean trade data
    logger.info(f"Clean trade data (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    
    (df_uv, df_q, report_clean, report_q_clean,non_kg_unit, 
      is_valid_kg, is_valid_q) = clean_trade(code, year, flow, config, logger)
    
    if not is_valid_kg and not is_valid_q:
        logger.warning("❌ Skipping both kg-based and non-kg-based analysis due to insufficient sample size.")

        if isinstance(report_q_clean, dict):
            report_q_clean = prefix_dict_keys(report_q_clean, prefix="q_")

        report_final = {}
        if isinstance(report_clean, dict):
            report_final.update(report_clean)
        if isinstance(report_q_clean, dict):
            report_final.update(report_q_clean)

        report_final["skip_reason"] = "Too few kg-based and non-kg-based records"
        
        save_report_dict(report_final, code, year, flow, config, logger)

        return report_final
    
    logger_time("Completed trade data cleaning", start_time, logger)
    # %% Step 2: Detect outliers
    logger.info(f"Outlier detection (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    
    # === Kg-based UV ===
    if is_valid_kg:
        df_filtered, df_outliers, report_outlier = detect_outliers(
            df_uv, "ln_uv", code, year, flow, logger, unit_label="USD/kg", 
            plot =True, save_path=True, file_format="pdf")
    
    # === Non-kg-based UV (if exists) ===
    if is_valid_q and df_q is not None and not df_q.empty:
        df_q_filtered, df_q_outliers, report_q_outlier = detect_outliers(
            df_q,"ln_uv_q",code,year,flow,logger,unit_label=non_kg_unit,
            plot =True, save_path=True,)
        
    logger_time("Completed outlier detection", start_time, logger)

    # %% Step 3: Histogram
    logger.info(f"Histogram (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    
    # === Kg-based UV ===
    if is_valid_kg:
        plot_histogram(df_filtered["ln_uv"],code,year,flow, unit_label="USD/kg",save_path=True, file_format="pdf")
    
    # === Non-kg-based UV (if exists) ===
    if is_valid_q and df_q is not None and not df_q.empty:
        plot_histogram(df_q_filtered["ln_uv_q"],code,year,flow,
                       unit_label=non_kg_unit,save_path=True, file_format="pdf")
        
    logger_time("Completed histogram", start_time, logger)
    # %% Step 4: Modality test
    logger.info(f"Modality Test (HS {code}, {year}, {flow.upper()})")
    print("Running modality test on unit values...")
    start_time = time.time()

    # === Kg-based UV ===
    if is_valid_kg:
        report_modality, modality_decision, is_borderline = modality_test(
        df_filtered,logger=logger)
        
        #report_modality2, modality_decision2 = modality_test2(
        #df_filtered,logger=logger)

    # === Non-kg-based UV (if exists) ===
    if is_valid_q and df_q is not None and not df_q.empty:
        report_q_modality, modality_q_decision, is_q_borderline = modality_test(
            df_q_filtered,col="ln_uv_q",logger=logger)
        
        #report_q_modality2, modality_q_decision2 = modality_test2(
        #df_q_filtered,col="ln_uv_q",logger=logger)
    else:
        modality_q_decision = "unknown"
        is_q_borderline = False

    logger_time("Completed modality test", start_time, logger)
    # %% Step 5: Distribution fit
    # %%% Kg-based UV
    if is_valid_kg:
        if modality_decision == "unimodal" or is_borderline:
            
            # === Fitting Unimodal distribution of kg-based UV ====
            logger.info(f"Fitting Unimodal distribution of kg-based UV ({year}, {flow.upper()})")
            start_time = time.time()
            
            (best_fit_name, report_best_fit_uni, report_all_uni_fit, raw_params_dict,
            ) = fit_all_unimodal_models(df_filtered["ln_uv"], logger=logger)
            
            logger.info(f"- {best_fit_name.capitalize()} distribution fits best based on AIC and BIC.")
            logger_time("Completed unimodal distribution fit (kg-based)", start_time, logger)

            # === Bootstrapping CI (kg-based) ====
            start_time = time.time()
            logger.info(f"Bootstrapping CI (kg-based) ({year}, {flow.upper()})")
            
            report_ci_uni = bootstrap_parametric_ci(
                df_filtered["ln_uv"], dist=best_fit_name, n_bootstraps=1000
            )
            
            logger_time("Completed ootstrapping CI (kg-based)", start_time, logger)

            # === Plotting unimodal distribution fit of kg-based UV ====
            logger.info(f"Plotting unimodal distribution fit of kg-based UV ({year}, {flow.upper()}))")
            start_time = time.time()
            
            plot_dist(
                df_filtered["ln_uv"],code,year,flow,unit_label="USD/kg",
                dist=None,
                best_fit_name=best_fit_name,
                report_best_fit_uni=report_best_fit_uni,
                report_all_uni_fit=report_all_uni_fit,
                raw_params_dict=raw_params_dict,
                ci=report_ci_uni,
                save_path=True,
                ax=None,
                file_format="pdf"
            )
            
            logger_time("Completed unimodal distribution fit plot for kg-based UV", start_time, logger)
            
            if is_borderline:
                
                # === Also fitting GMM on kg-based UV if borderline ===
                logger.info("⚠️ Borderline modality: Also fitting GMM on kg-based UV")
                start_time = time.time()
                
                optimal_k, bic_values, report_gmmf_1d = find_gmm_components(
                    df_filtered[["ln_uv"]], code, year, flow, "USD/kg", plot=True, save_path=True
                )
                
                report_gmm_1d = fit_gmm(
                    df_filtered,
                    ["ln_uv"],
                    optimal_k,
                    code,
                    year,
                    flow,
                    plot=True,
                    save_path=True,
                    n_init=10,
                    reg_covar=1e-3,
                    unit_label="USD/kg",
                )
                
                logger_time("Completed GMM fit on kg-based UV (borderline)", start_time, logger)
        else:
            logger.info("Fitting GMM on kg-based UV")
            start_time = time.time()
            
            optimal_k, bic_values, report_gmmf_1d = find_gmm_components(
                df_filtered[["ln_uv"]], code, year, flow, "USD/kg", plot=True, save_path=True
            )
            
            report_gmm_1d = fit_gmm(
                df_filtered,
                ["ln_uv"],
                optimal_k,
                code,
                year,
                flow,
                plot=True,
                save_path=True,
                n_init=10,
                reg_covar=1e-3,
                unit_label="USD/kg",
            )
            
            logger_time("Completed GMM fit on kg-based UV", start_time, logger)
    # %%% Non-kg-based UV
    if is_valid_q and df_q is not None and not df_q.empty:
        if modality_q_decision == "unimodal" or is_q_borderline:
            
            # === Fitting Unimodal distribution of non-kg-based UV ====
            logger.info(f"Fitting Unimodal distribution of non-kg-based UV ({year}, {flow.upper()})")
            start_time = time.time()
            (best_fit_name_q,
                report_q_best_fit_uni,
                report_q_all_uni_fit,
                raw_params_dict_q,
            ) = fit_all_unimodal_models(df_q_filtered["ln_uv_q"], logger=logger)
            
    
            logger.info(f"- {best_fit_name_q.capitalize()} distribution fits best based on AIC and BIC.")
            logger_time("Completed unimodal distribution fit (non-kg-based)", start_time, logger)
            
            # === Bootstrapping CI (non-kg-based) ====
            start_time = time.time()
            logger.info(f"Bootstrapping CI (non-kg-based) ({year}, {flow.upper()})")
            
            report_q_ci_uni = bootstrap_parametric_ci(
                df_q_filtered["ln_uv_q"], dist=best_fit_name_q, n_bootstraps=1000
            )
            
            logger_time("Completed bootstrapping CI (non-kg-based)", start_time, logger)
            # === Plotting unimodal distribution fit of non-kg-based UV ====
            logger.info(f"Plotting unimodal distribution fit of non-kg-based UV ({year}, {flow.upper()}))")
            start_time = time.time()
            plot_dist(
                df_q_filtered["ln_uv_q"],
                code,
                year,
                flow,
                unit_label=non_kg_unit,
                dist=None,
                best_fit_name=best_fit_name_q,
                report_best_fit_uni=report_q_best_fit_uni,
                report_all_uni_fit=report_q_all_uni_fit,
                raw_params_dict=raw_params_dict_q,
                ci=report_q_ci_uni,
                save_path=True,
                ax=None,
                file_format="pdf"
            )
            
            logger_time("Completed unimodal distribution fit plot for non-kg-based UV", start_time, logger)
                
            if is_q_borderline:
                
                # === Also fitting GMM on non-kg-based UV if borderline ===
                logger.info("⚠️ Borderline modality: Also fitting GMM on non-kg-based UV")
                start_time = time.time()
                
                optimal_k_q, bic_values_q, report_q_gmmf_1d = find_gmm_components(
                    df_q_filtered[["ln_uv_q"]], code, year, flow, non_kg_unit, plot=True, save_path=True
                )
               
                report_q_gmm_1d = fit_gmm(
                    df_q_filtered,
                    ["ln_uv_q"],
                    optimal_k_q,
                    code,
                    year,
                    flow,
                    plot=True,
                    save_path=True,
                    n_init=10,
                    reg_covar=1e-3,
                    unit_label=non_kg_unit,
                )

                logger_time("Completed GMM fit on non-kg-based UV (borderline)", start_time, logger)
        else:
            logger.info("Fitting GMM on non-kg-based UV")
            start_time = time.time()
            
            optimal_k_q, bic_values_q, report_q_gmmf_1d = find_gmm_components(
                df_q_filtered[["ln_uv_q"]], code, year, flow, non_kg_unit, plot=True, save_path=True
            )
            
            report_q_gmm_1d = fit_gmm(
                df_q_filtered,
                ["ln_uv_q"],
                optimal_k_q,
                code,
                year,
                flow,
                plot=True,
                save_path=True,
                n_init=10,
                reg_covar=1e-3,
                unit_label=non_kg_unit,
            )
           
            logger_time("Completed GMM fit on non-kg-based UV", start_time, logger)
    # %% Step 6: Return final report
    
    report_final = {}
    # Reports to merge (non-kg ones will be prefixed)
    report_keys_kg = [
        "report_clean", "report_outlier", "report_modality", 
        "report_all_uni_fit", "report_ci_uni", "report_gmmf_1d", "report_gmm_1d"
    ]

    report_keys_q = [
        "report_q_clean", "report_q_outlier", "report_q_modality", 
        "report_q_all_uni_fit", "report_q_ci_uni", "report_q_gmmf_1d", "report_q_gmm_1d"
    ]

    # Merge kg-based reports directly
    for name in report_keys_kg:
        r = locals().get(name)
        if isinstance(r, dict):
            report_final.update(r)

    # Merge q-based reports with prefix
    for name in report_keys_q:
        r = locals().get(name)
        if isinstance(r, dict):
            r_prefixed = prefix_dict_keys(r, prefix="q_")
            report_final.update(r_prefixed)
            
    save_report_dict(report_final, code, year, flow, config, logger)
          
    return report_final