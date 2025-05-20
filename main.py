"""
UVPicker - Unit Value Analysis Tool
Copyright (c) 2025 Kai Li
Licensed under LGPL v3.0 – See LICENSE file for details.
Funding Acknowledgment:
- European Horizon Project (No. 101060142) "RESOURCE – REgional project development aSsistance fOr the Uptake of an aRagonese Circular Economy"
- Financial support from CML, Leiden University, for full access to the UN Comtrade database
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
    ensure_cov_matrix,
    fit_gmm2_flexible,
)

from uv_visualization import (
    plot_histogram,
    plot_dist,
    plot_gdpcountry,
    plot_gmm1d_country,
)


# subscription_key = "4a624b220f67400c9a6ef19b1890f1f9"
# path = 'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_2023_numpy'
code = "390110"
year = "2022"
flow = "x"

config = load_config(input_dir="C:/Users/lik6/Data/ComtradeTariffline/merge/")

logger = logger_setup(
    code=code, year=year, flow=flow, log_dir=config["dirs"]["logs"]
)


def cmltrade_uv(code, year, flow):
    zero_time = time.time()  # Starting the total analysis timer
    print(f"Starting analysis for HS code {code} in year {year}...\n")

    # %% Step 1: Clean trade data
    logger.info(f"Clean trade data (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    df_uv, df_q, report_clean, non_kg_unit = clean_trade(
        code, year, flow, config, logger
    )
    logger_time("Completed trade data cleaning", start_time, logger)

    # %% Step 2: Detect outliers
    logger.info(f"Outlier detection (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    df_filtered, df_outliers, report_outlier = detect_outliers(
        df_uv, ["ln_uv"], code, year, flow, logger, label="Kg-based UV"
    )
    if df_q is not None and not df_q.empty:
        df_q_filtered, df_q_outliers, report_q_outlier = detect_outliers(
            df_q,
            ["ln_uv_q"],
            code,
            year,
            flow,
            logger,
            label="Non-kg-based UV",
        )

    logger_time("Completed outlier detection", start_time, logger)

    # %% Step 3: Histogram
    logger.info(f"Histogram (HS {code}, {year}, {flow.upper()})")
    start_time = time.time()
    plot_histogram(
        df_filtered["ln_uv"],
        code,
        year,
        flow,
        unit_label="USD/kg",
    )
    if df_q is not None and not df_q.empty:
        plot_histogram(
            df_q_filtered["ln_uv_q"],
            code,
            year,
            flow,
            unit_label=non_kg_unit,
        )
    logger_time("Completed histogram", start_time, logger)

    # %% Step 4: Modality test
    logger.info(f"Modality Test (HS {code}, {year}, {flow.upper()})")
    print("Running modality test on unit values...")
    start_time = time.time()
    report_modality, modality_decision = modality_test(
        df_filtered, logger=logger
    )
    if df_q is not None and not df_q.empty:
        report_q_modality, modality_q_decision = modality_test(
            df_q_filtered, col="ln_uv_q", logger=logger
        )
    else:
        logger.info(
            "⚠️ Skipping quantity-weighted modality test: df1_q is None or empty."
        )
    logger_time("Completed modality test", start_time, logger)

    # %% Step 5: Distribution fit
    # %%% Kg-based UV
    if modality_decision == "unimodal":
        # === Fitting Unimodal distribution of kg-based UV====
        logger.info(
            f"Fitting Unimodal distribution of kg-based UV ({year}, {flow.upper()})"
        )
        start_time = time.time()
        (
            best_fit_name,
            report_best_fit_uni,
            report_all_uni_fit,
            raw_params_dict,
        ) = fit_all_unimodal_models(df_filtered["ln_uv"], logger=logger)
        logger.info(
            f"- {best_fit_name.capitalize()} distribution fits best based on AIC and BIC."
        )
        logger_time(
            "Completed unimodal distribution fit (kg-based)",
            start_time,
            logger,
        )

        # === Bootstrapping CI (kg-based) ====
        start_time = time.time()
        logger.info(f"Bootstrapping CI (kg-based) ({year}, {flow.upper()})")
        report_ci_uni = bootstrap_parametric_ci(
            df_filtered["ln_uv"], dist=best_fit_name, n_bootstraps=1000
        )
        logger_time(
            "Completed ootstrapping CI (kg-based)", start_time, logger
        )

        # === Plotting unimodal distribution fit of kg-based UV ====
        logger.info(
            f"Plotting unimodal distribution fit of kg-based UV "
            f"({year}, {flow.upper()}))"
        )
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
            ax=None,
        )
        logger_time(
            "Completed unimodal distribution fit plot for kg-based UV",
            start_time,
            logger,
        )
    # === Fitting GMM of kg-based UV====
    else:
        logger.info("Fitting GMM on kg-based UV")
        start_time = time.time()
        optimal_k, bic_values, report_gmmf_1d = find_gmm_components(
                          df_filtered[["ln_uv"]], code, year, flow, "USD/kg", 
                          plot=True,save_path=True)
        
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
        logger_time(
            "Completed GMM fit on kg-based UV", start_time, logger
        )
    # %%% Non-kg-based UV
    if df_q is not None and not df_q.empty:
        if modality_q_decision == "unimodal":
            # === Fitting Unimodal distribution of non-kg-based UV====
            logger.info(
                f"Fitting Unimodal distribution of non-kg-based UV ({year}, {flow.upper()})"
            )
            start_time = time.time()
            (
                best_fit_name_q,
                report_q_best_fit_uni,
                report_q_all_uni_fit,
                raw_params_dict_q,
            ) = fit_all_unimodal_models(df_filtered["ln_uv_q"], logger=logger)
            logger.info(
                f"- {best_fit_name_q.capitalize()} distribution fits best based on AIC and BIC."
            )
            logger_time(
                "Completed unimodal distribution fit (non-kg-based)",
                start_time,
                logger,
            )

            # === Bootstrapping CI (non-kg-based) ====
            start_time = time.time()
            logger.info(
                f"Bootstrapping CI (non-kg-based) ({year}, {flow.upper()})"
            )
            report_q_ci_uni = bootstrap_parametric_ci(
                df_filtered["ln_uv_q"], dist=best_fit_name, n_bootstraps=1000
            )
            logger_time(
                "Completed ootstrapping CI (non-kg-based)", start_time, logger
            )

            # === Plotting unimodal distribution fit of kg-based UV ====
            logger.info(
                f"Plotting unimodal distribution fit of non-kg-based UV "
                f"({year}, {flow.upper()}))"
            )
            start_time = time.time()
            plot_dist(
                df_filtered["ln_uv_q"],
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
            )
            logger_time(
                "Completed unimodal distribution fit plot for non-kg-based UV",
                start_time,
                logger,
            )
        else:
            start_time = time.time()
            logger.info("Fitting GMM on non-kg-based UV")
            optimal_k_q, bic_values_q, report_q_gmmf_1d = find_gmm_components(
                df_q_filtered[["ln_uv_q"]], code, year, flow, non_kg_unit, plot=True,save_path=True,
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
            logger_time(
                "Completed GMM fit on non-kg-based UV",
                start_time,
                logger,
            )

