# -*- coding: utf-8 -*-
import pandas as pd
import os
import pickle

def load_config(
    country_file="./pkl/uv_mapping_country.pkl",
    unit_file="./pkl/uv_mapping_unit.pkl",
    unit_abbr_file="./pkl/uv_mapping_unitAbbr.pkl",
    input_dir=None,
    base_dir="."
    ):
    """Load ISO mappings, group list, quantity unit mappings, and thresholds from pickle files."""

    # === ISO country mapping ===
    df_cmap = pd.read_pickle(country_file)
    iso_map = df_cmap.set_index("Code")["IsoAlpha3"].str.strip().to_dict()

    # === Group codes to filter out ===
    lst_gp = [
        "_AC", "ATA", "_X", "X1", "R91", "A49", "E29", "R20", "X2", "A79",
        "NTZ", "A59", "F49", "O19", "F19", "E19", "ZA1", "XX", "F97", "W00",
        "R4", "EUR"
    ]

    # === Quantity unit mappings ===
    with open(unit_file, "rb") as f:
        unit_map = pickle.load(f)

    with open(unit_abbr_file, "rb") as f:
        unit_abbr_map = pickle.load(f)
    
    # === Define directory structure ===
    figures_dir = os.path.join(base_dir, "figures")
    logs_dir = os.path.join(base_dir, "logs")
    results_dir = os.path.join(base_dir, "results")
    reports_dir = os.path.join(base_dir, "reports")
    input_data_dir = input_dir if input_dir else os.path.join(base_dir, "input")  
    
    for d in [figures_dir, logs_dir, results_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
    # === Define essential columns for early-stage processing ===
    cols_to_keep_early = [
        "period", "reporterCode", "flowCategory", "partnerCode",
        "cmdCode", "qtyUnitCode", "qty", "netWgt", "cifValue", "fobValue"
    ]
    
    # === Rscript logic same as before ===
    rscript_exec = os.environ.get("RSCRIPT_EXEC", None)
    if not rscript_exec:
        local_paths = [
            "C:/Users/lik6/AppData/Local/Programs/R/R-4.5.0/bin/x64/Rscript.exe", # laptop path
            "C:/Program Files/R/R-4.4.1/bin/x64/Rscript.exe" # desktop path
        ]
        for path in local_paths:
            if os.path.exists(path):
                rscript_exec = path
                break
    if not rscript_exec or not os.path.exists(rscript_exec):
        raise RuntimeError("⚠️ Could not find a valid Rscript executable. Please set RSCRIPT_EXEC or verify local paths.")

    return {
        "iso_map": iso_map,
        "lst_gp": lst_gp,
        "unit_map": unit_map,
        "unit_abbr_map": unit_abbr_map,
        "cols_to_keep_early": cols_to_keep_early,
        "q_share_threshold": 0.10,
        "min_records_q": 100,
        "min_records_uv": 100,
        "dirs": {
            "figures": figures_dir,
            "logs": logs_dir,
            "results": results_dir,
            "reports": reports_dir,
            "input":input_data_dir
        },
        "rscript_exec": rscript_exec,
    }