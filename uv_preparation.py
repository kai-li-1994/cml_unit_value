import pandas as pd
import numpy as np
import glob
import os


def clean_trade(code, year, flow, config, logger):

    # === Locate file ===
    year_folder = f"split_by_hs_{year}_numpy"
    input_subdir = os.path.join(config["dirs"]["input"], year_folder)
    matches = glob.glob(os.path.join(input_subdir, f"*{code}*.csv"))

    if not matches:
        logger.error(f"No matching file found for code {code} in path {input_subdir}")
        raise FileNotFoundError
    logger.info(f"📄 Found file: {matches[0]}")
    df = pd.read_csv(matches[0])
        
    # === Early column trimming ===
    df = df[[col for col in df.columns if col in config["cols_to_keep_early"]]]

    # === Flow filter ===
    flow = flow.lower()
    if flow not in {"m", "x"}:
        logger.error(f"Invalid flow: {flow}")
        raise ValueError("Flow must be 'm' or 'x'")
    df = df[df["flowCategory"].str.lower() == flow]
    p1 = len(df)

    # === ISO mapping ===
    iso_map = config["iso_map"]
    df = df.rename(columns={"reporterCode": "reporterCodeRaw", "partnerCode": "partnerCodeRaw"})
    df["reporterISO"] = df["reporterCodeRaw"].map(iso_map)
    df["partnerISO"] = df["partnerCodeRaw"].map(iso_map)

    # === Country filter ===
    lst_gp = config["lst_gp"]
    df = df[(~df["partnerISO"].isin(lst_gp)) & (~df["reporterISO"].isin(lst_gp))]
    p2 = len(df)

    # === Trade value filter ===
    df = df[((df["cifValue"].fillna(0) > 0) | (df["fobValue"].fillna(0) > 0))]
    p4 = len(df)
    
    # === Logging progress ===
    logger.info("🧹 Cleaning Summary:")
    logger.info(f" - Initial rows: {p1}")
    logger.info(f" - Valid country filter: {p2} ({p2/p1:.2%})")
    logger.info(f" - Valid trade value filter: {p4} ({p4/p1:.2%})")

    # === Subset: kg-based UV ===
    df_uv = df[df["netWgt"].fillna(0) > 0].copy()
    p5 = len(df_uv)
    logger.info(f" - Valid kg-based UV rows: {p5} ({p5/p1:.2%})")
    if p5 < config.get("min_records_uv", 100):
        logger.warning(f"⚠️ Only {p5} kg-based UV records (<{config['min_records_uv']})")
    is_valid_kg = p5 >= config.get("min_records_uv", 100)
    
    if flow == "m":
        df_uv["uv"] = np.where(df_uv["cifValue"].fillna(0) > 0,
                               df_uv["cifValue"] / df_uv["netWgt"],
                               df_uv["fobValue"] / df_uv["netWgt"])
    else:
        df_uv["uv"] = np.where(df_uv["fobValue"].fillna(0) > 0,
                               df_uv["fobValue"] / df_uv["netWgt"],
                               df_uv["cifValue"] / df_uv["netWgt"])

    df_uv["ln_uv"] = np.log(df_uv["uv"])
    df_uv["ln_netWgt"] = np.log(df_uv["netWgt"])
    df_uv.drop(columns=["qty", "ln_qty"], errors="ignore", inplace=True)

    # === Subset: non-kg-based UV ===
    df_q_valid = df[df["qty"].fillna(0) > 0].copy()
    
    df_q = df.iloc[0:0] # Initialize an emplty placeholder
    share_pass = count_pass = False
    return_unit = "USD/kg"  # Default return unit
    unit_counts = df_q_valid["qtyUnitCode"].value_counts()
    alt_units = unit_counts[~unit_counts.index.isin([-1, 8])] # Exclude kg and unknown
    
    is_valid_q = False
    
    if not alt_units.empty: # non-kg unit exits
        top_unit = alt_units.idxmax() # keep only the top non-kg unit 
        top_count = alt_units[top_unit]
        top_share = top_count / len(df_q_valid)
        
        unit_desc = config['unit_map'].get(top_unit, f"Code {top_unit}")
        unit_abbr = config['unit_abbr_map'].get(top_unit, "N/A")
        
        # Evaluate thresholds
        share_pass = top_share >= config['q_share_threshold']
        count_pass = top_count >= config['min_records_q']

        logger.info("📊 Non-kg UV Subset:")
        logger.info(f" - Top unit: {unit_desc} ({unit_abbr})")
        logger.info(f" - Share of valid qty rows: {top_share:.2%} (Required: {config['q_share_threshold']:.0%})")
        logger.info(f" - Valid qty count: {top_count} (Required: {config['min_records_q']})")

        non_kg_top_unit = f"{unit_desc} ({unit_abbr})"
        non_kg_top_unit_share = round(top_share, 4)

        if share_pass and count_pass: # Non-kg unit meets both share and size requirements
            # build df_q
            df_q = df_q_valid[df_q_valid["qtyUnitCode"] == top_unit].copy()
            p6 = len(df_q)
            is_valid_q = True
            if flow == 'm':
                df_q['uv_q'] = np.where(df_q['cifValue'].fillna(0) > 0,
                                        df_q['cifValue'] / df_q['qty'],
                                        df_q['fobValue'] / df_q['qty'])
             
            else:
                df_q['uv_q'] = np.where(df_q['fobValue'].fillna(0) > 0,
                                        df_q['fobValue'] / df_q['qty'],
                                        df_q['cifValue'] / df_q['qty'])

            df_q["ln_uv_q"] = np.log(df_q["uv_q"])
            df_q["ln_qty"] = np.log(df_q["qty"])
            df_q.drop(columns=[
                "netWgt", "uv", "ln_uv", "ln_netWgt" ], errors="ignore", inplace=True)
            return_unit = f"USD/{unit_abbr}"  # ✅ Set return unit

            fail_reason_non_kg_uv = None
            logger.info("✅ Non-kg UV subset created.")

        else: # Non-kg unit fails to meet either share or size requirement
            fail_reasons = []
            if not share_pass:
                fail_reasons.append(f"share {top_share:.2%} < {config['q_share_threshold']:.0%}")
            if not count_pass:
                fail_reasons.append(f"count {top_count} < {config['min_records_q']}")

            fail_reason_non_kg_uv = " and ".join(fail_reasons)
            logger.warning(f"❌ No non-kg UV subset created: {' and '.join(fail_reasons)}")
    else: # No non-kg unit found in the column "qtyUnitCode"
        fail_reason_non_kg_uv = "No non-kg alternative units found"
        logger.warning("⚠️ No non-kg units found in qty rows.")

    logger.info(f"✅ Finished cleaning: HS {code}, Year {year}, Flow {flow.upper()}")
    # === Restructure final report ===
    
    report_base = {
        "hs_code": code,
        "year": year,
        "flow": flow,
        "c_initial_rows": p1,
        "c_valid_country_rows": p2,
        "c_valid_value_rows": p4
    }
    report_clean = {
    **report_base,
    "uv_type": "USD/kg",
    "c_valid_weight_rows": p5,
    "c_fail_reason_non_kg_uv": fail_reason_non_kg_uv
}
    report_q_clean = None     # default

    if share_pass and count_pass:
        report_q_clean = {
            **report_base,
            "uv_type_2": f"USD/{unit_abbr}",
            "c_top_unit": non_kg_top_unit,
            "c_top_unit_share": non_kg_top_unit_share,
            "c_valid_top_unit_rows": p6
        }
    return df_uv, df_q, report_clean, report_q_clean, return_unit, is_valid_kg, is_valid_q
    

def detect_outliers(df, value_column,  code, year, flow, logger,label="Data"):
    """
    Detect outliers in a DataFrame using the modified Z-score method.

    Args:
        df (pd.DataFrame): The input DataFrame to check.
        value_column: The column to use for outlier detection.
        label (str): Label for printing/logging purposes.

    Returns:
        df_filtered (pd.DataFrame): DataFrame after removing outliers.
        df_outliers (pd.DataFrame): DataFrame containing detected outliers.
        report_outlier (dict): Summary report of outlier detection.
    """
    
    # === Calculate modified Z-scores
    raw_data = df[value_column].values
    median = np.median(raw_data)
    mad = np.median(np.abs(raw_data - median))
    
    if mad == 0:
        logger.warning(f"⚠️ Outlier Detection: MAD=0 for {value_column} — skipping detection.")
        return df.copy().reset_index(drop=True), df.iloc[0:0].copy(), {
            "d_initial_rows": len(df),
            "d_outliers_removed": 0,
            "d_outlier_rate": 0.0,
            "d_rows_after_outliers": len(df)
        }
    
    modified_z_scores = 0.6745 * (raw_data - median) / mad
    
    # === Identify filtered data and outliers
    df_filtered = df[np.abs(modified_z_scores) <= 3.5]
    df_outliers = df[np.abs(modified_z_scores) > 3.5]
    dp_rate = (len(df_outliers) / len(df)) * 100

    # === Print Summary
    logger.info(f"📊 Outlier Detection on {label}:")
    logger.info(f"- Dropped {len(df_outliers)} rows ({dp_rate:.3f}%).")
    logger.info(f"- Remaining rows: {len(df_filtered)}.")
    
    # === Prepare outlier report
    report_outlier = {
        "d_initial_rows": len(df),
        "d_outliers_removed": len(df_outliers),
        "d_outlier_rate": dp_rate,
        "d_rows_after_outliers": len(df_filtered)
    }

    return df_filtered, df_outliers, report_outlier
