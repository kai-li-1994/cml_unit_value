import pandas as pd
import numpy as np
import comtradeapicall
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import textwrap
import glob
from uv_logger_setup import logger_setup, logger_time_info

logger = logger_setup("trade_logger")

def extract_trade(subscription_key, code, year, direction):
    """
    Extract monthly trade data from the comtrade API based on subscription key,
    commodity code, year, and trade direction.

    Args:
        subscription_key (str): API subscription key.
        code (str): 6-digit commodity code (HS code).
        year (int): The year for the trade data.
        direction (str): Trade direction ("m" or "x").

    Returns:
        pd.DataFrame: A DataFrame containing the trade data.
    """
    period = ",".join(f"{year}{month:02d}" for month in range(1, 13))
    data = comtradeapicall.getFinalData(
        subscription_key,
        typeCode="C",
        freqCode="M",
        clCode="HS",
        period=period,
        reporterCode=None,
        cmdCode=code,
        flowCode=direction.upper(),
        partnerCode=None,
        partner2Code=None,
        customsCode=None,
        motCode=None,
        format_output="JSON",
        aggregateBy=None,
        breakdownMode="plus",
        countOnly=None,
        includeDesc=True,
    )
    df = data
    text = df.cmdDesc.unique()[0]
    wrapped_text = textwrap.fill(text, width=50)
    print(
        f"{len(df)} records of monthly import trade retrieved\n"
        f"under the HS code {code}\n"
        f"({wrapped_text})\n"
        f"in the year {year} via the UN Comtrade database.\n"
    )

    return df


def extract_trade_tariff(subscription_key, code, year, direction):
    period = ",".join(f"{year}{month:02d}" for month in range(1, 13))
    data = comtradeapicall.getTarifflineData(
        subscription_key,
        typeCode="C",
        freqCode="M",
        clCode="HS",
        period=period,
        reporterCode=None,
        cmdCode=code,
        flowCode=direction.upper(),
        partnerCode=None,
        partner2Code=None,
        customsCode=None,
        motCode=None,
        format_output="JSON",
        countOnly=None,
        includeDesc=True,
    )
    df = data
    # text = df.cmdDesc.unique()[0]
    # wrapped_text = textwrap.fill(text, width=50)
    print(
        f"{len(df)} records of monthly import tarrif trade retrieved\n"
        f"under the HS code {code}\n"
        # f"({wrapped_text})\n"
        f"in the year {year} via the UN Comtrade database.\n"
    )

    return df


def clean_trade(df1):
    """
    Clean the trade data by filtering the sum-up rows, dropping NA values,
    and calculating the log-transformed unit price, trade volume, and GDP per
    capita.

    Args:
        df (pd.DataFrame): The raw trade data.

    Returns:
        pd.DataFrame: The cleaned trade data with added log-transformed
        columns.
    """
    c1 = len(df1)

    df_cgdp_95 = pd.read_pickle("df_cgdp_95.pkl")
    lst_gp = (
        df_cgdp_95.loc[df_cgdp_95["group"] == "Grouped", "IsoAlpha3"]
        .str.strip()
        .tolist()
    )
    df1["partnerISO"] = df1[
        "partnerISO"
    ].str.strip()  # Remove whitespace in 'partnerISO'
    df1 = df1[
        (df1["partnerDesc"] != "World") & (~df1["partnerISO"].isin(lst_gp))
    ]  # Romoving the trading partner as a group of countries, e.g. 'World', 'Other Europe, nes', ect.
    c2 = len(df1)

    def filter_sum_rows(group):
        if (group["partner2Desc"] == "World").any() and (
            group["partner2Desc"] != "World"
        ).any():
            group = group[group["partner2Desc"] != "World"]
        if (group["motDesc"] == "TOTAL MOT").any() and (
            group["motDesc"] != "TOTAL MOT"
        ).any():
            group = group[
                group["motDesc"] != "TOTAL MOT"
            ]  # Romove the sum-up rows for transport modes and coustoms procedures when the sub mode or precedure is reported in a given country, a partner country, and a year.
        if (group["customsDesc"] == "TOTAL CPC").any() and (
            group["customsDesc"] != "TOTAL CPC"
        ).any():
            group = group[group["customsDesc"] != "TOTAL CPC"]
        return group

    df1 = df1.groupby(
        ["reporterDesc", "partnerDesc", "period"], group_keys=False
    ).apply(filter_sum_rows)
    c3 = len(df1)

    df1 = df1[
        ~(
            (df1["netWgt"].isna() | (df1["netWgt"] == 0))
            & (df1["qty"].isna() | (df1["qty"] == 0))
        )
    ]  # Now drop rows where both netWgt and qty are 0 or NaN
    c4 = len(df1)

    df1["uv"] = (
        df1["primaryValue"] / df1["netWgt"]
    )  # Compute plain unit value.

    # Add GDP per capita of the trading partner country
    df1["refYear"] = df1["refYear"].astype(
        "int"
    )  # locate the column 2013 instead of '2023'

    def find_gdp(row, df_cgdp_95):
        # Try to find the corresponding GDP value, or raise an error if not found
        matched_data = df_cgdp_95.loc[
            df_cgdp_95["text"] == row["partnerDesc"], row["refYear"]
        ]  # Try to find the corresponding GDP value based on partnerDesc

        if matched_data.empty:
            matched_data = df_cgdp_95.loc[
                df_cgdp_95["IsoAlpha3"] == row["partnerISO"], row["refYear"]
            ]  # If no match found, try using partnerISO instead

        if matched_data.empty:  # If still no match found, raise an error
            raise ValueError(
                f"No matching GDP data found for partner: "
                f"{row['partnerDesc']} in year {row['refYear']} "
                f"with the index of {row.name}"
            )
        return matched_data.iloc[0]

    df1["gdp"] = df1.apply(lambda row: find_gdp(row, df_cgdp_95), axis=1)
    # df1["gdp"] = df1.apply(lambda row: df_cgdp_95.loc[df_cgdp_95['text'
    #          ]==row["partnerDesc"], row["refYear"]].iloc[0], axis=1)

    df1["ln_gdp"] = np.log(df1["gdp"])  # Log-transform
    df1["ln_uv"] = np.log(df1["uv"])
    df1["ln_netWgt"] = np.log(df1["netWgt"])

    print(
        f"- Dropped {c1-c2}({(c1-c2)/c1*100:.3f}%) trade with aggregated"
        " trading partners, e.g.'World', 'Other Europe, nes', ect.\n"
        f"- Dropped {c2-c3}({(c2-c3)/c1*100:.3f}%) trade with aggregated"
        " transport modes or customs procedures\n"
        f"- Dropped {c3-c4}({(c3-c4)/c1*100:.3f}%) trade with zero or empty net weight.\n"
        "- Adding log-transformed unit price, trade volume, and GDP per capital.\n"
        f"- Remaining {c4} trade records after cleaning.\n"
    )

    return df1

def load_config(
    country_file="./pkl/uv_mapping_country.pkl",
    unit_file="./pkl/uv_mapping_unit.pkl",
    unit_abbr_file="./pkl/uv_mapping_unitAbbr.pkl"):
    """Load ISO mappings, group list, quantity unit mappings, and thresholds from pickle files."""
    import pandas as pd
    import pickle

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

    # === Define essential columns for early-stage processing ===
    cols_to_keep_early = [
        "period", "reporterCode", "flowCategory", "partnerCode",
        "cmdCode", "qtyUnitCode", "qty", "netWgt", "cifValue", "fobValue"
    ]

    # === Define thresholds ===
    q_share_threshold = 0.10  # 10% for non-kg units
    min_records_q = 100       # 100 records for non-kg
    min_records_uv = 100      # 100 records for kg-based unit value

    return {
        "iso_map": iso_map,
        "lst_gp": lst_gp,
        "unit_map": unit_map,
        "unit_abbr_map": unit_abbr_map,
        "cols_to_keep_early": cols_to_keep_early,
        "q_share_threshold": q_share_threshold,
        "min_records_q": min_records_q,
        "min_records_uv": min_records_uv
    }

def clean_trade_tariff(path, code, year,flow,config):

    # === Locate file ===
    matches = glob.glob(f"{path}/*{code}*.csv")
    if not matches:
        logger.error(f"No matching file found for code {code} in path {path}")
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
    df_uv.drop(columns=["cifValue", "fobValue", "qty", "ln_qty"], errors="ignore", inplace=True)

    # === Subset: non-kg-based UV ===
    df_q_valid = df[df["qty"].fillna(0) > 0].copy()
    
    df_q = df.iloc[0:0] # Initialize an emplty placeholder
    share_pass = count_pass = False
    
    unit_counts = df_q_valid["qtyUnitCode"].value_counts()
    alt_units = unit_counts[~unit_counts.index.isin([-1, 8])] # Exclude kg and unknown

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
                "netWgt", "uv", "ln_uv", "ln_netWgt", "cifValue", "fobValue"
            ], errors="ignore", inplace=True)

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
    lst_report_cleaning = []
    report_base = {
        "hs_code": code,
        "year": year,
        "flow": flow,
        "uv_type": "USD/kg",
        "step_1_name": "data_cleaning",
        "step_1_initial_rows": p1,
        "step_1_valid_country_filter": p2,
        "step_1_valid_trade_value_filter": p4
    }
    report_kg = {
    **report_base,
    "step_1_valid_net_weight_filter": p5,
    "step_1_fail_reason_non_kg_uv": fail_reason_non_kg_uv
}

    lst_report_cleaning.append(report_kg)

    if share_pass and count_pass:
        lst_report_cleaning.append({
            **report_base,
            "uv_type": f"USD/{unit_abbr}",
            "step_1_non_kg_top_unit": non_kg_top_unit,
            "step_1_non_kg_top_unit_share": non_kg_top_unit_share,
            "step_1_valid_non_kg_filter": p6,
            "step_1_fail_reason_non_kg_uv": fail_reason_non_kg_uv
        })
    return df_uv, df_q, lst_report_cleaning
    

def detect_outliers(df, value_column, label="Data"):
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
            "step_2_initial_rows": len(df),
            "step_2_outliers_removed": 0,
            "step_2_outlier_rate": 0.0,
            "step_2_rows_after_outliers": len(df)
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
        "step_2_initial_rows": len(df),
        "step_2_outliers_removed": len(df_outliers),
        "step_2_outlier_rate": dp_rate,
        "step_2_rows_after_outliers": len(df_filtered)
    }

    return df_filtered, df_outliers, report_outlier
