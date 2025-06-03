# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:34:04 2025

@author: lik6
"""

import re
from datetime import datetime
from pathlib import Path
import pandas as pd

def raw_data_summary_download(year, base_dir):
    """Fast-access metadata: merged file count, reporting countries, update date, etc."""
    metadata = {
        "year": year,
        "n_files_merged": 0,
        "latest_update": "Unknown",
        "log_created_at": "Unknown",
        "n_reporting_countries": 0,
        "total_rows": None,
        "mismatched_hs_rows": None,
        "hs_matched_rows": None,
        "hs_match_ratio": None,
    }

    filename_pattern = re.compile(r"CM(\d{3})\d{4}(\d{2})H\d?\[(\d{4}-\d{2}-\d{2})\]")
    log_created_pattern = re.compile(r"Log Created At:\s*([\d\-]+\s[\d:]+)")
    total_row_pattern = re.compile(r"Total rows processed\s*:\s*([\d,]+)")
    unknown_hs_pattern = re.compile(r"Total unknown HS rows\s*:\s*([\d,]+)")

    # Initialize monthly map
    month_country_map = {f"{i:02}": set() for i in range(1, 13)}
    update_dates = []
    country_codes = set()

    # log_merge
    log_file = base_dir / f"log_merge_{year}.txt"
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            match = filename_pattern.search(line)
            if match:
                cc, mm, update_date = match.groups()
                country_codes.add(cc)
                month_country_map[mm].add(cc)
                update_dates.append(datetime.strptime(update_date, "%Y-%m-%d"))

        log_created = log_created_pattern.search("".join(lines))
        if log_created:
            metadata["log_created_at"] = log_created.group(1)
        if update_dates:
            metadata["latest_update"] = max(update_dates).strftime("%Y-%m-%d")
            metadata["n_files_merged"] = len(update_dates)
            metadata["n_reporting_countries"] = len(country_codes)
        for mm in sorted(month_country_map):
            metadata[f"n_countries_m{mm}"] = len(month_country_map[mm])

    # run_summary
    run_summary_file = base_dir / f"split_by_hs_{year}_numpy" / f"run_summary_{year}.txt"
    if run_summary_file.exists():
        with open(run_summary_file, encoding="utf-8") as f:
            content = f.read()
        total_match = total_row_pattern.search(content)
        unknown_match = unknown_hs_pattern.search(content)
        if total_match:
            metadata["total_rows"] = int(total_match.group(1).replace(",", ""))
        if unknown_match:
            metadata["mismatched_hs_rows"] = int(unknown_match.group(1).replace(",", ""))

        if metadata["total_rows"] is not None and metadata["mismatched_hs_rows"] is not None:
            matched = metadata["total_rows"] - metadata["mismatched_hs_rows"]
            metadata["hs_matched_rows"] = matched
            metadata["hs_match_ratio"] = round(matched / metadata["total_rows"], 4) if metadata["total_rows"] > 0 else None

    return metadata
#%%
def raw_data_summary_size(year, folder):
    """
    Extracts sample sizes for each HS-year-flow combination from a folder.
    
    Parameters:
        year (int): Year to process
        folder (Path): Path to split_by_hs_<year>_numpy folder
    
    Returns:
        pd.DataFrame: columns = ['hs_code', 'year', 'flow', 'sample_size']
    """
    records = []
    csv_files = list(folder.glob(f"*_[0-9][0-9][0-9][0-9][0-9][0-9]_{year}.csv"))

    for file in csv_files:
        try:
            hs_code = file.stem.split("_")[1]
            df = pd.read_csv(file, usecols=["flowCategory"])

            for flow in ["M", "X"]:
                count = (df["flowCategory"] == flow).sum()
                if count > 0:
                    records.append({
                        "hs_code": hs_code,
                        "year": year,
                        "flow": flow,
                        "sample_size": count
                    })
        except Exception:
            continue

    return pd.DataFrame(records)

#%%
base_dir = Path(r"C:\Users\lik6\OneDrive - Universiteit Leiden\PlasticTradeFlow\tradeflow\cml_trade\data_uncomtrade")
metadata_list = []

for year in range(2010, 2024):
    year_metadata = raw_data_summary_download(year, base_dir)
    metadata_list.append(year_metadata)

df = pd.DataFrame(metadata_list)
#%%
base_dir = Path(r"C:\Users\lik6\OneDrive - Universiteit Leiden\PlasticTradeFlow\tradeflow\cml_trade\data_uncomtrade")
summary_all_years = []

for year in range(2010, 2024):
    #year = 2010
    folder = base_dir / f"split_by_hs_{year}_numpy"
    df_year = raw_data_summary_size(year, folder)
    summary_all_years.append(df_year)

df_all = pd.concat(summary_all_years, ignore_index=True)