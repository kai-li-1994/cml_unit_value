import pandas as pd
import numpy as np
import comtradeapicall 
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import textwrap

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
    period = ','.join(f"{year}{month:02d}" for month in range(1, 13))
    data = comtradeapicall.getFinalData(subscription_key, typeCode='C', 
       freqCode='M', clCode='HS', period= period,
       reporterCode=None, cmdCode= code, flowCode= direction.upper(), 
       partnerCode=None, partner2Code=None, customsCode=None, motCode=None, 
       format_output='JSON', aggregateBy=None, breakdownMode='plus', 
       countOnly=None, includeDesc=True)   
    df = data
    text = df.cmdDesc.unique()[0]
    wrapped_text = textwrap.fill(text, width=50)
    print(f"{len(df)} records of monthly import trade retrieved\n"
          f"under the HS code {code}\n" 
          f"({wrapped_text})\n"
          f"in the year {year} via the UN Comtrade database.\n")
    
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
    lst_gp = df_cgdp_95.loc[df_cgdp_95['group'] == "Grouped", 
                            'IsoAlpha3'].str.strip().tolist()
    df1['partnerISO'] = df1['partnerISO'].str.strip()                         # Remove whitespace in 'partnerISO'              
    df1 = df1[(df1["partnerDesc"] != "World")&(~df1['partnerISO'].isin(lst_gp))] # Romoving the trading partner as a group of countries, e.g. 'World', 'Other Europe, nes', ect.  
    c2 = len(df1)                   

    def filter_sum_rows(group):
        if (group['partner2Desc'] == 'World').any() and (
                group['partner2Desc'] != 'World').any():
            group = group[group['partner2Desc'] != 'World'] 
        if (group['motDesc'] == 'TOTAL MOT').any() and (
                group['motDesc'] != 'TOTAL MOT').any():
            group = group[group['motDesc'] != 'TOTAL MOT']                    # Romove the sum-up rows for transport modes and coustoms procedures when the sub mode or precedure is reported in a given country, a partner country, and a year. 
        if (group['customsDesc'] == 'TOTAL CPC').any() and (
                group['customsDesc'] != 'TOTAL CPC').any():
            group = group[group['customsDesc'] != 'TOTAL CPC']
        return group
    df1 = df1.groupby(["reporterDesc", "partnerDesc", "period"],              
         group_keys=False).apply(filter_sum_rows)   
    c3 = len(df1)                                           
    
    df1.loc[(df1['netWgt'].isna()) | (df1['netWgt'] == 0), 'netWgt'
            ] = df1.loc[(df1['netWgt'].isna()) | (df1['netWgt'] == 0), 'qty'] # First, check where netWgt is 0 or NaN

    df1 = df1[~((df1['netWgt'].isna() | (df1['netWgt'] == 0)) & (
        df1['qty'].isna() | (df1['qty'] == 0)))]                              # Now drop rows where both netWgt and qty are 0 or NaN
    c4 = len(df1)
    
    df1['uv'] = df1['primaryValue'] / df1['netWgt']                           # Compute plain unit value.
    
    # Add GDP per capita of the trading partner country
    df1["refYear"] = df1["refYear"].astype("int")                             # locate the column 2013 instead of '2023'
    
    def find_gdp(row, df_cgdp_95):
        # Try to find the corresponding GDP value, or raise an error if not found
        matched_data = df_cgdp_95.loc[df_cgdp_95['text'
                       ] == row["partnerDesc"], row["refYear"]]               # Try to find the corresponding GDP value based on partnerDesc
        
        if matched_data.empty:
            matched_data = df_cgdp_95.loc[df_cgdp_95['IsoAlpha3'
                     ] == row["partnerISO"], row["refYear"]]                  # If no match found, try using partnerISO instead
        
        if matched_data.empty:                                                # If still no match found, raise an error
            raise ValueError(f"No matching GDP data found for partner: "
             f"{row['partnerDesc']} in year {row['refYear']} "
             f"with the index of {row.name}")
        return matched_data.iloc[0]
    
    df1["gdp"] = df1.apply(lambda row: find_gdp(row, df_cgdp_95), axis=1)
    # df1["gdp"] = df1.apply(lambda row: df_cgdp_95.loc[df_cgdp_95['text'
    #          ]==row["partnerDesc"], row["refYear"]].iloc[0], axis=1)
    
    df1["ln_gdp"] =np.log(df1["gdp"])                                         # Log-transform
    df1["ln_uv"] =np.log(df1["uv"])
    df1["ln_netWgt"] =np.log(df1["netWgt"])
    
    print(f"- Dropped {c1-c2}({(c1-c2)/c1*100:.3f}%) trade with aggregated"
    " trading partners, e.g.'World', 'Other Europe, nes', ect.\n"
    f"- Dropped {c2-c3}({(c2-c3)/c1*100:.3f}%) trade with aggregated"
     " transport modes or customs procedures\n" 
    f"- Dropped {c3-c4}({(c3-c4)/c1*100:.3f}%) trade with zero or empty net weight.\n"
     "- Adding log-transformed unit price, trade volume, and GDP per capital.\n"
          f"- Remaining {c4} trade records after cleaning.\n")

    return df1


def detect_outliers(df2, value_columns):
    """
    Detect outliers in a DataFrame using the modified Z-score method across multiple features.

    Args:
        df2 (pd.DataFrame): The input DataFrame containing the data.
        value_columns (list): The column names of the features to check for outliers.

    Returns:
        pd.DataFrame: The filtered DataFrame with outliers removed.
        pd.DataFrame: The DataFrame containing only the detected outliers.
        float: The percentage of detected outliers in the data.
    """
    # Check that the value columns exist in the DataFrame
    for column in value_columns:
        if column not in df2.columns:
            raise ValueError(f"'{column}' column is not in the DataFrame.")

    # Standardize the data (important for outlier detection in multivariate space)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df2[value_columns])

    # Calculate median and MAD for each feature
    medians = np.median(scaled_data, axis=0)
    mad = np.median(np.abs(scaled_data - medians), axis=0)

    # Compute modified Z-scores for each feature in the multivariate space
    modified_z_scores = 0.6745 * (scaled_data - medians) / mad

    # Compute the modified Z-scores in 3D space (use the maximum Z-score across features)
    max_z_scores = np.max(np.abs(modified_z_scores), axis=1)

    # Identify filtered data and outliers (Z-score > 3.5 is considered an outlier)
    filtered_df = df2[max_z_scores <= 3.5]
    outliers_df = df2[max_z_scores > 3.5]
    filtered_df = filtered_df.reset_index(drop = True)
    
    # Calculate the outlier percentage
    dp_rate = (len(outliers_df) / len(df2)) * 100
    
    print(f"- Dropped {len(outliers_df)} ({dp_rate:.3f}%) records as outliers.\n"
          f"- Remaining records after outlier detection: {len(filtered_df)}.\n")

    return filtered_df




