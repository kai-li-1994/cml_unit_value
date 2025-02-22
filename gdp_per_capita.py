"""
Kai Li 11/02/2025
"""
"""
Rationale for Selecting 1995 as the Starting Year for Unit Value Analysis
1. All modern sovereign countries exist

By 1995, three major geopolitical shifts had concluded, ensuring that trade 
and GDP data were being reported under modern country boundaries:
    
- Dissolution of the USSR (1991) → Former Soviet republics (Russia, Ukraine, 
Kazakhstan, etc.) transitioned to independent economic reporting.

- Breakup of Yugoslavia (1991–1992) → Successor states (Serbia, Croatia, 
Slovenia, etc.) started independent trade and GDP reporting.

- German Reunification (1990) → East and West Germany’s economic data 
was fully integrated.

2.  Trade data is fully standardized in HS codes              
In 1995, UN Comtrade required all UN members to report trade data using HS     # https://unstats.un.org/unsd/trade/dataextract/dataclass.htm
codes starting from 1988. This ensures that all trade data from 1995 onward 
is standardized in HS codes, making commodity-level unit values more reliable.

"""
# %% 1. Package importing
import requests
import pandas as pd
import numpy as np
import pycountry
import comtradeapicall 
import copy
from currency_converter import CurrencyConverter
from datetime import datetime, timedelta
import pickle
# %% 2. API calling for GDP per capita from the World Bank

# Define the API endpoint and parameters
# url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD"       # GDP per capita in current US dollars (nominal GDP per capita)
# params = {
#     "date": "1995:2024",  # Specify the year range
#     "format": "json",
#     "per_page": "30000"  # Ensure enough entries are returned
# }
    
# response = requests.get(url, params=params)                                    # Send the request to the World Bank API
# dict_wb = response.json()

# with open("dict_wb.pkl", "wb") as f:
#     pickle.dump(dict_wb, f)

with open("dict_wb.pkl", "rb") as f:
    dict_wb = pickle.load(f)
    
# Extract the relevant data
records = []
unmapped_records = []

for entry in dict_wb[1]:
    iso2 = entry['country']['id']  # ISO-2 code
    year = entry['date']
    value = entry['value']

    # Check if ISO-2 is a valid country code using pycountry
    try:
        # Attempt to convert ISO-2 to ISO-3 using pycountry
        iso3 = pycountry.countries.get(alpha_2=iso2).alpha_3
        # Append the record if it's a valid country
        records.append({"Country": iso3, "Year": int(year), 
                        "GDP_per_Capita": value})

    except AttributeError:
        # If no valid country is found, use the ISO-2 code itself
        iso3 = entry['country']['value']
        unmapped_records.append({"Country": iso3, "Year": int(year), 
                        "GDP_per_Capita": value})

# Create a DataFrame
gdp_df = pd.DataFrame(records)
gdp_df2 = pd.DataFrame(unmapped_records)
# Pivot the DataFrame to have countries as rows and years as columns
df_pivot_wb = gdp_df.pivot(index='Country', columns='Year', 
                           values='GDP_per_Capita')
df_pivot_wb2 = gdp_df2.pivot(index='Country', columns='Year', 
                           values='GDP_per_Capita')

# %% 3. API calling for GDP per capita from the IMF

# year_list = ",".join(str(year) for year in range(1995, 2023 + 1))
# url_imf = (f"https://www.imf.org/external/datamapper/api/v1/NGDPDPC/?periods={year_list}")

# # Send the request to the IMF API
# response = requests.get(url_imf)
# dict_imf = response.json()

# with open("dict_imf.pkl", "wb") as f:
#     pickle.dump(dict_imf, f)

# # Define the IMF API URLs
# url_imf_cc1 = 'https://www.imf.org/external/datamapper/api/v1/countries'
# url_imf_cc2 = 'https://www.imf.org/external/datamapper/api/v1/regions'
# url_imf_cc3 = 'https://www.imf.org/external/datamapper/api/v1/groups'

# # Send requests to get the data from all three URLs
# response_cc1 = requests.get(url_imf_cc1)
# response_cc2 = requests.get(url_imf_cc2)
# response_cc3 = requests.get(url_imf_cc3)

# # Parse the JSON responses
# data_cc1 = response_cc1.json()
# data_cc2 = response_cc2.json()
# data_cc3 = response_cc3.json()

# # Create dictionaries for each response
# country_dict = {key: value['label'] for key, value in data_cc1['countries'].items()}
# region_dict = {key: value['label'] for key, value in data_cc2['regions'].items()}
# group_dict = {key: value['label'] for key, value in data_cc3['groups'].items()}

# # Combine all three dictionaries into one
# dict_imf_cc = {**country_dict, **region_dict, **group_dict}

# with open("dict_imf_cc.pkl", "wb") as f:
#      pickle.dump(dict_imf_cc, f)

with open("dict_imf.pkl", "rb") as f:
    dict_imf = pickle.load(f)

with open("dict_imf_cc.pkl", "rb") as f:
    dict_imf_cc = pickle.load(f)
# Extract the relevant data
records = []
unmapped_records =[]
valid_iso3_codes = {country.alpha_3 for country in pycountry.countries}
# Process each country in the response
for imf_code, country_data in dict_imf['values']['NGDPDPC'].items():

    if imf_code not in valid_iso3_codes:
        for year, value in country_data.items():
            if value is not None:  # Ignore missing data
                unmapped_records.append({"Country": imf_code, "Year": int(year), 
                                "GDP_per_Capita": value})
    # Process GDP data by year
    for year, value in country_data.items():
        if value is not None:  # Ignore missing data
            records.append({"Country": imf_code, "Year": int(year), 
                            "GDP_per_Capita": value})
            
gdp_imf_df = pd.DataFrame(records)
gdp_imf_df2 = pd.DataFrame(unmapped_records)
df_pivot_imf = gdp_imf_df.pivot(index='Country', columns='Year', 
                                values='GDP_per_Capita')
df_pivot_imf2 = gdp_imf_df2.pivot(index='Country', columns='Year', 
                                values='GDP_per_Capita')

df_pivot_imf2['Economy_Label'] = df_pivot_imf2.index.map(dict_imf_cc)
df_pivot_imf2.insert(0, 'Economy_Label', df_pivot_imf2.pop('Economy_Label'))
# %% 4. API calling for GDP per capita from the UNCTAD                            # https://unctadstat.unctad.org/datacentre/dataviewer/US.GDPTotal

# Load the UN GDP per capita dataset
df_unctad = pd.read_csv("US.GDPTotal_20250211_113033.csv",index_col=0)
df_m49_iso3 = pd.read_csv("m49_iso3.csv")                                      # 248 elements where TWN is missing

df_m49_iso3["M49 code"] = df_m49_iso3["M49 code"].astype(int)                  # Ensure M49 code in df_m49_iso3 is of integer type for matching

dict_m49_to_iso3 = dict(zip(df_m49_iso3["M49 code"], 
                            df_m49_iso3["ISO-alpha3 code"]))                   # Create a mapping dictionary from M49 to ISO3

df_unctad.index = df_unctad.index.map(lambda x: dict_m49_to_iso3.get(x,x))

df_unctad.columns = [df_unctad.columns[0]] + list(range(1995, 2024))

df_pivot_unctad2 = df_unctad[~df_unctad.index.isin(dict_m49_to_iso3.values())]

df_pivot_unctad = df_unctad[df_unctad.index.isin(dict_m49_to_iso3.values())]

df_pivot_unctad = df_pivot_unctad[list(range(1995, 2024))]

# %% 5. Check data missing in WB and fill with IMF and UNdata

no_1 = df_pivot_wb.isna().sum().sum()

df_filled_imf = df_pivot_wb.combine_first(df_pivot_imf)                        # Fill missing values from IMF and log changes
no_2 = df_filled_imf.isna().sum().sum()

df_pivot = df_filled_imf.combine_first(df_pivot_unctad)                            # Fill remaining missing values from UN and log changes
no_3 = df_pivot.isna().sum().sum()

log_message = f"""
GDP per Capita Data Merging Summary:
-------------------------------------
- Initial missing values in World Bank: {no_1}
- Filled from IMF: {no_1 - no_2}
- Filled from UNdata: {no_2 - no_3}
- Remaining missing values after merging: {no_3}
"""
# Print and save log
print(log_message)
# %% 6. Filling the missing data for Gibratar and other countries
"""
References of GDP (Factor Cost per Capita) (£) for Gibratar:
https://www.gibraltar.gov.gi/statistics/key-indicators
https://www.gibraltar.gov.gi/uploads/statistics/2023/
National%20Income/GDP%20Estimates%202007-08%20-%202022-23.pdf
https://www.gibraltar.gov.gi/statistics/downloads
"""
df_GIB= pd.DataFrame({                                                        
    "Year": [
       1995, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 
       2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 
       2017, 2018, 2019, 2020, 2021, 2022, 2023
   ],
    "GDP (Factor Cost per Capita) (£)": [
        11680, 14526, 15091, 15863, 16608, 17770, 19552, 20629, 22328, 24859,
        26714, 29357, 32570, 34247, 37369, 40381, 45032, 48522, 53433, 59768,
        61000, 63000, 65000, 67000, 69000, 80517, 85614
        ]
})

df_GIB = df_GIB.set_index("Year").reindex(range(1995, 2024))                   # Add missing years (1996, 1997) with NaN values

# Apply linear interpolation to fill missing values
df_GIB["GDP (Factor Cost per Capita) (£)"] = df_GIB[
    "GDP (Factor Cost per Capita) (£)"].interpolate().round(0).astype(int)     # Apply linear interpolation & round to integers

df_GIB = df_GIB.reset_index()                                                  # Reset index to restore "Year" column

df_gbp_usd = pd.read_csv(
    "Exchange_Rate_IMF.tsv", 
    sep="\t", 
    encoding="ISO-8859-1", 
    skiprows=2,   # Skip metadata at the top
    skipfooter=13, # Adjust based on how many extra rows exist at the bottom
    engine="python"  # Required for `skipfooter`
)
df_gbp_usd.reset_index(inplace=True)
df_gbp_usd = df_gbp_usd[['Date', 'U.K. pound(GBP)']]

df_gbp_usd["Date"] = pd.to_datetime(df_gbp_usd["Date"], format="%d-%b-%Y")       # Convert the "Date" column to datetime format

# Extract the year from the Date column
df_gbp_usd["Year"] = df_gbp_usd["Date"].dt.year

df_gbp_usd["U.K. pound(GBP)"] = df_gbp_usd["U.K. pound(GBP)"
                                          ].astype(str).str.strip()            # extra spaces found would interupt pd.to_numeric()
df_gbp_usd["U.K. pound(GBP)"] = pd.to_numeric(df_gbp_usd["U.K. pound(GBP)"])

df_ex_avg = df_gbp_usd.groupby("Year")["U.K. pound(GBP)"].mean().reset_index()  # Compute annual average exchange rate

df_GIB = df_GIB.merge(df_ex_avg, on = 'Year')

df_GIB['GDP (Factor Cost per Capita) (USD)'] = df_GIB["U.K. pound(GBP)"
                            ]*df_GIB["GDP (Factor Cost per Capita) (£)"]

df_pivot.loc["GIB"] = df_GIB["GDP (Factor Cost per Capita) (USD)"].values      # Assign the 1995-2023 array to the missing place


df_pivot.loc['MAF', list(range(2011, 2022))] = df_pivot.loc[
                      'MAF', list(range(2011, 2022))].interpolate()           # Interpolate data of 'MAF' between 2011 and 2022

countries_ffill = ['ASM', 'GUM', 'IMN', 'MNP', 'PRK', 'VIR','MAF']            # Fill the missing data with the previous year's data

for country in countries_ffill:
    df_pivot.loc[country] = df_pivot.loc[country].ffill()
    
df_pivot.to_pickle('df_pivot.pkl')
print("✅ GDP per capita data saved as df_pivot.pkl")

# %% API calling for GDP per capita from World Bank

df_p = comtradeapicall.getReference('partner')
df_r = comtradeapicall.getReference('reporter')

df_p = df_p.rename(columns={
    'PartnerCode': 'Code',
    'PartnerDesc': 'Desc',
    'partnerNote': 'Note',
    'PartnerCodeIsoAlpha2': 'IsoAlpha2',
    'PartnerCodeIsoAlpha3': 'IsoAlpha3'
})

df_r = df_r.rename(columns={
    'reporterCode': 'Code',
    'reporterDesc': 'Desc',
    'reporterNote': 'Note',
    'reporterCodeIsoAlpha2': 'IsoAlpha2',
    'reporterCodeIsoAlpha3': 'IsoAlpha3'
})

# Concatenate the dataframes vertically
df_rp= pd.concat([df_p, df_r], ignore_index=True)

# Drop duplicates
df_rp = df_rp.drop_duplicates(subset=['id', 'text', 'Code', 'Desc', 'Note', 
'IsoAlpha2', 'IsoAlpha3','entryEffectiveDate','entryExpiredDate'])

no1 = len(df_rp)

df_gdp = pd.read_pickle("df_pivot.pkl")

df_cgdp = df_rp.merge(df_gdp, left_on='IsoAlpha3', right_index=True, how='left')

# Strip leading and trailing spaces from the 'text', 'Desc', and 'Note' columns
df_cgdp['text'] = df_cgdp['text'].str.strip()
df_cgdp['Desc'] = df_cgdp['Desc'].str.strip()
df_cgdp['Note'] = df_cgdp['Note'].str.strip()

# Create a pattern to extract the year from the 'text' column
df_cgdp['entryExpiredDate'] = pd.to_datetime(df_cgdp['entryExpiredDate'], errors='coerce')

df_cgdp_95 = df_cgdp[(df_cgdp['entryExpiredDate'] >= '1995-01-01') | pd.isna(df_cgdp['entryExpiredDate'])]

no2 = len(df_cgdp_95)

# Manually filling missing data on Åland Islands of Finland
df_cgdp_95.loc[df_cgdp_95['text']=='Channel Islands', list(range(1995, 2024))
               ] = df_pivot_wb2.loc['Channel Islands',:].values
df_cgdp_95.loc[df_cgdp_95['text']=='Kosovo', list(range(1995, 2024))
               ] = df_pivot_imf2.loc['UVK',list(range(1995, 2024))].values
df_cgdp_95.loc[df_cgdp_95['text']=='Netherlands Antilles (...2010)', 
    list(range(1995, 2024))] = df_pivot_unctad2.loc[530,list(range(1995, 2024))
                                                 ].values
df_cgdp_95.loc[df_cgdp_95['text']=='Serbia and Montenegro (...2005)', 
          list(range(1995, 2024))] = df_pivot_unctad2.loc[891,list(range(1995, 2024))].values

df_cgdp_95.loc[df_cgdp_95['text']=='Other Asia, nes', 
          list(range(1995, 2024))] = df_cgdp_95.loc[
            df_cgdp_95['text']=='Taiwan, Province of China', 
                       list(range(1995, 2024))].values   
                                            
# Reconstruct the sovereign country mapping dictionary
sovereign_country_mapping = {
    # Sovereign Countries (ISO3 codes), sorted alphabetically
    "Bonaire": "NLD",                                                          # Bonaire is a special municipality of the Netherlands located in the Caribbean.
    "Bouvet Island": "NOR",                                                    # Bouvet Island is a dependency of Norway located in the South Atlantic Ocean.
    "Br. Antarctic Terr.": "GBR",                                              # British Antarctic Territory is an overseas territory of the UK.
    "Br. Indian Ocean Terr.": "GBR",                                           # The British Indian Ocean Territory is an overseas territory of the UK.
    "Christmas Isds": "AUS",                                                   # Christmas Island is an external territory of Australia in the Indian Ocean.
    "Cocos Isds": "AUS",                                                       # The Cocos (Keeling) Islands are an external territory of Australia.
    "Falkland Isds (Malvinas)": "GBR",                                         # The Falkland Islands are a British Overseas Territory in the South Atlantic.
    "Fr. South Antarctic Terr.": "FRA",                                        # The French Southern and Antarctic Lands are an overseas territory of France.
    "French Guiana (Overseas France)": "FRA",                                  # French Guiana is an overseas department of France in South America.
    "Guadeloupe (Overseas France)": "FRA",                                     # Guadeloupe is an overseas department of France in the Caribbean.
    "Guernsey": "GBR",                                                         # Guernsey is a Crown dependency of the UK in the Channel Islands.
    "Heard Island and McDonald Islands": "AUS",                                # These islands are an external territory of Australia in the Indian Ocean.
    "Holy See (Vatican City State)": "ITA", 
    "Jersey": "GBR",                                                           # Jersey is a Crown dependency of the UK in the Channel Islands.
    "Martinique (Overseas France)": "FRA",                                     # Martinique is an overseas department of France in the Caribbean.
    "Mayotte (Overseas France)": "FRA",                                        # Mayotte is an overseas department of France in the Indian Ocean.
    "Midway Islands": "USA",                                                   # Midway Atoll is an unincorporated territory of the United States.
    "Niue": "NZL",                                                             # Niue is a self-governing territory in free association with New Zealand.
    "Norfolk Isds": "AUS",                                                     # Norfolk Island is an external territory of Australia.
    "Pitcairn": "GBR",                                                         # The Pitcairn Islands are a British Overseas Territory in the Pacific.
    "Ryukyu Isd": "JPN",                                                       # The Ryukyu Islands are a chain of Japanese islands including Okinawa.
    "Réunion (Overseas France)": "FRA",                                        # Réunion is an overseas department of France in the Indian Ocean.
    "Saint Barthélemy": "FRA",                                                 # Saint Barthélemy is a French overseas collectivity in the Caribbean.
    "Saint Helena": "GBR",                                                     # Saint Helena is a British Overseas Territory in the South Atlantic Ocean.
    "Saint Maarten":"NLD",
    "Saint Martin (French part)": "FRA",    
    "Saint Pierre and Miquelon": "FRA",                                        # A self-governing French overseas collectivity near Canada.
    "Sarawak": "MYS",                                                          # Sarawak joined Malaysia in 1963 as part of East Malaysia.
    "South Georgia and the South Sandwich Islands": "GBR",                     # British Overseas Territory in the South Atlantic Ocean.
    "Svalbard and Jan Mayen Islands": "NOR",                                   # Svalbard is under Norwegian sovereignty with special status.
    "Tokelau": "NZL",                                                          # Tokelau is a dependent territory of New Zealand.
    "US Misc. Pacific Isds": "USA",                                            # Various unincorporated territories of the US in the Pacific.
    "United States Minor Outlying Islands": "USA",                             # Various US unincorporated Pacific islands.
    "Wake Island": "USA",                                                      # A US territory in the Pacific Ocean.
    "Wallis and Futuna Isds": "FRA",                                           # A French overseas territory in the South Pacific.
    "Western Sahara": "MAR",                                                   # Western Sahara is a disputed territory mostly controlled by Morocco.
    "Åland Islands": "FIN",                                                    # The Åland Islands are an autonomous region of Finland.

    # Grouped Entities, sorted alphabetically
    "Africa CAMEU region, nes": "Grouped",
    "Antarctica": "Grouped",
    "Areas, nes": "Grouped",
    "ASEAN": "Grouped",
    "Bunkers": "Grouped",
    "CACM, nes": "Grouped",
    "Caribbean, nes": "Grouped",
    "Eastern Europe, nes": "Grouped",
    "Europe EFTA, nes": "Grouped",
    "European Union": "Grouped",
    "Free Zones": "Grouped",
    "LAIA, nes": "Grouped",
    "Neutral Zone": "Grouped",
    "North America and Central America, nes": "Grouped",
    "Northern Africa, nes": "Grouped",
    "Oceania, nes": "Grouped",
    "Other Africa, nes": "Grouped",
    "Other Europe, nes": "Grouped",
    "Pacific Isds (...1991)": "Grouped",
    "Panama-Canal-Zone (...1977)": "Grouped",
    "Rest of America, nes": "Grouped",
    "Southern African Customs Union (...1999)": "Grouped",
    "Special Categories": "Grouped",
    "Western Asia, nes": "Grouped",
    "World": "Grouped",
    }

df_cgdp_95.insert(0,'group',df_cgdp_95['text'].map(sovereign_country_mapping))


# Loop through the rows where 'group' is neither 'Grouped' nor NaN
for idx, row in df_cgdp_95[df_cgdp_95['group'].notna() & (df_cgdp_95['group'] != 'Grouped')].iterrows():
    iso3_code = row['group']  # Get the ISO3 code
    df_cgdp_95.loc[idx, list(range(1995, 2024))] = df_cgdp_95.loc[df_cgdp_95['IsoAlpha3'] == iso3_code, list(range(1995, 2024))].iloc[0].values

df_cgdp_95.to_pickle('df_cgdp_95.pkl')
print("✅ Data for country name and GDP per capita saved as df_cgdp_95.pkl")