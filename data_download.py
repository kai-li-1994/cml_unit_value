"""
Author: Kai Li
Created on Fri Apr  4 10:01:19 2025
Contact: k.li@cml.leidenuniv.nl (xiaoshancqu@gmail.com)
"""
#%% Import packages
from datetime import date 
from datetime import timedelta
import datetime
import comtradeapicall
import pandas as pd
import glob
import re
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
from collections import defaultdict
import requests
import json
import psutil
import cpuinfo
import wmi
#%% Data request comtradeapicall

year = '2023'
period = ",".join(f"{year}{month:02d}" for month in range(1, 13))

path = f"C:/Users/lik6/Data/ComtradeTariffline/tariff_{year}"
#path = "./month/test"
os.makedirs(path, exist_ok=True)
comtradeapicall.bulkDownloadTarifflineFile("707b66b3161940889e89edcea764320b", 
                                          path,                                # <-- ✅ Save directly to C disk
                                     typeCode='C', freqCode='M', clCode='HS',
                            period='202312', reporterCode=None, decompress=True)
#%% Merge annual data
start_time = time.time()
input_folder  = f'C:/Users/lik6/Data/ComtradeTariffline/tariff_{year}'                                                        # Set your folder path
output_folder = 'C:/Users/lik6/Data/ComtradeTariffline/merge/' 
os.makedirs(output_folder, exist_ok=True)

file_list = glob.glob(os.path.join(input_folder, '*.txt'))                     # Find all text files
file_list = sorted(file_list)                                                  # Sort files to ensure correct month order

output_path = os.path.join(output_folder, f'all_{year}_merged.txt')        # Build the path and filename where the merged file will be saved
log_path    = os.path.join(output_folder, f'log_merge_{year}.txt')
print(f"🔵 Merging {len(file_list)} files for year {year}...")

with open(output_path, 'w', encoding='utf-8') as outfile:
    first_file = True
    for fname in tqdm(file_list, desc=f"Merging {year}"):
        with open(fname, 'r', encoding='utf-8') as infile:
            for idx, line in enumerate(infile):
                if first_file or idx > 0:  # Keep header only for the first file
                    outfile.write(line)
        first_file = False  # After first file done

print(f"✅ Saved merged file: {output_path}")
    
cpu_info = cpuinfo.get_cpu_info()['brand_raw']
cpu_cores = psutil.cpu_count(logical=False)
cpu_threads = psutil.cpu_count(logical=True)
ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)  # in GB
total_time = time.time() - start_time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(log_path, 'w', encoding='utf-8') as log:
    log.write("Merge Log Report\n")
    log.write("====================\n")
    log.write(f"Year Merged: {year}\n")
    log.write(f"Log Created At: {current_time}\n")
    log.write(f"Total Merging Time: {total_time:.2f} seconds\n\n")

    log.write("System Information\n")
    log.write("----------------------\n")
    log.write(f"CPU: {cpu_info}\n")
    log.write(f"RAM: {ram_gb} GB\n")
    log.write(f"CPU Cores: {cpu_cores}\n")
    log.write(f"CPU Threads: {cpu_threads}\n\n")

    log.write(f"Merged Files ({len(file_list)} total):\n")
    log.write("----------------------\n")
    for fname in file_list:
        log.write(f"{fname}\n")

print(f"📝 Merge log saved at: {log_path}")
#%% split by HS6 using Numpy (fast processing)
# === Settings ===
input_path = f'C:/Users/lik6/Data/ComtradeTariffline/merge/all_{year}_merged.txt'
output_folder = f'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_{year}_numpy'
#output_folder = './month/merge/split_by_hs_2023_numpy'
checkpoint_file = f'C:/Users/lik6/Data/ComtradeTariffline/merge/split_by_hs_{year}_numpy_checkpoint.txt'
#checkpoint_file = './month/merge/split_by_hs_2023_numpy_checkpoint.txt'
hs_code_cache_folder = 'C:/Users/lik6/Data/hs_code_reference'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(hs_code_cache_folder, exist_ok=True)

chunk_size = 100000
bytes_per_row_empirical = 128  # Based on previous full run

# === Helper function ===
def format_minutes(minutes):
    minutes = int(minutes)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

# === Step 1: Extract year from input_path ===
match = re.search(r'all_(\d{4})_merged', input_path)
if match:
    data_year = int(match.group(1))
    print(f"🔵 Detected year: {data_year}")
else:
    raise ValueError("❗ Could not detect year from input_path filename.")

# === Step 2: Map year → HS version ===
year_to_hs_version = {
    1988: 'H0', 1989: 'H0', 1990: 'H0', 1991: 'H0', 1992: 'H0',
    1993: 'H1', 1994: 'H1', 1995: 'H1', 1996: 'H1', 1997: 'H1',
    1998: 'H2', 1999: 'H2', 2000: 'H2', 2001: 'H2', 2002: 'H2',
    2003: 'H3', 2004: 'H3', 2005: 'H3', 2006: 'H3', 2007: 'H3',
    2008: 'H4', 2009: 'H4', 2010: 'H4', 2011: 'H4', 2012: 'H4',
    2013: 'H5', 2014: 'H5', 2015: 'H5', 2016: 'H5', 2017: 'H5', 2018: 'H5',
    2019: 'H6', 2020: 'H6', 2021: 'H6', 2022: 'H6', 2023: 'H6'
}

if data_year not in year_to_hs_version:
    raise ValueError(f"❗ Year {data_year} not mapped to HS version.")
hs_version = year_to_hs_version[data_year]
print(f"🔵 Mapped to HS version: {hs_version}")

# === Step 3: Load or download HS codes ===
hs_code_cache_path = os.path.join(hs_code_cache_folder, f'{hs_version}.json')

if not os.path.exists(hs_code_cache_path):
    print(f"🔵 Downloading official HS code list for {hs_version}...")
    url = f'https://comtradeapi.un.org/files/v1/app/reference/{hs_version}.json'
    response = requests.get(url)
    response.raise_for_status()
    with open(hs_code_cache_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    print(f"✅ Saved {hs_version} code list to: {hs_code_cache_path}")

# Load HS codes
print(f"🔵 Loading {hs_version} codes...")
with open(hs_code_cache_path, 'r', encoding='utf-8') as f:
    hs_data = json.load(f)

valid_hs_codes = set(item['id'] for item in hs_data['results'] if item['aggrlevel'] == 6)
print(f"✅ Loaded {len(valid_hs_codes)} official {hs_version} codes.\n")

# === Step 4: Estimate total rows quickly ===
print("🔵 Estimating total rows (empirical)...")
file_size_bytes = os.path.getsize(input_path)
estimated_total_rows = file_size_bytes / bytes_per_row_empirical
print(f"🔵 Estimated total rows: {estimated_total_rows:,.0f}\n")

# === Step 5: Check for existing checkpoint ===
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        last_finished_chunk = int(f.read().strip())
    print(f"🔵 Found checkpoint. Last finished chunk: {last_finished_chunk}\n")
else:
    last_finished_chunk = -1
    print("🔵 No checkpoint found. Starting fresh!\n")

# === Step 6: Prepare reading chunks ===
chunk_iterator = pd.read_csv(
    input_path,
    sep='\s+',
    engine='c',
    dtype=str,
    chunksize=chunk_size,
    low_memory=False,
    memory_map=True
)

# === Initialize counters ===
start_time = time.time()
chunk_count = 0
row_count = 0
unknown_row_count = 0

# Path for unknown HS codes
unknown_hs_path = os.path.join(output_folder, f'{hs_version}_unknown_{data_year}.csv')

# === Main Loop ===
for chunk_idx, chunk in enumerate(tqdm(chunk_iterator, 
                    desc=f"Splitting chunks from {last_finished_chunk+1}")):

    if chunk_idx <= last_finished_chunk:
        continue  # 🔵 Skip already processed

    chunk_count += 1
    row_count += len(chunk)

    data_np = chunk.values
    cmdCode_idx = chunk.columns.get_loc('cmdCode')
    cmdcode_column = data_np[:, cmdCode_idx].astype(str)
    hs_codes = np.array([s[:6] for s in cmdcode_column])

    hs_groups = defaultdict(list)
    unknown_rows = []

    for i, hs_code in enumerate(hs_codes):
        if hs_code in valid_hs_codes:
            hs_groups[hs_code].append(data_np[i])
        else:
            unknown_rows.append(data_np[i])

    # --- Save identified HS codes ---
    for hs_code, rows in hs_groups.items():
        output_path_hs = os.path.join(output_folder, 
                            f'{hs_version}_{hs_code}_{data_year}.csv')
        rows_np = np.array(rows)

        if not os.path.exists(output_path_hs):
            header = ','.join(chunk.columns)
            np.savetxt(output_path_hs, rows_np, fmt='%s', delimiter=','
                       , header=header, comments='')
        else:
            with open(output_path_hs, 'a', encoding='utf-8') as f:
                np.savetxt(f, rows_np, fmt='%s', delimiter=',')

    # --- Save unknown HS codes ---
    if unknown_rows:
        rows_np = np.array(unknown_rows)
        unknown_row_count += len(rows_np)

        if not os.path.exists(unknown_hs_path):
            header = ','.join(chunk.columns)
            np.savetxt(unknown_hs_path, rows_np, fmt='%s', delimiter=','
                       , header=header, comments='')
        else:
            with open(unknown_hs_path, 'a', encoding='utf-8') as f:
                np.savetxt(f, rows_np, fmt='%s', delimiter=',')

    # --- Save checkpoint ---
    with open(checkpoint_file, 'w') as f:
        f.write(str(chunk_idx))

    # --- Progress report ---
    if chunk_count % 5 == 0:
        elapsed_time = time.time() - start_time
        seconds_per_chunk = elapsed_time / chunk_count
        estimated_total_chunks = estimated_total_rows / chunk_size
        estimated_chunks_left = estimated_total_chunks - chunk_count
        estimated_total_time = estimated_total_chunks * seconds_per_chunk
        estimated_time_left = estimated_total_time - elapsed_time

        print("\n--- Progress Report ---")
        print(f"Chunks processed     : {chunk_count}")
        print(f"Rows processed       : {row_count:,}")
        print(f"Unknown HS rows      : {unknown_row_count:,}")
        print(f"Elapsed time         : {format_minutes(elapsed_time/60)}")
        print(f"Est. chunks remaining: {round(estimated_chunks_left):,}")
        print(f"Est. time remaining  : {format_minutes(estimated_time_left/60)}")
        print("------------------------\n")

# === Final steps ===
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

total_time = time.time() - start_time
print("\n✅✅ All chunks processed and files safely closed!")
print(f"✅ Total rows processed   : {row_count:,}")
print(f"✅ Total unknown HS rows  : {unknown_row_count:,}")
print(f"✅ Total time taken       : {format_minutes(total_time/60)}")

# === Gather system info ===
cpu_info = cpuinfo.get_cpu_info()['brand_raw']
cpu_cores = psutil.cpu_count(logical=False)
cpu_threads = psutil.cpu_count(logical=True)
c = wmi.WMI()
disks = c.Win32_DiskDrive()
disk_info = []
for d in disks:
    disk_info.append(f"{d.Model} ({d.InterfaceType}, {round(int(d.Size) / (1024**3))} GB)")
memory = c.Win32_PhysicalMemory()
ram_info = []
for m in memory:
    ram_info.append(f"{m.Manufacturer.strip()} {int(int(m.Capacity)/(1024**3))}GB @ {int(m.Speed)}MHz")
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# === Prepare log content ===
log_content = f"""
Run summary ({current_time})
===========================
Input file               : {os.path.basename(input_path)}
Data year detected       : {data_year}
HS version used          : {hs_version}

Total rows processed   : {row_count:,}
Total unknown HS rows  : {unknown_row_count:,}
Total time taken       : {format_minutes(total_time/60)}

System info
---------------------------
CPU                      : {cpu_info}
Physical cores           : {cpu_cores}
Logical processors       : {cpu_threads}
RAM Modules              : {ram_info[0]}
Disk Hardware            : {disk_info[0]}
===========================
"""

# === Write to log file ===
log_path = os.path.join(output_folder, f'run_summary_{data_year}.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(log_content)

print(f"\n✅✅✅ Summary log saved at: {log_path}")
