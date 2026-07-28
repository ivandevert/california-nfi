# get_station_catalog.py
# 
# This script downloads station metadata from NCEDC and SCEDC.

# %%
# Import built-in libraries
import os
from os.path import join
import urllib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from obspy import UTCDateTime as UTC
from obspy.clients.fdsn import Client

from tqdm import trange, tqdm


# --- Load config --- #
import sys
from pathlib import Path
config_path = str(Path(__file__).resolve().parents[2] / "config.yaml")
lib_dir = str(Path(__file__).resolve().parents[2] / "lib")
sys.path.insert(0, lib_dir)
from helper_functions import load_config
cfg = load_config(config_path)
# --- End load config --- #

# Load config parameters
cfg_paths = cfg["paths"]
cfg_params = cfg['station_downloads']

project_dir = cfg_paths["project_dir"]
catalogs_dir = cfg_paths["catalogs_dir"]
station_dir = join(catalogs_dir, "station_fdsn/")
src_dir = join(station_dir, "src/")

os.makedirs(station_dir, exist_ok=True)
os.makedirs(src_dir, exist_ok=True)

starttime = cfg_params['starttime']
endtime = cfg_params['endtime']

allowed_sample_rates = cfg_params['allowed_sample_rates']
allowed_units = cfg_params['allowed_units']
allowed_dips = cfg_params['allowed_dips']

script_name = os.path.basename(__file__)

# %%
# This isn't really data availability, but changes in station gain/responses.
# I might want to change later on.
ncedc_url = "https://service.ncedc.org/fdsnws/station/1/query?level=channel&format=text&includeavailability=false"
scedc_url = "https://service.scedc.caltech.edu/fdsnws/station/1/query?level=channel&format=text&includeavailability=false&nodata=404"

ncedc_src_filepath = join(src_dir, "ncedc.csv")
scedc_src_filepath = join(src_dir, "scedc.csv")

print("Loading station information...\n")

if not os.path.exists(ncedc_src_filepath):
    with urllib.request.urlopen(ncedc_url) as response, open(ncedc_src_filepath, 'w') as out_file:
        text = response.read().decode('utf-8')
        out_file.write(text)
else:
    print("ncedc.csv already exists, skipping download.")

if not os.path.exists(scedc_src_filepath):
    with urllib.request.urlopen(scedc_url) as response, open(scedc_src_filepath, 'w') as out_file:
        text = response.read().decode('utf-8')
        out_file.write(text)
else:
    print("scedc.csv already exists, skipping download.")

# read in the station data
dfs = []
ncedc_df = pd.read_csv(ncedc_src_filepath, sep='|', low_memory=False)
scedc_df = pd.read_csv(scedc_src_filepath, sep='|', low_memory=False)
ncedc_df['source'] = 'ncedc'
scedc_df['source'] = 'scedc'

### Clean the data
# rename fields
rename_dict = {
    "#Network "     : "net",
    " Station "     : "sta",
    " Location "    : "loc",
    " Channel "     : "cha",
    " Latitude "    : "slat",
    " Longitude "   : "slon",
    " Elevation "   : "sele",
    " Depth "       : "sdep",
    " Azimuth "     : "sazi",
    " Dip "         : "sdip",
    " Instrument "  : "instrument",
    " Scale "       : "scale",
    " ScaleFreq "   : "scalefreq",
    " ScaleUnits "  : "units",
    " SampleRate "  : "sample_rate",
    " StartTime "   : "starttime",
    " EndTime"      : "endtime"
}
ncedc_df = ncedc_df.rename(columns=rename_dict, errors="raise")
scedc_df = scedc_df.rename(columns=rename_dict, errors="raise")

# replace NaNs in 'sloc' with empty string
ncedc_df['loc'] = ncedc_df['loc'].fillna('')
scedc_df['loc'] = scedc_df['loc'].fillna('')

# convert start and end times to pd datetimes
# first, clip endtimes to 2030-01-01T00:00:00
ncedc_df.loc[ncedc_df['endtime'].str[:4].astype(int)>=2030, 'endtime'] = "2030-01-01T00:00:00"
scedc_df.loc[scedc_df['endtime'].str[:4].astype(int)>=2030, 'endtime'] = "2030-01-01T00:00:00"
ncedc_df['starttime'] = pd.to_datetime(ncedc_df['starttime'])
scedc_df['starttime'] = pd.to_datetime(scedc_df['starttime'])
ncedc_df['endtime'] = pd.to_datetime(ncedc_df['endtime'])
scedc_df['endtime'] = pd.to_datetime(scedc_df['endtime'])

# remove entries with column 'starttime' > endtime and column 'endtime' < starttime
ncedc_df = ncedc_df[(ncedc_df['starttime'] <= endtime) & (ncedc_df['endtime'] >= starttime)]
scedc_df = scedc_df[(scedc_df['starttime'] <= endtime) & (scedc_df['endtime'] >= starttime)]

ncedc_df['channel_name'] = ncedc_df.apply(lambda row: f"{row['net']}.{row['sta']}.{row['loc']}.{row['cha']}", axis=1)
ncedc_df['station_name'] = ncedc_df.apply(lambda row: f"{row['net']}.{row['sta']}", axis=1)


scedc_df['channel_name'] = scedc_df.apply(lambda row: f"{row['net']}.{row['sta']}.{row['loc']}.{row['cha']}", axis=1)
scedc_df['station_name'] = scedc_df.apply(lambda row: f"{row['net']}.{row['sta']}", axis=1)

# unique identifiers for each row
cols = ['channel_name', 'station_name', 'starttime']

ncedc_in_scedc = pd.merge(ncedc_df, scedc_df, how='inner', on=cols, suffixes=('_n', '_s'))

# Prefer the NCEDC entries since they seem to be more accurate
# find the indices of scedc_df, where entry is also in ncedc_df. Match on 'channel_name' and 'starttime'
# scedc_df_indexed = scedc_df.copy()
# scedc_df_indexed['index'] = scedc_df_indexed.index

# scedc_in_ncedc = pd.merge(scedc_df_indexed, ncedc_df, how='inner', on=cols, suffixes=('_s', '_n'))
# remove_idx = scedc_in_ncedc['index'].values

# # remove those. 
# scedc_df = scedc_df.drop(remove_idx).reset_index(drop=True)

scedc_only = scedc_df.merge(ncedc_df[cols], on=cols, how='left', indicator=True)
scedc_df = scedc_only[scedc_only['_merge'] == 'left_only'].drop(columns='_merge').reset_index(drop=True)

# Now, merge (concat)
combined = pd.concat([ncedc_df, scedc_df], join='outer', ignore_index=True)

# Make sure no duplicates in combined, based on channel_name and starttime
assert combined.drop_duplicates(subset=['channel_name', 'starttime'], keep='first').shape == combined.shape
print("No duplicates remaining in combined, based on channel_name and starttime")



# %% [markdown]
# # Filter out unwanted stations for a second catalog

# %%
keep_columns = ['channel_name', 'station_name', 'slat', 'slon', 'sele', 
    'sdep', 'sazi', 'sdip', 'sample_rate', 'units', 'scale', 'scalefreq', 
    'starttime', 'endtime', 'source']

# Drop entries with NaN units (these are DAS instruments in the CA dataset)
combined_clean = combined[combined['units'].notna()].reset_index(drop=True)

# Drop entries with weird units
combined_clean['units'] = [el.lower() for el in combined_clean['units'].values]
combined_clean = combined_clean[combined_clean['units'].isin(allowed_units)].reset_index(drop=True)

# Drop entries with weird sdip
combined_clean = combined_clean[combined_clean['sdip'].isin(allowed_dips)].reset_index(drop=True)

# Drop entries with weird sample rates
print(len(combined_clean))
combined_clean = combined_clean[combined_clean['sample_rate'].isin(allowed_sample_rates)].reset_index(drop=True)
print(len(combined_clean))

combined_clean = combined_clean[keep_columns].reset_index(drop=True)
# sort by channel_name (primary) and starttime descending (secondary)
csv_kwargs = {
    'index': False,
    'sep': '|',
    'mode': 'a',
    'index_label': 'index'
}

channel_catalog_filepath        = paths['channel_catalog_filepath']
channel_epoch_catalog_filepath  = join(catalogs_dir, "channel_epoch_catalog.csv")
station_catalog_filepath        = paths['station_catalog_filepath']


combined_clean = combined_clean.sort_values(['channel_name', 'starttime'], ascending=[True, False])

# drop duplicated and keep the first entry, based on channel name alone
combined_clean_out = combined_clean.drop_duplicates(subset=['channel_name'], keep='first').reset_index(drop=True)
combined_clean_out = combined_clean_out.drop(['starttime', 'endtime'], axis=1)
with open(channel_catalog_filepath, 'w') as f:
    f.write(f"# generated by {os.getcwd()}/{script_name}\n")
combined_clean_out.to_csv(channel_catalog_filepath, **csv_kwargs)

# write combined_clean to a file
with open(channel_epoch_catalog_filepath, 'w') as f:
    f.write(f"# generated by {os.getcwd()}/{script_name}\n")
combined_clean.to_csv(channel_epoch_catalog_filepath, **csv_kwargs)

stations_out = combined_clean_out.drop_duplicates(subset='station_name', keep='first')
stations_out = stations_out[['station_name', 'slat', 'slon', 'sele', 'sdep', 'source']].reset_index(drop=True)
with open(station_catalog_filepath, 'w') as f:
    f.write(f"# generated by {os.getcwd()}/{script_name}\n")
stations_out.to_csv(station_catalog_filepath, **csv_kwargs)

station_json_filepath = join(station_dir, "station_catalog.json")
channel_json_filepath = join(station_dir, "channel_catalog.json")

# make a lookup dict for station locations station_name: (slon, slat, sele)
station_locations = dict(zip(stations_out['station_name'], zip(stations_out['slon'], stations_out['slat'], stations_out['sele'])))
with open(station_json_filepath, 'w') as f_out:
    json.dump(station_locations, f_out, indent=4)

# make a lookup dict for channels: (slon, slat, sele)
channel_dict = dict(zip(combined_clean_out['channel_name'], zip(combined_clean_out['sazi'], combined_clean_out['sdip'], combined_clean_out['sample_rate'], combined_clean_out['units'], combined_clean_out['scale'])))
with open(channel_json_filepath, 'w') as f_out:
    json.dump(channel_dict, f_out, indent=4)

# %% [markdown]
# ## Download and organize station response

# %%


response_output_dir = join(catalogs_dir, "response/")
download_dir = join(response_output_dir, "src/")
xml_output_dir = join(response_output_dir, "xml/")

for d in [response_output_dir, download_dir, xml_output_dir]:
    os.makedirs(d, exist_ok=True)


channel_catalog_filepath = join(catalogs_dir, "channel_catalog.csv")
cha_df = pd.read_csv(channel_catalog_filepath, sep='|', comment='#')

# filter to only include components beginning with H or E
cha_df['component'] = cha_df['channel_name'].str.split(".").str[-1]
cha_df = cha_df[cha_df['component'].str[0].isin(['H', 'E'])].reset_index(drop=True)
station_names = np.unique(cha_df['station_name'])


# %%
channels = "HHZ,HHN,HHE,HNZ,HNN,HNE,EHZ,EHN,EHE,ENZ,ENN,ENE,HLZ,HLN,HLE"
providers = ['NCEDC', "SCEDC", "EARTHSCOPE"]

starttime = UTC(starttime)
endtime = UTC(endtime)

clients = [Client(provider) for provider in providers]

def fetch_station(st_code):
    filepath = join(xml_output_dir, f"{st_code}.xml")
    if os.path.isfile(filepath):
        return
    net, sta = st_code.split(".")
    errors = []
    if net=="CI":
        use_clients = [clients[1], clients[2], clients[0]]
    else:
        use_clients = clients
    for client in use_clients:
        try:
            inv = client.get_stations(
                network=net, station=sta, location="*", channel=channels,
                starttime=starttime, endtime=endtime, level="response"
            )
            inv.write(filepath, format="STATIONXML")
            return
        except Exception as e:
            errors.append(f"{client}: {e}")
    print(f"All clients failed for {st_code}")# {'; '.join(errors)}")

max_workers = min(16, len(station_names))
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(fetch_station, st_code): st_code for st_code in station_names}
    for future in tqdm(as_completed(futures), total=len(futures)):
        future.result()  # re-raises unexpected exceptions
