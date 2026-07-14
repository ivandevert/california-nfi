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

# %% [markdown]
# ## Generate response curve lookup

# # %%
# from obspy import read_inventory
# from collections import defaultdict

# def build_response_lookup(xml_paths, eval_freqs):
#     lookup = defaultdict(list)

#     for path in tqdm(xml_paths):
#         inv = read_inventory(path, format="STATIONXML")
#         for net in inv:
#             for sta in net:
#                 for cha in sta:
#                     key = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
#                     correction = get_correction(cha, eval_freqs)

#                     if correction is None:
#                         print(f"{net.code}.{sta.code}.{cha.location_code}.{cha.code} {cha.start_date} to {cha.end_date}: Failed to get correction")
#                     else:
#                         lookup[key].append((cha.start_date, cha.end_date, correction))

#     for key in lookup:
#         lookup[key].sort(key=lambda x: x[0])

#     return lookup

# def get_correction(cha, eval_freqs):
#     pi = 3.141592653589793
#     resp = cha.response

#     # Some channels have no response stages. Skip them unless they are HH
#     if not resp.response_stages and not cha.code.startswith("HH"):
#         return None
#     elif not resp.response_stages and cha.code.startswith("HH"):
#         sensitivity = resp.instrument_sensitivity
#         if isinstance(sensitivity, obspy.core.inventory.response.InstrumentSensitivity):
#             sensitivity = sensitivity.value
#         correction = sensitivity * (2 * pi * eval_freqs) # HH* are velocity
#     else:
#         # Otherwise, get the correction
#         try:
#             resp.recalculate_overall_sensitivity(10)
#             correction = resp.get_evalresp_response_for_frequencies(
#                 eval_freqs, 
#                 output="DISP",
#                 hide_sensitivity_mismatch_warning=True
#             )
#         except Exception as e:
#             return None
    
#     # convert from time-series correction to frequency-domain correction
#     correction = np.abs(correction) ** 2

#     return correction
    


# # %%
# response_dir = join(response_output_dir, "xml/")

# eval_freqs = np.arange(0, 50.6666, 2./3.)

# # get a list of all xml files
# response_files = [response_dir + f for f in os.listdir(response_dir) if f.endswith(".xml")]
# lookup = build_response_lookup(response_files, eval_freqs)

# # %%
# def get_epoch_correction(key, event_time, response_lookup):
#     corrections = response_lookup.get(key, [])

#     if not corrections:
#         return None

#     for start, end, correction in corrections:
#         if event_time >= start and event_time < end:
#             return correction

#     return None

# %%
# get_epoch_correction("CI.CCC..HHZ", UTC("2002-01-01T00:00:00"), lookup)

# %% [markdown]
# ## Test response removal
# - Remove instrument response from a precomputed power spectrum by subtracting 2 * log10(|H(f)|), where H(f) is the full response (poles/zeros × gain × sensitivity) evaluated at each frequency via get_evalresp_response_for_frequencies(freqs, output="DISP")
# - The factor of 2 arises because power ∝ amplitude²: dividing the time series by |H| reduces power by |H|²
# - For an amplitude spectrum, subtract 1 * log10(|H(f)|) instead
# - H(f) includes the overall sensitivity (counts/m for DISP), so no separate sensitivity correction is needed
# - This is equivalent to what obspy.remove_response() does via time-domain deconvolution, minus the water level and pre-filter stabilization applied at band edges
# 
# 
# When full response is not available, for HH* channels (velocity) we can approximate the response as the following:
# ```
# 2*np.log10((channel_resp.instrument_sensitivity.value)*(2 * np.pi * f)**1)
# ```
# where $(2\pi f)^1$ is the velocity to displacement correction, and channel_resp.instrument_sensitivity.value is the scalar sensitivity correction. Since the response is approximately flat in the frequencies we're concerned with (about 2--18 Hz) this should work.

# %%

# import obspy
# from obspy import read_inventory
# import scipy 
# import scipy.signal as signal
# from scipy.signal.windows import hann
# from obspy.signal import freqattributes
# import multitaper.utils as utils

# def mtspec_helper(x, nw, kspec, dt, nfft, iadapt, vn, lamb):
#     # assumptions:
#     # 1) x is real
#     # 2) nfft is odd (nwin is even)

#     npts = len(x)

#     x = x[:, np.newaxis]

#     nf = int((nfft+1)/2)
#     f = scipy.fft.fftfreq(nfft, dt)
#     df = f[1] - f[0]

#     x = signal.detrend(x, axis=0, type='constant')
#     xvar = np.var(x)

#     yk, sk  = utils.eigenspec(x, vn, lamb, nfft)

#     spec, se, wt = utils.adaptspec(yk, sk, lamb, iadapt)
#     sscal = np.sum(spec)*df
#     sscal = xvar/sscal
#     spec  = sscal*spec

#     freq     = np.zeros((nf,1), dtype=float)
#     freq     = f[0:nf]


#     # spec = np.zeros((nf,1), dtype=float)
#     spec = spec[0:nf,0]
#     spec[0:nf]   = 2.0 * spec[0:nf]
#     spec[0]      = 0.5 * spec[0]
    
#     return freq, spec

# t_buffer = -0.15                # time relative to pick for signal window
# t_window_desired = 1.5                  # length of desired signal and noise windows (s)
# sampling_rate = 100                    # desired sampling rate for output spectra

# dt = 1 / sampling_rate
# nwin = int(t_window_desired * sampling_rate)
# nf = int(nwin / 2 + 1)
# f_nyquist = sampling_rate / 2

# nw = 3.5                    # time bandwidth product
# kspec = 5                   # number of tapers
# nfft = int(2*t_window_desired*sampling_rate + 1)   # number of FFT points to pad to
# iadapt = 0                  # multitaper method: 0=adaptive, 1=unweighted, 2=wt by the eigenvalue of DPSS (see docs)

# # variable time window
# min_twin = 1.5
# max_twin = 4.5
# d_twin = 0.25
# M_min_twin = 3.5
# M_max_twin = 4.5

# # Number of time window lengths
# ntwin = int((max_twin - min_twin) / d_twin + 1)
# # Time window lengths
# twins = np.linspace(min_twin, max_twin, ntwin)
# nwins = (twins * sampling_rate).astype(int)
# nffts = (2*twins*sampling_rate + 1).astype(int)

# # Precompute slepian tapers
# vn = [[]] * ntwin
# lamb = [[]] * ntwin
# for i in range(ntwin):
#     vn[i], lamb[i] = utils.dpss(nwins[i], nw, kspec)

# f_resample = np.linspace(0, f_nyquist, nf)

# params = {
#     'nw': nw,
#     'kspec': kspec,
#     'nfft': nfft,
#     'iadapt': iadapt,
#     'nffts': nffts,
#     'vn': vn,
#     'lamb': lamb,
# }

# event_name = "n72429611"

# ms_path = f"/zdata/datasets/ca_seismicity/waveforms/2015/100/{event_name}/*.ms"

# st = obspy.read(ms_path)
# # st = st.select(station='MHD')
# st = st.select(network="NC", station="BBGB", channel="HHZ")
# st.detrend("linear")
# st.taper(type="hamming", max_percentage=0.05)
# st.filter(type="highpass", freq=0.5)
# tr = st[0]

# # t0 = 12.3
# t0 = 19.5
# t1 = t0 + 1.5

# twin_idx = 0

# it0 = np.argmin(np.abs(st[0].times() - t0))
# it1 = np.argmin(np.abs(st[0].times() - t1))
# print(it0, it1, it1-it0)


# t = st[0].times()[it0:it1]
# d = st[0].data[it0:it1]
# f, s_orig = mtspec_helper(d, nw=params['nw'], kspec=params['kspec'], dt=tr.stats.delta, nfft=params['nffts'][twin_idx], iadapt=params['iadapt'], vn=params['vn'][twin_idx], lamb=params['lamb'][twin_idx])
# s_orig = np.log10(s_orig)

# # attach response and correct
# # station_name = "NC.MHD"
# station_name = "NC.BBGB"
# net, sta = station_name.split(".")

# resp = read_inventory(join(xml_output_dir, f"{station_name}.xml"))

# eval_freqs = f[1:]
# net, sta, loc, cha = tr.id.split(".")
# # channel_resp = resp.select(network=net, station=sta, location=loc, channel=cha)[0][0][25].response
# channel_resp = resp.select(network=net, station=sta, location=loc, channel=cha)[0][0][0].response
# H = channel_resp.get_evalresp_response_for_frequencies(
#     eval_freqs,
#     output="DISP",
# )
# corr = np.log10(np.abs(H))

# st2 = st.copy()
# st2.remove_response(output="DISP", water_level=60, inventory=resp, plot=True, pre_filt=[0.1, 0.5, 40, 50])
# tr2 = st2[0]
# # tr2.filter("highpass", freq=1, zerophase=True, corners=4)
# d2 = tr2.data[it0:it1]
# d2 = d2 = d2 - np.mean(d2)
# # d2 = d2 * taper

# f2, s2_orig = mtspec_helper(d2, nw=params['nw'], kspec=params['kspec'], dt=tr2.stats.delta, nfft=params['nffts'][twin_idx], iadapt=params['iadapt'], vn=params['vn'][twin_idx], lamb=params['lamb'][twin_idx])
# s2_orig = np.log10(s2_orig)

# f_corr = f

# # convert from time-series correction to frequency-domain correction
# corr = 2 * corr
# corr_full = np.insert(corr, 0, corr[0]-0.5)


# yrng = 5

# ylim1 = [np.max(s_orig)+0.5-yrng, np.max(s_orig)+0.5]
# ylim2 = [np.max(s2_orig)+0.5-yrng, np.max(s2_orig)+0.5]

# fig, axs = plt.subplots(2,2, figsize=(12, 4))
# axs[0,0].plot(st[0].times(), st[0].data, color='k', lw=0.5)
# axs[0,0].axvline(t0, color='r', lw=0.5)
# axs[0,0].axvline(t1, color='r', lw=0.5)
# axs[0,0].plot(t, d, color='k', lw=1.5)
# axs[0,0].set_xlim([t0-2, t1+2])
# axs[0,0].set_title(f"{event_name} {tr.id}")

# axs[1,0].plot(tr2.times(), tr2.data, color='k', lw=0.5)
# axs[1,0].plot(t, d2, color='b', lw=1.5)
# axs[1,0].axvline(t0, color='r', lw=0.5)
# axs[1,0].axvline(t1, color='r', lw=0.5)
# axs[1,0].set_xlim([t0-2, t1+2])


# axs[0,1].plot(f, s_orig, color='k', lw=1.5, label="Vel")
# axs[0,1].set_xlim([2, 40])
# axs[0,1].set_ylim(ylim1)
# axs[0,1].set_xscale('log')
# axs[0,1].legend(loc="lower left", fontsize=10)

# axs[1,1].plot(f2, s2_orig, color='b', lw=1.5, label="Disp (using resp)")
# axs[1,1].plot(f, s_orig - corr_full, color='r', lw=1.5, label="Disp (using corr)")

# axs[1,1].set_xlim([2, 40])
# axs[1,1].set_ylim(ylim2)
# axs[1,1].set_xscale('log')
# axs[1,1].legend(loc="lower left", fontsize=10)


# plt.show()

# channel_resp.recalculate_overall_sensitivity(10)


# corr_flat = 2*np.log10((channel_resp.instrument_sensitivity.value) * (2 * np.pi * f)**1)
# plt.figure()
# # plt.plot(f_corr, corr_full, color='g', lw=1.5, label="Correction")
# plt.plot(f2, s2_orig, color='b', lw=1.5, label="Disp (using resp)")
# # plt.plot(f, s_orig - corr_full - np.log10(channel_resp.instrument_sensitivity.value), color='r', lw=1.5, label="Disp (using corr)")
# plt.plot(f, s_orig - corr_full, color='r', lw=1.5, label="Disp (using corr)")
# plt.plot(f, s_orig - corr_flat, color='lime', lw=1.5, label="Disp (using flat corr)")
# plt.xscale('log')
# plt.xlim([2, 50])
# plt.show()





# # %%
# a = 2*np.log10((channel_resp.instrument_sensitivity.value)*(2 * np.pi * f)**1)


# plt.figure()
# plt.plot(f_corr, corr_full, color='r', lw=1.5)
# plt.plot(f_corr, a, color='lime', lw=1.5)
# plt.xscale('log')
# plt.xlim([2, 50])
# plt.show()

# # %%
# st = obspy.read(ms_path)
# # st = st.select(station='MHD')
# st = st.select(channel="HN*")
# st.plot()

# # %%
# channel_resp.recalculate_overall_sensitivity(10)
# channel_resp.instrument_sensitivity.value

# # %%
# dir(channel_resp)

# # %%
# np.log10(3.98637e+08)

# # %%
# s_orig - corr_full - s2_orig

# # %%
# eval_freqs

# # %%
# net, sta, loc, cha = tr.id.split(".")
# channel_resp = resp.select(network=net, station=sta, location=loc, channel=cha)[0][0][0].response
# channel_resp

# # %%
# # check for moving stations
# unique_channels = np.unique(combined_clean['channel_name'])
# moving_idx = []
# tol = 0.1 # in ~km

# for i in trange(len(unique_channels)):
#     matching_rows = combined_clean[combined_clean['channel_name'] == unique_channels[i]]
#     if len(matching_rows) > 1:
#         slat = matching_rows['slat'].values
#         slon = matching_rows['slon'].values

#         if len(np.unique(slat)) > 1 or len(np.unique(slon)) > 1:
#             slat_std = np.std(slat)
#             slon_std = np.std(slon)
#             if slat_std > tol or slon_std > tol:
#                 print(f"moving station: {matching_rows.iloc[0]['channel_name']}, std(slat): {slat_std*111.0} (~km), std(slon): {slon_std*111.0} (~km)")
#             # print(matching_rows[['channel_name', 'slat', 'slon']])

# # %%
# # If all is well above, save combined_clean as the final catalog

# # sort by channel_name (primary) and starttime descending (secondary)
# csv_kwargs = {
#     'index': True,
#     'sep': '|',
#     'mode': 'a',
#     'index_label': 'index'
# }

# combined_clean = combined_clean.sort_values(['channel_name', 'starttime'], ascending=[True, False])

# # drop duplicated and keep the first entry, based on channel name alone
# combined_clean_out = combined_clean.drop_duplicates(subset=['channel_name'], keep='first').reset_index(drop=True)
# combined_clean_out = combined_clean_out.drop(['starttime', 'endtime'], axis=1)
# # with open('station_catalogs/california_channel_catalog.csv', 'w') as f:
# #     f.write(f"generated by {os.getcwd()}/get_station_catalog.ipynb\n")
# # combined_clean_out.to_csv('station_catalogs/california_channel_catalog.csv', **csv_kwargs)

# # write combined_clean to a file
# # with open('station_catalogs/california_channel_epoch_catalog.csv', 'w') as f:
# #     f.write(f"generated by {os.getcwd()}/get_station_catalog.ipynb\n")
# # combined_clean.to_csv('station_catalogs/california_channel_epoch_catalog.csv', **csv_kwargs)

# stations_out = combined_clean_out.drop_duplicates(subset='station_name', keep='first')
# stations_out = stations_out[['station_name', 'slat', 'slon', 'sele', 'sdep', 'source']].reset_index(drop=True)
# # with open('station_catalogs/california_station_catalog.csv', 'w') as f:
# #     f.write(f"generated by {os.getcwd()}/get_station_catalog.ipynb\n")
# # stations_out.to_csv('station_catalogs/california_station_catalog.csv', **csv_kwargs)

# # %% [markdown]
# # # slat differences

# # %%
# quantity = 'slat'
# tol = 0.1/111.0
# unit = 'deg'

# mismatched_rows = ncedc_in_scedc[ncedc_in_scedc[f'{quantity}_n']!=ncedc_in_scedc[f'{quantity}_s']][['channel_name', 'starttime', 'endtime_n', 'endtime_s', f'{quantity}_n', f'{quantity}_s']]
# err = mismatched_rows[f'{quantity}_n'] - mismatched_rows[f'{quantity}_s']


# plt.figure()
# plt.hist(err, bins=20)
# plt.xlabel(f"Difference in {quantity} ({unit})")
# plt.ylabel("Count")
# plt.show()

# # station NP.1879 has a known difference in slat as reported by NCEDC and SCEDC.
# # I'd believe the NCEDC slat is more accurate since it is at Moffett Field
# # https://ds.iris.edu/mda/NP/1879/?starttime=2024-12-31T01:15:00&endtime=3000-01-01T00:00:00

# print(f"These entries have > {tol} {unit} difference in {quantity}: ")
# mismatched_rows[err.abs() > tol]

# # %% [markdown]
# # # slon differences

# # %%
# quantity = 'slon'
# tol = 0.1/111.0
# unit = 'deg'

# mismatched_rows = ncedc_in_scedc[ncedc_in_scedc[f'{quantity}_n']!=ncedc_in_scedc[f'{quantity}_s']][['channel_name', 'starttime', 'endtime_n', 'endtime_s', f'{quantity}_n', f'{quantity}_s']]
# err = mismatched_rows[f'{quantity}_n'] - mismatched_rows[f'{quantity}_s']


# plt.figure()
# plt.hist(err, bins=20)
# plt.xlabel(f"Difference in {quantity} ({unit})")
# plt.ylabel("Count")
# plt.show()

# # station NP.1879 has a known difference in slat as reported by NCEDC and SCEDC.
# # I'd believe the NCEDC slat is more accurate since it is at Moffett Field
# # https://ds.iris.edu/mda/NP/1879/?starttime=2024-12-31T01:15:00&endtime=3000-01-01T00:00:00

# print(f"These entries have > {tol} {unit} difference in {quantity}: ")
# mismatched_rows[err.abs() > tol]

# # %% [markdown]
# # # sele

# # %%
# quantity = 'sele'
# tol = 1
# unit = 'm'

# mismatched_rows = ncedc_in_scedc[ncedc_in_scedc[f'{quantity}_n']!=ncedc_in_scedc[f'{quantity}_s']][['channel_name', 'starttime', 'endtime_n', 'endtime_s', f'{quantity}_n', f'{quantity}_s']]
# err = mismatched_rows[f'{quantity}_n'] - mismatched_rows[f'{quantity}_s']


# plt.figure()
# plt.hist(err, bins=20)
# plt.xlabel(f"Difference in {quantity} ({unit})")
# plt.ylabel("Count")
# plt.show()

# # station NP.1879 has a known difference in slat as reported by NCEDC and SCEDC.
# # I'd believe the NCEDC slat is more accurate since it is at Moffett Field
# # https://ds.iris.edu/mda/NP/1879/?starttime=2024-12-31T01:15:00&endtime=3000-01-01T00:00:00

# print(f"These entries have > {tol} {unit} difference in {quantity}: ")
# mismatched_rows[err.abs() > tol]

# # %% [markdown]
# # # sdep

# # %%
# quantity = 'sdep'
# tol = 1
# unit = 'm'

# mismatched_rows = ncedc_in_scedc[ncedc_in_scedc[f'{quantity}_n']!=ncedc_in_scedc[f'{quantity}_s']][['channel_name', 'starttime', 'endtime_n', 'endtime_s', f'{quantity}_n', f'{quantity}_s']]
# err = mismatched_rows[f'{quantity}_n'] - mismatched_rows[f'{quantity}_s']


# plt.figure()
# plt.hist(err, bins=20)
# plt.xlabel(f"Difference in {quantity} ({unit})")
# plt.ylabel("Count")
# plt.show()

# # station NP.1879 has a known difference in slat as reported by NCEDC and SCEDC.
# # I'd believe the NCEDC slat is more accurate since it is at Moffett Field
# # https://ds.iris.edu/mda/NP/1879/?starttime=2024-12-31T01:15:00&endtime=3000-01-01T00:00:00

# print(f"These entries have > {tol} {unit} difference in {quantity}: ")
# mismatched_rows[err.abs() > tol]

# # %%
# quantity = 'sample_rate'
# tol = 1
# unit = 'Hz'

# mismatched_rows = ncedc_in_scedc[ncedc_in_scedc[f'{quantity}_n']!=ncedc_in_scedc[f'{quantity}_s']][['channel_name', 'starttime', 'endtime_n', 'endtime_s', f'{quantity}_n', f'{quantity}_s']]
# err = mismatched_rows[f'{quantity}_n'] - mismatched_rows[f'{quantity}_s']


# plt.figure()
# plt.hist(err, bins=20)
# plt.xlabel(f"Difference in {quantity} ({unit})")
# plt.ylabel("Count")
# plt.show()

# # station NP.1879 has a known difference in slat as reported by NCEDC and SCEDC.
# # I'd believe the NCEDC slat is more accurate since it is at Moffett Field
# # https://ds.iris.edu/mda/NP/1879/?starttime=2024-12-31T01:15:00&endtime=3000-01-01T00:00:00

# print(f"These entries have > {tol} {unit} difference in {quantity}: ")
# mismatched_rows[err.abs() > tol]

# # %% [markdown]
# # # endtime

# # %%
# quantity = 'endtime'
# tol = 1
# unit = 'days'

# mismatched_rows = ncedc_in_scedc[ncedc_in_scedc[f'{quantity}_n']!=ncedc_in_scedc[f'{quantity}_s']][['channel_name', 'starttime', f'{quantity}_n', f'{quantity}_s']]
# err = np.array([UTCDateTime(n) - UTCDateTime(s) for n, s in zip(mismatched_rows['endtime_n'], mismatched_rows['endtime_s'])]) / 86400


# plt.figure()
# plt.hist(err, bins=20)
# plt.xlabel(f"Difference in {quantity} ({unit})")
# plt.ylabel("Count")
# plt.show()

# # station NP.1879 has a known difference in slat as reported by NCEDC and SCEDC.
# # I'd believe the NCEDC slat is more accurate since it is at Moffett Field
# # https://ds.iris.edu/mda/NP/1879/?starttime=2024-12-31T01:15:00&endtime=3000-01-01T00:00:00

# print(f"These entries have > {tol} {unit} difference in {quantity}: ")
# mismatched_rows[np.abs(err) > tol]

