# %% [markdown]
# # get_aws_waveforms.ipynb
# 
# This notebook helps download event-based waveforms from both NCEDC and SCEDC for a combined dataset. The data comes from AWS services:
# 
# https://ncedc.org/db/cloud.html
# 
# https://scedc.caltech.edu/data/cloud.html
# 
# See **Parameters** section below for prerequisite catalogs.

# %%
import numpy as np
import pandas as pd
import os
from os.path import join
import traceback
from datetime import datetime, timezone

from matplotlib import pyplot as plt

import obspy
from obspy import UTCDateTime, Stream
from obspy.io.mseed.core import ObsPyMSEEDFilesizeTooSmallError, InternalMSEEDError

# --- AWS SDK imports ---#
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config

import sys
from pathlib import Path

# --- Load config --- #
config_path = str(Path(__file__).resolve().parents[2] / "config.yaml")
lib_dir = str(Path(__file__).resolve().parents[2] / "lib")
sys.path.insert(0, lib_dir)
from helper_functions import load_config
cfg = load_config(config_path)
# --- End load config --- #

# --- Globals --- #
# Define status codes for database
STATUS = {
    'new': 0,
    'downloaded': 1,
    'failed': -99,
    'processed': 2, # unused but retain for possible future use
    'no_data': -90,
    'redundant': -91
}



# %%
def get_scedc_bucket_key_filename_event(event_id, date, ext):
    # date is a pandas.Timestamp
    # Tested, working
    # example key: event_waveforms/2022/2022_001/38438519.ms
    julday = str(date.dayofyear).zfill(3)
    year = date.year
    
    prefix = f"event_waveforms/{year}/{year}_{julday}/"
    remote_filename = f"{event_id}.{ext}"
    key = prefix + remote_filename
    filename = f"s{event_id}.ms"

    return key, filename

def get_ncedc_bucket_key_filename_event(event_id, date, ext):
    # https://ncedc.org/db/cloud.html
    julday = str(date.dayofyear).zfill(3)
    year = date.year

    prefix = f"event_waveforms/{year}/{year}.{julday}/"
    remote_filename = f"NC.{event_id}.NCSS.{ext}"
    filename = f"n{event_id}.ms"

    key = prefix + remote_filename

    return key, filename

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    tested: working
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km

def get_waveform_path(event_name, edatetime):
    # return event path, ex:
    # event_name = 's1234' on Jan 23, 2022
    # event_partial_dir = {year}/{julianday}/{event_name}/
    year, julday = edatetime.year, str(edatetime.dayofyear).zfill(3)
    event_partial_dir = f"{year}/{julday}/{event_name}/"
    return event_partial_dir

def download_aws_event(event_name, date, download_dir, boto3_res, ext='ms',
                       err_log_path=None):
    """
    Download an event from AWS.
 
    Parameters
    ----------
    event_name : str
        Event name, e.g. 'n38438519' or 's38438519'
    date : pandas.Timestamp
        Event origin time
    download_dir : str
        Directory to download .ms files into
    boto3_res : boto3.resource
        AWS resource object
    ext : str, optional
        Remote file extension, by default 'ms'
    err_log_path : str, optional
        If given, 'failed' outcomes are logged here with the S3 key and
        full traceback.
 
    Returns
    -------
    code : str
        'downloaded', 'no_data', 'failed'
    """
    event_id = int(event_name[1:])  # remove leading 'n' or 's'
    source = event_name[0]          # 'n' or 's'
    ext = ext.strip('.')
    if source == 'n':
        key, _ = get_ncedc_bucket_key_filename_event(event_id, date, ext)
        source_name = 'ncedc'
        bucket = 'ncedc-pds'
    elif source == 's':
        key, _ = get_scedc_bucket_key_filename_event(event_id, date, ext)
        source_name = 'scedc'
        bucket = 'scedc-pds'
    else:
        raise ValueError(f"Unknown source '{source}' for event {event_name}")
 
    ms_filename = f"{event_name}_{source_name}.ms"
    save_filepath = join(download_dir, ms_filename)
 
    # Stale temp file from an interrupted run: delete and retry
    if os.path.exists(save_filepath):
        os.remove(save_filepath)
 
    try:
        boto3_res.Bucket(bucket).download_file(key, save_filepath)
        code = 'downloaded'
    except botocore.exceptions.ClientError as e:
        err_code = e.response['Error']['Code']
        if err_code == 'LimitExceededException':
            raise  # let the caller stop the run
        elif err_code in ('404', 'NoSuchKey'):
            code = 'no_data'
        else:
            code = 'failed'
            print(event_name, e)
            if err_log_path is not None:
                log_error(err_log_path, event_name,
                          f"download_aws_event ClientError '{err_code}' "
                          f"for s3://{bucket}/{key}", e)
    except Exception as e:
        code = 'failed'
        print(f"Error downloading {event_id} ({event_name}): {e}")
        if err_log_path is not None:
            log_error(err_log_path, event_name,
                      f"download_aws_event unexpected {type(e).__name__} "
                      f"for s3://{bucket}/{key}", e)
 
    return code

def update_status_file(df, filepath):
    """
    Update the status file with the current status of the dataframe df
    """
    status_columns = ['event_name', 'aws_n', 'aws_s', 'ntries']
    tmp = filepath + ".tmp"
    df[status_columns].to_csv(tmp, index=False)
    os.replace(tmp, filepath)

def clean_stream(st_downloaded, event_origin, program_params, event_name):
    """
    input a downloaded Obspy stream
    Perform the following actions:
    1) Remove traces farther than max_distance
    2) Remove traces with weird sampling rates
    3) Slice stream to desired time window

    event_origin should be (elon, elat, edep, edatetime)
    easily accessed in lookup dict: event_origins[event_name]

    need 'from obspy import Stream'

    event_name is used only for error logging.

    """
    errlog = program_params['err_log_path']
    disallowed_channel_prefixes = ["D", "C"]

    station_locations = program_params['station_locations']
    max_distance = program_params['max_distance']
    allowed_sample_rates = program_params['allowed_sample_rates']
    minimum_trace_length = program_params['minimum_trace_length']

    elon, elat, edep, edatetime = event_origin

    t1 = UTCDateTime(str(edatetime)) - program_params['pre_event_time']
    t2 = UTCDateTime(str(edatetime)) + program_params['post_event_time']

    st_clean = Stream()

    ntr = len(st_downloaded)
    dist = np.zeros(ntr, dtype=float)

    # 1) Filter out far traces
    for i, tr in enumerate(st_downloaded):
        try:
            channel_name = tr.get_id()
            channel_code = channel_name.split('.')[-1]
            if channel_code[0] not in disallowed_channel_prefixes:
                station_name = '.'.join(channel_name.split('.')[:2])
                try:
                    slon, slat, _ = station_locations[station_name]
                except KeyError:
                    slon, slat = 90.0, -60.0 # arbitrary location far from project
                dist[i] = haversine_np(elon, elat, slon, slat)
                tr.stats.distance = dist[i] * 1E3
                if dist[i] <= max_distance:
                    st_clean.append(tr.copy())
        except KeyError as e:
            log_error(errlog, event_name,
                      f"clean_stream KeyError for trace {tr.get_id()}", e)
        
        except Exception as e:
            log_error(errlog, event_name,
                      f"clean_stream error for trace {tr.get_id()}", e)
            raise

    
    # 2) Remove traces with weird sampling rates
    st_clean = Stream(
        [tr for tr in st_clean if tr.stats.sampling_rate in allowed_sample_rates]
    )

    # 3) Slice stream to desired time window
    st_clean.trim(t1, t2)

    st_clean = Stream(
        [tr for tr in st_clean if tr.stats['npts']*tr.stats['delta'] >= minimum_trace_length]
    )

    return st_clean

def log_error(errlog_path, event_name, context, exc):
    """
    Append an exception (with traceback) to the error log.

    Parameters
    ----------
    errlog_path : str
        Path to the error log file
    event_name : str
        Name of the event
    context : str
        Where the error occurred, e.g. a processing stage or trace ID
    exc : Exception
        The exception being handled (logged via the active traceback)
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(errlog_path, 'a') as f:
        f.write(f"\n[{timestamp}] Error processing event {event_name}: {context}\n")
        f.write(traceback.format_exc())
        f.write("-" * 60 + "\n")

# %% 
# Parameters 

params = cfg['aws_downloads']

# === USER CONFIG ===
# Don't download events below mag_range[0]
mag_range                   = params['mag_range']
# Events between dates times will be downloaded if available ("YYYY-MM-DD")
starttime                   = params['starttime']
endtime                     = params['endtime']
# Instruments with other sampling rates aren't typically seismometers.
# 0.0 is kept intentionally: some stations report 0.0 due to metadata errors
# but the underlying instrument and data appear to be fine.
allowed_sample_rates        = params['allowed_sample_rates']
# Only use instruments with a vertical component
allowed_dips                = params['allowed_dips']
# Channels with D and C prefixes aren't seismometers (?)
disallowed_channel_prefixes = params['disallowed_channel_prefixes']
# Define the time window to slice waveforms at, relative to origin time
pre_event_time              = params['pre_event_time']
post_event_time             = params['post_event_time']
# Only use events within max_distance km
max_distance                = params['max_distance'] # km
# Basic data cleaning
minimum_trace_length        = params['minimum_trace_length']
# Set to True if you want to retry failed downloads
retry_failed_downloads      = params['retry_failed_downloads']

# ---------- Don't change anything below this line ----------
project_dir                 = cfg['paths']['project_dir']
catalogs_dir                = cfg['paths']['catalogs_dir']
figure_dir                  = cfg['paths']['figure_dir']

waveforms_dir               = cfg['paths']['waveforms_dir']
waveforms_temp_dir          = cfg['paths']['waveforms_temp_dir']

# This contains all earthquakes in the catalog, not necessarily ones I want
# to use.
eq_catalog_path             = cfg['paths']['eq_catalog_path']
station_catalog_path        = cfg['paths']['station_catalog_path']

err_log_path                = join(waveforms_dir, "get_aws_waveforms_error_log.txt")

# Throw an error if important dirs don't exist yet:
for d in [catalogs_dir, project_dir]:
    if not os.path.exists(d):
        raise Exception(f"Directory {d} does not exist. Please check your paths.")

# Verify catalogs exist
for c in [eq_catalog_path, station_catalog_path]:
    if not os.path.isfile(c):
        raise Exception(f"Catalog {c} does not exist. Please check your paths.")

# Make other directories if they don't exist
for d in [waveforms_temp_dir, waveforms_dir, figure_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

print("--- INPUT PARAMETERS ---")
print(f"Start time: {starttime}")
print(f"End time: {endtime}")
print(f"Magnitude range: M{mag_range[0]} - M{mag_range[1]}")
print("\n", end='')

print("--- WAVEFORM INFORMATION ---")
print("Acceptable sample rates: ", end='')
for sr in allowed_sample_rates: print(f"{sr}", end=', ')
print("\n", end='')
print(f"Waveform begins {pre_event_time} seconds before event origin and ends {post_event_time} seconds after.")
print(f"Station-event distance must be less than {max_distance} km")
print("\n", end='')
print("--- PATHS AND URLS ---")
print(f"Waveform storage location: {waveforms_dir}")
print(f"Temporary waveform download location: {waveforms_temp_dir}")
print(f"Error log: {err_log_path}")
print(f"Figure save location: {figure_dir}")
print("\n", end='')

# %%
# Load event catalog and filter out too small events, create event_name
print("Reading earthquake catalog...", end="")
eq_df = pd.read_csv(eq_catalog_path)
len0 = len(eq_df)
print(f"Done. {len0:,} events loaded.")
eq_df = eq_df[eq_df['emag'] >= mag_range[0]].reset_index(drop=True)
eq_df['source'] = eq_df['event_name'].str[0]
eq_df['edatetime'] = pd.to_datetime(eq_df['edatetime']) 
print(f"{len0 - len(eq_df):,} events with magnitude < {mag_range[0]} discarded. {len(eq_df):,} events remaining.")

assert len(np.unique(eq_df['event_name'].values)) == len(eq_df['event_name'].values), "event_name must be unique"

# Filter out events before start time and after end time
len0 = len(eq_df)
m = (eq_df['edatetime'] >= pd.to_datetime(starttime)) & (eq_df['edatetime'] <= pd.to_datetime(endtime))
eq_df = eq_df[m].reset_index(drop=True)
print(f"{len0 - len(eq_df):,} events outside of time range {starttime} to {endtime} discarded. {len(eq_df):,} events remaining.")

print("Reading station catalog...", end="")
all_sta_catalog = pd.read_csv(station_catalog_path, sep='|', skiprows=1)
if 'index' in all_sta_catalog.columns: all_sta_catalog = all_sta_catalog.drop('index', axis=1)
print(f"Done. {len(all_sta_catalog):,} unique stations loaded with columns {list(all_sta_catalog.columns)}")

# Construct dict for station locations station_name: (slon, slat, sele)
station_locations = dict(zip(all_sta_catalog['station_name'], zip(all_sta_catalog['slon'], all_sta_catalog['slat'], all_sta_catalog['sele'])))
# Now, station locations can be accessed quickly using the station_name:
# slat, slon, sele = station_locations[station_name]

# Make a lookup dict for event origins event_name: (elon, elat, edep, edatetime)
event_origins = dict(zip(eq_df['event_name'], zip(eq_df['elon'], eq_df['elat'], eq_df['edep'], eq_df['edatetime'])))

program_params = {
    'waveforms_dir': waveforms_dir,
    'eq_catalog': eq_df, 
    'allowed_sample_rates': allowed_sample_rates,
    'pre_event_time': pre_event_time,
    'post_event_time': post_event_time,
    'max_distance': max_distance,
    'STATUS': STATUS,
    'event_origins': event_origins,
    'station_locations': station_locations,
    'disallowed_channel_prefixes': disallowed_channel_prefixes,
    'waveforms_temp_dir': waveforms_temp_dir,
    'err_log_path': err_log_path,
    'minimum_trace_length': minimum_trace_length
}

# --- status file handling ---
# First, check for status file in waveforms_dir
status_filepath = join(waveforms_dir, 'dataset_status.csv')

# Try to load a current status file
if os.path.isfile(status_filepath):
    print(f"Found existing status file {status_filepath}. Loading...")
    df_status = pd.read_csv(status_filepath)
    assert df_status['event_name'].isin(eq_df['event_name']).all(), "Status file contains events not in the catalog. Delete the status file and try again."
    
    eq_df = pd.merge(eq_df, df_status, how='left', on='event_name')
    assert not eq_df['aws_n'].isna().any(), "Catalog contains events not in the status file. Delete the status file and try again."
    print("Done.")

# Otherwise, make a new status file
else:
    print("No status file found. Making a new one...")
    eq_df['aws_n'] = STATUS['new']
    eq_df['aws_s'] = STATUS['new']
    eq_df['ntries'] = STATUS['new']

    # save as .csv
    update_status_file(eq_df, status_filepath)
    print(f"Done. Status file saved to {status_filepath}.")


# %%
# Process each event. First, check status codes in eq_df. Then check if 
# there are downloaded files. If not, download from the following using 
# this logic:
# 1) if source=='n', download from AWS NCEDC (bucket = 'ncedc-pds'). 
#   Do nothing for source=='s'. suffix: '_ncedc.ms'
# 2) if source=='s', download from AWS SCEDC (bucket = 'scedc-pds'). 
#   Do nothing for source=='n'. suffix: '_scedc.ms'
# 3) Update status codes in eq_df and save to status file after n_save_status events.

boto3_res = boto3.resource(
            's3', 
            config=Config(signature_version=UNSIGNED)
        )

nevents = len(eq_df)

n_save_status = 100 # save status file every n_save_status events
sym_success = "\u2705"
sym_error = "\u274C"
yr, mon = -1, -1
n_updates = 0 # counts every status change, not just successful downloads

for ievent in range(nevents):
    # Save at the top of the loop so 'continue' paths can't skip it;
    # resetting the counter avoids repeated full-CSV writes
    if n_updates >= n_save_status:
        update_status_file(eq_df, status_filepath)
        n_updates = 0
    ev = eq_df.loc[ievent].copy()

    # This simply prints a status message by checking if year or month is 
    # different from the previous event
    if ev['edatetime'].year != yr or ev['edatetime'].month != mon:
        yr, mon = ev['edatetime'].year, ev['edatetime'].month
        print(f"Processing year {yr} month {mon}")

    # Check the status log if the event has already been downloaded
    if (ev['aws_n'] == STATUS['downloaded']) or (ev['aws_s'] == STATUS['downloaded']):
        continue

    # Check if the event has been attempted; skip or keep going based on retry_failed_downloads
    # any iteration that passes this will be attempted
    if (ev['aws_n'] != STATUS['new']) or (ev['aws_s'] != STATUS['new']):
        if retry_failed_downloads:
            pass
        else:
            continue
    
    # try to download the event, update status, save to disk
    try:
        event_name  = ev['event_name']
        edatetime   = ev['edatetime']
        source      = ev['source']

        # check for existing files
        event_partial_dir = get_waveform_path(event_name, edatetime)
        event_waveform_dir = join(waveforms_dir, event_partial_dir)
        if not os.path.exists(event_waveform_dir):
            os.makedirs(event_waveform_dir, exist_ok=True)
        
        # check if files exist, update status codes
        existing = os.listdir(event_waveform_dir)
        if event_name+'_ncedc.ms' in existing:
            ev['aws_n'] = STATUS['downloaded']
            eq_df.loc[ievent] = ev
            n_updates += 1
            # print("WARNING: this should not happen")
            continue
        if event_name+'_scedc.ms' in existing:
            ev['aws_s'] = STATUS['downloaded']
            eq_df.loc[ievent] = ev
            n_updates += 1
            # print("WARNING: this should not happen")
            continue

        ev['ntries'] = ev['ntries'] + 1

        # Try to download event
        if source == 'n': # and ev['aws_n'] == STATUS['new']:
            code = download_aws_event(event_name, edatetime, waveforms_temp_dir, boto3_res, err_log_path=err_log_path)
            # if no data, try .mseed extension
            if code=='no_data':
                code2 = download_aws_event(event_name, edatetime, waveforms_temp_dir, boto3_res, ext='mseed', err_log_path=err_log_path)
                if code2 != 'no_data':
                    code = code2
            ev['aws_n'] = STATUS[code]
            ev['aws_s'] = STATUS['redundant']
        elif source == 's': # and ev['aws_s'] == STATUS['new']:
            code = download_aws_event(event_name, edatetime, waveforms_temp_dir, boto3_res, err_log_path=err_log_path)
            # if no data, try .mseed extension
            if code=='no_data':
                code2 = download_aws_event(event_name, edatetime, waveforms_temp_dir, boto3_res, ext='mseed', err_log_path=err_log_path)
                if code2 != 'no_data':
                    code = code2
            ev['aws_s'] = STATUS[code]
            ev['aws_n'] = STATUS['redundant']

        # Read the downloaded data
        temp_files = [el for el in os.listdir(waveforms_temp_dir)
                        if el.startswith(f"{event_name}_")]
        if len(temp_files) > 1:
            raise ValueError(f"More than one file for event {event_name} in {waveforms_temp_dir}")
        if len(temp_files) == 0:
            # nothing downloaded (no_data/failed); status already set above
            eq_df.loc[ievent] = ev
            n_updates += 1
            continue
        else:
            try:
                st_aws = obspy.read(join(waveforms_temp_dir, temp_files[0]), format="MSEED")
            except (ObsPyMSEEDFilesizeTooSmallError, InternalMSEEDError) as e:
                ev[f"aws_{source}"] = STATUS['failed']
                print(f"Failed to load {temp_files[0]}: {type(e).__name__} (logged)")
                log_error(err_log_path, event_name,
                          f"obspy.read failed for {temp_files[0]}", e)
                os.remove(join(waveforms_temp_dir, temp_files[0]))
                eq_df.loc[ievent] = ev
                n_updates += 1
                continue

        # clean stream, resave data
        st_clean = clean_stream(st_aws, event_origins[event_name], program_params, event_name)
        
        # save to disk
        if source == 'n':
            source_long = 'ncedc'
        elif source == 's':
            source_long = 'scedc'
        output_filepath = join(event_waveform_dir, f"{event_name}_{source_long}.ms")
        if len(st_clean) > 0:
            st_clean.write(output_filepath, format='MSEED')
        else:
            ev[f"aws_{source}"] = STATUS['no_data']

        # remove temp files
        for temp_file in temp_files:
            os.remove(join(waveforms_temp_dir, temp_file))

        # update eq_df
        eq_df.loc[ievent] = ev
        n_updates += 1

        # print status line
        aws_status = sym_success if ev[f'aws_{source}'] == STATUS['downloaded'] else sym_error
        aws_status_line = f"AWS_{source}: [{aws_status}]"
        status = f"{event_name:>10}  {str(edatetime.date()):>10} " \
            f"{aws_status_line}"
        print(status)

    except KeyboardInterrupt:
        # persist statuses before stopping so downloaded events aren't left 'new'
        update_status_file(eq_df, status_filepath)
        raise

    except Exception as e:
        if isinstance(e, botocore.exceptions.ClientError) and \
                e.response['Error']['Code'] == 'LimitExceededException':
            update_status_file(eq_df, status_filepath)  # don't lose progress on throttle stop
            raise  # AWS throttling: stop rather than burn through the catalog
        print(f"Error processing event {ev['event_name']}: {e} (logged, continuing)")
        log_error(err_log_path, ev['event_name'], "main download/processing loop", e)
        # mark as failed so this event isn't retried as 'new' on resume
        ev[f"aws_{ev['source']}"] = STATUS['failed']
        eq_df.loc[ievent] = ev
        n_updates += 1

# Final save so the last chunk of events isn't lost
update_status_file(eq_df, status_filepath)

# %%
# Validate the downloaded data
from tqdm import trange

df = eq_df.copy().drop(['aws_n', 'aws_s', 'ntries'], axis=1)

# Check for downloaded waveform files per event
has_data = np.zeros(len(df), dtype=int)
event_names = df['event_name'].values
edatetimes = df['edatetime']
for i in trange(len(df)):
    event_name = event_names[i]
    event_waveform_dir = join(waveforms_dir, get_waveform_path(event_name, edatetimes[i]))
    if not os.path.isdir(event_waveform_dir):
        continue
    if any(event_name in el for el in os.listdir(event_waveform_dir)):
        has_data[i] = 1

df['has_data'] = has_data

# Summary stats by source
for src, label in [('s', 'SCEDC'), ('n', 'NCEDC')]:
    sub = df[df['source'] == src]
    n = len(sub)
    n_aws = sub['has_data'].sum()
    print(f"{label}: {n} events, {n_aws} with AWS data ({100 * n_aws / n:.2f}%)")

# Assign each event an integer day index relative to the catalog start
t = df['edatetime'].astype('datetime64[ns]')
t0 = t.min().normalize()  # midnight of the first day
df['day_idx'] = ((t - t0) // pd.Timedelta(days=1)).astype(int)
ndays = int(df['day_idx'].max()) + 1

# Decimal year, including time of day, reused for both ratio and event scatters
def to_decimal_year(times):
    year_start = pd.to_datetime(times.dt.year.astype(str) + "-01-01")
    year_end = pd.to_datetime((times.dt.year + 1).astype(str) + "-01-01")
    return times.dt.year + (times - year_start) / (year_end - year_start)

def daily_ratio(sub):
    """sum(has_data)/count per day, indexed 0..ndays-1, NaN where no events."""
    g = sub.groupby('day_idx')['has_data']
    return (g.sum() / g.size()).reindex(range(ndays)).values

df['dyear'] = to_decimal_year(t)

# Per-day bin centers in decimal years for the ratio series
bin_years = to_decimal_year(pd.Series(t0 + pd.to_timedelta(np.arange(ndays), unit='D')))

# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True,
                        layout='constrained')

panels = [
    (df,                       "All events",           "#1f77b4"),
    (df[df['source'] == 'n'],  "NCEDC (source = 'n')", "#2ca02c"),
    (df[df['source'] == 's'],  "SCEDC (source = 's')", "#d62728"),
]

for ax, (sub, title, color) in zip(axs, panels):
    # Left axis: fraction of events with downloaded data per day
    ratio = daily_ratio(sub)
    ax.scatter(bin_years, ratio, s=4, color=color, alpha=1, edgecolors='none',
               zorder=3)
    ax.plot(bin_years, ratio, color=color, lw=0.5, zorder=3)
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines[['top']].set_visible(False)
    ax.ticklabel_format(useOffset=False, axis='x')

    # Right axis: magnitude vs origin time for this source
    ax2 = ax.twinx()
    ax2.scatter(sub['dyear'], sub['emag'], s=1, color='k', alpha=0.5,
                edgecolors='none', zorder=1)
    ax2.set_ylabel("Magnitude", color='0.4')
    ax2.tick_params(axis='y', labelcolor='0.4')
    ax2.set_zorder(ax.get_zorder() - 1)  # keep ratio points visually on top
    ax.patch.set_visible(False)          # let the twin axis show through

axs[-1].set_xlabel("Year")
fig.supylabel("Fraction of events with downloaded data")
plt.savefig(join(figure_dir, "aws_data_ratio_by_source.png"), dpi=600)
plt.show()

print("Done. Figure saved to ", join(figure_dir, "aws_data_ratio_by_source.png"))
