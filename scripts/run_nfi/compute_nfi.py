# %%
import time

t0_script = time.time()
import numpy as np
import pandas as pd

import sys
import os
from os.path import join
from tqdm import trange, tqdm

# --- Load config --- #
from pathlib import Path

config_path = str(Path(__file__).resolve().parents[2] / "config.yaml")
lib_dir = str(Path(__file__).resolve().parents[2] / "lib")
sys.path.insert(0, lib_dir)
from helper_functions import load_config

cfg = load_config(config_path)
cfg_paths = cfg["paths"]
# --- End load config --- #

# --- Import nfi --- #
nfi_path = cfg_paths['local_nfi_dir']
if nfi_path != "":
    sys.path.insert(0, nfi_path)
from nfi import nFIEstimator
import nfi

# %%
# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

# --- Run name ---
# Label for this run. Used to name output directories and files so results from
# different parameter choices are kept separate and reproducible.
run_name = "run_22"

# --- Signal-to-noise threshold ---
# Require the signal-to-noise ratio to be greater than or equal to stn_req
# in a defined frequency band, stn_band.
# STN = median(signal spectrum) / median(noise spectrum) in the stn_band. 
stn_req = 3.0
stn_band = (2.667, 6.0)

# --- Nyquist frequency ---
# Set this to half the sampling rate. For the Ridgecrest and California
# datasets, all waveforms are 100 Hz (or resampled to 100 Hz), so f_nyquist
# is set to 50.
f_nyquist = 50.0

# --- Magnitude correction method ---
# Which method to use for the magnitude correction step. "smoothedspline" is
# the default, and applies a correction (not actually a spline) to all the 
# data at once. "source-separate" applied the same correction, but separately
# to events with different source columns (ex: SCEDC and NCEDC events will
# have different corrections, as explained in the paper).
mag_corr_method = "source-separate"

# --- Cylindrical calibration parameters ---
# Search method "cylinder" (default):
#   Searches in a cylinder shape around the target event (within radius 
#   calib_hdist_max and depth +/- calib_zdist_max). 
calib_search_method = "cylinder"
calib_hdist_max = 10.0
calib_zdist_max = 2.0

# --- Nearest-neighbor calibration parameters ---
# Only used when calib_search_method = "nearest-neighbor". calib_nn sets the
# number of nearest calibration events to select for each target event.
# calib_depth_scale weights vertical distance relative to horizontal distance
# when computing neighbor proximity, so a value of 5 makes 1 km of depth
# separation count as 5 km of epicentral distance (compensating for the
# poorer resolution of depth relative to horizontal location).
# calib_search_method         = "nearest-neighbor"
# calib_nn                    = 10
# calib_depth_scale           = 5

# --- Time-filtered calibration ---
# Restrict calibration events to those occurring near the target event in time.
# Set calib_time_filter = False to disable time filtering entirely (calibration
# events are drawn from the full catalog regardless of origin time).
# calib_time_method "interval": keep calibration events within +/- 
# calib_time_ndays of the target event's origin time.
# calib_time_ndays: half-width of the time window in days (180 = +/- ~6 months).
# No other methods implemented yet.
calib_time_filter = False
calib_time_method = "interval"
calib_time_ndays = 180

# --- Parallelization ---
# Number of worker processes for the parallel processing pool. Tune to 
# available resources.
n_workers = 16

# --- Gain correction ---
# If True, use gain-corrected waveforms. This uses the spectra corrected for 
# undocumented network gain changes. If False, use the raw counts as stored.
# This doesn't affect the final results.
use_gain_corrected = True

# --- Channel selection ---
# Restrict processing to a single channel code (ex: "HHZ"). Set to "" to process
# all available channels.
channel = ""

# --- Test mode ---
# Set N_test = 0 to process the full catalog. Set N_test = N (a positive integer)
# to process only the first N records, for quick testing and debugging of the
# pipeline before a full run.
N_test = 0

# --- Spatial subsetting ---
# Restrict processing to events within a geographic bounding box. Each is a
# (min, max) tuple in decimal degrees; set to None to process the full extent.
# This is mostly used for testing.
lat_range = None
lon_range = None

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
project_dir = cfg_paths["project_dir"]

p_results_dir = join(project_dir, "results", run_name)  # Parent output directory
catalogs_dir = join(project_dir, "catalogs")  # Catalog directory
arrays_dir = join(
    project_dir, "proc", "compspec"
)  # Directory containing s1.npy, s2.npy, etc. files

if use_gain_corrected:
    arrays_dir = join(arrays_dir, "gain_corrected/")

if os.path.exists(p_results_dir):
    print(f"Directory {p_results_dir} already exists. Continue? (y/n)")
    if input() != "y":
        sys.exit()

# Make output directories if they don't exist
for pth in [p_results_dir]:
    if not os.path.exists(pth):
        os.makedirs(pth)

# print out parameters summary and paths summary with descriptions
print("Parameters:")
print(f"  stn_req: {stn_req}")
print(f"  stn_band: {stn_band[0]} - {stn_band[1]} Hz")
print(f"  f_nyquist: {f_nyquist} Hz")
print("\nPaths:")
print(
    f"  run_name: {run_name} - Name of this analysis run, used for organizing outputs."
)
print(f"  project_dir: {project_dir} - Base directory for the project.")
print(
    f"  p_results_dir: {p_results_dir} - Directory where {run_name} results will be saved."
)
print(
    f"  catalogs_dir: {catalogs_dir} - Directory containing earthquake and station catalogs."
)
print(
    f"  arrays_dir: {arrays_dir} - Directory containing precomputed arrays (s1.npy, s2.npy, etc.) for nFI computation."
)

if N_test != 0:
    print("\n")
    print(f"DEBUGGING MODE ACTIVE. Test run with {N_test} records.")

# %%
# ----------------------------------------------------------------
# Load catalogs
# ----------------------------------------------------------------
t0 = time.time()
print("Reading catalogs...", end="")

eq_df = pd.read_csv(join(catalogs_dir, "event_catalog.csv"))

if lat_range is not None and lon_range is not None:
    print(f"Filtering by lat/lon range: {lat_range}, {lon_range}")
    l0 = len(eq_df)
    eq_df = eq_df[
        (eq_df["elat"] >= lat_range[0])
        & (eq_df["elat"] <= lat_range[1])
        & (eq_df["elon"] >= lon_range[0])
        & (eq_df["elon"] <= lon_range[1])
    ].reset_index(drop=True)
    print(
        f"Removed {l0-len(eq_df):,} events based on lat/lon range. Remaining: {len(eq_df):,} events."
    )
else:
    print("No lat/lon filtering applied.")


sta_catalog = pd.read_csv(
    join(catalogs_dir, "station_catalog.csv"), sep="|", skiprows=1
)

print(
    f"Done. {len(eq_df):,} events, {len(sta_catalog):,} stations. "
    f"({time.time()-t0:.2f}s)"
)


# %%
# ----------------------------------------------------------------
# Load arrays — use mmap_mode='r' to avoid reading everything into RAM
# up front. We'll only materialize the rows we actually keep.
# ----------------------------------------------------------------
t0 = time.time()
print("Loading spectra arrays (mmap)...", end="")

spectra_mmap = np.load(join(arrays_dir, "s2.npy"), mmap_mode="r")
noise_mmap = np.load(join(arrays_dir, "s1.npy"), mmap_mode="r")
event_names = np.load(join(arrays_dir, "event_names.npy"), allow_pickle=True)
channel_names = np.load(join(arrays_dir, "channel_names.npy"), allow_pickle=True)
deldist = np.load(join(arrays_dir, "deldist.npy"))

if N_test != 0:
    spectra_mmap = spectra_mmap[:N_test, :]
    noise_mmap = noise_mmap[:N_test, :]
    event_names = event_names[:N_test]
    channel_names = channel_names[:N_test]
    deldist = deldist[:N_test]


nf = spectra_mmap.shape[1]
f = np.linspace(0, f_nyquist, nf)

print(f"Done. Shape: {spectra_mmap.shape}. ({time.time()-t0:.2f}s)")

# %%

# print("finding where event_names change")
# change_inds = np.append(0, np.where(event_names[1:] != event_names[:-1])[0] + 1)
# print("finding unique event_names")
# # unique_event_names = np.unique(event_names)
# # assert len(unique_event_names) == len(change_inds), 'uh oh'

# diffs = np.diff(np.append(change_inds, len(event_names)))
df_event_names = set(eq_df["event_name"].values)

print("building inds array")
evinds_in_df = np.array(
    [i for i in range(len(event_names)) if event_names[i] in df_event_names]
)

# filter the arrays to only include the events we care about
print("filtering arrays")
spectra_mmap = spectra_mmap[evinds_in_df, :]
noise_mmap = noise_mmap[evinds_in_df, :]
event_names = event_names[evinds_in_df]
channel_names = channel_names[evinds_in_df]
deldist = deldist[evinds_in_df]
print("done")

# Filter out by channel if specified
if channel != "":
    print(f"Filtering by channel: {channel}")
    l0 = len(spectra_mmap)
    channel_codes = np.char.rpartition(channel_names.astype("U"), ".")[:, -1]
    mask = channel_codes == channel
    spectra_mmap = spectra_mmap[mask]
    noise_mmap = noise_mmap[mask]
    event_names = event_names[mask]
    channel_names = channel_names[mask]
    deldist = deldist[mask]
    print(
        f"Removed {l0-len(spectra_mmap):,} records based on channel. Remaining: {len(spectra_mmap):,} records."
    )

# %%

# ----------------------------------------------------------------
# Build metadata DataFrame — single merge pass
# ----------------------------------------------------------------
t0 = time.time()
print("Building metadata DataFrame...", end="")

df = pd.DataFrame(
    {
        "event_name": event_names,
        "channel_name": channel_names,
        "deldist": deldist,
    }
)

# Extract station_name for the merge, then drop it
df["station_name"] = df["channel_name"].str.rsplit(".", n=2).str[0]
df = df.merge(sta_catalog, how="left", on="station_name")
df.drop(columns=["station_name"], inplace=True)

# Merge event metadata (only the columns we need)
eq_cols_needed = [
    "event_name",
    "edatetime",
    "etype",
    "emag",
    "emagtype",
    "elat",
    "elon",
    "edep",
]
eq_cols_available = [c for c in eq_cols_needed if c in eq_df.columns]
df = df.merge(eq_df[eq_cols_available], how="left", on="event_name")

print(f"Done. {len(df):,} records. ({time.time()-t0:.2f}s)")


# %%
# ----------------------------------------------------------------
# Single-pass STN filtering
#
# Instead of: copy spectra → index with keep_inds → compute STN →
#             filter → index again with new keep_inds → copy again
# We: compute STN on the mmap'd arrays → build one final boolean mask
#     → index spectra/noise exactly once into a contiguous copy.
# ----------------------------------------------------------------
t0 = time.time()
print("Computing STN and filtering...", end="")

stn_i1 = np.argmin(np.abs(f - stn_band[0]))
stn_i2 = np.argmin(np.abs(f - stn_band[1]))

# Compute STN directly on the (possibly mmap'd) arrays.
# np.median on mmap'd arrays will read only the columns in the slice,
# which is fine — the bottleneck is I/O, not computation.
nrows = len(df)
stn_values = np.empty(nrows, dtype=np.float64)

# For very large arrays, computing median on the full array at once is
# faster than row-by-row, and numpy handles mmap transparently.
sig_band = spectra_mmap[:nrows, stn_i1 : stn_i2 + 1]  # triggers read of just these cols
noi_band = noise_mmap[:nrows, stn_i1 : stn_i2 + 1]
stn_values = np.median(sig_band, axis=1) / np.median(noi_band, axis=1)

# Build the keep mask
keep_mask = stn_values >= stn_req
n_kept = keep_mask.sum()

# Filter DataFrame
df_p = df.loc[keep_mask].reset_index(drop=True)

# Index spectra/noise ONCE — this is the only full copy we make.
# np.where gives sorted indices, which helps with sequential disk reads
# on mmap'd arrays.
keep_indices = np.where(keep_mask)[0]
p_spectra = np.array(spectra_mmap[keep_indices, :])  # contiguous copy
# noise not needed downstream — don't copy it
# p_noise = np.array(noise_mmap[keep_indices, :])

# Release mmap references
del spectra_mmap, noise_mmap, sig_band, noi_band, stn_values

print(f"Done. {n_kept:,}/{nrows:,} records kept. ({time.time()-t0:.2f}s)")

# %%
# ----------------------------------------------------------------
# Final column cleanup
# ----------------------------------------------------------------
if "sele" in df_p.columns:
    df_p.rename(columns={"sele": "selev"}, inplace=True)

# Drop columns that nFIEstimator doesn't need
cols_to_drop = [c for c in ["sdep", "source", "emagtype", "nst"] if c in df_p.columns]
if cols_to_drop:
    df_p.drop(columns=cols_to_drop, inplace=True)


# Convert high-cardinality string columns to categoricals to save memory.
# event_name and channel_name are repeated many times across records.
for col in ["event_name", "channel_name"]:
    if col in df_p.columns:
        df_p[col] = df_p[col].astype("category")

print(
    f"\nFinal: {len(df_p):,} records, "
    f"{df_p['event_name'].nunique():,} events, "
    f"{df_p['channel_name'].nunique():,} channels"
)
print(f"Spectra array: {p_spectra.shape}, {p_spectra.nbytes/1e9:.2f} GB")
print(f"DataFrame memory: {df_p.memory_usage(deep=True).sum()/1e9:.2f} GB")

# %%
# ----------------------------------------------------------------
# Run nFIEstimator
# ----------------------------------------------------------------

p_results = nFIEstimator(
    df_records=df_p,
    spectra=p_spectra,
    f=f,
    save_dir=p_results_dir,
    compute_uncertainty=True,
    calib_search_method=calib_search_method,
    calib_hdist_max=calib_hdist_max,
    calib_zdist_max=calib_zdist_max,
    n_workers=n_workers,
    calib_time_filter=calib_time_filter,
    calib_time_method=calib_time_method,
    calib_time_ndays=calib_time_ndays,
    mag_corr_method=mag_corr_method,
)
p_results = p_results.compute(recompute=True)

print(f"Done. Total time: {time.time()-t0_script:,.2f}s")
