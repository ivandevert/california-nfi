# %% [markdown]
# formerly compspec.ipynb
# Todo: update to use hdf5 instead of .mspec

# %%

# --- Built-in libraries --- #
import time
import os
from os import listdir
from os.path import join, listdir, isdir
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from glob import glob

# --- Third-party libraries --- #
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import obspy
from obspy import read
import scipy
import scipy.signal as signal
import multitaper.utils as utils
from obspy import UTCDateTime as UTC
from obspy import read_inventory
import msgpack
import msgpack_numpy

# --- Load config --- #
import sys
from pathlib import Path
config_path = str(Path(__file__).resolve().parents[2] / "config.yaml")
lib_dir = str(Path(__file__).resolve().parents[2] / "lib")
sys.path.insert(0, lib_dir)
from helper_functions import load_config
cfg = load_config(config_path)
# --- End load config --- #

# %%
# # Program paths & parameters
cfg_paths = cfg["paths"]
cfg_params = cfg["compute_spectra"]

# Project directory
project_dir = cfg_paths["project_dir"]

# Directory for processing files (file lists, etc.)
proc_dir = join(project_dir, "proc/compspec/")

# Parent directory of waveforms. Individual event miniseed files are at:
#    <waveforms_dir>/<year>/<jday>/<event_name>_<method>.ms
waveforms_dir = cfg_paths["waveforms_dir"]

catalogs_dir = cfg_paths["catalogs_dir"]

# Parent directory of phase files. Monthly .miniphase files are at:
#    <phase_dir>/<full_source>/processed/<year>.<month>.miniphase
phase_dir = join(catalogs_dir, "phase/combined/")

# Directory holding station response .xml files
response_dir = join(catalogs_dir, "response/xml/")

# Path of the entire earthquake catalog
eq_catalog_path = cfg_paths["eq_catalog_filepath"]

# Might not need all these; check later
channel_catalog_path = join(catalogs_dir, "channel_catalog.csv")
station_catalog_path = join(catalogs_dir, "station_catalog.csv")

spectra_save_dir = join(proc_dir, "spectra/")

# Program parameters
t_buffer = cfg_params["t_buffer"]
t_window_desired = cfg_params["t_window_desired"]
sampling_rate = cfg_params["sampling_rate"]
phase = cfg_params["phase"]
allowed_components = cfg_params["allowed_components"]
allow_simple_correct = cfg_params["allow_simple_correct"]

dt = 1 / sampling_rate
nwin = int(t_window_desired * sampling_rate)
nf = int(nwin / 2 + 1)
f_nyquist = sampling_rate / 2
#


# Multitaper parameters
nw = cfg_params["nw"]
kspec = cfg_params["kspec"]
nfft = int(2 * t_window_desired * sampling_rate + 1)  # number of FFT points to pad to
iadapt = cfg_params["iadapt"]

# variable time window
min_twin = cfg_params["min_twin"]
max_twin = cfg_params["max_twin"]
d_twin = cfg_params["d_twin"]
M_min_twin = cfg_params["M_min_twin"]
M_max_twin = cfg_params["M_max_twin"]

for d in [proc_dir, spectra_save_dir]:
    if not isdir(d):
        os.makedirs(d)

# %%
# Number of time window lengths
ntwin = int((max_twin - min_twin) / d_twin + 1)
# Time window lengths
twins = np.linspace(min_twin, max_twin, ntwin)
nwins = (twins * sampling_rate).astype(int)
nffts = (2 * twins * sampling_rate + 1).astype(int)

# Precompute slepian tapers
vn = [[]] * ntwin
lamb = [[]] * ntwin
for i in range(ntwin):
    vn[i], lamb[i] = utils.dpss(nwins[i], nw, kspec)

f_resample = np.linspace(0, f_nyquist, nf)


# %%
program_params = {
    "waveforms_dir": waveforms_dir,
    "spectra_save_dir": spectra_save_dir,
    "min_twin": min_twin,
    "max_twin": max_twin,
    "d_twin": d_twin,
    "M_min_twin": M_min_twin,
    "M_max_twin": M_max_twin,
    "t_buffer": t_buffer,
    "t_window_desired": t_window_desired,
    "sampling_rate": sampling_rate,
    "nw": nw,
    "kspec": kspec,
    "nfft": nfft,
    "iadapt": iadapt,
    "nffts": nffts,
    "f_resample": f_resample,
    "vn": vn,
    "lamb": lamb,
    "allowed_components": allowed_components,
}

# %%
# Function definitions


# assign approximate p and s arrival times based on travel time tables
def mtspec_helper(x, nw, kspec, dt, nfft, iadapt, vn, lamb):
    # assumptions:
    # 1) x is real
    # 2) nfft is odd (nwin is even)

    npts = len(x)

    x = x[:, np.newaxis]

    nf = int((nfft + 1) / 2)
    f = scipy.fft.fftfreq(nfft, dt)
    df = f[1] - f[0]

    x = signal.detrend(x, axis=0, type="constant")
    xvar = np.var(x)

    yk, sk = utils.eigenspec(x, vn, lamb, nfft)

    spec, se, wt = utils.adaptspec(yk, sk, lamb, iadapt)
    sscal = np.sum(spec) * df
    sscal = xvar / sscal
    spec = sscal * spec

    freq = np.zeros((nf, 1), dtype=float)
    freq = f[0:nf]

    spec = spec[0:nf, 0]
    spec[0:nf] = 2.0 * spec[0:nf]
    spec[0] = 0.5 * spec[0]

    return freq, spec


# simplified variable time window length: assign time window lengths to
# each record based on emag. emag <=M_min_twin should have t_window = M_min_twin
# emag >= M_max_twin should have t_window = M_max_twin, emag between should be
# assigned a t_window between M_min_twin and M_max_twin
def get_time_window_length(emag, min_twin, max_twin, M_min_twin, M_max_twin, d_twin):
    ntwin = int((max_twin - min_twin) / d_twin + 1)
    if emag <= M_min_twin:
        return min_twin, 0
    elif emag >= M_max_twin:
        return max_twin, int(ntwin - 1)
    else:
        dMwin = (M_max_twin - M_min_twin) / (ntwin - 2)
        n = np.ceil((emag - M_min_twin) / dMwin)
        t_window = d_twin * n + min_twin
        return t_window, int(n)


def read_mphase(mphase_filepath):
    return json.load(open(mphase_filepath, "r"))


def load_picks_from_dir(phase_dir):
    """
    read all .mphase files in phase_dir and return a pick_dict:
    pick_dict[event_name] = [edatetime, [stnames], [phases], [picks], [qualities]]
    """

    phase_filepaths = [el for el in listdir(phase_dir) if el.endswith(".mphase")]
    phase_filepaths.sort()

    n_events_picked = 0
    n_picks = 0
    pick_dict = {}
    for f in tqdm(phase_filepaths, desc="Loading picks into memory"):
        D = read_mphase(join(phase_dir, f))
        pick_dict.update(D)
        n_events_picked += len(D)
        n_picks += sum([len(v[3]) for k, v in D.items()])
    print(
        f"Done. Picks for {len(pick_dict):,} events loaded from {len(phase_filepaths):,} .mphase files."
    )
    return pick_dict, n_events_picked, n_picks


def attach_distance_to_stream(st, elon, elat, station_locations_dict):
    for i, tr in enumerate(st):
        channel_name = tr.get_id()
        station_name = ".".join(channel_name.split(".")[:2])
        slon, slat, _ = station_locations_dict[station_name]
        dist = haversine_np(elon, elat, slon, slat)
        st[i].stats.distance = dist * 1e3

    return st


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

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km


def get_station_id(trace_id):
    return ".".join(trace_id.split(".")[:2])


def save_mspec(filepath, event_meta, channel_names, deldist, x1, x2, s1, s2):
    # x should be of size (ntraces, nt)
    # s should be of size (ntraces, nf)

    assert x1.shape == x2.shape
    assert s1.shape == s2.shape
    assert x1.shape[0] == s1.shape[0] == len(channel_names)
    data = {
        "event_name": event_meta["event_name"],
        "elat": event_meta["elat"],
        "elon": event_meta["elon"],
        "edep": event_meta["edep"],
        "emag": event_meta["emag"],
        "edatetime_ns": int(event_meta["edatetime"].ns),
        "channel_names": list(channel_names),
        "deldist": deldist.astype(np.float32),
        "x1": x1.astype(np.float32),  # (ntraces, nt)
        "x2": x2.astype(np.float32),
        "s1": s1.astype(np.float32),  # (ntraces, nf)
        "s2": s2.astype(np.float32),
    }
    with open(filepath, "wb") as f:
        f.write(msgpack.packb(data))


def load_mspec(filepath):
    with open(filepath, "rb") as f:
        return msgpack.unpackb(f.read())


### Response functions
def build_response_lookup(xml_paths, eval_freqs):
    lookup = defaultdict(list)

    for path in tqdm(xml_paths):
        inv = read_inventory(path, format="STATIONXML")
        for net in inv:
            for sta in net:
                for cha in sta:
                    key = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"
                    correction = get_correction(cha, eval_freqs)

                    if correction is None:
                        print(
                            f"{net.code}.{sta.code}.{cha.location_code}.{cha.code} {cha.start_date} to {cha.end_date}: Failed to get correction"
                        )
                    else:
                        lookup[key].append((cha.start_date, cha.end_date, correction))

    for key in lookup:
        lookup[key].sort(key=lambda x: x[0])

    return lookup


def get_correction(cha, eval_freqs):
    pi = 3.141592653589793
    resp = cha.response

    # Some channels have no response stages. Skip them unless they are HH
    if not resp.response_stages and not cha.code.startswith("HH"):
        return None
    elif not resp.response_stages and cha.code.startswith("HH"):
        # HH channels have mostly flat response in the band of interest
        sensitivity = resp.instrument_sensitivity
        if isinstance(sensitivity, obspy.core.inventory.response.InstrumentSensitivity):
            sensitivity = sensitivity.value
        correction = sensitivity * (2 * pi * eval_freqs)  # HH* are velocity
    else:
        # Otherwise, get the correction
        try:
            resp.recalculate_overall_sensitivity(10)
            correction = resp.get_evalresp_response_for_frequencies(
                eval_freqs, output="DISP", hide_sensitivity_mismatch_warning=True
            )
        except Exception as e:
            return None

    # convert from time-series correction to frequency-domain correction
    correction = np.abs(correction) ** 2

    return correction


def get_epoch_correction(key, event_time, response_lookup):
    corrections = response_lookup.get(key, [])

    if not corrections:
        return None

    for start, end, correction in corrections:
        if event_time >= start and event_time < end:
            return correction

    return None


# %%
# First, get directories of each event that has waveforms (skip empty dirs)
if not os.path.exists(join(proc_dir, "event_reldirs.npy")):
    event_reldirs = []
    years = [el for el in listdir(waveforms_dir) if isdir(join(waveforms_dir, el))]
    for i in trange(len(years), desc="Parsing waveforms directory for events"):
        year = years[i]
        year_dir = join(waveforms_dir, year)
        jdays = listdir(year_dir)
        for jday in jdays:
            jday_dir = join(year_dir, jday)
            event_reldirs += [
                join(year, jday, el)
                for el in listdir(jday_dir)
                if len(listdir(join(waveforms_dir, year, jday, el))) > 0
            ]
    event_reldirs.sort()
    np.save(join(proc_dir, "event_reldirs.npy"), np.array(event_reldirs))
else:
    event_reldirs = np.load(join(proc_dir, "event_reldirs.npy")).tolist()
    print(
        f"Loaded {len(event_reldirs):,} event directories from {proc_dir}event_reldirs.npy"
    )

file_event_names = [event_reldir.split("/")[-1] for event_reldir in event_reldirs]
file_event_names_set = set(file_event_names)

# Load earthquake catalog
print("Reading earthquake catalog...", end="")
eq_df = pd.read_csv(eq_catalog_path)
# eq_df['event_name'] = eq_df['source'].values + eq_df['event_id'].values.astype(str)
len0 = len(eq_df)
print(f"Done. {len0:,} events loaded.")

event_names_set = set(eq_df["event_name"].values)

# confirm that no downloaded events are missing from catalog
missing_event_names = file_event_names_set - event_names_set
# assert len(missing_event_names) == 0, "ERROR: events missing from catalog"

# Drop missing events from catalog
len0 = len(eq_df)
eq_df = eq_df[eq_df["event_name"].isin(file_event_names_set)].reset_index(drop=True)
eq_df = eq_df.reset_index(drop=True)
eq_df["edatetime"] = [UTC(el) for el in eq_df["edatetime"].values.astype(str) + "Z"]
print(
    f"{len(eq_df):,} events with waveforms discovered. {len0 - len(eq_df):,} events dropped."
)

# Create mon and jday columns to help finding picks and waveforms
eq_df["mon"] = [el.month for el in eq_df["edatetime"].values]
eq_df["jday"] = [el.julday for el in eq_df["edatetime"].values]

# Make a lookup dict for event origins event_name: (elon, elat, edep, edatetime, emag)
event_origins = dict(
    zip(
        eq_df["event_name"],
        zip(
            eq_df["elon"],
            eq_df["elat"],
            eq_df["edep"],
            eq_df["edatetime"],
            eq_df["emag"],
        ),
    )
)

event_names_set = set(eq_df["event_name"].values)


print("Reading channel catalog...", end="")
cha_catalog = pd.read_csv(channel_catalog_path, sep="|", skiprows=1)
print(
    f"Done. {len(cha_catalog):,} unique channels loaded with columns {list(cha_catalog.columns)}"
)

print("Reading station catalog...", end="")
sta_catalog = pd.read_csv(station_catalog_path, sep="|", skiprows=1)
print(
    f"Done. {len(sta_catalog):,} unique stations loaded with columns {list(sta_catalog.columns)}"
)

# make a lookup dict for station locations station_name: (slon, slat, sele)
station_locations = dict(
    zip(
        sta_catalog["station_name"],
        zip(sta_catalog["slon"], sta_catalog["slat"], sta_catalog["sele"]),
    )
)
# PFR is missing from the catalog. This is the real location, but it doesn't even matter since it only has strainmeters not used in this study
station_locations["NC.PFR"] = (-120.487366, 35.910728, 434.0)

# make a lookup dict for channel locations channel_name: (slon, slat, sele)
channel_locations = dict(
    zip(
        cha_catalog["channel_name"],
        zip(cha_catalog["slon"], cha_catalog["slat"], cha_catalog["sele"]),
    )
)


# # Now, load .mphase files
# phase_filepaths = [el for el in listdir(phase_dir) if el.endswith('.mphase')]
# phase_filepaths.sort()

# n_events_picked = 0
# n_picks = 0
# pick_dict = {}
# for f in tqdm(phase_filepaths, desc="Loading picks into memory"):
#     D = read_mphase(join(phase_dir, f))
#     pick_dict.update(D)
#     n_events_picked += len(D)
#     n_picks += sum([len(v[3]) for k, v in D.items()])
# print(f"Done. Picks for {len(pick_dict):,} events loaded from {len(phase_filepaths):,} .mphase files.")

pick_dict, n_events_picked, n_picks = load_picks_from_dir(phase_dir)

print("")
print("Pick information:")
print(f"Total events with picks: {n_events_picked:,}")
print(f"Total picks: {n_picks:,}")
print(f"Average picks per event: {n_picks/n_events_picked:.2f}")

# %%
pick_dict["s3201047"]

# %%
phase_dir

# %%
# Finally, load the response lookup dict
eval_freqs = f_resample.copy()
eval_freqs[0] = f_resample[1] / 2
response_files = [
    response_dir + f for f in os.listdir(response_dir) if f.endswith(".xml")
]
response_lookup = build_response_lookup(response_files, eval_freqs)
# Response lookup:


# %%
def process_event(event_meta, pick_info):
    params = _worker_params
    sta_locs = _worker_sta_locs
    response_lookup = _worker_response_lookup

    event_name = event_meta["event_name"]
    elon = event_meta["elon"]
    elat = event_meta["elat"]
    edep = event_meta["edep"]
    emag = event_meta["emag"]
    edatetime = event_meta["edatetime"]
    eyear, ejulday = str(edatetime.year), f"{edatetime.julday:03d}"

    allowed_components = params["allowed_components"]
    f_corr = params["f_resample"].copy()
    f_corr[0] = f_corr[1] / 2  # fudge first point to avoid division by zero

    if len(pick_info[1]) == 0:
        return event_name, "no_picks"

    # Store the event directory
    event_dir = join(params["waveforms_dir"], eyear, ejulday, event_name)
    if not os.path.exists(event_dir):
        return event_name, "missing_event_dir"
    mspec_filename = f"{event_name}.mspec"
    output_filepath = join(params["spectra_save_dir"], eyear, ejulday, mspec_filename)

    if os.path.exists(output_filepath):
        return event_name, "already_processed"

    # get time window length for this event based on magnitude
    t_window, twin_idx = get_time_window_length(
        emag,
        params["min_twin"],
        params["max_twin"],
        params["M_min_twin"],
        params["M_max_twin"],
        params["d_twin"],
    )

    # load waveforms for this event
    if len([el for el in listdir(event_dir) if el.endswith(".ms")]) == 0:
        return event_name, "no_miniseed_files"
    try:
        st = read(join(event_dir, "*.ms"), format="MSEED")
    except Exception as e:
        print(str(e))
        return event_name, "failed_to_read_miniseed"

    st = obspy.Stream(
        [
            tr
            for tr in st
            if tr.get_id().split(".")[-1][-1].upper() in allowed_components
        ]
    )

    st = attach_distance_to_stream(st, elon, elat, sta_locs)

    # get picks for this event
    _, stnames, phases, picks, probs = pick_info
    pickdict = dict(zip(stnames, zip(phases, picks)))

    # filter stream to only include traces with picks, and only the channels we want
    st_picks = obspy.Stream()
    for tr in st:
        station_id = get_station_id(tr.get_id())
        if station_id in pickdict:
            phase, pick = pickdict[station_id]

            # hardcoded P only for now
            if phase == "P":
                # if tr.get_id() in channel_names:
                st_picks.append(tr.copy())

    if len(st_picks) == 0:
        return event_name, "no_picks_on_available_channels"

    # compute spectra
    deldist = []
    x1_list = []
    x2_list = []
    s1_list = []
    s2_list = []
    channel_names = []

    for tr in st_picks:
        tr_id = tr.get_id().lower()

        # filter out non-vertical components
        if tr_id[-1] != "z":
            continue

        corr = get_epoch_correction(tr.get_id(), edatetime, response_lookup)

        # Retry for CI network with location code 00 if no correction found
        if corr is None and tr.stats.location == "00" and tr.stats.network == "CI":
            net, sta, loc, cha = tr.get_id().upper().split(".")
            tr_id = ".".join([net, sta, "", cha])
            corr = get_epoch_correction(tr_id, edatetime, response_lookup)

        if corr is None:
            if allow_simple_correct:
                ### OLD METHOD: simple correction from input units to displacement
                # Determine correction to be applied to spectrum
                fcorr = f_corr.copy()
                fcorr[0] = fcorr[1] / 2  # fudge first point to avoid division by zero
                if tr_id[-2] == "n":
                    power = -2
                elif tr_id[-2] == "h":
                    power = -1
                else:
                    continue
                corr = (2 * np.pi * fcorr) ** power
                corr = corr**2
            else:
                continue

        if corr[0] == 0.0:
            corr[0] = corr[1] / 2

        # Resample if necessary
        if tr.stats.sampling_rate != params["sampling_rate"]:
            if tr.stats.sampling_rate == params["sampling_rate"] * 2:
                tr = tr.decimate(2)
            else:
                tr = tr.resample(params["sampling_rate"])

        # Get the pick time, relative to origin time
        _, t_pick = pickdict[get_station_id(tr.get_id())]

        # Compute time between origin and starttime (positive if starttime before origin)
        tdif = edatetime - tr.stats.starttime

        # make sure start time is before origin
        if tdif < 0:
            continue

        # Time array, relative to origin time
        t = tr.times() - tdif

        # Determine first and last indices of signal window
        is0 = np.argmin(np.abs((t) - (t_pick + params["t_buffer"])))
        is1 = is0 + int(t_window / tr.stats.delta)

        # Determine first and last indices of noise window, immediately preceding signal window
        in1 = is0
        in0 = in1 - int(t_window / tr.stats.delta)

        # Extract signal and noise windows
        x2 = tr.data[is0:is1]
        x1 = tr.data[in0:in1]

        t2 = t[is0:is1]
        t1 = t[in0:in1]

        if t1[0] < 0 and t1[-1] > 0:
            continue  # noise window straddles origin

        # If either window is too short, discard
        if (len(x2) * tr.stats.delta < t_window) | (
            len(x1) * tr.stats.delta < t_window
        ):
            continue

        #### SPECTRUM COMPUTATION ####
        # Direct use of mtspec functions, muuuch faster
        f2, s2_orig = mtspec_helper(
            x2,
            nw=params["nw"],
            kspec=params["kspec"],
            dt=tr.stats.delta,
            nfft=params["nffts"][twin_idx],
            iadapt=params["iadapt"],
            vn=params["vn"][twin_idx],
            lamb=params["lamb"][twin_idx],
        )
        f1, s1_orig = mtspec_helper(
            x1,
            nw=params["nw"],
            kspec=params["kspec"],
            dt=tr.stats.delta,
            nfft=params["nffts"][twin_idx],
            iadapt=params["iadapt"],
            vn=params["vn"][twin_idx],
            lamb=params["lamb"][twin_idx],
        )

        # Interpolate noise and signal to desired frequency resolution
        # Possible issue: interpolating a power spectrum in linear frequency
        # space smooths across bins non-uniformly
        if len(s2_orig) != len(params["f_resample"]):
            s2 = np.interp(params["f_resample"], f2, s2_orig)
            s1 = np.interp(params["f_resample"], f1, s1_orig)
        else:
            s2 = s2_orig
            s1 = s1_orig

        s2 = s2 / corr
        s1 = s1 / corr

        assert len(s2) == len(params["f_resample"])

        deldist.append(tr.stats.distance / 1000)
        x1_list.append(x1)
        x2_list.append(x2)
        s1_list.append(s1)
        s2_list.append(s2)
        channel_names.append(tr.get_id())

    if len(channel_names) == 0:
        return event_name, "no_remaining_traces"
    else:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        save_mspec(
            output_filepath,
            event_meta,
            channel_names,
            np.array(deldist),
            np.array(x1_list),
            np.array(x2_list),
            np.array(s1_list),
            np.array(s2_list),
        )
        return event_name, "processed"


# %%

warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", module="multitaper"
)

_worker_params = None
_worker_sta_locs = None
_worker_response_lookup = None


def _init_worker(params, sta_locs, response_lookup):
    global _worker_params, _worker_sta_locs, _worker_response_lookup
    _worker_params = params
    _worker_sta_locs = sta_locs
    _worker_response_lookup = response_lookup


def _make_args(event_name):
    elon, elat, edep, edatetime, emag = event_origins[event_name]
    event_meta = {
        "event_name": event_name,
        "elon": elon,
        "elat": elat,
        "edep": edep,
        "emag": emag,
        "edatetime": edatetime,
    }
    event_picks = pick_dict.get(event_name, (None, [], [], [], []))
    return event_meta, event_picks


msgpack_numpy.patch()  # enables numpy array support in msgpack

nevents = len(file_event_names)

events_to_process = file_event_names

initargs = (program_params, station_locations, response_lookup)

with ProcessPoolExecutor(
    max_workers=32, initializer=_init_worker, initargs=initargs
) as executor:
    futures = {}
    for event_name in tqdm(events_to_process, desc="Submitting events to process pool"):
        try:
            event_meta, event_picks = _make_args(event_name)
        except Exception as e:
            print(f"Error making args for event {event_name}: {e}")
            continue
        fut = executor.submit(process_event, event_meta, event_picks)
        futures[fut] = event_name

    counts = {}
    for fut in tqdm(
        as_completed(futures), total=len(futures), desc="Processing events"
    ):
        try:
            event_name, status = fut.result()
        except Exception as e:
            event_name = futures[fut]
            status = f"exception: {str(e)[:80]}"
        counts[status] = counts.get(status, 0) + 1

# print out counts summary
print("\nProcessing summary:")
total_processed = sum(counts.values())
for status, count in counts.items():
    print(f"{status}: {count} events ({count/total_processed:.2%})")

# %%

mspec_files = sorted(glob(f"{spectra_save_dir}/**/*.mspec", recursive=True))

# Estimate total records (adjust if you have a better guess)
n_est = 15_000_000
dn = 10_000_000

assert nf == len(f_resample), "error with nf"

s1 = np.empty((n_est, nf), dtype=np.float32)
s2 = np.empty((n_est, nf), dtype=np.float32)
deldist = np.empty(n_est, dtype=np.float32)
event_names = np.empty(n_est, dtype=object)
channel_names = np.empty(n_est, dtype=object)

idx = 0
for filepath in tqdm(mspec_files):
    data = load_mspec(filepath)
    n = len(data["channel_names"])
    end = idx + n
    if end > n_est:
        # add more space to arrays
        s1 = np.vstack((s1, np.empty((dn, nf), dtype=np.float32)))
        s2 = np.vstack((s2, np.empty((dn, nf), dtype=np.float32)))
        deldist = np.hstack((deldist, np.empty(dn, dtype=np.float32)))
        event_names = np.hstack((event_names, np.empty(dn, dtype=object)))
        channel_names = np.hstack((channel_names, np.empty(dn, dtype=object)))
        n_est += dn

    s1[idx:end] = data["s1"]
    s2[idx:end] = data["s2"]
    deldist[idx:end] = data["deldist"]
    event_names[idx:end] = data["event_name"]  # scalar, broadcast
    channel_names[idx:end] = data["channel_names"]
    idx = end

# Trim to actual size
s1 = s1[:idx]
s2 = s2[:idx]
deldist = deldist[:idx]
event_names = event_names[:idx]
channel_names = channel_names[:idx]

print(f"{idx:,} records from {len(mspec_files):,} events")


# %%
def UTC_to_decimal_year(t):  # t is UTCDateTime
    y = t.year
    start = UTC(y, 1, 1).timestamp
    end = UTC(y + 1, 1, 1).timestamp
    return y + (t.timestamp - start) / (end - start)


# make a dict mapping for event_name to
eyears = {
    e: UTC_to_decimal_year(t) for e, t in zip(eq_df["event_name"], eq_df["edatetime"])
}

# ---------- Sort all arrays by event time ----------
t0 = time.time()
print("Sorting arrays by event time...", end="")

# Build time array from event_names via the eyears dict
# (eyears must be defined before this block)
event_times = np.array([eyears[e] for e in event_names])

# Lexsort: primary key is time, secondary is channel_name so that within
# a given event, channels are grouped (stable, deterministic ordering).
sort_idx = np.lexsort((channel_names, event_times))

s1 = s1[sort_idx]
s2 = s2[sort_idx]
event_names = event_names[sort_idx]
channel_names = channel_names[sort_idx]
deldist = deldist[sort_idx]

print(f"Done. ({time.time()-t0:.2f}s)")

# Find rows with any NaNs in s1 or s2

print("Checking for NaNs...", end="")
t0 = time.time()
bad = np.isnan(s1).any(axis=1) | np.isnan(s2).any(axis=1)
keep = ~bad

s1 = s1[keep]
s2 = s2[keep]
event_names = event_names[keep]
channel_names = channel_names[keep]
deldist = deldist[keep]

print(
    f"Dropping {bad.sum():,} rows with NaNs ({bad.mean():.2%}), keeping {keep.sum():,} rows. ({time.time()-t0:.2f}s)"
)

print("Saving arrays...", end="")
np.save(join(proc_dir, "s1.npy"), s1)
np.save(join(proc_dir, "s2.npy"), s2)
np.save(join(proc_dir, "event_names.npy"), event_names)
np.save(join(proc_dir, "channel_names.npy"), channel_names)
np.save(join(proc_dir, "deldist.npy"), deldist)
print("Done.")
