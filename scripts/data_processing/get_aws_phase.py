# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import time
from tqdm import tqdm, trange
import unlzw3
import json

from obspy import UTCDateTime

# --- AWS SDK imports ---#
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config

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
def parse_phase_event_line(line):
    # Y2000 summary format (also used as header in archive file) page 98 of HYPOINVERSE-2000
    # these lines are of lengths 180 (mostly) or 170 (rarely)
    year = int(line[0:4].strip())
    month = int(line[4:6].strip())
    day = int(line[6:8].strip())
    hour = int(line[8:10].strip())
    minute = int(line[10:12].strip())
    second = float(line[12:14] + "." + line[14:16])
    edatetime = UTCDateTime(year, month, day, hour, minute) + second
    # edep = float(line[31:34] + '.' + line[34:36])

    event_id = line[138:146].strip()
    # print(event_id, edatetime)
    return event_id, edatetime


def parse_station_line(line):
    # Y2000 (station) archive format on pages 99-100 of the User's Guide to HYPOINVERSE-2000
    # lengths of 121, possibly 82
    net = line[5:7].strip()
    sta = line[0:5].strip()
    loc = line[111:113].strip()
    cha = line[9:12].strip()
    if loc == "--":
        loc = ""
    channel_name = ".".join([net, sta, loc, cha])

    year = int(line[17:21].strip())
    month = int(line[21:23].strip())
    day = int(line[23:25].strip())
    hour = int(line[25:27].strip())
    minute = int(line[27:29].strip())
    p_sec_str = line[29:34].strip()
    s_sec_str = line[41:46].strip()
    if p_sec_str:
        p_sec = float(p_sec_str[:-2] + "." + p_sec_str[-2:])
    else:
        p_sec = None

    if s_sec_str:
        s_sec = float(s_sec_str[:-2] + "." + s_sec_str[-2:])
    else:
        s_sec = None

    epi_dist_str = line[74:78].strip()
    if epi_dist_str:
        epi_dist = float(epi_dist_str[:-1] + "." + epi_dist_str[-1:])
    else:
        epi_dist = None

    if p_sec:
        p_arrival = UTCDateTime(year, month, day, hour, minute) + p_sec
    else:
        p_arrival = None

    if s_sec:
        s_arrival = UTCDateTime(year, month, day, hour, minute) + s_sec
    else:
        s_arrival = None

    # If blank in input file, p_arrival, s_arrival, or epi_dist will be None
    return channel_name, p_arrival, s_arrival, epi_dist


def write_events_picks_to_file(
    filepath, event_id, origin_time, channel_names, phases, picks
):
    # append a single event's picks to an existing file
    assert len(channel_names) == len(phases) == len(picks)
    with open(filepath, "a") as f:
        f.write(f"# {event_id:<10} {str(origin_time):27}\n")
        for channel_name, phase, pick in zip(channel_names, phases, picks):
            f.write(f"{channel_name:<15} {phase:>2} {pick:7.3f}\n")


def process_ncedc_phase_file(phase_filepath, mphase_output_filepath):
    # these are formatted using HYPOINVERSE-2000 formats

    EVENT_HEADER_LENGTHS = [180, 170]
    STATION_LENGTHS = [121, 82, 63]
    with open(phase_filepath, "r") as f:
        lines = f.readlines()
    # first, identify which lines represent events
    header_idx = []
    for l, line in enumerate(lines):
        if len(line) in EVENT_HEADER_LENGTHS:
            header_idx.append(l)
    header_idx.append(len(lines))  # add a final index to make for loop easier

    # # Then, if it exists, open mphase_output_filepath and check for existing event IDs
    # existing_event_ids = []
    # if os.path.exists(mphase_output_filepath):
    #     with open(mphase_output_filepath, 'r') as f:
    #         existing_lines = f.readlines()
    #     for eline in existing_lines:
    #         if eline.startswith('#'):
    #             existing_event_ids.append(int(eline.split()[1].strip()))

    D = {}

    # loop over each event
    for i in range(len(header_idx) - 1):
        eline = lines[header_idx[i]]
        event_id, edatetime = parse_phase_event_line(eline)
        event_name = f"n{str(event_id)}"

        # if int(event_id) in existing_event_ids: continue
        # there's a line at the end of the station lines; skip it
        station_lines = lines[header_idx[i] + 1 : header_idx[i + 1] - 1]

        # channel_names = []
        station_names = []
        phases = []
        picks = []
        qualities = []

        for j in range(len(station_lines)):
            sline = station_lines[j]

            channel_name, p_arrival, s_arrival, _ = parse_station_line(sline)
            station_name = ".".join(channel_name.split(".")[:2])

            if p_arrival:
                p_pick = p_arrival - edatetime
                if p_pick > 0:
                    station_names.append(station_name)
                    phases.append("P")
                    picks.append(p_pick)
                    qualities.append(1)

            if s_arrival:
                s_pick = s_arrival - edatetime
                if s_pick > 0:
                    station_names.append(station_name)
                    phases.append("S")
                    picks.append(s_pick)
                    qualities.append(1)

        if len(station_names) > 0:
            D[event_name] = (str(edatetime), station_names, phases, picks, qualities)
            # write_events_picks_to_file(mphase_output_filepath, event_id, edatetime, station_names, phases, picks)
    write_mphase(D, mphase_output_filepath)


def get_ncedc_phase_keys(starttime, endtime):
    # keys are in event_phases/2017/2017.01.phase.Z
    y1 = starttime.year
    y2 = endtime.year
    prefix = "event_phases"

    keys = []
    for year in range(y1, y2 + 1):
        for mon in range(1, 13):
            key = f"{prefix}/{year}/{year:04d}.{mon:02d}.phase.Z"
            keys.append(key)
    return keys


def download_phase_ncedc(keys, download_dir):
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

    filepaths = [os.path.join(download_dir, key.split("/")[-1]) for key in keys]
    bucket = "ncedc-pds"

    # for key, filepath in zip(keys, filepaths):
    for i in trange(len(filepaths), desc="Downloading NCEDC phase files"):
        key = keys[i]
        filepath = filepaths[i]
        if not os.path.exists(filepath):
            # print("Downloading", key, "to", filepath, end='')
            s3.Bucket(bucket).download_file(key, filepath)
            # print("...Done.")
    print(f"All NCEDC phase files downloaded to {download_dir}.")


def write_mphase(D, mphase_filepath):
    def round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 2)
        if isinstance(obj, list):
            return [round_floats(x) for x in obj]
        if isinstance(obj, tuple):
            return [round_floats(x) for x in obj]
        return obj

    D_rounded = {k: round_floats(v) for k, v in D.items()}

    with open(mphase_filepath, "w") as fout:
        json.dump(D_rounded, fout)


def read_mphase(mphase_filepath):
    return json.load(open(mphase_filepath, "r"))


# %%
cfg_paths = cfg["paths"]
cfg_params = cfg["get_aws_phase"]

catalog_dir = cfg_paths["catalogs_dir"]
event_catalog_filepath = cfg_paths["eq_catalog_filepath"]

# === Path definitions ===
phase_dir = join(catalog_dir, "phase")
ncedc_phase_dir = join(phase_dir, "ncedc/")
scedc_phase_dir = join(phase_dir, "scedc/")
ncedc_originals_dir = join(ncedc_phase_dir, "src/")
scedc_originals_dir = join(scedc_phase_dir, "src/")

scedc_status_filepath = join(phase_dir, "scedc_status.txt")
ncedc_status_filepath = join(phase_dir, "ncedc_status.txt")

starttime = UTCDateTime(cfg_params["starttime"])
endtime = UTCDateTime(cfg_params["endtime"])

mag_range = cfg_params["mag_range"]

for d in [
    phase_dir,
    ncedc_phase_dir,
    scedc_phase_dir,
    ncedc_originals_dir,
    scedc_originals_dir,
]:
    if not os.path.exists(d):
        os.makedirs(d)

print(f"Phase related files will be stored in the directory: {phase_dir}")

# %%
# Download all NCEDC phase to temp_download_dir

keys = get_ncedc_phase_keys(starttime, endtime - 1)

download_phase_ncedc(keys, ncedc_originals_dir)

# convert into .mphase files
compressed_phase_files = os.listdir(ncedc_originals_dir)
compressed_phase_files = [el for el in compressed_phase_files if el.endswith(".Z")]
compressed_phase_files = [el for el in compressed_phase_files if not el.startswith(".")]
compressed_phase_files.sort()

for i in trange(len(compressed_phase_files)):
    cfile = compressed_phase_files[i]
    phase_filepath = os.path.join(ncedc_originals_dir, cfile[:-2])
    new_phase_filepath = os.path.join(
        ncedc_phase_dir, ".".join(cfile.split(".")[:2]) + ".mphase"
    )

    if not os.path.exists(new_phase_filepath):
        # uncompress
        with open(join(ncedc_originals_dir, cfile), "rb") as f_in:
            with open(phase_filepath, "wb") as f_out:
                f_out.write(unlzw3.unlzw(f_in.read()))

        process_ncedc_phase_file(phase_filepath, new_phase_filepath)

        # delete uncompressed file
        os.remove(phase_filepath)
print("All NCEDC phase files processed.")


# %%


def get_scedc_phase_keys(eq_catalog):
    # keys are in event_phases/2017/2017_123/eventid.phase
    prefix = "event_phases"

    edatetimes = eq_catalog["edatetime"].values
    event_ids = eq_catalog["event_id"].values

    eyears = np.array([dt.year for dt in edatetimes])
    ejuldays = np.array([dt.julday for dt in edatetimes])

    keys = [
        f"{prefix}/{eyears[i]}/{eyears[i]:04d}_{ejuldays[i]:03d}/{event_ids[i]}.phase"
        for i in range(len(event_ids))
    ]

    return keys


def download_phase_scedc(s3, keys, download_dir):
    bucket = "scedc-pds"
    status = np.zeros(len(keys), dtype=int)

    # append a timestamp to the status file
    with open(scedc_status_filepath, "a") as f:
        f.write(f"\n=== SCEDC phase download attempt at {UTCDateTime.now()} ===\n")
    for i in trange(len(keys)):
        _, _, year_julday, event_id = keys[i].split("/")
        year, julday = year_julday.split("_")
        month = UTCDateTime(year=int(year), julday=int(julday)).month
        month = f"{month:02d}"
        event_id = event_id.split(".")[0]
        filedir = os.path.join(download_dir, f"{year}_{month}")
        filename = f"{event_id}.phase"
        filepath = os.path.join(filedir, filename)
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        try:
            if not os.path.exists(filepath):
                s3.Bucket(bucket).download_file(keys[i], filepath)
                print(f"Downloaded {event_id} successfully.")
                status[i] = 1  # Downloaded successfully
            else:
                status[i] = 10  # File already exists
        except:
            status[i] = -1  # Download failed

            with open(scedc_status_filepath, "a") as f:
                f.write(f"Failed to download {keys[i]}\n")
            # wait a bit
            time.sleep(0.1)
            pass
    print(f"All SCEDC phase files downloaded to {download_dir}.")
    return status


def read_scedc_phase_file(filepath):
    # read a single SCEDC .phase file

    with open(filepath, "r") as f:
        lines = f.readlines()

    eline_parts = lines[0].strip().split()
    # print(eline_parts)

    event_id = int(eline_parts[0])

    # 40965784.phase is empty:
    if event_id == 0:
        return None, None, None, None, None, None
    try:
        edate, etime = eline_parts[3].split(",")
    except:
        # this is weird: event ID 39321911 is missing a magnitude type
        print(event_id, f"has no magnitude type", end="")
        edate, etime = eline_parts[2].split(",")
        print(f"edate, etime = {edate}, {etime}")
    origin_time = UTCDateTime(f"{edate} {etime}")

    station_names = []
    phases = []
    picks = []
    qualities = []
    for l, sline in enumerate(lines[1:]):
        sline_parts = sline.strip().split()
        net = sline_parts[0]
        sta = sline_parts[1]
        cha = sline_parts[2]
        loc = sline_parts[3]
        if loc == "--":
            loc = ""
        station_name = ".".join([net, sta])
        phase = sline_parts[7]
        quality = float(sline_parts[10])
        pick = float(sline_parts[12])

        if pick > 0:
            station_names.append(station_name)
            phases.append(phase)
            picks.append(pick)
            qualities.append(quality)

    return event_id, origin_time, station_names, phases, picks, qualities


def process_scedc_phase_files(phase_filepaths, mphase_output_filepath):
    """
    SCEDC phase info is stored in one file per event. I need phase to
    be in one file per month of data like the .mphase files.

    phase_filepaths: list of paths to .phase files to be combined
    mphase_output_filepath: path of new .mphase file

    """

    nfiles = len(phase_filepaths)

    D = {}
    for i in range(nfiles):
        phase_filepath = phase_filepaths[i]
        event_id, origin_time, station_names, phases, picks, qualities = (
            read_scedc_phase_file(phase_filepath)
        )
        event_name = f"s{str(event_id)}"
        if event_id:
            # write_events_picks_to_file(mphase_output_filepath, event_id, origin_time, channel_names, phases, picks)
            D[event_name] = (str(origin_time), station_names, phases, picks, qualities)
        else:
            print(f"File {phase_filepath} is empty; skipping.")
    write_mphase(D, mphase_output_filepath)


# %%
print("Reading SCEDC earthquake catalog...", end="")
eq_df = pd.read_csv(event_catalog_filepath)
eq_df = eq_df[eq_df["source"] == "s"].reset_index(drop=True)
eq_df["edatetime"] = [UTCDateTime(el) for el in eq_df["edatetime"].values]
len0 = len(eq_df)
print(f"Done. {len0:,} events loaded.")
eq_df = eq_df[eq_df["emag"] >= mag_range[0]].reset_index(drop=True)
print(
    f"{len0 - len(eq_df):,} events with magnitude < {mag_range[0]} discarded. {len(eq_df):,} events remaining."
)

# %%
s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

# Download all SCEDC phase to scedc_originals_dir

keys = get_scedc_phase_keys(eq_df)
status = download_phase_scedc(s3, keys, scedc_originals_dir)

# print status report: status==0: not attempted, 1: downloaded, -1: failed, 10: redundant
print("Download status report:")
print(f"  {np.sum(status==0)} files not attempted.")
print(f"  {np.sum(status==1)} files downloaded.")
print(f"  {np.sum(status==-1)} files failed to download.")
print(f"  {np.sum(status==10)} files already existed.")


# %%
# convert scedc phase files to .mphase files
# dir format:
# scedc_phase_dir/src/year_month/event_id.phase
# processed files to:
# scedc_phase_dir/year.month.mphase

if not os.path.exists(scedc_phase_dir):
    os.makedirs(scedc_phase_dir)

phase_dirs = os.listdir(scedc_originals_dir)
phase_dirs.sort()
phase_dirs = [el for el in phase_dirs if not el.startswith(".")]

# each phase dir is a month; each dir will become one .mphase file

ndir = len(phase_dirs)
for i in trange(ndir):
    pdir = phase_dirs[i]
    year, month = pdir.split("_")
    phase_filepaths = []
    pdir_full = os.path.join(scedc_originals_dir, pdir)
    for f in os.listdir(pdir_full):
        if f.endswith(".phase") and not f.startswith("."):
            phase_filepaths.append(os.path.join(pdir_full, f))
    phase_filepaths.sort()
    new_phase_filepath = os.path.join(scedc_phase_dir, f"{year}.{month}.mphase")
    if not os.path.exists(new_phase_filepath):
        process_scedc_phase_files(phase_filepaths, new_phase_filepath)

print("All SCEDC phase files processed.")

# %%
# load eq catalog
print("Reading earthquake catalog...", end="")
eq_df = pd.read_csv(event_catalog_filepath)
eq_df["event_name"] = eq_df["source"] + eq_df["event_id"].astype(str)
# eq_df['edatetime'] = [UTCDateTime(el) for el in eq_df['edatetime'].values]
len0 = len(eq_df)
print(f"Done. {len0:,} events loaded.")
# eq_df = eq_df[eq_df['emag'] >= mag_range[0]].reset_index(drop=True)
# print(f"{len0 - len(eq_df):,} events with magnitude < {mag_range[0]} discarded. {len(eq_df):,} events remaining.")

# make a lookup dict such that eq[event_name] = emag
eq = dict(zip(eq_df["event_name"].values, eq_df["emag"].values.astype(float)))


# %%
# Process both SCEDC and NCEDC .mphase files: for each file, remove events that are too small

# Combine scedc and ncedc catalog pick files and store in phase_dir/combined/

combined_phase_dir = join(phase_dir, "combined/")
if not os.path.exists(combined_phase_dir):
    os.makedirs(combined_phase_dir)

for year in np.arange(starttime.year, endtime.year):
    # print(f"Combining {year} picks")
    for month in np.arange(1, 13):
        mphase_filename = f"{year}.{month:0>2}.mphase"

        combined_filepath = join(combined_phase_dir, mphase_filename)

        scedc_filepath = join(scedc_phase_dir, mphase_filename)
        ncedc_filepath = join(ncedc_phase_dir, mphase_filename)

        if os.path.isfile(combined_filepath):
            continue
        print(f"Processing {mphase_filename}..., ", end="")
        scedc_D = read_mphase(scedc_filepath)
        ncedc_D = read_mphase(ncedc_filepath)
        slen0 = len(scedc_D)
        nlen0 = len(ncedc_D)

        # remove events that are not in the earthquake catalog
        scedc_D = {k: v for k, v in scedc_D.items() if k in eq}
        ncedc_D = {k: v for k, v in ncedc_D.items() if k in eq}

        # remove events that are too small
        scedc_D = {k: v for k, v in scedc_D.items() if eq[k] >= mag_range[0]}
        ncedc_D = {k: v for k, v in ncedc_D.items() if eq[k] >= mag_range[0]}
        print(
            f"    {slen0 - len(scedc_D):,}/{slen0:,} SCEDC picks dropped, {nlen0 - len(ncedc_D):,}/{nlen0:,} NCEDC dropped"
        )
        D = {**scedc_D, **ncedc_D}

        # raise ValueError()

        # save to combined directory
        write_mphase(D, combined_filepath)
print("All combined .mphase files written.")

