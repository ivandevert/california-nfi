# california-nfi

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21632654.svg)](https://doi.org/10.5281/zenodo.21632654)

This repository holds the codes (and some of the data) required for reproducing the results of the 2026 California nFI paper (submitted to JGR: Solid Earth). See [py-nfi](https://github.com/ivandevert/py-nfi) for details on the normalized frequency index method and installation.

## Repository layout
```
california-nfi/
├── catalogs/               # Earthquake and station catalogs are saved here
├── comparison_datasets/    # External datasets used to compare to nFI results
├── lib/                    # Shared helper modules imported by the scripts
├── scripts/                # Workflow and analysis scripts
│   ├── data_processing/    # Catalog and waveform preparation, see notes below
│   │   ├── download_eq_catalogs.py
│   │   ├── get_station_catalog.py
│   │   ├── get_aws_waveforms.py
│   │   ├── get_aws_phase.py
│   │   └── compute_spectra.py
│   └── run_nfi/            # Holds scripts for runs of computing nFI
│       └── compute_nfi.py  # Template for computing nFI
├── config.yaml             # Settings shared by data processing scripts
├── LICENSE                 
└── readme.md               # This file
```

## Data processing scripts
Each of the scripts in the `scripts/data_processing/` folder downloads, cleans, processes, and saves part of the required dataset.

### download_eq_catalogs.py
This script does the following:
- Downloads earthquake catalogs from NCEDC and SCEDC servers and saves in a consistent format
- Relocates events based on two relocated catalogs: [Hauksson et al. (2012)](https://scedc.caltech.edu/data/alt-2011-dd-hauksson-yang-shearer.html) for southern California, and [Waldhauser and Schaff (2008)](http://nocalDD.ldeo.columbia.edu/catalog) for northern California
- Discard SCEDC earthquakes outside of, and NCEDC earthquakes inside a hand-drawn polygon in southern California, approximately separating the natural break in norcal and socal seismicity
- Saves the cleaned, combined catalog as `catalogs/event_catalog.csv`

---

**event_catalog.csv** column description:
1) **event_name**: unique event identifier (string). Format is `{source}{event_id}`, where `event_id` is the event ID or event name reported by either the NCEDC or SCEDC, and `source` is `n` for NCEDC events and `s` for SCEDC events
2) **edatetime**: UTC origin time (string). Format is `YYYY-MM-DD HH:MM:SS.SSS`
3) **emag**: event catalog magnitude (float). Reported by source network.
4) **emagtype**: magnitude type reported by the source network (string), e.g. `ml`, `mw`, `md` (among others)
5) **elat**: event latitude in decimal degrees (float).
6) **elon**: event longitude in decimal degrees (float).
7) **edep**: event depth in kilometers (float). Positive is below geoid height.
8) **nst**: number of stations (or phase picks) used in the event location (integer)
9) **etype**: event type (string), e.g. `eq` for earthquake. Assigned by source network.
10) **relocated**: whether the event location was replaced with a relocated catalog position (integer), `1` if origin is relocated, `0` if the original network location was retained.

---

### get_station_catalog.py
This script:
- Downloads station metadata and response information from NCEDC
- Cleans and saves **station** metadata to `catalogs/station_catalog.csv`
- Cleans and saves **channel** metadata to `catalogs/channel_catalog.csv`. Station location information is redundant to that in `station_catalog.csv`, but this file is created for convenience.
- Saves response information in `catalogs/station/xml/`

---

**station_catalog.csv** column description:

This file is pipe-delimited (`|`), with a one-line provenance header above the
column names (hence `skiprows=1` when reading it).

1) **index**: row index (integer), assigned on write; not a meaningful station identifier
2) **station_name**: station identifier (string). Format is `{network}.{station}`, e.g. `CI.CCA`
3) **slat**: station latitude in decimal degrees (float).
4) **slon**: station longitude in decimal degrees (float).
5) **sele**: station elevation in meters (float). Positive is above geoid height.
6) **sdep**: emplacement depth of the sensor below the surface in meters (float). Mostly unused in these codes, but retained.
7) **source**: data center the station metadata was retrieved from (string), e.g. `ncedc` or `scedc`. 

---

**channel_catalog.csv** column description:

Also pipe-delimited (`|`) with a one-line provenance header (`skiprows=1`). One
row per channel; the station-level columns (`slat`, `slon`, `sele`, `sdep`,
`source`) repeat the values from `station_catalog.csv` for the parent station.

1) **channel_name**: channel identifier (string). Format is `{network}.{station}.{location}.{channel}`, e.g. `CI.CCA..HHZ`
2) **station_name**: parent station identifier (string), `{network}.{station}`. Joins to `station_catalog.csv`
3) **slat**: station latitude in decimal degrees (float).
4) **slon**: station longitude in decimal degrees (float).
5) **sele**: station elevation in meters (float). Positive is above geoid height.
6) **sdep**: emplacement depth of the sensor below the surface in meters (float). Mostly unused in these codes, but retained.
7) **sazi**: channel azimuth in degrees (float).
8) **sdip**: channel dip in degrees (float). 
9) **sample_rate**: channel sampling rate in samples per second (float).
10) **units**: physical units of the instrument-corrected data (string), e.g. `m/s` for a velocity sensor.
11) **scale**: overall sensitivity relating raw counts to `units` at frequency `scalefreq` (float).
12) **scalefreq**: frequency in Hz at which `scale` is defined (float).
13) **source**: data center the channel metadata was retrieved from (string), e.g. `ncedc` or `scedc`.

**Note:** station_name has the format `NET.STA`. This uniquely identifies the *station*; there may be multiple channels recording here but they are all assumed to have the same latitude, longitude, and elevation. 

---

### get_aws_waveforms.py
This script:
- Loads event and station information
- Downloads event-based miniSEED waveforms from the public AWS buckets hosted by [NCEDC](https://ncedc.org/db/cloud.html) (`ncedc-pds`) and [SCEDC](https://scedc.caltech.edu/data/cloud.html) (`scedc-pds`); no AWS credentials are required
- Cleans each downloaded stream: removes traces with large station-epicentral distances, removes non-seismometer channels and unexpected sampling rates, and trims waveforms to a window around the origin time
- Saves cleaned waveforms to `waveforms/{year}/{julian_day}/{event_name}/{event_name}_{ncedc|scedc}.ms`. One miniseed file per event.
- Tracks per-event download status in `waveforms/dataset_status.csv`, so the script can be safely interrupted and re-run to resume where it left off
- Produces a validation figure (`figures/aws_data_ratio_by_source.png`) showing the daily fraction of events with downloaded data, alongside event magnitudes, for the full catalog and for each data center separately

---

### get_aws_phase.py
This script downloads P- and S-wave phase-arrival picks for the events in the
earthquake catalog from the NCEDC (`ncedc-pds`) and SCEDC (`scedc-pds`) public
AWS buckets; no AWS credentials are required. It parses each network's format,
merges the two sources, and writes consolidated monthly pick files to
`catalogs/phase/combined/{year}.{month}.mphase`.

**.mphase file format:** JSON object mapping `event_name` to a 5-element list:
`[origin_time, station_names, phases, picks, qualities]`, where `station_names`,
`phases` (`"P"`/`"S"`), `picks` (travel time in seconds), and `qualities` are
parallel lists, one entry per pick. Floats are rounded to 2 decimals on write.

---

### compute_spectra.py
This script computes signal and noise spectra for every picked, vertical-component
waveform. For each event it loads the cleaned miniSEED (from `get_aws_waveforms.py`)
and phase picks (from `get_aws_phase.py`), selects a P-phase signal window and a
preceding noise window of equal length (the window length scales with event
magnitude), computes a multitaper power spectrum for each, removes the instrument
response, and resamples to a common frequency grid. Per-event results are written
as `.mspec` files, then concatenated across all events into the arrays consumed
by `compute_nfi.py`.

**Output arrays** (saved to `proc/compspec/`, one row per station-channel record,
sorted by event time then channel):
- `s2.npy` — signal spectra, shape `(nrecords, nf)`, float32
- `s1.npy` — noise spectra, shape `(nrecords, nf)`, float32
- `event_names.npy` — event identifier per row (object)
- `channel_names.npy` — `NET.STA.LOC.CHA` per row (object)
- `deldist.npy` — station-epicentral distance in km per row (float32)

The frequency grid is `nf` points linearly spaced from 0 to the Nyquist frequency
(half the common sampling rate). The gain-corrected variants used by the paper are
written to `proc/compspec/gain_corrected/`.


## Reproducing results of the 2026 California nFI paper (submitted to JGR: Solid Earth)

Run the following in order. See **Data processing scripts** above for details and
output formats. Paths and parameters are set in `config.yaml`.

1) `scripts/data_processing/download_eq_catalogs.py` — build the earthquake catalog
   → `catalogs/event_catalog.csv`
2) `scripts/data_processing/get_station_catalog.py` — download station/channel metadata and responses
   → `catalogs/station_catalog.csv`, `catalogs/channel_catalog.csv`, `catalogs/response/xml/`
3) `scripts/data_processing/get_aws_waveforms.py` — download and clean event waveforms
   → `waveforms/{year}/{jday}/{event_name}/`
4) `scripts/data_processing/get_aws_phase.py` — download and merge phase picks
   → `catalogs/phase/combined/{year}.{month}.mphase`
5) `scripts/data_processing/compute_spectra.py` — compute signal/noise spectra
   → `proc/compspec/` (s1, s2, event_names, channel_names, deldist)
6) `scripts/run_nfi/compute_nfi.py` — compute nFI and uncertainties
   → `results/{run_name}/`