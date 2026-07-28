# California nFI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21632655.svg)](https://doi.org/10.5281/zenodo.21632655)

## Reproducing results
1) **Download and save the earthquake catalog.** Run `scripts/data_processing/download_eq_catalogs.py`, which does the following:
    - Downloads earthquake catalogs from NCEDC and SCEDC servers and puts in a consistent format
    - Relocates events based on two relocated catalogs: [Hauksson et al. (2012)](https://scedc.caltech.edu/data/alt-2011-dd-hauksson-yang-shearer.html) for southern California, and [Waldhauser and Schaff (2008)](http://nocalDD.ldeo.columbia.edu/catalog) for northern California
    - Discard SCEDC earthquakes outside of, and NCEDC earthquakes inside a hand-drawn polygon in southern California, approximately separating the natural break in norcal and socal seismicity
    - Save as `catalogs/event_catalog.csv`
2) **Download and save the station catalog.**
3) **Download and save the earthquake waveforms.** Run `scripts/get_aws_waveforms.py`, which does the following:
- Loads `catalogs/event_catalog.csv` and `catalogs/station_catalog.csv`, then filters events by magnitude (M ≥ 1.0) and time range (1995–2025 by default; see the **Parameters** cell)
- Downloads event-based miniSEED waveforms from the public AWS buckets hosted by [NCEDC](https://ncedc.org/db/cloud.html) (`ncedc-pds`) and [SCEDC](https://scedc.caltech.edu/data/cloud.html) (`scedc-pds`); no AWS credentials are required
- Cleans each downloaded stream: removes traces with large station-epicentral distances, removes non-seismometer channels and unexpected sampling rates, and trims waveforms to a window around the origin time
- Saves cleaned waveforms to `waveforms/{year}/{julian_day}/{event_name}/{event_name}_{ncedc|scedc}.ms`
- Tracks per-event download status in `waveforms/dataset_status.csv`, so the notebook can be safely interrupted and re-run to resume where it left off
- Produces a validation figure (`figures/aws_data_ratio_by_source.png`) showing the daily fraction of events with downloaded data, alongside event magnitudes, for the full catalog and for each data center separately
