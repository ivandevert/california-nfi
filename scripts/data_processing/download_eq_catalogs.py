# download_eq_catalogs.py
# Download and combine earthquake catalogs from NCEDC and SCEDC, replace
# origin locations with double-difference relocations where available, and
# save the combined catalog as a single CSV.
#
# Relocated catalogs (URLs are pinned to specific versions):
#   - SoCal:  Hauksson, Yang, and Shearer (updated through 2025)
#             https://scedc.caltech.edu/data/alt-2011-dd-hauksson-yang-shearer.html
#   - NorCal: Waldhauser and Schaff, NCAeqDD v202112.1 (ends Dec 2021)
#             https://nocaldd.ldeo.columbia.edu/
#
# Note: no magnitude cut is applied here; the full catalog is saved and
# filtered downstream.
#
# project_dir/
# ├── catalogs/
# │   ├── event_catalog.csv          # Full combined catalog (CSV format)
# │   └── event/
# │       ├── relocated/             # Downloaded relocated catalogs (gzipped)
# │       ├── scedc/                 # Downloaded SCEDC yearly catalogs
# │       └── ncedc/                 # Downloaded NCEDC yearly catalogs


# ### Imports

# %%
import gzip
import os
import shutil
from os.path import join

import numpy as np
import pandas as pd
import requests
from shapely import contains_xy
from shapely.geometry import Polygon
from tqdm import trange

# --- Load config --- #
import sys
from pathlib import Path
config_path = str(Path(__file__).resolve().parents[2] / "config.yaml")
lib_dir = str(Path(__file__).resolve().parents[2] / "lib")
sys.path.insert(0, lib_dir)
from helper_functions import load_config
cfg = load_config(config_path)
# --- End load config --- #

# I drew this polygon in QGIS, roughly following the natural break in events
# with a computed nFI
scedc_poly = Polygon([(-120.965911, 33.794442),
              (-119.572042, 35.433123),
              (-118.526641, 36.309047),
              (-113.704995, 38.545147),
              (-111.308394, 30.483274),
              (-121.435274, 30.385167),
              (-121.421051, 32.915314)])


# %%
def download_file(url, dest_path, encoding=None):
    """Download `url` to `dest_path`. If `encoding` is given, decode the
    response with that encoding (replacing invalid characters) and write
    text; otherwise write raw bytes."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    if encoding is None:
        with open(dest_path, "wb") as f:
            f.write(response.content)
    else:
        with open(dest_path, "w") as f:
            f.write(response.content.decode(encoding, errors="replace"))


def relocate_events(df_orig, df_reloc):
    """Replace origin coordinates (elat, elon, edep) in `df_orig` with
    relocated values from `df_reloc`, matched on event_name. Adds a
    `relocated` column (1 = relocated, 0 = original location)."""
    assert df_reloc['event_name'].is_unique, "Relocated catalog has non-unique event_names"

    # Merge on event_name (left join keeps all df_orig rows)
    merged = df_orig.merge(
        df_reloc[['event_name', 'elat', 'elon', 'edep']],
        on='event_name',
        how='left',
        suffixes=('', '_reloc'),
    )

    # Flag relocated events
    has_reloc = merged['elat_reloc'].notna()
    merged['relocated'] = has_reloc.astype(int)

    # Overwrite origin columns where relocation exists
    for col in ['elat', 'elon', 'edep']:
        merged.loc[has_reloc, col] = merged.loc[has_reloc, f'{col}_reloc']

    # Drop temporary columns
    merged.drop(columns=['elat_reloc', 'elon_reloc', 'edep_reloc'], inplace=True)

    print(f"Relocated: {has_reloc.sum():,} / {len(df_orig):,} events ({has_reloc.mean()*100:.1f}%)")

    return merged

# %% [markdown]
# ### Parameters and paths

# %%
cfg_paths = cfg['paths']
cfg_params = cfg['download_eq_catalogs']

catalog_output_dir = cfg_paths['catalogs_dir']
partial_output_dir = join(catalog_output_dir, "event")

# Year range of yearly catalogs to download. starttime is inclusive and
# endtime is exclusive, so the last catalog year is that of (endtime - 1 day)
first_year = int(cfg_params['starttime'].split('-')[0])
last_year = (pd.to_datetime(cfg_params['endtime']) - pd.Timedelta(days=1)).year

print(f"Downloading catalogs from {first_year} to {last_year} (inclusive)")

# Full catalog output path
full_catalog_path_csv = cfg_paths['eq_catalog_filepath']

# Directories for downloaded source catalogs
relocated_dir = join(partial_output_dir, "relocated")
scedc_dir = join(partial_output_dir, "scedc")
ncedc_dir = join(partial_output_dir, "ncedc")

# Prefixes for event names, such that event_name = event_name_prefix + event_id
# e.g. s12345678 is event ID 12345678 from the SCEDC catalog
SCEDC_event_name_prefix = 's'
NCEDC_event_name_prefix = 'n'

combine_columns = ['event_id', 'edatetime', 'emag', 'emagtype', 'elat', 'elon', 'edep', 'nst', 'etype']
output_columns = ['event_name', 'edatetime', 'emag', 'emagtype', 'elat', 'elon', 'edep', 'nst', 'etype', 'wrong_network', 'relocated']

for d in [catalog_output_dir, partial_output_dir, relocated_dir, scedc_dir, ncedc_dir]:
    os.makedirs(d, exist_ok=True)

# %% [markdown]
# ## Load and prepare relocated catalogs

# %%
# ===================== SoCal relocated catalog =====================
# https://scedc.caltech.edu/data/alt-2011-dd-hauksson-yang-shearer.html
socal_reloc_url = "https://scedc.caltech.edu/ftp/catalogs/hauksson/Socal_DD/sc_1981_2025_1d_3d_gc_soda_noqb_10_1_SCSN.gc"
reloc_cols = ["event_name", "edatetime", "elat", "elon", "edep"]


# File format:
#
# 012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
# 1981 01 01 17 45 59.096   3301578  33.50471 -116.76482   6.020  1.50    7832      19     836   105  1151   656  0.25  0.20   0.109   0.343   0.019    33.50487 -116.76460   6.065 le h gc 58
# 1981 01 01 18 57 18.260  12249959  33.54833 -117.76867  12.600  1.85      10                                        59.600   1.200   6.400   0.170                                le   3d


socal_catalog_filepath = join(relocated_dir, "hauksson_relocated_eq.gc")
socal_catalog_zfilepath = join(relocated_dir, "hauksson_relocated_eq.gc.gz")

# Download the catalog if needed. Only a gzipped copy is kept on disk to save
# space; the uncompressed file is deleted after reading.
if not os.path.exists(socal_catalog_zfilepath):
    print("Downloading Hauksson et al. (SoCal) relocated catalog...")
    print(f"URL: {socal_reloc_url}")
    download_file(socal_reloc_url, socal_catalog_filepath)
    print(f"Downloaded to {socal_catalog_filepath}")

    print("Compressing catalog...")
    with open(socal_catalog_filepath, "rb") as f_in:
        with gzip.open(socal_catalog_zfilepath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Compressed to {socal_catalog_zfilepath}")
else:
    print("Uncompressing SoCal relocated catalog...")
    with gzip.open(socal_catalog_zfilepath, "rb") as f_in:
        with open(socal_catalog_filepath, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Uncompressed to {socal_catalog_filepath}")


socal_columns = [
    'year', 'month', 'day', 'hour', 'minute', 'second',
    'event_id',
    'elat', 'elon', 'edep',
    'emag',
    'qid', 'cid', 'nbranch',
    'qnpair', 'qndiff_p', 'qndiff_s',
    'rms_p', 'rms_s',
    'eh', 'ez', 'et',
    'elat_orig', 'elon_orig', 'edep_orig',
    'etype', 'idk', 'reloc_method', 'number',
]

socal_colspecs = [
    (0,    4),   # year
    (4,    7),   # month
    (7,   10),   # day
    (10,  13),   # hour
    (13,  16),   # minute
    (16,  23),   # second
    (23,  33),   # event_id
    (33,  43),   # elat
    (43,  54),   # elon
    (54,  62),   # edep
    (62,  68),   # emag
    (68,  76),   # qid
    (76,  84),   # cid
    (84,  92),   # nbranch
    (92,  98),   # qnpair
    (98,  104),  # qndiff_p
    (104, 110),  # qndiff_s
    (110, 116),  # rms_p
    (116, 122),  # rms_s
    (122, 130),  # eh
    (130, 138),  # ez
    (138, 146),  # et
    (146, 158),  # elat_orig
    (158, 169),  # elon_orig
    (169, 177),  # edep_orig
    (177, 180),  # etype
    (180, 182),  # idk
    (182, 185),  # reloc_method
    (185, 188),  # number
]
assert len(socal_columns) == len(socal_colspecs), "Column/colspec length mismatch (SoCal)"

df_socal = pd.read_fwf(socal_catalog_filepath, colspecs=socal_colspecs, names=socal_columns)

# Remove the uncompressed file; the gzipped copy is kept
os.remove(socal_catalog_filepath)

df_socal = df_socal[df_socal['year'] >= first_year]
df_socal['event_name'] = 's' + df_socal['event_id'].astype(str)
# Note: pd.to_datetime rolls seconds >= 60 over into the next minute
df_socal['edatetime'] = pd.to_datetime(df_socal[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_socal = df_socal[reloc_cols].reset_index(drop=True)
print("SoCal relocated catalog head:")
print(df_socal.head(10), "\n")

# ===================== NorCal relocated catalog ====================
norcal_catalog_url = "https://nocaldd.ldeo.columbia.edu/catalog/NCAeqDD.v202112.1.gz"
norcal_catalog_filepath = join(relocated_dir, "NCAeqDD.v202112.1")
norcal_catalog_zfilepath = join(relocated_dir, "NCAeqDD.v202112.1.gz")

norcal_columns = [
    'year', 'month', 'day', 'hour', 'minute', 'second',
    'elat', 'elon', 'edep',
    'eh1', 'eh2', 'eaz', 'ez', 'emag', 'event_id',
]

norcal_colspecs = [
    (0,    4),   # year
    (4,    7),   # month
    (7,   10),   # day
    (10,  13),   # hour
    (13,  16),   # minute
    (16,  23),   # second
    (23,  35),   # elat
    (35,  48),   # elon
    (48,  57),   # edep
    (57,  65),   # eh1
    (65,  73),   # eh2
    (73,  77),   # eaz
    (77,  85),   # ez
    (85,  90),   # emag
    (90, 101),   # event_id
]
assert len(norcal_columns) == len(norcal_colspecs), "Column/colspec length mismatch (NorCal)"

# Download the catalog if needed (already gzipped at the source)
if not os.path.exists(norcal_catalog_zfilepath):
    print("Downloading Waldhauser and Schaff (NorCal) relocated catalog...")
    print(f"URL: {norcal_catalog_url}")
    download_file(norcal_catalog_url, norcal_catalog_zfilepath)
    print(f"Downloaded to {norcal_catalog_zfilepath}")

print("Uncompressing NorCal relocated catalog...")
with gzip.open(norcal_catalog_zfilepath, "rb") as f_in:
    with open(norcal_catalog_filepath, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
print(f"Uncompressed to {norcal_catalog_filepath}")

# The file starts with a 97-line header
df_norcal = pd.read_fwf(
    norcal_catalog_filepath, colspecs=norcal_colspecs, names=norcal_columns, skiprows=97
)
# Remove the uncompressed file; the gzipped copy is kept
os.remove(norcal_catalog_filepath)

df_norcal = df_norcal[df_norcal['year'] >= first_year]
df_norcal['event_name'] = 'n' + df_norcal['event_id'].astype(str)
df_norcal['edatetime'] = pd.to_datetime(df_norcal[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_norcal = df_norcal[reloc_cols].reset_index(drop=True)
print("NorCal relocated catalog head:")
print(df_norcal.head(10), "\n")

# ===================== Combine relocated catalogs ==================
df_relocated = (
    pd.concat([df_socal, df_norcal], axis=0)
    .sort_values(by='edatetime')
    .reset_index(drop=True)
)

print("Combined relocated catalog head:")
print(df_relocated.head(10), "\n")


# %% [markdown]
# ## Download and prepare SCEDC and NCEDC catalogs

# %%
# ===================== SCEDC full catalog ==========================
# info:   https://github.com/SCEDC/SCEDC-catalogs
# format: https://scedc.caltech.edu/eq-catalogs/docs/scec_dc.html

years = np.arange(first_year, last_year + 1, 1).astype(str)

scedc_column_names = ['eymd', 'ehms', 'etype', 'egtype', 'emag', 'emagtype', 'elat',
                'elon', 'edep', 'equal', 'event_id', 'nphase', 'nst']
scedc_urls = ["https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/" + year + ".catalog" for year in years]
scedc_file_paths = [join(scedc_dir, year + '.catalog') for year in years]

if not all([os.path.exists(fp) for fp in scedc_file_paths]):
    for i in trange(len(scedc_urls), desc='Downloading SCEDC catalogs: '):
        if not os.path.exists(scedc_file_paths[i]):
            download_file(scedc_urls[i], scedc_file_paths[i], encoding='utf-8')
else:
    print(f"SCEDC catalogs already downloaded to {scedc_dir}")

# Load all yearly catalogs into one dataframe. Each file has a 10-line header
# and a 1-line footer.
scedc_yearly_dfs = [
    pd.read_csv(fp, skiprows=10, sep=r'\s+',
                names=scedc_column_names, skipfooter=1, engine='python')
    for fp in scedc_file_paths
]
scedc_df = pd.concat(scedc_yearly_dfs, axis=0).reset_index(drop=True)
assert len(scedc_df) == sum([len(el) for el in scedc_yearly_dfs])

# Correct the formats of certain columns:
# eymd column: use '-' date separators so pd.to_datetime can parse it
scedc_df['eymd'] = scedc_df['eymd'].apply(lambda x: x.replace('/', '-'))
# ehms column: for some reason, some events have seconds = 60.0. Replace
# these with 59.999
scedc_df['ehms'] = scedc_df['ehms'].apply(lambda x: x.replace('60.00', '59.999'))

# Entries with seconds > 60 are real wonky. idek, but get rid of them
bad_seconds = np.array([float(el.split(":")[-1]) for el in scedc_df['ehms'].values]) > 60.0
print("The following entries will be deleted because their seconds are greater than 60:")
if bad_seconds.sum() == 0:
    print("None.")
else:
    print(scedc_df[bad_seconds])
scedc_df = scedc_df[~bad_seconds].reset_index(drop=True)

# Properly format edatetime and keep only the columns used downstream
scedc_df['edatetime'] = pd.to_datetime(scedc_df['eymd'] + 'T' + scedc_df['ehms'])
scedc_df = scedc_df.drop(['eymd', 'ehms'], axis=1)
scedc_df = scedc_df[combine_columns]

# Make sure event IDs are unique
assert scedc_df['event_id'].is_unique

# Add source column. event_name can be formed with source + event_id
scedc_df['source'] = SCEDC_event_name_prefix

print(f"Total number of SCEDC events: {len(scedc_df):,}")


# ===================== NCEDC full catalog ==========================
# info: https://github.com/NCEDC/NCSS-catalogs

ncedc_column_names = ['edatetime', 'elat', 'elon', 'edep', 'emag', 'emagtype',
                'nst', 'gap', 'dmin', 'rms', 'net', 'event_id', 'updated',
                'place', 'etype', 'horiz_error', 'depth_error', 'mag_error',
                'mag_nst', 'status', 'location_source', 'mag_source']
ncedc_urls = ["https://ncedc.org/pub/catalogs/NCSS-catalogs/" + year + ".ehpcsv" for year in years]
ncedc_file_paths = [join(ncedc_dir, year + '.catalog') for year in years]

if not all([os.path.exists(fp) for fp in ncedc_file_paths]):
    for i in trange(len(ncedc_urls), desc='Downloading NCEDC catalogs: '):
        if not os.path.exists(ncedc_file_paths[i]):
            download_file(ncedc_urls[i], ncedc_file_paths[i], encoding='ascii')
else:
    print(f"NCEDC catalogs already downloaded to {ncedc_dir}")

# Load all yearly catalogs into one dataframe. Each file has a 1-line header.
ncedc_yearly_dfs = [
    pd.read_csv(fp, skiprows=1, sep=',', names=ncedc_column_names)
    for fp in ncedc_file_paths
]
ncedc_df = pd.concat(ncedc_yearly_dfs, axis=0).reset_index(drop=True)
assert len(ncedc_df) == sum([len(el) for el in ncedc_yearly_dfs])

ncedc_df['edatetime'] = pd.to_datetime(ncedc_df['edatetime']).dt.tz_localize(None)
ncedc_df = ncedc_df[combine_columns]

# Filter out earthquakes without a location or magnitude
len0 = len(ncedc_df)
ncedc_df = ncedc_df[np.logical_and(ncedc_df['emagtype'] != 'Unk', ncedc_df['elon'] != 0.0)].reset_index(drop=True)
print(f"{len0 - len(ncedc_df):,} events have no location or magnitude and are discarded")

# Make sure event IDs are unique
assert ncedc_df['event_id'].is_unique

# Add source column. event_name can be formed with source + event_id
ncedc_df['source'] = NCEDC_event_name_prefix

print(f"Total number of NCEDC events: {len(ncedc_df):,}")

# ===================== Combine, relocate, and save =================
eq_df = pd.concat([scedc_df, ncedc_df], axis=0).reset_index(drop=True)
eq_df['event_name'] = eq_df['source'] + eq_df['event_id'].astype(str)

# Replace origin locations with relocated locations where available
eq_df = relocate_events(eq_df, df_relocated)

eq_df = eq_df.sort_values(by='edatetime').reset_index(drop=True)

# Mark events outside their network's authoritative region as 'wrong_network':
# NCEDC events inside the SCEDC polygon, and SCEDC events outside it
inside = contains_xy(scedc_poly, eq_df['elon'].to_numpy(), eq_df['elat'].to_numpy())
source = eq_df['source'].to_numpy()

wrong_net_ncedc = inside & (source == NCEDC_event_name_prefix)   # NCEDC events inside polygon
wrong_net_scedc = ~inside & (source == SCEDC_event_name_prefix)  # SCEDC events outside polygon

eq_df['wrong_network'] = (wrong_net_ncedc | wrong_net_scedc).astype('int8')

eq_df[output_columns].to_csv(full_catalog_path_csv, index=False)
print(f"Saved full catalog to {full_catalog_path_csv}")

# ===================== Final report ================================
# Note: the NorCal relocated catalog (v202112.1) ends in Dec 2021, so NCEDC
# events after that date cannot be relocated
n_ncedc = (eq_df['source'] == NCEDC_event_name_prefix).sum()
n_scedc = (eq_df['source'] == SCEDC_event_name_prefix).sum()

print("")
print("------ Final report ------")
print(f"Total number of events: {len(eq_df):,}")
print(f"Total NCEDC events: {n_ncedc:,}")
print(f"Total SCEDC events: {n_scedc:,}")
print(f"NCEDC events inside polygon: {wrong_net_ncedc.sum():,} ({wrong_net_ncedc.sum() / n_ncedc * 100:.2f}% of NCEDC events)")
print(f"SCEDC events outside polygon: {wrong_net_scedc.sum():,} ({wrong_net_scedc.sum() / n_scedc * 100:.2f}% of SCEDC events)")
print(f"Total number of relocated events: {eq_df['relocated'].sum():,}")

print("\n ------------------------------------------- \n")
eq_df_m1 = eq_df[eq_df['emag'] >= 1.0].reset_index(drop=True)
print(f"M>=1 number of events: {len(eq_df_m1):,}")
print(f"M>=1 NCEDC events: {(eq_df_m1['source'] == NCEDC_event_name_prefix).sum():,}")
print(f"M>=1 SCEDC events: {(eq_df_m1['source'] == SCEDC_event_name_prefix).sum():,}")
print(f"M>=1 number of relocated events: {eq_df_m1['relocated'].sum():,} ({eq_df_m1['relocated'].sum() / len(eq_df_m1) * 100:.2f}%)")
