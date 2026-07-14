# download_eq_catalogs.py
# This is a notebook to download & combine earthquake catalogs from both
# NCEDC and SCEDC. Need to add relocation file downloading later.
# 
# project_dir/
# ├── catalogs/
# │   ├── event_catalog.csv          # Full catalog (CSV format)
# │   └── event/
# │       ├── scedc_{first_year}_{last_year}.pkl # SCEDC combined catalog
# │       ├── ncedc_{first_year}_{last_year}.pkl # NCEDC combined catalog
# │       ├── relocated/             # 
# │       ├── scedc/                 # Downloaded SCEDC catalog data
# │       └── ncedc/                 # Downloaded NCEDC catalog data


# ### Imports

# %%
import numpy as np 
import os
from os.path import join
import pandas as pd

import json

from tqdm import trange
import urllib.request
from shapely.geometry import Polygon
from shapely import contains_xy

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
def relocate_events(df_orig, df_reloc):
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
partial_output_dir = join(catalog_output_dir, "event/")

first_year = int(cfg_params['starttime'].split('-')[0])
last_year = int(cfg_params['endtime'].split('-')[0] - 1) # inclusive

# Full catalog output path
full_catalog_path_csv = cfg_paths['eq_catalog_filepath']

# Relocated events output path
relocated_dir = join(partial_output_dir, "relocated/")

# eq_json_path = join(partial_output_dir, 'event_catalog.json')

# Prefixes for event names, such that event_name = event_name_prefix + event_id
# e.g. s12345678 is event ID 12345678 from the SCEDC catalog
SCEDC_event_name_prefix = 's'
NCEDC_event_name_prefix = 'n'

combine_columns = ['event_id', 'edatetime', 'emag', 'emagtype', 'elat', 'elon', 'edep', 'nst', 'etype']
output_columns = ['event_name', 'edatetime', 'emag', 'emagtype', 'elat', 'elon', 'edep', 'nst', 'etype', 'relocated']

os.makedirs(catalog_output_dir, exist_ok=True)


# %% [markdown]
# ## Load and prepare relocated catalogs

# %%
# https://scedc.caltech.edu/data/alt-2011-dd-hauksson-yang-shearer.html
reloc_cols = ["event_name", "edatetime", "elat", "elon", "edep"]


# File format:
# 
# 012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
# 1981 01 01 17 45 59.096   3301578  33.50471 -116.76482   6.020  1.50    7832      19     836   105  1151   656  0.25  0.20   0.109   0.343   0.019    33.50487 -116.76460   6.065 le h gc 58
# 1981 01 01 18 57 18.260  12249959  33.54833 -117.76867  12.600  1.85      10                                        59.600   1.200   6.400   0.170                                le   3d   



# ===================== SoCal relocated catalog =====================
socal_catalog_filepath = join(relocated_dir, "hauksson_relocated_eq.gc")

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
df_socal = df_socal[df_socal['year'] >= 1995]
df_socal['event_name'] = 's' + df_socal['event_id'].astype(str)
df_socal['edatetime'] = pd.to_datetime(df_socal[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_socal = df_socal[reloc_cols].reset_index(drop=True)
print("Socal catalog head:")
print(df_socal.head(10), "\n")

# ===================== NorCal relocated catalog ====================
norcal_catalog_filepath = join(relocated_dir, "NCAeqDD.v202112.1")

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

df_norcal = pd.read_fwf(
    norcal_catalog_filepath, colspecs=norcal_colspecs, names=norcal_columns, skiprows=97
)
df_norcal = df_norcal[df_norcal['year'] >= 1995]
df_norcal['event_name'] = 'n' + df_norcal['event_id'].astype(str)
df_norcal['edatetime'] = pd.to_datetime(df_norcal[['year', 'month', 'day', 'hour', 'minute', 'second']])
df_norcal = df_norcal[reloc_cols].reset_index(drop=True)
print("NorCal catalog head:")
print(df_norcal.head(10), "\n")

# ===================== Combine and save ============================
df_relocated = (
    pd.concat([df_socal, df_norcal], axis=0)
    .sort_values(by='edatetime')
    .reset_index(drop=True)
)

print("Combined catalog head:")
print(df_relocated.head(10), "\n")


# %% [markdown]
# ## Download and prepare NCEDC and SCEDC catalogs

# %%
### Download SCEDC full catalogs #######################################

os.makedirs(catalog_output_dir, exist_ok=True)
os.makedirs(partial_output_dir, exist_ok=True)

years = np.arange(first_year, last_year+1, 1).astype(str)

# info https://github.com/SCEDC/SCEDC-catalogs
# format https://scedc.caltech.edu/eq-catalogs/docs/scec_dc.html

scedc_column_names = ['eymd', 'ehms', 'etype', 'egtype', 'emag', 'emagtype', 'elat', 
                'elon', 'edep', 'equal', 'event_id', 'nphase', 'nst']
scedc_urls = ["https://service.scedc.caltech.edu/ftp/catalogs/SCEC_DC/" + el + ".catalog" for el in years]
file_names = [partial_output_dir + 'scedc/' + year + '.catalog' for year in years]
scedc_catalog_filename = f"scedc_{first_year}_{last_year}.pkl"
scedc_catalog_filepath = os.path.join(partial_output_dir, scedc_catalog_filename)

os.makedirs(partial_output_dir + 'scedc/', exist_ok=True)

if not os.path.exists(scedc_catalog_filepath):
    if not all([os.path.exists(file_name) for file_name in file_names]):
        for i in trange(len(scedc_urls), desc='Downloading SCEDC catalogs: '):
            if not os.path.exists(file_names[i]):
                # Download the file from `url` and save it locally under `file_name`:
                with urllib.request.urlopen(scedc_urls[i]) as response, open(file_names[i], 'w') as out_file:
                    text = response.read().decode('utf-8')
                    out_file.write(text)
            else:
                print(f"SCEDC catalog {file_names[i]} already downloaded")
    else:
        print("SCEDC catalogs already downloaded to " + partial_output_dir + "scedc/")

    # load all year catalogs into one dataframe
    scedc_df = pd.DataFrame()
    df = [[]] * len(years)
    for i, year in enumerate(years):
        # header for scedc catalogs is 10 lines
        df[i] = pd.read_csv(file_names[i], skiprows=10, sep=r'\s+', 
                        names=scedc_column_names, skipfooter=1, engine='python')
    scedc_df = pd.concat(df, axis=0).reset_index(drop=True)
    assert len(scedc_df) == sum([len(el) for el in df])

    # write full catalog .pkl file
    scedc_df.to_pickle(scedc_catalog_filepath)
else:
    print(f"Reading SCEDC catalog {scedc_catalog_filepath}")
    scedc_df = pd.read_pickle(scedc_catalog_filepath)

scedc_df_out = scedc_df.copy()

# correct the formats of certain columns
# edatetime column
scedc_df_out['eymd'] = scedc_df_out['eymd'].apply(lambda x: x.replace('/', '-'))
# replace '60.00' with '59.999' in ehms column - for some reason, some 
# events have seconds = 60.0. Replace these with 59.999
scedc_df_out['ehms'] = scedc_df_out['ehms'].apply(lambda x: x.replace('60.00', '59.999'))
bad_idx = np.where(np.array([float(el.split(":")[-1]) for el in scedc_df_out['ehms'].values]) > 60.0)[0]

# these entries are real wonky. idek, but get rid of them
print("The following entries will be deleted because their seconds are greater than 60:")
if len(bad_idx) == 0:
    print("None.")
else:
    print(scedc_df_out.iloc[bad_idx])
scedc_df_out = scedc_df_out.drop(bad_idx, axis=0).reset_index(drop=True)

# properly format edatetime and get rid of other columns
scedc_df_out['edatetime'] = pd.to_datetime(scedc_df_out['eymd'] + 'T' + scedc_df_out['ehms'])
scedc_df_out = scedc_df_out.drop(['eymd', 'ehms'], axis=1)
scedc_df_out = scedc_df_out[combine_columns]

# make sure event IDs are unique
assert len(scedc_df_out['event_id'].unique()) == len(scedc_df_out)

# add source column. event_name can be formed with source + event_id
scedc_df_out['source'] = SCEDC_event_name_prefix

# print out some statistics
print(f"Total number of SCEDC events: {len(scedc_df_out):,}")

#####################################################

### Download NCEDC full catalogs #######################################
# info https://github.com/NCEDC/NCSS-catalogs

ncedc_column_names = ['edatetime', 'elat', 'elon', 'edep', 'emag', 'emagtype',
                'nst', 'gap', 'dmin', 'rms', 'net', 'event_id', 'updated',
                'place', 'etype', 'horiz_error', 'depth_error', 'mag_error', 
                'mag_nst', 'status', 'location_source', 'mag_source']
ncedc_urls = ["https://ncedc.org/pub/catalogs/NCSS-catalogs/" + el + ".ehpcsv" for el in years]
file_names = [partial_output_dir + 'ncedc/' + year + '.catalog' for year in years]
ncedc_catalog_filename = f"ncedc_{first_year}_{last_year}.pkl"
ncedc_catalog_filepath = os.path.join(partial_output_dir, ncedc_catalog_filename)

os.makedirs(partial_output_dir + 'ncedc/', exist_ok=True)
if not os.path.exists(ncedc_catalog_filepath):
    if not all([os.path.exists(file_name) for file_name in file_names]):
        for i in trange(len(ncedc_urls), desc='Downloading NCEDC catalogs: '):
            if not os.path.exists(file_names[i]):
                # Download the file from `url` and save it locally under `file_name`:
                with urllib.request.urlopen(ncedc_urls[i]) as response, open(file_names[i], 'w') as out_file:
                    text = response.read().decode('ascii', errors='replace')
                    out_file.write(text)
                    # weird issue with 2024, downloaded manually
            else:
                print(f"NCEDC catalog {file_names[i]} already downloaded.")
    else:
        print("NCEDC catalogs already downloaded to " + partial_output_dir + "ncedc/")

    # load all catalogs into one dataframe
    ncedc_df = pd.DataFrame()
    df = [[]] * len(years)
    for i, year in enumerate(years):
        df[i] = pd.read_csv(file_names[i], skiprows=1, sep=',', 
                            names=ncedc_column_names,)
    ncedc_df = pd.concat(df, axis=0).reset_index(drop=True)
    assert len(ncedc_df) == sum([len(el) for el in df])

    # write full catalog .pkl file
    ncedc_df.to_pickle(ncedc_catalog_filepath)
else:
    print(f"Reading NCEDC catalog {ncedc_catalog_filepath}")
    ncedc_df = pd.read_pickle(ncedc_catalog_filepath)

ncedc_df_out = ncedc_df.copy()
ncedc_df_out['edatetime'] = pd.to_datetime(ncedc_df['edatetime']).dt.tz_localize(None)
ncedc_df_out = ncedc_df_out[combine_columns]

# filter out earthquakes without a location or magnitude
len0 = len(ncedc_df_out)
ncedc_df_out = ncedc_df_out[np.logical_and(ncedc_df_out['emagtype']!='Unk', ncedc_df_out['elon']!=0.0)].reset_index(drop=True)
print(f"{len0 - len(ncedc_df_out):,} events have no location or magnitude and are discarded")
ncedc_df_out['source'] = NCEDC_event_name_prefix

print(f"Total number of NCEDC events: {len(ncedc_df_out):,}")

### combine catalogs
eq_df = pd.concat([scedc_df_out, ncedc_df_out], axis=0).reset_index(drop=True)
eq_df['event_name'] = eq_df['source'] + eq_df['event_id'].astype(str)

# Relocate events
eq_df = relocate_events(eq_df, df_relocated)

eq_df = eq_df.sort_values(by='edatetime').reset_index(drop=True)

len0 = len(eq_df)

# remove SCEDC events outside polygon, and NCEDC events inside polygon
inside = contains_xy(scedc_poly, eq_df['elon'].to_numpy(), eq_df['elat'].to_numpy())
source = eq_df['source'].to_numpy()

drop_n = inside & (source == NCEDC_event_name_prefix) # NCEDC events inside polygon
drop_s = ~inside & (source == SCEDC_event_name_prefix) # SCEDC events outside polygon
drop_mask = drop_n | drop_s

eq_df = eq_df.loc[~drop_mask].reset_index(drop=True)

print(f"Total number of events: {len(eq_df):,}")

eq_df[output_columns].to_csv(full_catalog_path_csv, index=False)

print("")
print("------ Final report ------")
print(f"Total number of events: {len(eq_df):,}")
print(f"Total NCEDC events: {len(eq_df[eq_df['source'] == NCEDC_event_name_prefix]):,}")
print(f"Total SCEDC events: {len(eq_df[eq_df['source'] == SCEDC_event_name_prefix]):,}")
print(f"Total dropped NCEDC events inside polygon: {drop_n.sum():,} ({drop_n.sum() / len(eq_df) * 100:.2f}%)")
print(f"Total dropped SCEDC events outside polygon: {drop_s.sum():,} ({drop_s.sum() / len(eq_df) * 100:.2f}%)")
print(f"Total number of relocated events: {sum(eq_df['relocated']):,}")

print("\n ------------------------------------------- \n")
df = eq_df[eq_df['emag']>=1.0].reset_index(drop=True)
print(f"M>=1 number of events: {len(df):,}")
print(f"M>=1 NCEDC events: {len(df[df['source'] == NCEDC_event_name_prefix]):,}")
print(f"M>=1 SCEDC events: {len(df[df['source'] == SCEDC_event_name_prefix]):,}")
print(f"M>=1 number of relocated events: {sum(df['relocated']):,} ({sum(df['relocated'])/len(df)*100:.2f}%)")



