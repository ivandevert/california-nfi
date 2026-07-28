import os
import yaml
from pathlib import Path
 
 
def load_config(config_path=None):
    """Load config.yaml, expand ~ and $VARS, and resolve repo-relative paths.
 
    Repo-relative paths (figure_dir, waveforms_temp_dir) are resolved relative
    to the directory containing the config file, so a fresh clone works with no
    edits. Absolute paths in the config are left untouched. dataset_dir and its
    sub-paths are joined into full paths for convenience.
    """
    # Locate config.yaml next to this file. In a .py script, __file__ works.
    # In a notebook, __file__ is undefined, so fall back to the cwd.
    if config_path is None:
        try:
            here = Path(__file__).resolve().parent
        except NameError:  # running in a notebook
            here = Path.cwd()
        config_path = here / "config.yaml"
    config_path = Path(config_path).resolve()
 
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found. Copy config.example.yaml to config.yaml "
            f"and edit it for your machine."
        )
 
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
 
    base = config_path.parent  # repo-relative paths resolve against this
 
    def resolve(p):
        """Expand ~ and $VARS; make relative paths relative to the config dir."""
        p = Path(os.path.expandvars(os.path.expanduser(str(p))))
        return p if p.is_absolute() else (base / p)
 
    paths = cfg["paths"]
    storage_dir = resolve(paths["storage_dir"])
    project_dir = resolve(paths["project_dir"])

    assert storage_dir.is_dir(), f"storage_dir {storage_dir} does not exist."
    assert project_dir.is_dir(), f"project_dir {project_dir} does not exist."

    project_dir_subdirs = cfg['paths']['project_dir_subdirs']
    storage_dir_subdirs = cfg['paths']['storage_dir_subdirs']

    # resolve all the required subdirs
    for subdir in ["catalogs", "figures", "results", "waveforms", "waveforms_temp"]:
        if subdir in project_dir_subdirs:
            name = subdir + "_dir"
            paths[name] = resolve(project_dir / subdir)
        elif subdir in storage_dir_subdirs:
            name = subdir + "_dir"
            paths[name] = resolve(storage_dir / subdir)
        else:
            raise ValueError(f"Subdirectory {subdir} not found in project_dir_subdirs or storage_dir_subdirs.")

    paths['figure_dir'] = paths['figures_dir']
    paths['eq_catalog_filepath'] = paths['catalogs_dir'] / paths['eq_catalog_filename']
    paths['station_catalog_filepath'] = paths['catalogs_dir'] / paths['station_catalog_filename']
    paths['channel_catalog_filepath'] = paths['catalogs_dir'] / paths['channel_catalog_filename']
    cfg['paths'] = {**paths, **cfg["paths"]}

    return cfg
