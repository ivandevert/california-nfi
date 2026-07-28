import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from matplotlib import colors as mcolors
import matplotlib as mpl

# scm_vik_colors = np.loadtxt("/Users/ivandevert/projects/california-nfi/resources/scm_vik.txt")
# nfi_cmap = mcolors.ListedColormap(scm_vik_colors, name='SCM/vik_r')

# try:
#     plt.colormaps.register(cmap=nfi_cmap, name='SCM/vik_r')
# except ValueError:
#     pass  # already registered

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return r * 2 * np.arcsin(np.sqrt(a))

def add_coastlines(ax, lon_bounds, lat_bounds, color='k', linewidth=0.5, **kwargs):
    """Draw coastlines & borders clipped to lon/lat bounds on a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    lon_bounds : tuple of (lon_min, lon_max)
    lat_bounds : tuple of (lat_min, lat_max)
    color : str
        Line color.
    linewidth : float
        Line width.
    **kwargs
        Passed to ax.plot().
    """
    import cartopy.io.shapereader as shpreader

    features = []
    for category, name in [
        ('physical', 'coastline'),
        ('cultural', 'admin_1_states_provinces_lines'),
        ('cultural', 'admin_0_boundary_lines_land'),
    ]:
        try:
            shpfile = shpreader.natural_earth(
                resolution='50m', category=category, name=name)
            reader = shpreader.Reader(shpfile)
            for geom in reader.geometries():
                features.append(geom)
        except Exception:
            pass

    for geom in features:
        parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            coords = np.array(part.coords)
            if len(coords) < 2:
                continue
            lons, lats = coords[:, 0], coords[:, 1]
            inside = (
                (lons >= lon_bounds[0]) & (lons <= lon_bounds[1]) &
                (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
            )
            if not inside.any():
                continue
            splits = np.where(np.diff(inside.astype(int)) != 0)[0] + 1
            segments = np.split(np.arange(len(lons)), splits)
            for seg in segments:
                if len(seg) < 2 or not inside[seg[0]]:
                    continue
                ax.plot(lons[seg], lats[seg], color=color, 
                        linewidth=linewidth, **kwargs)

def make_discrete_nfi_cmap(n_colors=6, clim=(-0.333, 0.333)):
    """6-bin discrete SCM/vik colormap centered on zero."""
    # base_cmap = plt.cm.coolwarm_r
    base_cmap = mpl.colormaps['SCM/vik'].reversed()
    bounds = np.linspace(clim[0], clim[1], n_colors + 1)
    norm_sample = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    colors = [base_cmap(norm_sample((bounds[i] + bounds[i + 1]) / 2))
              for i in range(n_colors)]
    cmap = mcolors.ListedColormap(colors, name='nFI_discrete')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def make_grid(x_range, y_range, box_size_km=10.0, units='deg'):
    """Create a regular grid with approximately square cells.

    Parameters
    ----------
    x_range, y_range : array-like, length 2
        Min/max bounds for each axis.
    box_size_km : float or (float, float)
        Cell size in km.  Scalar → square cells; pair → (dx_km, dy_km).
    units : {'deg', 'km'}
        If 'deg', convert km sizes to degrees using mean latitude.
        If 'km', use km values directly.
    """
    if np.isscalar(box_size_km):
        dx_km, dy_km = box_size_km, box_size_km
    else:
        dx_km, dy_km = box_size_km

    if units == 'deg':
        mean_lat = np.mean(y_range)
        dy = dy_km / 111.0
        dx = dx_km / (111.0 * np.cos(np.radians(mean_lat)))
    elif units == 'km':
        dx = dx_km
        dy = dy_km
    else:
        raise ValueError("units must be 'deg' or 'km'")

    x_edges = np.arange(x_range[0], x_range[1] + dx, dx)
    y_edges = np.arange(y_range[0], y_range[1] + dy, dy)
    return x_edges, y_edges, dx, dy

def _latlon_grid():
    pass

def bin_to_grid(elon, elat, values, lon_edges, lat_edges, stat='median'):
    """Assign events to grid cells and compute a statistic."""
    from scipy.stats import binned_statistic_2d
    result = binned_statistic_2d(
        elon, elat, values,
        statistic=stat,
        bins=[lon_edges, lat_edges]
    )
    return result.statistic, result.x_edge, result.y_edge

def boxplot(x, y, xbins, ax=None, color='r', **kwargs):
    if ax is None:
        ax = plt.gca()

    x = np.asarray(x)
    y = np.asarray(y)

    # filter to values within bin range
    mask = (x >= xbins[0]) & (x <= xbins[-1])
    x, y = x[mask], y[mask]

    inds = np.digitize(x, bins=xbins) - 1
    # digitize returns len(xbins)-1 for x == xbins[-1]; clamp to last bin
    inds = np.clip(inds, 0, len(xbins) - 2)

    unique_inds = np.unique(inds)
    xmids = (xbins[1:] + xbins[:-1]) / 2
    xwidth = xbins[1] - xbins[0]

    X = [y[inds == i] for i in unique_inds]

    defaults = {
        'positions': xmids[unique_inds],
        'manage_ticks': False,
        'widths': xwidth * 0.8,
        'sym': '',
        'whis': (5, 95),
        'boxprops': dict(color=color),
        'medianprops': dict(color=color, linewidth=2),
        'whiskerprops': dict(color=color),
        'capprops': dict(color=color),
    }
    # **kwargs overrides **defaults
    return ax.boxplot(X, **{**defaults, **kwargs})

def errorbar_binned(x, y, xbins, ax=None, color='r', n_sigma=2, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Clip to bins range
    mask = (x >= xbins[0]) & (x <= xbins[-1])
    x, y = x[mask], y[mask]
    inds = np.digitize(x, bins=xbins) - 1
    # digitize returns len(xbins) for x == xbins[-1]; clamp to last bin
    inds = np.clip(inds, 0, len(xbins) - 2)
    xmids_all = (xbins[1:] + xbins[:-1]) / 2
    # Only keep non-empty bins
    occupied = np.unique(inds)
    medians = np.array([np.median(y[inds == i]) for i in occupied])
    sems = np.array([np.std(y[inds == i], ddof=1) / np.sqrt(np.sum(inds == i))
                     if np.sum(inds == i) > 1 else np.nan
                     for i in occupied])
    xmids = xmids_all[occupied]
    defaults = dict(fmt='o', color=color, capsize=3)
    defaults.update(kwargs)
    return ax.errorbar(xmids, medians, yerr=n_sigma * sems, **defaults)

def median_nn(lon, lat, dep, v, N):
    """Compute median of v using N nearest neighbors for each point.
    
    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude in degrees.
    dep : array-like
        Depth in km.
    v : array-like
        Values to compute median of.
    N : int
        Number of nearest neighbors (including self).
    
    Returns
    -------
    v_median : np.ndarray
        Median of v over N nearest neighbors for each point.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    dep = np.asarray(dep, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Convert to Cartesian (km) for proper 3D distance
    R = 6371.0
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = (R - dep) * np.cos(lat_r) * np.cos(lon_r)
    y = (R - dep) * np.cos(lat_r) * np.sin(lon_r)
    z = (R - dep) * np.sin(lat_r)
    
    coords = np.column_stack([x, y, z])  # (npts, 3)
    
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, inds = tree.query(coords, k=N, workers=-1)
    
    v_median = np.median(v[inds], axis=1)
    
    return v_median

def make_interpolation(lon, lat, values, bin_km=10.0, grid_km=10.0,
                       rbf_epsilon=2, mean_lat=None, force_binning=False,
                       kernel='thin_plate_spline', smoothing=0):
    """
    Fit an RBF interpolator to scattered (lon, lat, values) data and
    evaluate it on a regular grid.

    Bins to a median grid first if duplicate locations exist.

    Parameters
    ----------
    lon, lat, values : array-like
        Coordinates and values of input data.
    bin_km : float
        Approximate grid cell size in km for binning duplicates.
    grid_km : float
        Approximate grid cell size in km for the output grid.
    rbf_epsilon : float
        Shape parameter for RBFInterpolator.
    mean_lat : float or None
        Reference latitude for km-to-degree conversion. If None, uses mean of lat.

    Returns
    -------
    dict with keys:
        'rbf'         : RBFInterpolator, the fitted interpolator
        'lon_edges'   : ndarray, longitude bin edges for pcolormesh
        'lat_edges'   : ndarray, latitude bin edges for pcolormesh
        'lon_centers' : ndarray, longitude bin centers
        'lat_centers' : ndarray, latitude bin centers
        'grid'        : ndarray (nlat, nlon), interpolated values on the grid
        'lon_fit'     : ndarray, lon coordinates used for fitting
        'lat_fit'     : ndarray, lat coordinates used for fitting
        'v_fit'       : ndarray, values used for fitting
        'binned'      : bool, whether binning was applied before fitting
    """
    from scipy.interpolate import RBFInterpolator

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    values = np.asarray(values, dtype=float)

    if mean_lat is None:
        mean_lat = np.mean(lat)
    cos_lat = np.cos(np.radians(mean_lat))

    # Check for duplicate locations
    coords = np.column_stack([lon, lat])
    _, _, unique_counts = np.unique(
        coords, axis=0, return_index=True, return_counts=True
    )
    has_duplicates = np.any(unique_counts > 1)

    if has_duplicates or force_binning:
        from scipy.stats import binned_statistic_2d

        dlat_bin = bin_km / 111.0
        dlon_bin = dlat_bin / cos_lat

        bin_lon_edges = np.arange(lon.min(), lon.max() + dlon_bin, dlon_bin)
        bin_lat_edges = np.arange(lat.min(), lat.max() + dlat_bin, dlat_bin)

        bin_grid = binned_statistic_2d(
            lon, lat, values, statistic='median',
            bins=[bin_lon_edges, bin_lat_edges]
        ).statistic.T  # (nlon, nlat) -> (nlat, nlon)

        bin_lon_centers = (bin_lon_edges[1:] + bin_lon_edges[:-1]) / 2
        bin_lat_centers = (bin_lat_edges[1:] + bin_lat_edges[:-1]) / 2
        X, Y = np.meshgrid(bin_lon_centers, bin_lat_centers)

        mask = ~np.isnan(bin_grid.ravel())
        lon_fit = X.ravel()[mask]
        lat_fit = Y.ravel()[mask]
        v_fit = bin_grid.ravel()[mask]
    else:
        lon_fit = lon
        lat_fit = lat
        v_fit = values

    # Fit interpolator
    xy_fit = np.column_stack([lon_fit, lat_fit])
    rbf = RBFInterpolator(xy_fit, v_fit, epsilon=rbf_epsilon, kernel=kernel,
                          smoothing=smoothing)

    # Build output grid
    dlat_grid = grid_km / 111.0
    dlon_grid = dlat_grid / cos_lat

    lon_edges = np.arange(lon.min(), lon.max() + dlon_grid, dlon_grid)
    lat_edges = np.arange(lat.min(), lat.max() + dlat_grid, dlat_grid)
    lon_centers = (lon_edges[1:] + lon_edges[:-1]) / 2
    lat_centers = (lat_edges[1:] + lat_edges[:-1]) / 2

    Xg, Yg = np.meshgrid(lon_centers, lat_centers)
    grid = rbf(np.column_stack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    return {
        'rbf': rbf,
        'lon_edges': lon_edges,
        'lat_edges': lat_edges,
        'lon_centers': lon_centers,
        'lat_centers': lat_centers,
        'lon_g': Xg,
        'lat_g': Yg,
        'grid': grid,
        'lon_fit': lon_fit,
        'lat_fit': lat_fit,
        'v_fit': v_fit,
        'binned': has_duplicates,
    }

# PyVista functions

def make_discrete_cmap(n_colors=6, clim=(-0.333, 0.333)):
    """6-bin discrete coolwarm_r colormap centered on zero."""
    base_cmap = plt.cm.coolwarm_r
    bounds = np.linspace(clim[0], clim[1], n_colors + 1)
    norm_sample = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
    colors = [base_cmap(norm_sample((bounds[i] + bounds[i + 1]) / 2))
              for i in range(n_colors)]
    cmap = mcolors.ListedColormap(colors, name='nFI_discrete')
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def get_coastlines_pv(lon_bounds, lat_bounds, z=0.0):
    """Extract coastlines & borders clipped to lon/lat bounds.

    Coordinates outside the bounding box are discarded, and contiguous
    in-bounds segments are returned as separate line meshes (no markers).

    Parameters
    ----------
    lon_bounds : tuple of (lon_min, lon_max)
    lat_bounds : tuple of (lat_min, lat_max)
    z : float
        Constant z-value for all coastline vertices.

    Returns
    -------
    list of pv.PolyData
        Each element is a polyline mesh with proper line cells.
    """
    import cartopy.io.shapereader as shpreader
    import pyvista as pv
    features = []
    for category, name in [
        ('physical', 'coastline'),
        ('cultural', 'admin_1_states_provinces_lines'),
        ('cultural', 'admin_0_boundary_lines_land'),
    ]:
        try:
            shpfile = shpreader.natural_earth(
                resolution='10m', category=category, name=name)
            reader = shpreader.Reader(shpfile)
            for geom in reader.geometries():
                features.append(geom)
        except Exception:
            pass

    meshes = []
    for geom in features:
        if hasattr(geom, 'geoms'):
            parts = list(geom.geoms)
        else:
            parts = [geom]

        for part in parts:
            coords = np.array(part.coords)
            if len(coords) < 2:
                continue
            lons, lats = coords[:, 0], coords[:, 1]

            # Mask: which vertices fall inside the bounding box
            inside = (
                (lons >= lon_bounds[0]) & (lons <= lon_bounds[1]) &
                (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
            )
            if not inside.any():
                continue

            # Split into contiguous runs of in-bounds vertices
            splits = np.where(np.diff(inside.astype(int)) != 0)[0] + 1
            segments = np.split(np.arange(len(lons)), splits)

            for seg in segments:
                if len(seg) < 2 or not inside[seg[0]]:
                    continue
                seg_pts = np.column_stack([
                    lons[seg], lats[seg], np.full(len(seg), z)
                ])
                n = len(seg_pts)
                # Build line cells: each cell is [2, i, i+1]
                cells = np.column_stack([
                    np.full(n - 1, 2, dtype=np.int64),
                    np.arange(n - 1, dtype=np.int64),
                    np.arange(1, n, dtype=np.int64),
                ]).ravel()
                meshes.append(pv.PolyData(seg_pts, lines=cells))

    return meshes

def get_qfaults_pv(lon_bounds, lat_bounds, z=0.0,
                   simplify_tol=0.001,
                   shp_path='/Users/ivandevert/projects/california-nfi/comparison_datasets/qfaults/SHP/Qfaults_US_Database.shp'):
    """Extract USGS Quaternary faults clipped to lon/lat bounds.

    Returns a single pv.PolyData containing all fault segments as disjoint
    line cells. Wrapped in a 1-element list for API compatibility with
    get_coastlines_pv.
    """
    import geopandas as gpd
    import pyvista as pv
    from shapely.geometry import LineString, MultiLineString

    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    gdf = gpd.read_file(shp_path, bbox=(lon_min, lat_min, lon_max, lat_max))
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    clipped = gdf.geometry.clip_by_rect(lon_min, lat_min, lon_max, lat_max)
    clipped = clipped[~clipped.is_empty & clipped.notna()]

    if simplify_tol is not None:
        clipped = clipped.simplify(simplify_tol, preserve_topology=False)

    # Flatten Multi* to single LineStrings, collect coord arrays
    coord_arrays = []
    for geom in clipped:
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            c = np.asarray(geom.coords)
            if len(c) >= 2:
                coord_arrays.append(c)
        else:  # MultiLineString
            for p in geom.geoms:
                c = np.asarray(p.coords)
                if len(c) >= 2:
                    coord_arrays.append(c)

    if not coord_arrays:
        return [pv.PolyData()]

    # Concatenate all vertices into one array
    lengths = np.array([len(c) for c in coord_arrays])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total_pts = offsets[-1]

    pts = np.empty((total_pts, 3))
    pts[:, 2] = z
    for c, off, n in zip(coord_arrays, offsets[:-1], lengths):
        pts[off:off + n, :2] = c

    # Build line cells for each segment. For a polyline of n vertices,
    # VTK wants [n, v0, v1, ..., v_{n-1}].
    n_segs = len(coord_arrays)
    n_pairs = int((lengths - 1).sum())  # total line segments (pairs)
    # Each segment contributes one cell of length n+1 ints: [n, idx0..idx_{n-1}]
    cell_buf = np.empty(int(lengths.sum()) + n_segs, dtype=np.int64)
    pos = 0
    for off, n in zip(offsets[:-1], lengths):
        cell_buf[pos] = n
        cell_buf[pos + 1:pos + 1 + n] = np.arange(off, off + n)
        pos += n + 1

    mesh = pv.PolyData(pts, lines=cell_buf)
    return [mesh]

def pyvista_scatter(
        lon, 
        lat, 
        dep, 
        values, 
        depth_exag=1, 
        clim=None, 
        bg='white', 
        point_size=3, 
        save=None, 
        plotvaluename="Values", 
        plot_faults=False,
        feature_padding=1.5,
    ):
    import pyvista as pv
    if clim is None:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        absmax = max(abs(vmin), abs(vmax))
        clim = (-absmax, absmax)


    # Coordinate system: x=lon, y=lat, z=-depth (with exaggeration)
    # Scale depth so that it's visible relative to lat/lon degrees.
    # ~1 degree ≈ 111 km, so depth_exag converts km to "degree-like" units.
    z = -dep / 111.0 * depth_exag

    # ── PyVista point cloud ──
    points = np.column_stack([lon, lat, z])
    cloud = pv.PolyData(points)
    cloud['values'] = values

    # ── Colormap ──
    cmap, norm, bounds = make_discrete_cmap(
        n_colors=6, clim=clim)

    # ── Coastlines ──
    pad = feature_padding
    lon_bounds = (lon.min() - pad, lon.max() + pad)
    lat_bounds = (lat.min() - pad, lat.max() + pad)
    # lon_bounds = (-125.0, -114.0)
    # lat_bounds = (32.0, 43.0)
    print("Loading coastlines & borders...")
    coast_meshes = get_coastlines_pv(lon_bounds, lat_bounds, z=0.0)
    print(f"  {len(coast_meshes)} line segments loaded.")

    if plot_faults:
        print("Loading faults...")
        fault_meshes = get_qfaults_pv(lon_bounds, lat_bounds, z=0.0)
        print(f"  {len(fault_meshes)} line segments loaded.")

    # ── Build the plotter ──
    pv.global_theme.background = bg
    pv.global_theme.font.color = 'black'
    pv.global_theme.font.size = 14

    pl = pv.Plotter()

    # Add earthquake points
    pl.add_mesh(
        cloud,
        scalars='values',
        cmap=cmap,
        clim=clim,
        point_size=point_size,
        render_points_as_spheres=False,
        scalar_bar_args={
            'title': plotvaluename,
            'n_labels': len(bounds),
            'fmt': '%.3f',
            'width': 0.3,
            'height': 0.06,
            'position_x': 0.35,
            'position_y': 0.02,
            'title_font_size': 16,
            'label_font_size': 12,
        },
    )

    # Add coastlines
    for mesh in coast_meshes:
        pl.add_mesh(mesh, color='black', line_width=1.5)

    if plot_faults:
        for mesh in fault_meshes:
            pl.add_mesh(mesh, color='grey', line_width=1.5)

    # Optional: add a translucent surface at z=0 for geographic context
    x_grid = np.linspace(lon_bounds[0], lon_bounds[1], 50)
    y_grid = np.linspace(lat_bounds[0], lat_bounds[1], 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    zz = np.zeros_like(xx)
    surface = pv.StructuredGrid(xx, yy, zz)
    pl.add_mesh(surface, color='whitesmoke', opacity=0.15)

    # Axis labels
    pl.add_axes(
        xlabel='Longitude', ylabel='Latitude', zlabel='Depth',
        line_width=2,
    )
    # Enable tracking and link callback
    pl.view_xy()
    # pl.track_click_position(callback=mouse_click_callback, side='right')
    # Show or save
    if save:
        pl.export_html(save)
    else:
        pl.show()

def correlation_matrix_plot(
    dataframes,
    labels,
    id_col='event_name',
    value_col='nfi',
    figsize=None,
    hist_bins=20,
    hist_color='steelblue',
    scatter_s=4,
    scatter_color='steelblue',
    scatter_alpha=0.5,
    fit_color='magenta',
    fit_lw=1.5,
    corr_fontsize=12,
    label_fontsize=11,
    diagonal='histogram',
    upper_content='corr',
    savefig=None,
    dpi=150,
    diag_lim=None,
    M_min=None
):
    """
    Plot a correlation matrix with scatter plots on one triangle,
    correlation coefficients on the other, and histograms on the diagonal.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Each must contain `id_col` and `value_col`.
    labels : list of str
        Display names for each dataframe (axis labels).
    id_col : str
        Column used to merge dataframes (inner join).
    value_col : str
        Column whose values are compared.
    figsize : tuple or None
        Figure size. Defaults to (2.4*N, 2.4*N).
    hist_bins : int
        Number of bins for diagonal histograms.
    hist_color : str
        Histogram bar colour.
    scatter_s : float
        Marker size for scatter plots.
    scatter_color : str
        Scatter point colour.
    scatter_alpha : float
        Scatter point alpha.
    fit_color : str
        Colour of the linear‑fit line.
    fit_lw : float
        Line width for the fit line.
    corr_fontsize : float
        Font size of the correlation coefficient text.
    label_fontsize : float
        Font size of axis labels.
    diagonal : str
        'histogram' (default) or 'none'.
    upper_content : str
        'corr' to show Pearson r in the upper triangle, or 'none' to leave
        those panels blank.
    savefig : str or None
        If provided, save figure to this path.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig, axs
    """
    N = len(dataframes)
    if figsize is None:
        figsize = (2.4 * N, 2.4 * N)
    
    if M_min is not None:
        dataframes = [df[df['emag'] >= M_min].reset_index(drop=True) for df in dataframes]

    # --- Clean per-dataframe values (drop NaNs in value_col) --------------
    clean = []
    for k in range(N):
        tmp = dataframes[k][[id_col, value_col]].dropna(subset=[value_col])
        clean.append(tmp)

    # --- Pairwise merge helper --------------------------------------------
    def _get_pair(i, j):
        """Return matched (x, y) arrays for clean[j] vs clean[i], or None."""
        m = clean[j].merge(clean[i], on=id_col, how='inner',
                           suffixes=('_x', '_y'))
        if len(m) < 2:
            return None, None
        return m[f'{value_col}_x'].values, m[f'{value_col}_y'].values

    # --- Build figure -----------------------------------------------------
    fig, axs = plt.subplots(
        N, N, figsize=figsize, squeeze=False,
        gridspec_kw={'hspace': 0.08, 'wspace': 0.08},
    )

    for i in range(N):
        for j in range(N):
            ax = axs[i, j]

            # ---- Diagonal: histogram ------------------------------------
            if i == j:
                v = clean[i][value_col].values
                if diagonal == 'histogram' and len(v) > 0:
                    ax.hist(
                        v, bins=hist_bins,
                        color=hist_color, edgecolor='white', linewidth=0.3,
                    )
                    ax.text(
                        0.05, 0.95, f'n={len(v):,}',
                        transform=ax.transAxes, ha='left', va='top',
                        fontsize=9,
                    )
                else:
                    ax.axis('off')

            # ---- Lower triangle: scatter + fit --------------------------
            elif i > j:
                x, y = _get_pair(i, j)
                if x is not None:
                    ax.scatter(
                        x, y, s=scatter_s, c=scatter_color,
                        alpha=scatter_alpha, edgecolors='none', rasterized=True,
                    )
                    m, b = np.polyfit(x, y, 1)
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                    # ax.plot(xfit, m * xfit + b, color=fit_color, lw=fit_lw)

                    if diag_lim is not None:
                        ax.plot([-diag_lim, diag_lim], [-diag_lim, diag_lim], c='k', lw=1)

            # ---- Upper triangle: correlation text -----------------------
            else:
                if upper_content == 'corr':
                    x, y = _get_pair(i, j)
                    if x is not None:
                        r = np.corrcoef(x, y)[0, 1]
                        ax.text(
                            0.5, 0.5, f'{r:.2f}',
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=corr_fontsize, fontweight='bold',
                        )
                ax.axis('off')

            # ---- Axis labels & ticks ------------------------------------
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=label_fontsize)
            else:
                ax.set_ylabel(labels[i], fontsize=label_fontsize)
                # ax.tick_params(labelleft=False)

            if i == N - 1 and j != N - 1:
                ax.set_xlabel(labels[j], fontsize=label_fontsize)
            else:
                ax.set_xlabel(labels[j], fontsize=label_fontsize)
                # ax.tick_params(labelbottom=False)

            ax.tick_params(labelsize=8)

    if savefig is not None:
        fig.savefig(savefig, dpi=dpi, bbox_inches='tight')

    return fig, axs


def load_qfaults(
    lat_range=None,
    lon_range=None,
    faults_filepath="/Users/ivandevert/projects/california-nfi/comparison_datasets/qfaults/SHP/Qfaults_US_Database.shp",
    offshore_filepath="/Users/ivandevert/projects/california-nfi/comparison_datasets/qfaults/SHP/ca_offshore.shp",
):
    """
    Load onshore and offshore Quaternary faults, harmonize their columns,
    and optionally restrict to faults intersecting a lat/lon bounding box.

    Parameters
    ----------
    lat_range : tuple(float, float), optional
        (min_lat, max_lat). If given with lon_range, faults entirely outside
        the box are discarded; faults partially inside are kept in full.
    lon_range : tuple(float, float), optional
        (min_lon, max_lon).
    faults_filepath : str
        Path to the onshore Qfaults shapefile.
    offshore_filepath : str
        Path to the offshore faults shapefile.

    Returns
    -------
    geopandas.GeoDataFrame
        Combined faults with consistent columns. No type filtering applied.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    gdf_onshore = gpd.read_file(faults_filepath)
    gdf_offshore = gpd.read_file(offshore_filepath)

    # Normalize offshore columns to match onshore conventions
    gdf_offshore.columns = gdf_offshore.columns.str.lower()
    gdf_offshore = gdf_offshore.rename(columns={"shape_leng": "Shape_Leng"})
    gdf_offshore["Shape_Leng"] = 1000 * gdf_offshore["Shape_Leng"]
    gdf_offshore = gdf_offshore.rename(
        columns={"flt_age": "age", "line_type": "linetype"}
    )
    gdf_offshore = gdf_offshore.drop(
        columns=[
            "fault_zone",
            "other_name",
            "mapped_sca",
            "expression",
            "flt_source",
            "location",
        ],
        errors="ignore",
    )

    # Harmonize categorical values
    age_map = {
        'pre-Quaternary': 'unspecified',
        "undifferentiated Quaternary": "undifferentiated Quaternary",
        "latest Quaternary": "latest Quaternary",
        "historic": "historic",
        "late Quaternary": "late Quaternary",
        "pre-Quaternary": "pre-Quaternary"
    }
    slip_rate_map = {
        "unspecified": "Unspecified",
        "Unspecified": "Unspecified",
        "<0.2": "Less than 0.2 mm/yr",
        "0.2-1.0": "Between 0.2 and 1.0 mm/yr",
        "1-5": "Between 1.0 and 5.0 mm/yr",
        ">5": "Greater than 5.0 mm/yr",
    }
    linetype_map = {
        "Accurately Located": "Well Constrained",
        "Approximately Located": "Moderately Constrained",
        "Inferred": "Inferred",
        "unspecified": "Inferred",
    }
    offshore_slip_sense_map = {
        'unspecified': 'Unspecified',
        'D':           'Right lateral',        # dextral
        'S':           'Left lateral',         # sinistral
        'R':           'Reverse',
        'N':           'Normal',
        'DR':          'Unspecified',        # dextral-dominant oblique → closest is Right lateral
        'RD':          'Unspecified',              # reverse-dominant oblique → closest is Reverse
        'RS':          'Unspecified',              # reverse-dominant, sinistral component
        'SR':          'Unspecified',         # sinistral-dominant, reverse component — maps to Left lateral; but could argue Strike slip
        'ND':          'Unspecified',               # normal-dominant, dextral component
    }

    onshore_slip_sense_map = {
        'Normal':                'Normal',
        'Right lateral':         'Right lateral',
        'Left lateral':          'Left lateral',
        'Thrust':                'Reverse',          # thrust is low-angle reverse
        'Reverse':               'Reverse',
        'Unspecified':           'Unspecified',
        'Strike slip':           'Strike slip',      # unresolved sense (left vs right)
        'Anticline':             'Unspecified',      # fold, not a slip sense
        'Left lateral; Normal':  'Unspecified',      # oblique
        'Syncline':              'Unspecified',      # fold
        'Unspecified; Normal':   'Unspecified',      # oblique/partial
        'Reverse; Left lateral': 'Unspecified',      # oblique
        'Monocline':             'Unspecified',      # fold
        'No data':               'Unspecified',
        'Thrust; Strike slip':   'Unspecified',      # oblique
        'Homocline':             'Unspecified',      # fold
    }

    gdf_offshore["age"] = gdf_offshore["age"].map(age_map)
    gdf_offshore["slip_rate"] = gdf_offshore["slip_rate"].map(slip_rate_map)
    gdf_offshore["linetype"] = gdf_offshore["linetype"].map(linetype_map)
    gdf_offshore['slip_sense'] = gdf_offshore['slip_sense'].map(offshore_slip_sense_map)

    gdf_onshore['slip_sense'] = gdf_onshore['slip_sense'].map(onshore_slip_sense_map)

    gdf = pd.concat([gdf_onshore, gdf_offshore], ignore_index=True).reset_index(
        drop=True
    )

    # Spatial filter: keep faults whose geometry intersects the box
    if lat_range is not None and lon_range is not None:
        bbox = box(min(lon_range), min(lat_range), max(lon_range), max(lat_range))
        gdf = gdf[gdf.intersects(bbox)].reset_index(drop=True)

    return gdf