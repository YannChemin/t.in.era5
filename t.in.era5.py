#!/usr/bin/env python3
############################################################################
#
# MODULE:       t.in.era5
# AUTHOR:       Yann Chemin
# PURPOSE:      Imports one or more daily climate variables as GRASS
#               space-time raster datasets (STRDS), one per variable.
#               For each requested variable, ERA5-Land is tried first
#               via the Copernicus Climate Data Store (its ~9km grid is
#               generally preferable for land applications); if that
#               fails for a given period (temporal gap, no ~/.cdsapirc
#               configured, or the CDS "derived daily statistics"
#               product's known restriction against accumulated
#               variables -- see NOTES), plain ERA5 (~31km grid, longer
#               record) is used instead for that period -- by default
#               read straight from the public, no-login-required
#               ARCO-ERA5 Google Cloud Zarr archive, or from CDS's own
#               plain-ERA5 product if -c is given. Requests are chunked
#               per (variable, year, month) so a fallback only affects
#               the affected chunk, and so repeat runs can reuse a
#               local cache instead of re-downloading.
# COPYRIGHT:    (C) 2026 by Yann Chemin
#               Released into the public domain -- see LICENSE (Unlicense).
#
############################################################################

# %module
# % description: Imports ERA5-Land (falling back to plain ERA5, by default via the public ARCO-ERA5 Google Cloud archive) daily climate variables as one space-time raster dataset per variable.
# % keyword: temporal
# % keyword: import
# % keyword: climate
# % keyword: hydrology
# %end
# %option
# % key: variables
# % type: string
# % multiple: yes
# % required: yes
# % options: precipitation,temperature,temperature_min,temperature_max,dewpoint_temperature,potential_evaporation,solar_radiation,wind_u,wind_v,surface_pressure,snowfall
# % description: Climate variables to import, one STRDS per variable (named <output_prefix>_<variable>)
# %end
# %option
# % key: start
# % type: string
# % required: yes
# % description: Start date, YYYY-MM-DD
# %end
# %option
# % key: end
# % type: string
# % required: yes
# % description: End date, YYYY-MM-DD (inclusive)
# %end
# %option
# % key: area
# % type: string
# % required: no
# % key_desc: north,west,south,east
# % description: Bounding box in WGS84 degrees (north,west,south,east); default derived from the current region
# %end
# %option
# % key: output_prefix
# % type: string
# % required: yes
# % answer: era5
# % description: STRDS name prefix; each variable becomes <output_prefix>_<variable>
# %end
# %option G_OPT_M_DIR
# % key: cache_dir
# % required: no
# % description: Directory to cache downloaded NetCDF files in (default a temporary, run-scoped directory -- pass a persistent path to avoid re-downloading on repeat runs)
# %end
# %option G_OPT_M_DIR
# % key: arco_world_cache
# % required: no
# % description: Directory to permanently cache full-world ARCO-ERA5 monthly data in, reused across every future run regardless of area/output_prefix (default $HOME/RSDATA/ERA5World) -- ARCO-ERA5 downloads a whole-globe chunk per hour touched no matter how small the requested area is, so keeping the whole globe instead of discarding it costs nothing extra and means that time period is never fetched from Google Cloud again for any region
# %end
# %flag
# % key: e
# % description: Force plain ERA5 for every variable (skip the ERA5-Land attempt entirely)
# %end
# %flag
# % key: c
# % description: Use the CDS API instead of ARCO-ERA5 (Google Cloud) for the plain-ERA5 tier -- requires a working ~/.cdsapirc for that tier too; by default that tier reads the public, no-login ARCO-ERA5 Zarr archive instead
# %end
# %flag
# % key: h
# % description: Hourly output instead of daily -- one raster per hour (~24x more STRDS maps for the same period, budget cache space and t.register time accordingly). Supported by every source tier (ERA5-Land raw-hourly, ARCO-ERA5, and plain-ERA5 raw-hourly via CDS); accumulated fields (precipitation, potential_evaporation, solar_radiation, snowfall) are correctly de-accumulated to genuine per-hour amounts, not just relabeled daily totals -- see NOTES in t.in.era5.md for each source's forecast-cycle reset schedule
# %end

import atexit
import calendar
import datetime
import glob
import os
import shutil
import sys


def _add_bundled_venv_to_path():
    """If this is an installed addon built with `make venv` (see
    Makefile), a dedicated virtualenv holding cdsapi/xarray/gcsfs/zarr/
    netCDF4 lives at ../etc/t.in.era5/venv relative to this script's own
    install location. GRASS always runs addon scripts with GRASS's own
    Python, which normally has none of these, and a plain system-wide
    `pip install` into that Python is blocked outright on modern
    Debian/Ubuntu by PEP 668's externally-managed-environment guard --
    so prepend the venv's site-packages instead, letting every lazy
    `import cdsapi`/`import xarray`/etc. below resolve without ever
    touching GRASS's own Python installation. A no-op (nothing to find)
    for a from-source dev checkout that never ran `make venv`, or where
    those packages are already available directly."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(script_dir, "..", "etc", "t.in.era5", "venv")
    for site_packages in glob.glob(
        os.path.join(venv_dir, "lib", "python*", "site-packages")
    ) + glob.glob(os.path.join(venv_dir, "Lib", "site-packages")):
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)


_add_bundled_venv_to_path()

import numpy as np

import grass.script as gs

TMP_RASTERS = []
TMP_DIR = None


def cleanup():
    if TMP_RASTERS:
        gs.run_command(
            "g.remove",
            flags="f",
            type="raster",
            name=TMP_RASTERS,
            quiet=True,
            errors="ignore",
        )
    if TMP_DIR and os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)


# Nominal native grid spacing in degrees, used as a write_geotiff()
# fallback when a requested area is smaller than one grid cell (CDS then
# returns a single point along that axis, with nothing to diff against).
NATIVE_RESOLUTION_DEG = {
    "era5land": 0.1, "era5": 0.25, "arco": 0.25, "arco_gap_cds": 0.25,
}

# Public, no-auth-required Google Cloud Zarr mirror of ERA5 (plain, not
# ERA5-Land): a straight GRIB->Zarr conversion of the raw archive with no
# reprocessing, per the arco-era5 README -- so, unlike CDS's
# derived-era5-single-levels-daily-statistics (which returns already
# hour-ending values), accumulated fields here are still in ECMWF's raw
# forecast-accumulation form and must be de-accumulated locally (see
# fetch_arco()). See https://github.com/google-research/arco-era5.
ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Default location for fetch_arco_world_month()'s permanent, area-
# independent cache of full-globe ARCO-ERA5 monthly data (see that
# function's docstring for why keeping the whole globe instead of just
# the requested area is free: ARCO-ERA5's chunking already forces a
# whole-globe download per hour touched, area or no area).
ARCO_WORLD_CACHE_DEFAULT = os.path.expanduser(os.path.join("~", "RSDATA", "ERA5World"))


def _deaccumulate_hourly(da, time_dim, reset_hours):
    """De-accumulates a raw ECMWF forecast-accumulated hourly field into
    genuine per-hour amounts. ECMWF's raw accumulated fields (total
    precipitation, potential_evaporation, solar_radiation, snowfall) are
    NOT hourly increments -- each value is a running total since the
    most recent "reset" hour, climbing monotonically until the next
    reset. So: the reading at a reset hour is itself already a genuine
    1-hour amount (nothing to diff against -- the previous reading
    belongs to a different accumulation cycle entirely); every other
    reading must be differenced from the immediately preceding hour to
    recover that hour's own amount.

    `reset_hours` is the set of UTC hours (0..23) at which a fresh
    accumulation cycle begins. Two real, DIFFERENT schedules are used by
    this module's two raw-hourly sources -- do not conflate them:
      - ARCO-ERA5 / plain-ERA5's raw forecast-cycle data: two 12h cycles
        per day, based at 06 and 18 UTC, each cycle's first VALID
        (non-zero) hour at 07 and 19 UTC respectively -> reset_hours =
        {7, 19} (see fetch_arco_world_month()'s docstring; the same
        convention applies to plain-ERA5's raw hourly CDS product,
        since ARCO is an unmodified Zarr mirror of that same archive).
      - ERA5-Land's raw hourly data: a single reset per calendar day, at
        01 UTC -> reset_hours = {1} (see load_daily()'s original
        daily-reduction comment, confirmed by inspection there: values
        climb from near-zero at 01 UTC to a peak at 00 UTC the
        following day).

    Returns a DataArray one element shorter than `da` along `time_dim`
    (xr.diff()'s usual length reduction) -- the first output element
    corresponds to da's SECOND input timestamp, since there is nothing
    to output for the very first raw reading (it's consumed as the base
    point for whatever comes after it, unless it happens to itself be a
    reset hour, in which case it's returned as-is rather than diffed)."""
    import xarray as xr

    hour = da[time_dim].dt.hour
    diffed = da.diff(time_dim)
    is_reset = hour[1:].isin(list(reset_hours))
    hourly = xr.where(is_reset, da.isel({time_dim: slice(1, None)}), diffed)
    return hourly.assign_coords({time_dim: da[time_dim].isel({time_dim: slice(1, None)})})


# ECMWF raw forecast-cycle accumulated fields (ARCO-ERA5 and plain-ERA5's
# raw hourly CDS product both use this schedule -- see
# _deaccumulate_hourly()'s docstring): two 12h cycles/day based at 06/18
# UTC, first valid (genuine, non-diffed) hour at 07/19 UTC.
ECMWF_CYCLE_RESET_HOURS = {7, 19}
# ERA5-Land's raw hourly accumulated fields: single daily reset at 01 UTC.
ERA5LAND_RESET_HOURS = {1}


class ArcoDataUnavailable(Exception):
    """Raised when a requested month is beyond what ARCO-ERA5 has
    ingested at all yet (neither final nor preliminary ERA5T) -- see
    _arco_availability()."""


def _arco_availability(ds_full, month_end_excl):
    """Classifies a calendar month's data quality in the ARCO-ERA5 store
    using its own valid_time_stop / valid_time_stop_era5t attributes
    (confirmed present via a live read: e.g. valid_time_stop=2026-04-30
    for the final/stable ERA5 record, valid_time_stop_era5t=2026-08-17
    for the preliminary ERA5T extension ARCO also carries in the same
    array). ERA5T values are provisional and get silently revised *in
    place* once the final ERA5 ships for that period, typically ~3
    months later -- so they must never be conflated with permanent,
    never-changing final data in the world cache. Returns "final",
    "era5t", or "unavailable" (nothing ingested yet, even provisionally
    -- see fetch_month()'s CDS fallback for that case). A dataset with
    no valid_time_stop attribute at all (e.g. a synthetic test dataset
    with no real-world metadata) is treated as unconditionally "final",
    since there is nothing to check availability against."""
    valid_stop_str = ds_full.attrs.get("valid_time_stop")
    if valid_stop_str is None:
        return "final"
    month_last_day = month_end_excl - np.timedelta64(1, "D")
    valid_stop = np.datetime64(valid_stop_str)
    if month_last_day <= valid_stop:
        return "final"
    valid_stop_era5t = np.datetime64(
        ds_full.attrs.get("valid_time_stop_era5t", valid_stop_str)
    )
    if month_last_day <= valid_stop_era5t:
        return "era5t"
    return "unavailable"

# variable key -> CDS variable name (same name used for both ERA5-Land and
# ERA5), whether it's an accumulated (flux/depth-since-last-step) field
# (which the "derived ... daily-statistics" products refuse to compute
# for ERA5-Land, forcing a raw-hourly-then-resample path instead), the
# daily statistic to request for non-accumulated fields, and the unit
# conversion applied to match the units r.hydro.hbv/HBV expect.
VARIABLES = {
    "precipitation": dict(
        cds_name="total_precipitation",
        accumulated=True,
        convert=lambda x: x * 1000.0,  # m -> mm
        description="Total precipitation, daily sum (mm/d)",
    ),
    "temperature": dict(
        cds_name="2m_temperature",
        accumulated=False,
        daily_statistic="daily_mean",
        convert=lambda x: x - 273.15,  # K -> degC
        description="2m air temperature, daily mean (deg C)",
    ),
    "temperature_min": dict(
        cds_name="2m_temperature",
        accumulated=False,
        daily_statistic="daily_minimum",
        convert=lambda x: x - 273.15,
        description="2m air temperature, daily minimum (deg C)",
    ),
    "temperature_max": dict(
        cds_name="2m_temperature",
        accumulated=False,
        daily_statistic="daily_maximum",
        convert=lambda x: x - 273.15,
        description="2m air temperature, daily maximum (deg C)",
    ),
    "dewpoint_temperature": dict(
        cds_name="2m_dewpoint_temperature",
        accumulated=False,
        daily_statistic="daily_mean",
        convert=lambda x: x - 273.15,
        description="2m dewpoint temperature, daily mean (deg C)",
    ),
    "potential_evaporation": dict(
        cds_name="potential_evaporation",
        accumulated=True,
        # ERA5(-Land) potential_evaporation is negative-down (loss);
        # take the magnitude and convert m -> mm.
        convert=lambda x: np.abs(x) * 1000.0,
        description="Potential evapotranspiration, daily sum (mm/d)",
    ),
    "solar_radiation": dict(
        cds_name="surface_solar_radiation_downwards",
        accumulated=True,
        convert=lambda x: x / 1.0e6,  # J/m2 -> MJ/m2
        description="Surface solar radiation downwards, daily sum (MJ/m2/d)",
    ),
    "wind_u": dict(
        cds_name="10m_u_component_of_wind",
        accumulated=False,
        daily_statistic="daily_mean",
        convert=lambda x: x,
        description="10m U wind component, daily mean (m/s)",
    ),
    "wind_v": dict(
        cds_name="10m_v_component_of_wind",
        accumulated=False,
        daily_statistic="daily_mean",
        convert=lambda x: x,
        description="10m V wind component, daily mean (m/s)",
    ),
    "surface_pressure": dict(
        cds_name="surface_pressure",
        accumulated=False,
        daily_statistic="daily_mean",
        convert=lambda x: x / 1000.0,  # Pa -> kPa
        description="Surface pressure, daily mean (kPa)",
    ),
    "snowfall": dict(
        cds_name="snowfall",
        accumulated=True,
        convert=lambda x: x * 1000.0,  # m (of water equiv.) -> mm
        description="Snowfall, daily sum (mm/d water equivalent)",
    ),
}


def month_chunks(start_date, end_date):
    """Yields (year, month, [day, day, ...]) tuples covering every
    calendar month touched by [start_date, end_date] inclusive. Always
    the *complete* month (every day, 1..last), not just the days inside
    [start_date, end_date] -- so a cached month is safe to reuse for any
    other request that also touches that month, regardless of exactly
    which days that request needs (the caller filters to its own exact
    date range after loading); fetching a day range narrower than a full
    month, keyed only by (variable, year, month), would otherwise let a
    later request silently reuse an incomplete cached month."""
    cur = start_date.replace(day=1)
    while cur <= end_date:
        _, n_days = calendar.monthrange(cur.year, cur.month)
        yield cur.year, cur.month, list(range(1, n_days + 1))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)


def cds_client():
    import cdsapi

    return cdsapi.Client()


def fetch_era5land_instantaneous(client, var_info, year, month, days, area, out_nc):
    client.retrieve(
        "derived-era5-land-daily-statistics",
        {
            "variable": var_info["cds_name"],
            "year": str(year),
            "month": "%02d" % month,
            "day": ["%02d" % d for d in days],
            "daily_statistic": var_info["daily_statistic"],
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": area,
        },
        out_nc,
    )


def fetch_era5land_raw_hourly(client, var_info, year, month, days, area, out_nc):
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": var_info["cds_name"],
            "year": str(year),
            "month": "%02d" % month,
            "day": ["%02d" % d for d in days],
            "time": ["%02d:00" % h for h in range(24)],
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        out_nc,
    )


def fetch_era5(client, var_info, year, month, days, area, out_nc):
    daily_statistic = "daily_sum" if var_info["accumulated"] else var_info["daily_statistic"]
    client.retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": var_info["cds_name"],
            "year": str(year),
            "month": "%02d" % month,
            "day": ["%02d" % d for d in days],
            "daily_statistic": daily_statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": area,
        },
        out_nc,
    )


def arco_client():
    """chunks=None (not "auto") matches Google's own arco-era5 usage
    examples for this exact store: since the array's own on-disk chunks
    already span the full globe per hour (see fetch_arco()'s docstring),
    "auto" only adds a dask-array wrapping layer that groups those
    already-maximal chunks into even bigger ones (triggering xarray's
    "specified chunks separate the stored chunks" warning) without
    reducing bytes transferred, and pulls in a dask dependency for no
    benefit. chunks=None instead reads directly through zarr's own
    lazy/indexed access, materializing a plain numpy array as soon as a
    .sel()/.isel() slice is taken -- confirmed faster in practice, not
    just simpler (no dask needed at all)."""
    import xarray as xr

    return xr.open_zarr(ARCO_ERA5_URL, chunks=None, storage_options=dict(token="anon"))


def _arco_select_area(da, area):
    """Subsets an ARCO-ERA5 DataArray to the requested [north, west, south,
    east] WGS84 bbox, converting from the module's -180..180 convention to
    ARCO's native 0..360 longitude and handling the antimeridian-wrap case
    (west > east once both are folded into 0..360). Returns the slice with
    longitude renormalized back to -180..180 ascending, to match the
    convention CDS downloads already use elsewhere in this module."""
    north, west, south, east = area
    da = da.sel(latitude=slice(north, south))

    w360 = west % 360.0
    e360 = east % 360.0
    if w360 <= e360:
        da = da.sel(longitude=slice(w360, e360))
    else:
        import xarray as xr

        da = xr.concat(
            [
                da.sel(longitude=slice(w360, 360.0)),
                da.sel(longitude=slice(0.0, e360)),
            ],
            dim="longitude",
        )

    new_lon = ((da["longitude"].values + 180.0) % 360.0) - 180.0
    return da.assign_coords(longitude=new_lon).sortby("longitude")


def fetch_arco_world_month(ds_full, var_info, year, month, days, world_cache_dir, hourly=False):
    """Fetches (unless already cached) one variable's FULL-GLOBE data
    (daily-reduced, or genuinely hourly if `hourly=True`) for one
    calendar month straight out of the ARCO-ERA5 Zarr store, and caches
    it *permanently* under world_cache_dir, keyed by (cds_name,
    accumulated-or-statistic, year, month, daily-or-hourly) --
    deliberately independent of `area`, `output_prefix`, or anything
    else about the calling run. Returns the path to that cached global
    NetCDF. The daily and hourly caches for the same (variable, month)
    are kept as entirely separate files (see base_tag below) so a run
    requesting one granularity never collides with or is mistaken for a
    cache of the other.

    When `hourly=True`: accumulated fields are de-accumulated to genuine
    per-hour amounts via _deaccumulate_hourly() (ECMWF_CYCLE_RESET_HOURS
    schedule -- see that function's docstring) instead of being summed
    to a daily total; non-accumulated fields are returned as their raw
    hourly readings, with no daily_statistic reduction applied at all.

    Why cache the whole globe instead of just the requested area: per
    arco-era5's own README, this array's on-disk Zarr chunking is
    {"time": 1, "latitude": 721, "longitude": 1440} -- one chunk *per
    hour*, spanning the entire globe (confirmed directly against the
    store's .zarray metadata: ~4.15 MB uncompressed, ~1.7-1.9 MB
    compressed per hourly chunk for a 2D surface field). There is no
    bbox small enough to avoid downloading and decompressing that
    whole-globe chunk for every hour touched -- a small `area` only
    trims what's kept *after* each hourly chunk is already in memory,
    not what gets transferred. So keeping the whole globe costs nothing
    beyond what a bbox-only fetch already pays, and means this exact
    (variable, month) is never fetched from Google Cloud again for any
    region, ever, no matter what area a later run asks for. A full
    calendar month is ~720-745 such chunks, so budget roughly 1.2-1.3 GB
    transferred (once) per accumulated variable per month on first
    request. See the ARCO-ERA5 (Google Cloud) SOURCE section in
    t.in.era5.md/.html for the tradeoff against -c (CDS, which subsets
    server-side and so never builds up a local world archive).

    ARCO-ERA5 is an unmodified GRIB->Zarr conversion of the raw archive,
    so accumulated fields (precipitation, evaporation, radiation, ...)
    are still ECMWF's raw forecast-accumulation: two 12h forecast cycles
    per day, based at 06 and 18 UTC, each accumulating from zero at its
    first valid hour (07 and 19 UTC respectively) up to its last (18 and
    06 UTC next day). The value AT hours 07/19 is already a genuine
    1-hour amount; every other hour must be differenced from the
    previous hour to recover that hour's own amount, per ECMWF's
    published conversion guidance for these fields (see
    https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790).
    This is a different reset pattern than reanalysis-era5-land's raw
    hourly (single reset per calendar day, at 01 UTC -- see load_daily())
    -- do not conflate the two.

    Raises ArcoDataUnavailable if ARCO-ERA5 has not ingested this month
    at all yet, not even provisionally -- see _arco_availability() and
    fetch_month()'s CDS fallback for that case. A month that IS present
    but only as ERA5T (preliminary, subject to later in-place revision
    -- see _arco_availability()) is cached under a distinct "_era5t"-
    suffixed filename, kept apart from permanent/final months so a
    cache hit never silently serves data that ARCO itself may since
    have superseded. A "final" cache hit is trusted forever without
    re-checking anything (final data never changes, so there is nothing
    to gain by re-touching ds_full) -- but an "_era5t" cache hit DOES
    re-check ds_full.attrs (cheap: attrs are already loaded when the
    store is opened, no data fetch) to see whether the final release
    has landed since it was cached; if so, the final month is fetched
    fresh, cached under its own "final" name, and the stale "_era5t"
    file is deleted -- no leftover provisional data lingering once a
    better version exists on disk."""
    varname = var_info["cds_name"]
    reduction = "accum" if var_info["accumulated"] else var_info["daily_statistic"]

    n_days = days[-1]
    month_start = np.datetime64("%04d-%02d-01" % (year, month))
    month_end_excl = month_start + np.timedelta64(n_days, "D")

    base_tag = "%s_%s_%04d%02d%s" % (
        varname, reduction, year, month, "_hourly" if hourly else "",
    )
    final_nc = os.path.join(world_cache_dir, base_tag + ".nc")
    era5t_nc = os.path.join(world_cache_dir, base_tag + "_era5t.nc")

    if os.path.exists(final_nc):
        return final_nc

    era5t_cached = os.path.exists(era5t_nc)
    if era5t_cached:
        availability = _arco_availability(ds_full, month_end_excl)
        if availability != "final":
            return era5t_nc  # still provisional -- nothing has changed
        # else: final has landed since -- fall through to fetch+cache it,
        # replacing the now-superseded provisional file below.
    else:
        availability = _arco_availability(ds_full, month_end_excl)
        if availability == "unavailable":
            raise ArcoDataUnavailable(
                "%s %04d-%02d is beyond ARCO-ERA5's current coverage "
                "(valid_time_stop_era5t=%s)"
                % (varname, year, month, ds_full.attrs.get("valid_time_stop_era5t"))
            )
    world_nc = final_nc if availability == "final" else era5t_nc

    import xarray as xr

    da = ds_full[varname]  # no area .sel() here -- keep the whole globe

    if var_info["accumulated"]:
        fetch_start = month_start - np.timedelta64(1, "h")
    else:
        fetch_start = month_start
    da = da.sel(time=slice(fetch_start, month_end_excl - np.timedelta64(1, "h")))

    if hourly:
        if var_info["accumulated"]:
            result = _deaccumulate_hourly(da, "time", ECMWF_CYCLE_RESET_HOURS)
        else:
            result = da  # raw hourly readings ARE the hourly values, no reduction
        result = result.sel(
            time=slice(month_start, month_end_excl - np.timedelta64(1, "h"))
        )
    elif var_info["accumulated"]:
        hourly_vals = _deaccumulate_hourly(da, "time", ECMWF_CYCLE_RESET_HOURS)
        result = hourly_vals.resample(time="1D").sum()
        result = result.sel(time=slice(month_start, month_end_excl - np.timedelta64(1, "D")))
    else:
        stat_fn = {
            "daily_mean": lambda r: r.mean(),
            "daily_minimum": lambda r: r.min(),
            "daily_maximum": lambda r: r.max(),
        }[var_info["daily_statistic"]]
        result = stat_fn(da.resample(time="1D"))
        result = result.sel(time=slice(month_start, month_end_excl - np.timedelta64(1, "D")))

    result.name = varname

    os.makedirs(world_cache_dir, exist_ok=True)
    # write-then-rename: a run killed mid-write never leaves a corrupt
    # file behind for a later run to mistake for a complete cache entry.
    tmp_nc = world_nc + ".tmp"
    result.load().to_netcdf(tmp_nc, encoding={varname: dict(zlib=True, complevel=4)})
    os.replace(tmp_nc, world_nc)

    if world_nc == final_nc and era5t_cached:
        # The final release just replaced a provisional ERA5T month we
        # already had cached -- remove the now-superseded file rather
        # than leaving stale provisional data sitting next to the real
        # thing forever.
        os.remove(era5t_nc)

    return world_nc


def fetch_arco(ds_full, var_info, year, month, days, area, out_nc, world_cache_dir, hourly=False):
    """Crops the permanently-cached full-world month for this variable
    (see fetch_arco_world_month()) to `area`, writing the small result
    to out_nc in the same daily-resolution shape (or hourly, if
    `hourly=True`) a CDS daily-statistics download would have (so
    load_daily() needs no ARCO-specific branch -- the NC this writes is
    already fully reduced/de-accumulated to its final granularity).
    Only fetch_arco_world_month() ever talks to Google Cloud; this
    function is a purely local crop, cheap regardless of area."""
    import xarray as xr

    world_nc = fetch_arco_world_month(
        ds_full, var_info, year, month, days, world_cache_dir, hourly=hourly
    )
    with xr.open_dataset(world_nc) as world_ds:
        da = _arco_select_area(world_ds[var_info["cds_name"]], area)
        da.load().to_netcdf(out_nc)


def _fatal_no_hourly_plain_era5(var_key, year, month):
    gs.fatal(
        "Hourly output (-h) requested for %s %04d-%02d, but the plain-ERA5 "
        "(non-ERA5-Land) fallback tier only supports daily output: CDS's "
        "'derived-era5-single-levels-daily-statistics' product (used by "
        "fetch_era5()) computes its daily reduction server-side, so there "
        "is no hourly data to recover from it. This is a known, "
        "documented gap, not a bug -- see the 'Known gaps / future work' "
        "section in t.in.era5.md/.html for what a fix would need (a new "
        "fetch function against CDS's raw 'reanalysis-era5-single-levels' "
        "product, mirroring fetch_era5land_raw_hourly()'s existing "
        "pattern). Work around this for now by ensuring ERA5-Land or "
        "ARCO-ERA5 coverage is available for this period instead (both "
        "support -h already)." % (var_key, year, month)
    )


def fetch_month(
    clients, var_key, var_info, year, month, days, area, cache_dir, force_era5, use_arco,
    world_cache_dir, hourly=False,
):
    """Downloads (or reuses a cached copy of) one variable's data for one
    calendar month, trying ERA5-Land first unless force_era5, then either
    CDS's plain ERA5 or ARCO-ERA5 (if use_arco) for the fallback tier --
    and if use_arco but ARCO-ERA5 has not ingested this month at all yet
    (see ArcoDataUnavailable), CDS as a one-month-only stopgap for that
    gap, cached separately (see below). Returns (path, source), source
    one of "era5land", "era5", "arco", "arco_gap_cds".

    `hourly=True` requests genuinely hourly (rather than daily) data.
    ERA5-Land and ARCO-ERA5 both support this (see fetch_era5land_raw_hourly()
    and fetch_arco_world_month()); the plain-ERA5-via-CDS tier
    (fetch_era5(), used both as the -c fallback and as the one-month
    ARCO-unavailable stopgap) does NOT -- see _fatal_no_hourly_plain_era5()."""
    hourly_tag = "_hourly" if hourly else ""
    tag = "%s_%04d%02d%s" % (var_key, year, month, hourly_tag)
    cache_land = os.path.join(cache_dir, tag + "_era5land.nc")
    cache_era5 = os.path.join(cache_dir, tag + ("_arco.nc" if use_arco else "_era5.nc"))
    fallback_source = "arco" if use_arco else "era5"

    if not force_era5 and os.path.exists(cache_land):
        return cache_land, "era5land"
    if os.path.exists(cache_era5):
        return cache_era5, fallback_source

    if not force_era5:
        try:
            # ERA5-Land's raw-hourly CDS product (reanalysis-era5-land)
            # carries every variable at native hourly resolution
            # regardless of accumulated/instantaneous -- so in hourly
            # mode it's the right fetch for BOTH cases, not just
            # accumulated ones (unlike the daily path below, which must
            # route non-accumulated fields through the separate
            # daily-statistics product since accumulated fields can't
            # use that product at all -- see VARIABLES' docstring note).
            if hourly or var_info["accumulated"]:
                fetch_era5land_raw_hourly(
                    clients["cds"](), var_info, year, month, days, area, cache_land
                )
            else:
                fetch_era5land_instantaneous(
                    clients["cds"](), var_info, year, month, days, area, cache_land
                )
            return cache_land, "era5land"
        except Exception as e:
            gs.warning(
                "ERA5-Land unavailable for %s %04d-%02d (%s) -- falling "
                "back to ERA5" % (var_key, year, month, e)
            )
            if os.path.exists(cache_land):
                os.remove(cache_land)

    if use_arco:
        cache_gap = os.path.join(cache_dir, tag + "_arco_gap_cds.nc")
        if os.path.exists(cache_gap):
            return cache_gap, "arco_gap_cds"
        try:
            fetch_arco(
                clients["arco"](), var_info, year, month, days, area, cache_era5,
                world_cache_dir, hourly=hourly,
            )
            return cache_era5, fallback_source
        except ArcoDataUnavailable as e:
            # Nothing ingested for this month yet, not even provisionally
            # (see ArcoDataUnavailable/_arco_availability()) -- CDS may
            # already have it. Cached separately from both the permanent
            # world archive and the normal per-run ARCO cache: there is
            # no guarantee CDS's value here is identical to whatever
            # ARCO eventually ingests for this same month (independent
            # pipelines), so it must never be mistaken for -- or later
            # silently trusted as -- genuine ARCO-sourced data.
            if hourly:
                _fatal_no_hourly_plain_era5(var_key, year, month)
            gs.warning(
                "%s -- falling back to CDS for this month only (needs "
                "~/.cdsapirc); not cached as ARCO data, since it may "
                "not exactly match what ARCO eventually ingests" % e
            )
            fetch_era5(clients["cds"](), var_info, year, month, days, area, cache_gap)
            return cache_gap, "arco_gap_cds"
    else:
        if hourly:
            _fatal_no_hourly_plain_era5(var_key, year, month)
        fetch_era5(clients["cds"](), var_info, year, month, days, area, cache_era5)
    return cache_era5, fallback_source


def load_daily(path, var_info, source, hourly=False):
    """Returns a (timestamps, values, lats, lons) tuple: timestamps a
    sorted list of timezone-aware datetime.datetime (UTC) -- one per DAY
    when hourly=False (always at implicit 00:00, the existing daily
    convention), one per HOUR when hourly=True -- values a matching list
    of 2D numpy arrays (native CDS units, not yet unit-converted),
    lats/lons the grid coordinates. Callers needing a plain
    datetime.date for the daily case can call .date() on each entry."""
    import xarray as xr

    ds = xr.open_dataset(path)
    varname = list(ds.data_vars)[0]
    da = ds[varname]
    time_dim = "valid_time" if "valid_time" in da.dims else "time"

    if source == "era5land" and var_info["accumulated"] and not hourly:
        # reanalysis-era5-land's raw hourly accumulated fields are NOT
        # hour-differenced increments -- confirmed by inspection: each
        # field resets near zero at hour 01 UTC and climbs
        # monotonically to a peak at hour 00 UTC the *following* day,
        # then resets again. That peak *is* the day's total (the whole
        # cycle's accumulation); summing all 24 raw hourly readings
        # (as before) adds up already-cumulative numbers on top of each
        # other and wildly overcounts (confirmed: one single day
        # inflated to ~230mm regional mean this way, versus a plausible
        # ~0.03-3mm/day for the same real values). Shifting every
        # timestamp back 1 hour realigns each 01h..00h(+1) cycle onto a
        # single calendar day, so grouping by day and taking the last
        # (i.e. peak) value per group recovers the correct daily total.
        # Edge effect: the last calendar day of a given fetched month
        # is missing its final (hour-00-of-next-month) reading, since
        # month chunks don't fetch across their own boundary -- that
        # day's total is undercounted by about one hour's worth of
        # accumulation, not the full day.
        shifted = da.assign_coords(
            {time_dim: da[time_dim] - np.timedelta64(1, "h")}
        )
        da = shifted.resample({time_dim: "1D"}).last()
        # the first group is always a spurious single leftover hour
        # (raw hour 00:00 on day 1 of this month's own file, shifted
        # to 23:00 the day before -- the tail end of the *previous*
        # month's cycle, not a real day here): for a month in the
        # middle of a multi-month request this exact date was already
        # correctly computed from the previous month's own file, so
        # keeping it here would register the same date twice and crash
        # t.register with a UNIQUE constraint violation.
        da = da.isel({time_dim: slice(1, None)})
    elif source == "era5land" and var_info["accumulated"] and hourly:
        # Same raw ERA5-Land data as above, but recovering genuine
        # PER-HOUR amounts instead of collapsing to a daily total -- see
        # _deaccumulate_hourly()'s docstring for the general method and
        # ERA5LAND_RESET_HOURS for this source's specific reset
        # schedule (single reset at 01 UTC, unlike ARCO/plain-ERA5's
        # 07/19 UTC two-cycle schedule). The very first fetched raw
        # reading (hour 00 on day 1 of this month's file) is dropped
        # implicitly by _deaccumulate_hourly()'s diff (it's the tail end
        # of the *previous* day's/month's cycle, not itself a reset hour
        # and not diffable against anything in this chunk -- the same
        # edge case the daily branch above documents, here it simply
        # never appears in the output rather than needing an explicit
        # extra slice).
        da = _deaccumulate_hourly(da, time_dim, ERA5LAND_RESET_HOURS)
    # else: non-accumulated fields (daily or hourly) and any source
    # other than era5land need no de-accumulation -- either the raw
    # readings are already instantaneous/non-cumulative, or (for
    # "arco"/"arco_gap_cds") fetch_arco_world_month() already fully
    # reduced/de-accumulated the data before writing this NC.

    lats = da["latitude"].values
    lons = da["longitude"].values
    timestamps = [
        datetime.datetime.fromtimestamp(
            t.astype("datetime64[s]").astype(int), tz=datetime.timezone.utc
        )
        for t in da[time_dim].values
    ]
    values = [da.isel({time_dim: i}).values for i in range(da.sizes[time_dim])]
    ds.close()
    return timestamps, values, lats, lons


def write_geotiff(path, array, lons, lats, nodata=-9999.0, fallback_res_deg=None):
    from osgeo import gdal, osr

    # A requested area smaller than the product's native grid spacing
    # (e.g. a few-km field against ERA5-Land's ~0.1 deg / ERA5's ~0.25
    # deg grid) makes CDS return a single point along that axis, so
    # there's no second point to diff. Fall back to the product's
    # nominal grid step: lon ascending (+), lat descending (-),
    # matching the ordering CDS returns for >1 point.
    if len(lons) > 1:
        dx = float(lons[1] - lons[0])
    else:
        dx = fallback_res_deg if fallback_res_deg else 0.1
    if len(lats) > 1:
        dy = float(lats[1] - lats[0])
    else:
        dy = -(fallback_res_deg if fallback_res_deg else 0.1)
    origin_x = float(lons[0]) - dx / 2.0
    origin_y = float(lats[0]) - dy / 2.0

    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)
    dst.SetGeoTransform((origin_x, dx, 0, origin_y, 0, dy))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dst.SetProjection(srs.ExportToWkt())

    arr = np.array(array, dtype=np.float32)
    arr = np.where(np.isnan(arr) | (np.abs(arr) > 1.0e30), nodata, arr)
    band = dst.GetRasterBand(1)
    # SetNoDataValue() must come *before* WriteArray(): calling it
    # after silently re-fills any not-yet-flushed blocks with the
    # nodata value, overwriting real data already written -- confirmed
    # by direct reproduction with a plain all-zero array (a real,
    # legitimate value for a dry day) coming back as all-nodata after
    # round-tripping through a fresh gdal.Open(), only when
    # SetNoDataValue() ran after WriteArray().
    band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    dst.FlushCache()
    dst = None


def default_area():
    """Falls back to the current region's bounding box, reprojected to
    WGS84 lat/lon (g.region -bg already does this regardless of the
    project's own CRS)."""
    info = gs.parse_command("g.region", flags="bg")
    return [
        float(info["ll_n"]),
        float(info["ll_w"]),
        float(info["ll_s"]),
        float(info["ll_e"]),
    ]


def main():
    options, flags = gs.parser()

    global TMP_DIR
    var_keys = options["variables"].split(",")
    start_date = datetime.datetime.strptime(options["start"], "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(options["end"], "%Y-%m-%d").date()
    if end_date < start_date:
        gs.fatal("end must not be before start")

    if options["area"]:
        area = [float(v) for v in options["area"].split(",")]
        if len(area) != 4:
            gs.fatal("area must be 'north,west,south,east'")
    else:
        area = default_area()

    cache_dir = options["cache_dir"]
    if not cache_dir:
        cache_dir = gs.tempdir()
        TMP_DIR = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    world_cache_dir = options["arco_world_cache"] or ARCO_WORLD_CACHE_DEFAULT

    force_era5 = bool(flags["e"])
    use_arco = not bool(flags["c"])
    hourly = bool(flags["h"])

    # Bounds for filtering loaded timestamps against [start_date,
    # end_date]: daily-mode timestamps compare as plain dates; hourly
    # timestamps need a full-precision, timezone-aware [00:00:00 of
    # start_date, 23:59:59 of end_date] UTC window instead, since a
    # bare datetime.date isn't comparable to a datetime.datetime.
    if hourly:
        start_bound = datetime.datetime.combine(
            start_date, datetime.time.min, tzinfo=datetime.timezone.utc
        )
        end_bound = datetime.datetime.combine(
            end_date, datetime.time.max, tzinfo=datetime.timezone.utc
        )
    else:
        start_bound, end_bound = start_date, end_date

    # Lazily built and cached on first use, so a run that only needs one
    # of the two backends (e.g. -e without -c, the default's plain-ERA5
    # tier, never touches CDS at all) never pays for the other's client
    # setup / credential check.
    lazy = {}

    def get_cds():
        return lazy.setdefault("cds", cds_client())

    def get_arco():
        return lazy.setdefault("arco", arco_client())

    clients = {"cds": get_cds, "arco": get_arco}

    for var_key in var_keys:
        var_info = VARIABLES[var_key]
        all_timestamps = []
        all_values = []
        lats = lons = None
        sources_used = set()

        for year, month, days in month_chunks(start_date, end_date):
            path, source = fetch_month(
                clients, var_key, var_info, year, month, days, area, cache_dir,
                force_era5, use_arco, world_cache_dir, hourly=hourly,
            )
            sources_used.add(source)
            timestamps, values, lats, lons = load_daily(path, var_info, source, hourly=hourly)
            for t, v in zip(timestamps, values):
                t_cmp = t if hourly else t.date()
                if start_bound <= t_cmp <= end_bound:
                    all_timestamps.append(t)
                    all_values.append(var_info["convert"](v))

        order = sorted(range(len(all_timestamps)), key=lambda i: all_timestamps[i])
        all_timestamps = [all_timestamps[i] for i in order]
        all_values = [all_values[i] for i in order]

        strds = "%s_%s" % (options["output_prefix"], var_key)
        gs.run_command(
            "t.create",
            output=strds,
            type="strds",
            temporaltype="absolute",
            title="ERA5(-Land) %s" % var_key,
            description=var_info["description"],
            overwrite=True,
        )

        if "era5land" in sources_used:
            fallback_res_deg = NATIVE_RESOLUTION_DEG["era5land"]
        elif "arco" in sources_used:
            fallback_res_deg = NATIVE_RESOLUTION_DEG["arco"]
        else:
            fallback_res_deg = NATIVE_RESOLUTION_DEG["era5"]

        raster_names = []
        name_fmt = "%Y%m%d%H" if hourly else "%Y%m%d"
        for t, v in zip(all_timestamps, all_values):
            base = "%s_%s" % (strds, t.strftime(name_fmt))
            tif = os.path.join(cache_dir, base + ".tif")
            write_geotiff(tif, v, lons, lats, fallback_res_deg=fallback_res_deg)
            gs.run_command(
                "r.import", input=tif, output=base, overwrite=True, quiet=True
            )
            os.remove(tif)
            raster_names.append((base, t))
            TMP_RASTERS.append(base)

        if raster_names:
            register_fmt = "%Y-%m-%d %H:%M:%S" if hourly else "%Y-%m-%d"
            maps_file = os.path.join(cache_dir, "%s_register.txt" % strds)
            with open(maps_file, "w") as f:
                for base, t in raster_names:
                    f.write("%s|%s\n" % (base, t.strftime(register_fmt)))
            gs.run_command("t.register", input=strds, file=maps_file)
            os.remove(maps_file)

        # rasters are now owned by the STRDS, not scratch -- don't
        # g.remove them on exit.
        del TMP_RASTERS[:]

        gs.message(
            "Wrote %d %s to STRDS <%s> (sources used: %s)"
            % (
                len(raster_names), "hours" if hourly else "days", strds,
                ", ".join(sorted(sources_used)),
            )
        )


if __name__ == "__main__":
    atexit.register(cleanup)
    sys.exit(main())
