#!/usr/bin/env python3
############################################################################
#
# MODULE:       t.in.era5
# AUTHOR:       Yann Chemin
# PURPOSE:      Imports one or more daily climate variables from the
#               Copernicus Climate Data Store as GRASS space-time raster
#               datasets (STRDS), one per variable. For each requested
#               variable, ERA5-Land is tried first (its ~9km grid is
#               generally preferable for land applications); if that
#               fails for a given period (temporal gap, or the CDS
#               "derived daily statistics" product's known restriction
#               against accumulated variables -- see NOTES), ERA5
#               (~31km grid, longer record, always supports daily
#               statistics of accumulated variables) is used instead
#               for that period. Requests are chunked per (variable,
#               year, month) so a fallback only affects the affected
#               chunk, and so repeat runs can reuse a local cache
#               instead of re-downloading.
# COPYRIGHT:    (C) 2026 by Yann Chemin
#               Released into the public domain -- see LICENSE (Unlicense).
#
############################################################################

# %module
# % description: Imports ERA5-Land (falling back to ERA5) daily climate variables from the Copernicus Climate Data Store as one space-time raster dataset per variable.
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
# %flag
# % key: e
# % description: Force plain ERA5 for every variable (skip the ERA5-Land attempt entirely)
# %end

import atexit
import calendar
import datetime
import os
import shutil
import sys

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


def fetch_month(client, var_key, var_info, year, month, days, area, cache_dir, force_era5):
    """Downloads (or reuses a cached copy of) one variable's data for one
    calendar month, trying ERA5-Land first unless force_era5. Returns
    (path, source) where source is "era5land" or "era5"."""
    tag = "%s_%04d%02d" % (var_key, year, month)
    cache_land = os.path.join(cache_dir, tag + "_era5land.nc")
    cache_era5 = os.path.join(cache_dir, tag + "_era5.nc")

    if not force_era5 and os.path.exists(cache_land):
        return cache_land, "era5land"
    if os.path.exists(cache_era5):
        return cache_era5, "era5"

    if not force_era5:
        try:
            if var_info["accumulated"]:
                fetch_era5land_raw_hourly(
                    client, var_info, year, month, days, area, cache_land
                )
            else:
                fetch_era5land_instantaneous(
                    client, var_info, year, month, days, area, cache_land
                )
            return cache_land, "era5land"
        except Exception as e:
            gs.warning(
                "ERA5-Land unavailable for %s %04d-%02d (%s) -- falling "
                "back to ERA5" % (var_key, year, month, e)
            )
            if os.path.exists(cache_land):
                os.remove(cache_land)

    fetch_era5(client, var_info, year, month, days, area, cache_era5)
    return cache_era5, "era5"


def load_daily(path, var_info, source):
    """Returns a (dates, values, lats, lons) tuple: dates is a sorted list
    of datetime.date, values a matching list of 2D numpy arrays (native
    CDS units, not yet unit-converted), lats/lons the grid coordinates."""
    import xarray as xr

    ds = xr.open_dataset(path)
    varname = list(ds.data_vars)[0]
    da = ds[varname]
    time_dim = "valid_time" if "valid_time" in da.dims else "time"

    if source == "era5land" and var_info["accumulated"]:
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

    lats = da["latitude"].values
    lons = da["longitude"].values
    dates = [
        datetime.datetime.fromtimestamp(
            t.astype("datetime64[s]").astype(int), tz=datetime.timezone.utc
        ).date()
        for t in da[time_dim].values
    ]
    values = [da.isel({time_dim: i}).values for i in range(da.sizes[time_dim])]
    ds.close()
    return dates, values, lats, lons


def write_geotiff(path, array, lons, lats, nodata=-9999.0):
    from osgeo import gdal, osr

    dx = float(lons[1] - lons[0])
    dy = float(lats[1] - lats[0])
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

    force_era5 = bool(flags["e"])
    client = cds_client()

    for var_key in var_keys:
        var_info = VARIABLES[var_key]
        all_dates = []
        all_values = []
        lats = lons = None
        sources_used = set()

        for year, month, days in month_chunks(start_date, end_date):
            path, source = fetch_month(
                client, var_key, var_info, year, month, days, area, cache_dir,
                force_era5,
            )
            sources_used.add(source)
            dates, values, lats, lons = load_daily(path, var_info, source)
            for d, v in zip(dates, values):
                if start_date <= d <= end_date:
                    all_dates.append(d)
                    all_values.append(var_info["convert"](v))

        order = sorted(range(len(all_dates)), key=lambda i: all_dates[i])
        all_dates = [all_dates[i] for i in order]
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

        raster_names = []
        for d, v in zip(all_dates, all_values):
            base = "%s_%s" % (strds, d.strftime("%Y%m%d"))
            tif = os.path.join(cache_dir, base + ".tif")
            write_geotiff(tif, v, lons, lats)
            gs.run_command(
                "r.import", input=tif, output=base, overwrite=True, quiet=True
            )
            os.remove(tif)
            raster_names.append((base, d))
            TMP_RASTERS.append(base)

        if raster_names:
            maps_file = os.path.join(cache_dir, "%s_register.txt" % strds)
            with open(maps_file, "w") as f:
                for base, d in raster_names:
                    f.write("%s|%s\n" % (base, d.strftime("%Y-%m-%d")))
            gs.run_command("t.register", input=strds, file=maps_file)
            os.remove(maps_file)

        # rasters are now owned by the STRDS, not scratch -- don't
        # g.remove them on exit.
        del TMP_RASTERS[:]

        gs.message(
            "Wrote %d days to STRDS <%s> (sources used: %s)"
            % (len(raster_names), strds, ", ".join(sorted(sources_used)))
        )


if __name__ == "__main__":
    atexit.register(cleanup)
    sys.exit(main())
