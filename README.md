# t.in.era5

A [GRASS GIS](https://grass.osgeo.org/) addon that downloads daily
climate variables from the Copernicus Climate Data Store (CDS) and
imports each as its own space-time raster dataset (STRDS), ready to
feed [r.hydro.hbv.forcing](https://github.com/YannChemin/HBV) (which
reduces a STRDS to a per-basin table for `r.hydro.hbv`), or any other
temporal GRASS workflow.

```
g.region n=34 s=31 e=49 w=47 res=0:06

t.in.era5 variables=precipitation,temperature,potential_evaporation \
  start=2001-06-01 end=2001-08-31 \
  output_prefix=karkheh cache_dir=$HOME/era5_cache
```

## Why

Given a variable list and a date range, this fetches the *whole* set
of STRDS a hydrological (or other daily-climate-driven) model needs in
one call, instead of hand-writing separate `t.rast.import`/`t.create`/
`t.register` invocations per variable per month.

## ERA5-Land first, ERA5 fallback

For each variable, **ERA5-Land** (~9km grid, generally preferable for
land applications) is tried first; if that fails for a given month
(explicit CDS error, network issue, or a genuine coverage gap), the
module automatically falls back to **ERA5** (~31km grid, longer
historical record) for that month only — other months/variables are
unaffected.

One CDS quirk drives part of this fallback even when ERA5-Land data
actually exists: the `derived-era5-land-daily-statistics` product used
for "instantaneous" variables (temperature, wind, pressure, ...)
explicitly refuses to compute daily statistics for *accumulated*
variables (precipitation, potential evaporation, solar radiation,
snowfall). For those, `t.in.era5` requests raw hourly
`reanalysis-era5-land` data instead and sums it to a daily total itself
— still ERA5-Land, just via a different CDS product — before falling
back to ERA5's own (accumulation-capable) daily-statistics product only
if that also fails.

Requests are chunked per (variable, year, month), so a single failure
doesn't force re-fetching a whole multi-year request, and repeat runs
against the same `cache_dir` reuse already-downloaded months instead of
re-downloading them — each cached month is always fetched in full
(every day of that calendar month), so a cached month is safe to reuse
for any other request touching it, regardless of exactly which days
that later request needs.

## Variables

| key | CDS variable | native units | converted to |
|---|---|---|---|
| `precipitation` | total_precipitation | m/day (accum.) | mm/d |
| `temperature` | 2m_temperature (daily mean) | K | °C |
| `temperature_min` | 2m_temperature (daily min) | K | °C |
| `temperature_max` | 2m_temperature (daily max) | K | °C |
| `dewpoint_temperature` | 2m_dewpoint_temperature | K | °C |
| `potential_evaporation` | potential_evaporation | m/day (accum.) | mm/d |
| `solar_radiation` | surface_solar_radiation_downwards | J/m² (accum.) | MJ/m²/d |
| `wind_u` | 10m_u_component_of_wind | m/s | m/s |
| `wind_v` | 10m_v_component_of_wind | m/s | m/s |
| `surface_pressure` | surface_pressure | Pa | kPa |
| `snowfall` | snowfall | m (accum.) | mm/d |

## CDS account setup

`t.in.era5` uses the `cdsapi` Python package, which reads credentials
from `~/.cdsapirc`. Create a free account at
[cds.climate.copernicus.eu](https://cds.climate.copernicus.eu), accept
the ERA5/ERA5-Land dataset licences on the CDS website (one-time, per
dataset), then copy your personal access token from your CDS profile
page into `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

(the `url`/`key` values above are the real format this file takes —
only the token itself is account-specific). Without this file (or with
an invalid token), every CDS request `t.in.era5` makes fails
immediately with an authentication error.

## Options

| Option | Description |
|---|---|
| `variables` | Comma-separated list from the table above |
| `start`, `end` | `YYYY-MM-DD`, inclusive |
| `output_prefix` | STRDS created as `<output_prefix>_<variable>` |
| `area` | `north,west,south,east` in WGS84 degrees; default derived from the current region |
| `cache_dir` | Directory to cache downloaded NetCDF files in; default a temporary, run-scoped directory |
| `-e` | Force ERA5 (skip ERA5-Land) for every variable/month |

## Requirements

- GRASS GIS with the temporal framework (`t.create`, `t.register`) and
  `r.import` (core)
- `cdsapi`, `xarray`, GDAL Python bindings (`osgeo.gdal`/`osgeo.osr`)
- A free CDS account and `~/.cdsapirc` (see above)

## Install

```
g.extension extension=t.in.era5 url=https://github.com/YannChemin/t.in.era5
```

## Testing

No standalone testsuite lives in this repo yet — the full
fetch → STRDS → zonal-mean → model pipeline is exercised end-to-end by
[r.hydro.hbv](https://github.com/YannChemin/HBV)'s
`testsuite/test_karkheh_era5_v2.py`, gated behind
`R_HYDRO_HBV_RUN_ERA5_TESTS=1` since it needs real network access and a
working `~/.cdsapirc`.

## License

Public domain — see [LICENSE](LICENSE) (Unlicense).

## See also

- [r.hydro.hbv](https://github.com/YannChemin/HBV) — the HBV
  hydrological model this module was built to feed climate forcing into
- [r.in.dem](https://github.com/YannChemin/r.in.dem) — the equivalent
  no-API-key global DEM importer for the same ecosystem
