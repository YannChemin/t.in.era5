# t.in.era5

## NAME

**t.in.era5** - Imports ERA5-Land (falling back to ERA5) daily climate
variables from the Copernicus Climate Data Store (CDS) as space-time
raster datasets (STRDS), one per variable.

## SYNOPSIS

**t.in.era5**\
**t.in.era5 --help**\
**t.in.era5** [**-e**] **variables**=*string*[,*string*,...]
**start**=*string* **end**=*string* [**area**=*north,west,south,east*]
**output_prefix**=*string* [**cache_dir**=*name*]

### Flags

**-e**
&nbsp;&nbsp;&nbsp;&nbsp;Force plain ERA5 for every variable (skip the ERA5-Land attempt
entirely).

## DESCRIPTION

*t.in.era5* downloads one or more daily climate variables from CDS and
imports each as its own STRDS (`<output_prefix>_<variable>`), ready to
feed *[r.hydro.hbv.forcing](r.hydro.hbv.forcing.md)* (which reduces a
STRDS to a per-basin table for *[r.hydro.hbv](r.hydro.hbv.md)*), or any
other temporal GRASS workflow.

For each variable, **ERA5-Land** (~9km grid, generally preferable for
land applications) is tried first; if that fails for a given month
(explicit CDS error, network issue, or a genuine coverage gap), the
module automatically falls back to **ERA5** (~31km grid, longer
historical record) for that month only -- other months/variables are
unaffected. One CDS quirk drives part of this fallback even when
ERA5-Land data actually exists: the `derived-era5-land-daily-statistics`
product used for "instantaneous" variables (temperature, wind,
pressure, ...) explicitly refuses to compute daily statistics for
*accumulated* variables (precipitation, potential evaporation, solar
radiation, snowfall). For those, *t.in.era5* requests raw hourly
`reanalysis-era5-land` data instead and sums it to a daily total itself
-- still ERA5-Land, just via a different CDS product -- before falling
back to ERA5's own (accumulation-capable) daily-statistics product only
if that also fails.

Requests are chunked per (variable, year, month), so a single failure
doesn't force re-fetching a whole multi-year request, and repeat runs
against the same **cache_dir** reuse already-downloaded months instead
of re-downloading them.

## VARIABLES

| key | CDS variable | native units | converted to |
|---|---|---|---|
| `precipitation` | total_precipitation | m/day (accum.) | mm/d |
| `temperature` | 2m_temperature (daily mean) | K | deg C |
| `temperature_min` | 2m_temperature (daily min) | K | deg C |
| `temperature_max` | 2m_temperature (daily max) | K | deg C |
| `dewpoint_temperature` | 2m_dewpoint_temperature | K | deg C |
| `potential_evaporation` | potential_evaporation | m/day (accum.) | mm/d |
| `solar_radiation` | surface_solar_radiation_downwards | J/m2 (accum.) | MJ/m2/d |
| `wind_u` | 10m_u_component_of_wind | m/s | m/s |
| `wind_v` | 10m_v_component_of_wind | m/s | m/s |
| `surface_pressure` | surface_pressure | Pa | kPa |
| `snowfall` | snowfall | m (accum.) | mm/d |

## CDS ACCOUNT SETUP

*t.in.era5* uses the `cdsapi` Python package, which reads credentials
from `~/.cdsapirc`. Create a free account at
<https://cds.climate.copernicus.eu>, accept the ERA5/ERA5-Land dataset
licences on the CDS website (one-time, per dataset), then copy your
personal access token from your CDS profile page into
`~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

(the `url`/`key` values above are the real format this file takes --
only the token itself is account-specific). Without this file (or with
an invalid token), every CDS request `t.in.era5` makes will fail
immediately with an authentication error.

## EXAMPLE

```sh
g.region n=34 s=31 e=49 w=47 res=0:06 # ~0.1 deg, matches ERA5-Land's native grid

t.in.era5 variables=precipitation,temperature,potential_evaporation \
  start=2001-06-01 end=2001-08-31 \
  output_prefix=karkheh cache_dir=$HOME/era5_cache

r.hydro.hbv.forcing strds=karkheh_precipitation basins=basins \
  basins_vector=basins_v output_table=karkheh_precip_table
```

## SEE ALSO

*[r.hydro.hbv](r.hydro.hbv.md)*,
*[r.hydro.hbv.forcing](r.hydro.hbv.forcing.md)*,
*[t.create](t.create.md)*, *[t.register](t.register.md)*,
*[r.import](r.import.md)*

## AUTHOR

Yann Chemin
