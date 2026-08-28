# t.in.era5

## NAME

**t.in.era5** - Imports ERA5-Land (falling back to plain ERA5) daily
climate variables as space-time raster datasets (STRDS), one per
variable.

## SYNOPSIS

**t.in.era5**\
**t.in.era5 --help**\
**t.in.era5** [**-e**] [**-c**] [**-h**] **variables**=*string*[,*string*,...]
**start**=*string* **end**=*string* [**area**=*north,west,south,east*]
**output_prefix**=*string* [**cache_dir**=*name*]
[**arco_world_cache**=*name*]

### Flags

**-e**
&nbsp;&nbsp;&nbsp;&nbsp;Force plain ERA5 for every variable (skip the ERA5-Land attempt
entirely).

**-c**
&nbsp;&nbsp;&nbsp;&nbsp;Use the CDS API instead of ARCO-ERA5 (Google Cloud) for the
plain-ERA5 tier. Requires a working `~/.cdsapirc` for that tier too. By
default that tier reads the public, no-login ARCO-ERA5 Zarr archive
instead.

**-h**
&nbsp;&nbsp;&nbsp;&nbsp;Hourly output instead of daily -- one raster per hour
(~24x more STRDS maps for the same period; see
[HOURLY OUTPUT](#hourly-output--h) below for data volume and per-source-tier
notes -- all three tiers support it).

## DESCRIPTION

*t.in.era5* downloads one or more daily climate variables and imports
each as its own STRDS (`<output_prefix>_<variable>`), ready to feed
*[r.hydro.hbv.forcing](r.hydro.hbv.forcing.md)* (which reduces a STRDS
to a per-basin table for *[r.hydro.hbv](r.hydro.hbv.md)*), or any other
temporal GRASS workflow.

For each variable, **ERA5-Land** (~9km grid, generally preferable for
land applications) is tried first via the Copernicus Climate Data
Store (CDS); if that fails for a given month (explicit CDS error, no
`~/.cdsapirc` configured, network issue, or a genuine coverage gap),
the module automatically falls back to **plain ERA5** (~31km grid,
longer historical record) for that month only -- other months/
variables are unaffected. By default that fallback tier is read
straight from the public **[ARCO-ERA5](#arco-era5-google-cloud-source)**
Google Cloud archive (no CDS account needed at all for it); pass **-c**
to use CDS's own plain-ERA5 product there instead.

One CDS quirk drives part of the ERA5-Land fallback even when
ERA5-Land data actually exists: the `derived-era5-land-daily-statistics`
product used for "instantaneous" variables (temperature, wind,
pressure, ...) explicitly refuses to compute daily statistics for
*accumulated* variables (precipitation, potential evaporation, solar
radiation, snowfall). For those, *t.in.era5* requests raw hourly
`reanalysis-era5-land` data instead and sums it to a daily total itself
-- still ERA5-Land, just via a different CDS product -- before falling
back to the plain-ERA5 tier only if that also fails.

Requests are chunked per (variable, year, month), so a single failure
doesn't force re-fetching a whole multi-year request, and repeat runs
against the same **cache_dir** reuse already-downloaded months instead
of re-downloading them.

## ARCO-ERA5 (Google Cloud) SOURCE

By default, the plain-ERA5 fallback tier is read directly from
[ARCO-ERA5](https://github.com/google-research/arco-era5), a public,
anonymous-access Zarr mirror of ERA5 on Google Cloud Storage
(`gs://gcp-public-data-arco-era5`), via `xarray`/`zarr`/`gcsfs` instead
of a CDS request. This needs no CDS account for that tier and sidesteps
the CDS download queue entirely, but ARCO-ERA5 only ever provides plain
ERA5 (~31km grid) -- it has no ERA5-Land counterpart. ERA5-Land is
still tried first via CDS unless **-e** is also given; **-e** alone
(without **-c**) means *t.in.era5* never contacts CDS at all, needing
no CDS account whatsoever. Pass **-c** to use CDS's own plain-ERA5
product for that tier instead (needs `~/.cdsapirc`).

*t.in.era5* opens the store with `xarray.open_zarr(..., chunks=None,
storage_options=dict(token="anon"))` -- matching
[Google's own arco-era5 usage examples](https://github.com/google-research/arco-era5)
for this exact dataset, and confirmed in practice to be both simpler
(no `dask` dependency at all) and faster than `chunks="auto"`, which
only adds a dask-array layer that regroups the store's own chunks
without reducing what gets transferred.

ARCO-ERA5 is an unmodified GRIB-to-Zarr conversion of the raw archive,
so accumulated variables (precipitation, potential evaporation, solar
radiation, snowfall) are still in ECMWF's raw forecast-accumulation
form and are de-accumulated locally to daily totals, following ECMWF's
published conversion guidance for these fields. This is a different
correction than the one applied to ERA5-Land's raw hourly data (see
above) -- do not assume the two behave the same way.

Reading ARCO-ERA5 requires the `gcsfs` and `zarr` Python packages in
addition to `xarray`/`netCDF4`. `make venv` (see the Makefile) builds a
dedicated virtualenv with all of these under the installed addon's own
`etc/t.in.era5/venv`, which *t.in.era5.py* detects and uses
automatically -- no manual `pip install` needed, and none is attempted
against GRASS's own Python installation (which a plain `pip install`
often can't touch anyway, blocked by PEP 668's
externally-managed-environment guard on modern Debian/Ubuntu).

### Cost/performance: whole-globe-per-hour chunking, and the world cache

Per arco-era5's own README, the `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
array this module reads is chunked `{"time": 1, "latitude": 721,
"longitude": 1440}` -- **one Zarr chunk per hour, spanning the entire
globe** (confirmed directly against the store's chunk metadata: ~4.15 MB
uncompressed, ~1.7-1.9 MB compressed per hourly chunk for a 2D surface
field). There is no bounding box small enough to avoid downloading and
decompressing that whole-globe chunk for every hour touched -- **area**
only trims what's *kept* after each hourly chunk is already in memory,
not what gets *transferred*. A full calendar month is ~720-745 such
chunks, so budget roughly **1.2-1.3 GB transferred per accumulated
variable per month** the *first* time that (variable, year, month) is
ever requested.

Since keeping the whole globe costs nothing beyond what a bbox-only
fetch already pays for, *t.in.era5* does exactly that: every ARCO-ERA5
fetch is cached permanently, at full world coverage, under
**arco_world_cache** (default `$HOME/RSDATA/ERA5World`), keyed only by
(variable, reduction, year, month) -- deliberately independent of
**area**, **output_prefix**, or which run asked for it. The very first
request for a given (variable, month) anywhere in the world pays the
full download; every later request for that same (variable, month) --
for *any* region, *any* project, *any* future run -- is served from
that local file with no network access at all. Confirmed in practice:
a second fetch for a different continent's bounding box, same month,
dropped from ~82s to ~0.05s. This directory grows over time by design
(each cached month is a genuine, reusable slice of world coverage,
compressed to a few MB per variable per month for typical fields) and
is never pruned automatically -- treat it as a small, permanent local
ERA5 mirror you're building up, and manage its size yourself if that
ever matters. A handful of days for a brand-new month is still fast (a
few minutes, paid once); only a large historical multi-month backfill
of variables/months never requested before is slow, and even then only
once -- for a true first-time bulk historical load, **-c** (CDS, which
subsets server-side and so is cheaper per single request but never
builds up a reusable local archive) may still be preferable.

#### Estimated arco_world_cache disk usage

Every cached month/variable is a full 721x1440 (0.25 deg)
daily-resolution global field, ~126 MB uncompressed, compressed to
roughly **60-65 MB** (measured: a real cached month of daily-mean
temperature came to 63.5 MB against a 124.6 MB uncompressed baseline,
~51% -- precipitation and potential evaporation are accumulated fields
with large near-zero areas and should compress at least as well,
likely somewhat better). For `precipitation`, `temperature`, and
`potential_evaporation` over a 2021-2026 span (72 months if requesting
through end of 2026 -- note ARCO itself only has data up to
`valid_time_stop_era5t`, currently a few months behind "now", so fewer
months will actually exist on disk until requested after ARCO catches
up):

| | per variable | x 3 variables |
|---|---|---|
| 72 months | ~4.6 GB | **~14 GB** |
| 65 months (roughly what's available as of writing) | ~4.2 GB | **~13 GB** |

Requesting `temperature_min`/`temperature_max` in addition to
`temperature` adds separate cache entries each of comparable size
(same underlying hourly source, but cached per daily reduction -- mean,
min, max are each their own file), roughly tripling the temperature
share of this estimate if all three are used.

### Data freshness: final vs. preliminary (ERA5T), and the CDS gap-fill

ARCO-ERA5 itself tracks two coverage boundaries in the store's own
metadata (`ds.attrs`, confirmed via a live read): `valid_time_stop`
(final, stable ERA5 -- permanent, never revised) and
`valid_time_stop_era5t` (preliminary ERA5T -- available sooner, but
**silently revised in place** once the final ERA5 ships for that
period, typically ~3 months later). A snapshot at time of writing:
`valid_time_stop=2026-04-30`, `valid_time_stop_era5t=2026-08-17`.

*t.in.era5* checks these on every month it fetches from ARCO-ERA5, and
caches accordingly:

- **Final** (within `valid_time_stop`): cached as
  `<var>_<reduction>_<year><month>.nc` -- trusted forever once cached,
  no re-check on later hits (final data never changes, so there's
  nothing to gain by re-touching ARCO for it).
- **ERA5T / preliminary** (beyond `valid_time_stop` but within
  `valid_time_stop_era5t`): cached separately, as
  `<var>_<reduction>_<year><month>_era5t.nc` -- this data is real and
  usable now, but is *not* guaranteed to be the exact values that will
  eventually be archived once ARCO ingests the final release for that
  month. Kept apart so a cache hit never silently serves stale-but-
  still-present-on-disk provisional data as if it were final: every
  `_era5t` cache hit re-checks ARCO's own `valid_time_stop` (cheap --
  metadata only, no data re-fetch), and once the final release has
  landed, *t.in.era5* automatically fetches and caches the real final
  month and **deletes the superseded `_era5t` file** -- no stale
  provisional data left behind for good cleaning practice, and no
  manual cleanup needed.
- **Not yet ingested at all** (beyond `valid_time_stop_era5t`):
  ARCO-ERA5 has nothing for this month yet -- not even provisionally.
  In this case *t.in.era5* automatically falls back to the CDS API for
  that one month only (needs `~/.cdsapirc`; a warning is printed), so
  the run still completes instead of silently producing a gap. **Is
  this the same data ARCO will eventually have for that month?** Not
  verifiably -- CDS and ARCO are independent pipelines (CDS's own
  ERA5T is itself provisional and can differ in fine detail from what
  ARCO eventually ingests), so this CDS-sourced fallback is cached
  under yet another distinct name (`<year><month>_arco_gap_cds.nc`, in
  the per-run **cache_dir**, *not* the permanent world cache) rather
  than being trusted as equivalent to genuine ARCO data. It is never
  reused as a substitute once ARCO does ingest that month -- a later
  run will fetch and cache the real ARCO version normally.

## HOURLY OUTPUT (-h)

By default *t.in.era5* reduces everything to one raster per day. Pass
**-h** to instead get one raster per HOUR -- the raw temporal
resolution ERA5(-Land) is actually observed/forecast at, before any
daily reduction. This roughly **multiplies the number of STRDS maps by
24** for the same date range: budget cache disk space, `t.register`
time, and downstream processing accordingly. A 3-month hourly pull is
already ~2160 maps per variable, versus ~90 for the same period daily.

Accumulated fields (`precipitation`, `potential_evaporation`,
`solar_radiation`, `snowfall`) are correctly **de-accumulated** to
genuine per-hour amounts, not just relabeled daily totals split evenly
-- ECMWF's raw archive stores these as running totals since a periodic
reset, not as hourly increments, so getting this right means
differencing consecutive raw readings and special-casing the reset
hour itself (which is already a genuine 1-hour amount, not a diff).
Two DIFFERENT reset schedules are involved, handled separately and
correctly for each source:

- **ERA5-Land** (`reanalysis-era5-land`, already used by the daily path
  for accumulated variables too): a single reset per calendar day, at
  01 UTC.
- **ARCO-ERA5 and plain-ERA5's own raw hourly CDS product**: two 12h
  forecast cycles per day, based at 06 and 18 UTC (first genuine,
  non-diffed hour at 07 and 19 UTC respectively). Confirmed the same
  schedule applies to both: ARCO-ERA5 is an unmodified Zarr mirror of
  this exact same raw ERA5 archive, not a separately-behaving product
  (see [ARCO-ERA5 SOURCE](#arco-era5-google-cloud-source) above), so
  plain-ERA5's raw hourly CDS product (`reanalysis-era5-single-levels`)
  reuses the identical `ECMWF_CYCLE_RESET_HOURS = {7, 19}` de-
  accumulation, not a separately-derived schedule.

### Per-source-tier support

| tier | hourly output | notes |
|---|---|---|
| ERA5-Land (default, tried first) | **yes** | via the same raw-hourly `reanalysis-era5-land` product the daily path already uses for accumulated variables -- in `-h` mode it's used for ALL variables, not just accumulated ones |
| ARCO-ERA5 (default fallback tier) | **yes** | genuinely hour-native at the Zarr store level; see [ARCO-ERA5 SOURCE](#arco-era5-google-cloud-source) above |
| plain ERA5 via CDS (**-c** fallback tier, and the one-month ARCO-unavailable stopgap) | **yes** | via `fetch_era5_raw_hourly()` against CDS's raw `reanalysis-era5-single-levels` product (not `fetch_era5()`'s daily-statistics product, which has no hourly output) -- see the real request-schema difference noted below |

All three source tiers now support `-h`.

**One real, non-obvious request-schema difference found while
implementing the plain-ERA5 raw-hourly fetch**: unlike
`reanalysis-era5-land` (no ensemble/product-type concept at all),
`reanalysis-era5-single-levels` is part of the same "single-levels"
product family as `fetch_era5()`'s derived-daily-statistics product,
and CDS's schema for that family requires an explicit
`product_type: "reanalysis"` request key (distinguishing the
deterministic reanalysis from CDS's separate ensemble-member products
for the same physical field). `fetch_era5_raw_hourly()` carries this
key, mirroring `fetch_era5()`'s own existing single-levels request;
`fetch_era5land_raw_hourly()` correctly does NOT, since ERA5-Land has
no such family/product-type distinction to make.

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

Only needed for the **ERA5-Land** tier (tried by default unless **-e**
is given) and for the plain-ERA5 fallback tier if **-c** is given --
with **-e -c** never needed at all, and with just **-e** (ARCO-ERA5
default) or no flags at all, a missing/invalid `~/.cdsapirc` only
causes the affected month to fall back further, with a warning, rather
than aborting the whole run.

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
only the token itself is account-specific).

## EXAMPLE

```sh
g.region n=34 s=31 e=49 w=47 res=0:06 # ~0.1 deg, matches ERA5-Land's native grid

t.in.era5 variables=precipitation,temperature,potential_evaporation \
  start=2001-06-01 end=2001-08-31 \
  output_prefix=karkheh cache_dir=$HOME/era5_cache

r.hydro.hbv.forcing strds=karkheh_precipitation basins=basins \
  basins_vector=basins_v output_table=karkheh_precip_table
```

Hourly output for a distributed hydrology model needing sub-daily
forcing (e.g. *r.hydro.rri*, not a per-basin-lumped model like
*r.hydro.hbv*):

```sh
t.in.era5 -h variables=precipitation start=2001-06-01 end=2001-06-07 \
  output_prefix=karkheh_hourly cache_dir=$HOME/era5_cache
# -> karkheh_hourly_precipitation, one raster per hour (168 maps for
#    this 7-day window instead of 7)
```

## SEE ALSO

*[r.hydro.hbv](r.hydro.hbv.md)*,
*[r.hydro.hbv.forcing](r.hydro.hbv.forcing.md)*,
*[t.create](t.create.md)*, *[t.register](t.register.md)*,
*[r.import](r.import.md)*

## AUTHOR

Yann Chemin
