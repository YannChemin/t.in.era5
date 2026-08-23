# t.in.era5

A [GRASS GIS](https://grass.osgeo.org/) addon that downloads daily
climate variables and imports each as its own space-time raster dataset
(STRDS), ready to feed [r.hydro.hbv.forcing](https://github.com/YannChemin/HBV)
(which reduces a STRDS to a per-basin table for `r.hydro.hbv`), or any
other temporal GRASS workflow.

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
`t.register` invocations per variable per month. By default it needs no
account or API key at all (see below).

## ERA5-Land first, plain ERA5 fallback

For each variable, **ERA5-Land** (~9km grid, generally preferable for
land applications) is tried first via the Copernicus Climate Data Store
(CDS); if that fails for a given month (explicit CDS error, no
`~/.cdsapirc` configured, network issue, or a genuine coverage gap), the
module automatically falls back to **plain ERA5** (~31km grid, longer
historical record) for that month only — other months/variables are
unaffected. By default that fallback tier is read straight from the
public [ARCO-ERA5](#arco-era5-google-cloud-source) Google Cloud archive
— no CDS account needed at all for it; pass `-c` to use CDS's own
plain-ERA5 product there instead.

One CDS quirk drives part of the ERA5-Land fallback even when ERA5-Land
data actually exists: the `derived-era5-land-daily-statistics` product
used for "instantaneous" variables (temperature, wind, pressure, ...)
explicitly refuses to compute daily statistics for *accumulated*
variables (precipitation, potential evaporation, solar radiation,
snowfall). For those, `t.in.era5` requests raw hourly
`reanalysis-era5-land` data instead and sums it to a daily total itself
— still ERA5-Land, just via a different CDS product — before falling
back to the plain-ERA5 tier only if that also fails.

Requests are chunked per (variable, year, month), so a single failure
doesn't force re-fetching a whole multi-year request, and repeat runs
against the same `cache_dir` reuse already-downloaded months instead of
re-downloading them — each cached month is always fetched in full
(every day of that calendar month), so a cached month is safe to reuse
for any other request touching it, regardless of exactly which days
that later request needs.

## ARCO-ERA5 (Google Cloud) source

By default, the plain-ERA5 fallback tier is read directly from
[ARCO-ERA5](https://github.com/google-research/arco-era5), a public,
anonymous-access Zarr mirror of ERA5 on Google Cloud Storage
(`gs://gcp-public-data-arco-era5`), via `xarray`/`zarr`/`gcsfs` instead
of a CDS request — no CDS account needed for that tier, and it
sidesteps the CDS download queue entirely. ARCO-ERA5 only ever provides
plain ERA5 (~31km grid); ERA5-Land is still tried first via CDS unless
`-e` is also given, so `-e` alone (without `-c`) means `t.in.era5`
never contacts CDS at all — needing no CDS account whatsoever. Pass
`-c` to use CDS's own plain-ERA5 product for that tier instead (needs
`~/.cdsapirc`).

`t.in.era5` opens the store with `xarray.open_zarr(..., chunks=None,
storage_options=dict(token="anon"))` — matching
[Google's own arco-era5 usage examples](https://github.com/google-research/arco-era5)
for this exact dataset, and confirmed in practice to be both simpler
(no `dask` dependency at all) and faster than `chunks="auto"`, which
only adds a dask-array layer that regroups the store's own chunks
without reducing what gets transferred.

ARCO-ERA5 is an unmodified GRIB-to-Zarr conversion of the raw archive,
so accumulated variables are still in ECMWF's raw forecast-accumulation
form and are de-accumulated locally to daily totals — a different
correction than the one ERA5-Land's raw hourly path applies, per
ECMWF's published conversion guidance for these fields.

### Cost/performance: whole-globe-per-hour chunking, and the world cache

Per arco-era5's own README, the `ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
array this module reads is chunked `{"time": 1, "latitude": 721,
"longitude": 1440}` — **one Zarr chunk per hour, spanning the entire
globe** (confirmed directly against the store's chunk metadata: ~4.15 MB
uncompressed, ~1.7–1.9 MB compressed per hourly chunk for a 2D surface
field). There is no bounding box small enough to avoid downloading and
decompressing that whole-globe chunk for every hour touched — `area`
only trims what's *kept* after each hourly chunk is already in memory,
not what gets *transferred*. A full calendar month is ~720–745 such
chunks, so budget roughly **1.2–1.3 GB transferred per accumulated
variable per month** the *first* time that (variable, year, month) is
ever requested.

Since keeping the whole globe costs nothing beyond what a bbox-only
fetch already pays for, `t.in.era5` does exactly that: every ARCO-ERA5
fetch is cached permanently, at full world coverage, under
`arco_world_cache` (default `$HOME/RSDATA/ERA5World`), keyed only by
(variable, reduction, year, month) — deliberately independent of
`area`, `output_prefix`, or which run asked for it. The very first
request for a given (variable, month) anywhere in the world pays the
full download; every later request for that same (variable, month) —
for *any* region, *any* project, *any* future run — is served from
that local file with no network access at all. Confirmed in practice: a
second fetch for a different continent's bounding box, same month,
dropped from ~82s to ~0.05s. This directory grows over time by design
(each cached month is a genuine, reusable slice of world coverage,
compressed to a few MB per variable per month for typical fields) and
is never pruned automatically — treat it as a small, permanent local
ERA5 mirror you're building up, and manage its size yourself if that
ever matters. A handful of days for a brand-new month is still fast (a
few minutes, paid once); only a large historical multi-month backfill
of variables/months never requested before is slow, and even then only
once — for a true first-time bulk historical load, `-c` (CDS, which
subsets server-side and so is cheaper per single request but never
builds up a reusable local archive) may still be preferable.

#### Estimated `arco_world_cache` disk usage

Every cached month/variable is a full 721×1440 (0.25°) daily-resolution
global field, ~126 MB uncompressed, compressed to roughly **60–65 MB**
(measured: a real cached month of daily-mean temperature came to 63.5 MB
against a 124.6 MB uncompressed baseline, ~51% — precipitation and
potential evaporation are accumulated fields with large near-zero areas
and should compress at least as well, likely somewhat better). For
`precipitation`, `temperature`, and `potential_evaporation` over a
2021–2026 span (72 months if requesting through end of 2026 — note ARCO
itself only has data up to `valid_time_stop_era5t`, currently a few
months behind "now", so fewer months will actually exist on disk until
requested after ARCO catches up):

| | per variable | × 3 variables |
|---|---|---|
| 72 months | ~4.6 GB | **~14 GB** |
| 65 months (roughly what's available as of writing) | ~4.2 GB | **~13 GB** |

Requesting `temperature_min`/`temperature_max` in addition to
`temperature` adds separate cache entries each of comparable size (same
underlying hourly source, but cached per daily reduction — mean, min,
max are each their own file), roughly tripling the temperature share
of this estimate if all three are used.

### Data freshness: final vs. preliminary (ERA5T), and the CDS gap-fill

ARCO-ERA5 itself tracks two coverage boundaries in the store's own
metadata (`ds.attrs`, confirmed via a live read):
`valid_time_stop` (final, stable ERA5 — permanent, never revised) and
`valid_time_stop_era5t` (preliminary ERA5T — available sooner, but
**silently revised in place** once the final ERA5 ships for that
period, typically ~3 months later). A snapshot at time of writing:
`valid_time_stop=2026-04-30`, `valid_time_stop_era5t=2026-08-17`.

`t.in.era5` checks these on every month it fetches from ARCO-ERA5, and
caches accordingly:

- **Final** (within `valid_time_stop`): cached as `<var>_<reduction>_
  <year><month>.nc` — trusted forever once cached, no re-check on later
  hits (final data never changes, so there's nothing to gain by
  re-touching ARCO for it).
- **ERA5T / preliminary** (beyond `valid_time_stop` but within
  `valid_time_stop_era5t`): cached separately, as `<var>_<reduction>_
  <year><month>_era5t.nc` — this data is real and usable now, but is
  *not* guaranteed to be the exact values that will eventually be
  archived once ARCO ingests the final release for that month. Kept
  apart so a cache hit never silently serves stale-but-still-present-
  on-disk provisional data as if it were final: every `_era5t` cache
  hit re-checks ARCO's own `valid_time_stop` (cheap — metadata only, no
  data re-fetch), and once the final release has landed, `t.in.era5`
  automatically fetches and caches the real final month and **deletes
  the superseded `_era5t` file** — no stale provisional data left
  behind for good cleaning practice, and no manual cleanup needed.
- **Not yet ingested at all** (beyond `valid_time_stop_era5t`): ARCO-
  ERA5 has nothing for this month yet — not even provisionally. In this
  case `t.in.era5` automatically falls back to the CDS API for that one
  month only (needs `~/.cdsapirc`; a warning is printed), so the run
  still completes instead of silently producing a gap. **Is this the
  same data ARCO will eventually have for that month?** Not verifiably
  — CDS and ARCO are independent pipelines (CDS's own ERA5T is itself
  provisional and can differ in fine detail from what ARCO eventually
  ingests), so this CDS-sourced fallback is cached under yet another
  distinct name (`<year><month>_arco_gap_cds.nc`, in the per-run
  `cache_dir`, *not* the permanent world cache) rather than being
  trusted as equivalent to genuine ARCO data. It is never reused as a
  substitute once ARCO does ingest that month — a later run will fetch
  and cache the real ARCO version normally.

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

Only needed for the **ERA5-Land** tier (tried by default unless `-e` is
given) and for the plain-ERA5 fallback tier if `-c` is given — with
`-e -c` never needed at all, and with just `-e` (ARCO-ERA5 default) or
no flags at all, a missing/invalid `~/.cdsapirc` only causes the
affected month to fall back further, with a warning, rather than
aborting the whole run.

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
only the token itself is account-specific).

## Options

| Option | Description |
|---|---|
| `variables` | Comma-separated list from the table above |
| `start`, `end` | `YYYY-MM-DD`, inclusive |
| `output_prefix` | STRDS created as `<output_prefix>_<variable>` |
| `area` | `north,west,south,east` in WGS84 degrees; default derived from the current region |
| `cache_dir` | Directory to cache downloaded NetCDF files in; default a temporary, run-scoped directory |
| `arco_world_cache` | Directory for the permanent, area-independent full-world ARCO-ERA5 monthly cache; default `$HOME/RSDATA/ERA5World` (see ARCO-ERA5 section above) |
| `-e` | Force plain ERA5 (skip ERA5-Land) for every variable/month |
| `-c` | Use the CDS API instead of ARCO-ERA5 (Google Cloud) for the plain-ERA5 tier |

## Requirements

- GRASS GIS with the temporal framework (`t.create`, `t.register`) and
  `r.import` (core)
- `xarray`, `netCDF4`, GDAL Python bindings (`osgeo.gdal`/`osgeo.osr`)
- `gcsfs` and `zarr` for the default ARCO-ERA5 source
- `cdsapi` and a free CDS account with `~/.cdsapirc` (see above) —
  needed only for the ERA5-Land tier and for `-c`; not needed at all
  when running with `-e` and without `-c`
- `make venv` (see the Makefile) builds a dedicated virtualenv with all
  the above (except the GDAL bindings, which come from GRASS) under the
  installed addon's own `etc/t.in.era5/venv`; *t.in.era5.py* detects and
  uses it automatically, so no manual `pip install` is needed — and none
  is attempted against GRASS's own Python installation, which a plain
  `pip install` often can't touch anyway (PEP 668's
  externally-managed-environment guard on modern Debian/Ubuntu)

## Install

```
g.extension extension=t.in.era5 url=https://github.com/YannChemin/t.in.era5
```

`g.extension` runs this addon's `Makefile`, whose `default` target
builds the bundled virtualenv (see Requirements) as well as installing
the script itself — no separate setup step needed.

## Testing

`tests/arco_test.py` covers the ARCO-ERA5 source (bbox/antimeridian
handling, forecast-accumulation de-accumulation, daily-statistic
reduction) with synthetic in-memory data — no network access needed.
Run with `pytest tests/` (needs `grass` on `PATH`, no live GRASS session
or `~/.cdsapirc` required for these).

Beyond that, no standalone testsuite lives in this repo yet — the full
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
