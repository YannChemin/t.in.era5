"""Tests for the -h (hourly output) support added to t.in.era5's ERA5-Land
raw-hourly path: _deaccumulate_hourly() (the shared de-accumulation
primitive) against ERA5-Land's specific reset schedule (a single reset
per calendar day at 01 UTC -- see that function's docstring and
ERA5LAND_RESET_HOURS), exercised through load_daily()'s "era5land" +
accumulated + hourly branch. Also the strongest available correctness
check for the whole -h feature: aggregating hourly output back to a
daily total must reproduce the SAME number the existing, already-trusted
daily-mode reduction (load_daily()'s original era5land branch) computes
for the identical synthetic input -- internal consistency between two
independently-coded reduction paths over the same raw data. All
synthetic, no network access.
"""

import numpy as np
import pytest


def _make_era5land_series(xr, np, n_days, increment, first_day_offset_hours=0):
    """A synthetic raw-ERA5-Land-style accumulated hourly series: resets
    to a genuine 1-hour amount at 01 UTC each day, then climbs by
    `increment` every subsequent hour through 00 UTC the following day
    (the peak = that cycle's full day total = 24*increment), matching
    the reset pattern load_daily()'s existing (already-trusted) daily
    branch documents and de-accumulates. `first_day_offset_hours` lets a
    test start the series mid-cycle (hour 0 of day 1, as a real month
    fetch would) so the "spurious leftover hour" edge case is covered
    too."""
    hours_per_cycle = 24  # 01,02,...,23,00(next day)
    n_hours = n_days * 24 + first_day_offset_hours
    start = np.datetime64("2021-06-01T00:00:00") - np.timedelta64(first_day_offset_hours, "h")
    times = start + np.arange(n_hours) * np.timedelta64(1, "h")
    hours_of_day = np.array(
        [t.astype("datetime64[h]").astype(object).hour for t in times]
    )
    # step within the current cycle: hour 01 -> step 1, hour 02 -> step 2,
    # ..., hour 00 (next day) -> step 24 (the peak/full-day total).
    steps = np.where(hours_of_day == 0, 24, hours_of_day)
    values = (steps * increment).astype(np.float64)
    data = np.broadcast_to(values[:, None, None], (n_hours, 2, 2)).copy()
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": times,
            "latitude": np.array([10.0, 0.0]),
            "longitude": np.array([0.0, 1.0]),
        },
        name="test_land_accum_var",
    )


def test_deaccumulate_hourly_era5land_reset_schedule(tin, xr, np):
    """_deaccumulate_hourly() with ERA5LAND_RESET_HOURS must recover the
    true per-hour increment at every hour, given ERA5-Land's single
    01-UTC-reset cycle."""
    increment = 0.05
    da = _make_era5land_series(xr, np, n_days=2, increment=increment)
    result = tin._deaccumulate_hourly(da, "time", tin.ERA5LAND_RESET_HOURS)

    # First output hour is at 01:00 (hour 00 is only the diff base point,
    # never itself an output -- see _deaccumulate_hourly()'s docstring).
    first_time = result["time"].values[0]
    assert np.datetime64(first_time, "h") == np.datetime64("2021-06-01T01")
    np.testing.assert_allclose(result.values, increment, rtol=1e-9)


def test_hourly_and_daily_deaccumulation_agree_on_the_same_synthetic_day(tin, xr, np, tmp_path):
    """The strongest correctness check available: sum a day's worth of
    HOURLY de-accumulated output and compare it against the existing,
    independently-coded DAILY reduction (load_daily()'s original
    shift-and-take-last logic) run on the exact same raw input. Two
    different algorithms over the same data landing on the same number
    is real evidence neither is quietly wrong in a self-consistent way."""
    increment = 0.02
    # 3 days so the middle day is unaffected by either edge (the first
    # day's leftover-hour trim, or the incomplete-last-day undercount
    # both existing branches document).
    da = _make_era5land_series(xr, np, n_days=3, increment=increment)
    ds = xr.Dataset({"test_land_accum_var": da})
    nc_path = tmp_path / "era5land_accum.nc"
    ds.to_netcdf(nc_path)

    var_info = {"cds_name": "test_land_accum_var", "accumulated": True}

    daily_timestamps, daily_values, _, _ = tin.load_daily(
        str(nc_path), var_info, "era5land", hourly=False
    )
    hourly_timestamps, hourly_values, _, _ = tin.load_daily(
        str(nc_path), var_info, "era5land", hourly=True
    )

    # Middle day: 2021-06-02.
    target_day = np.datetime64("2021-06-02")
    daily_idx = [i for i, t in enumerate(daily_timestamps) if np.datetime64(t.date()) == target_day]
    assert len(daily_idx) == 1
    daily_total = daily_values[daily_idx[0]][0, 0]  # scalar field, any cell

    hourly_idx = [
        i for i, t in enumerate(hourly_timestamps)
        if np.datetime64(t.date()) == target_day
        or (np.datetime64(t.date()) == target_day - np.timedelta64(1, "D") and t.hour == 0)
    ]
    # The 24 hours belonging to the cycle that PEAKS on 2021-06-02 are
    # hours 01..23 on 06-02 itself plus hour 00 on 06-03 (the cycle's
    # final reading) -- i.e. this day's cycle runs 06-02T01 .. 06-03T00.
    hourly_idx = [
        i for i, t in enumerate(hourly_timestamps)
        if (t.date() == target_day.astype(object) and t.hour >= 1)
        or (t.date() == (target_day + np.timedelta64(1, "D")).astype(object) and t.hour == 0)
    ]
    assert len(hourly_idx) == 24, "expected all 24 hours of the cycle peaking on 06-02"
    hourly_sum = sum(hourly_values[i][0, 0] for i in hourly_idx)

    np.testing.assert_allclose(hourly_sum, daily_total, rtol=1e-9)
    np.testing.assert_allclose(daily_total, 24 * increment, rtol=1e-9)


def test_load_daily_hourly_era5land_non_accumulated_needs_no_deaccumulation(tin, xr, np, tmp_path):
    """Non-accumulated ERA5-Land fields (temperature etc.) in hourly mode
    are raw readings already -- load_daily() must return them unchanged,
    not run them through the accumulated-field branch."""
    times = np.arange(
        np.datetime64("2021-06-01T00"), np.datetime64("2021-06-01T06"), np.timedelta64(1, "h")
    )
    vals = np.arange(len(times), dtype=np.float64)
    data = np.broadcast_to(vals[:, None, None], (len(times), 2, 2)).copy()
    da = xr.DataArray(
        data, dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": np.array([10.0, 0.0]), "longitude": np.array([0.0, 1.0])},
        name="test_land_inst_var",
    )
    ds = xr.Dataset({"test_land_inst_var": da})
    nc_path = tmp_path / "era5land_inst.nc"
    ds.to_netcdf(nc_path)

    var_info = {"cds_name": "test_land_inst_var", "accumulated": False, "daily_statistic": "daily_mean"}
    timestamps, values, _, _ = tin.load_daily(str(nc_path), var_info, "era5land", hourly=True)

    assert len(timestamps) == len(times)
    np.testing.assert_allclose([v[0, 0] for v in values], vals)
