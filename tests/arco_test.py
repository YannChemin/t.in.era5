"""Tests for the ARCO-ERA5 (public Google Cloud Zarr) source added to
t.in.era5: _arco_select_area()'s bbox/antimeridian handling,
fetch_arco_world_month()'s local de-accumulation of ERA5's raw
forecast-cycle accumulated fields plus its daily-statistic reduction for
instantaneous fields, and fetch_arco()'s permanent area-independent
world-cache contract (see fetch_arco_world_month()'s docstring in
t.in.era5.py for why: ARCO-ERA5 downloads a whole-globe chunk per hour
touched no matter how small the requested area is, so caching the whole
globe instead of just the bbox is free and means a given (variable,
month) is never re-fetched for any area, ever). All synthetic -- no
network access, no live GCS read: fetch_arco() is exercised against an
in-memory xarray.Dataset built to look like what arco_client() would
hand it, so these tests catch regressions in the math/reshaping/caching
without depending on the real archive being reachable.
"""

import os

import numpy as np
import pytest


def _cycle_step(hour):
    """Forecast-cycle step (1..12) for a plain-ERA5 hour-of-day, for
    cycles based at 06 and 18 UTC (first valid/accumulated hour is 07 and
    19 respectively) -- mirrors the reset pattern fetch_arco() assumes."""
    if 7 <= hour <= 18:
        return hour - 6
    if hour >= 19:
        return hour - 18
    return hour + 6  # hour in 0..6: tail of the previous day's 18z cycle


def _make_accumulated_series(xr, np, start, end, increment):
    """A synthetic raw-ERA5-style forecast-accumulated hourly series over
    [start, end) with a constant *true* hourly increment, so a correct
    de-accumulation recovers exactly `increment` at every hour and exactly
    24*increment per day."""
    times = np.arange(start, end, np.timedelta64(1, "h"))
    hours = times.astype("datetime64[h]").astype(object)
    steps = np.array([_cycle_step(t.hour) for t in hours])
    values = (steps * increment).astype(np.float64)
    data = np.broadcast_to(values[:, None, None], (len(times), 2, 2)).copy()
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": times,
            "latitude": np.array([10.0, 0.0]),
            "longitude": np.array([0.0, 1.0]),
        },
        name="test_accum_var",
    )


def test_arco_select_area_simple_bbox(tin, xr, np):
    da = xr.DataArray(
        np.arange(4 * 6).reshape(4, 6).astype(float),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.array([30.0, 20.0, 10.0, 0.0]),
            "longitude": np.arange(0.0, 360.0, 60.0),  # 0,60,120,180,240,300
        },
    )
    out = tin._arco_select_area(da, [25.0, 10.0, 5.0, 130.0])
    assert list(out["latitude"].values) == [20.0, 10.0]
    assert list(out["longitude"].values) == [60.0, 120.0]


def test_arco_select_area_wraps_antimeridian(tin, xr, np):
    da = xr.DataArray(
        np.arange(2 * 36).reshape(2, 36).astype(float),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.array([10.0, 0.0]),
            "longitude": np.arange(0.0, 360.0, 10.0),
        },
    )
    # west=-20 (=340 in 0..360), east=15: bbox straddles the 0 meridian.
    out = tin._arco_select_area(da, [10.0, -20.0, 0.0, 15.0])
    assert list(out["longitude"].values) == [-20.0, -10.0, 0.0, 10.0]
    # longitude renormalized to -180..180 and left ascending-sorted
    assert list(out["longitude"].values) == sorted(out["longitude"].values.tolist())


def test_fetch_arco_deaccumulates_and_sums_to_daily_total(tin, xr, np, tmp_path):
    year, month, days = 2021, 3, [1, 2, 3]
    month_start = np.datetime64("2021-03-01")
    fetch_start = month_start - np.timedelta64(1, "h")
    month_end_excl = month_start + np.timedelta64(len(days), "D")

    increment = 0.001  # true hourly amount (arbitrary units)
    series = _make_accumulated_series(xr, np, fetch_start, month_end_excl, increment)
    ds_full = xr.Dataset({"test_accum_var": series})

    var_info = {"cds_name": "test_accum_var", "accumulated": True}
    out_nc = tmp_path / "accum.nc"
    world_cache_dir = str(tmp_path / "world")
    tin.fetch_arco(
        ds_full, var_info, year, month, days, [10.0, 0.0, 0.0, 1.0], str(out_nc),
        world_cache_dir,
    )

    result = xr.open_dataset(out_nc)["test_accum_var"]
    assert result.sizes["time"] == len(days)
    expected_daily_total = 24 * increment
    np.testing.assert_allclose(result.values, expected_daily_total, rtol=1e-9)
    result.close()


def test_fetch_arco_daily_statistic_for_instantaneous_variable(tin, xr, np, tmp_path):
    year, month, days = 2021, 3, [1, 2]
    month_start = np.datetime64("2021-03-01")
    month_end_excl = month_start + np.timedelta64(len(days), "D")
    times = np.arange(month_start, month_end_excl, np.timedelta64(1, "h"))

    # hour-of-day as the value, identical every day -> known mean/min/max.
    hourly_vals = (times.astype("datetime64[h]").astype(object))
    hourly_vals = np.array([t.hour for t in hourly_vals], dtype=np.float64)
    data = np.broadcast_to(hourly_vals[:, None, None], (len(times), 2, 2)).copy()
    da = xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": times,
            "latitude": np.array([10.0, 0.0]),
            "longitude": np.array([0.0, 1.0]),
        },
        name="test_inst_var",
    )
    ds_full = xr.Dataset({"test_inst_var": da})
    world_cache_dir = str(tmp_path / "world")

    for stat, expected in (
        ("daily_mean", sum(range(24)) / 24.0),
        ("daily_minimum", 0.0),
        ("daily_maximum", 23.0),
    ):
        var_info = {
            "cds_name": "test_inst_var",
            "accumulated": False,
            "daily_statistic": stat,
        }
        out_nc = tmp_path / ("inst_%s.nc" % stat)
        tin.fetch_arco(
            ds_full, var_info, year, month, days, [10.0, 0.0, 0.0, 1.0], str(out_nc),
            world_cache_dir,
        )
        result = xr.open_dataset(out_nc)["test_inst_var"]
        assert result.sizes["time"] == len(days)
        np.testing.assert_allclose(result.values, expected)
        result.close()


def test_native_resolution_deg_has_arco_entry(tin):
    assert tin.NATIVE_RESOLUTION_DEG["arco"] == 0.25


def test_fetch_month_arco_branch_uses_distinct_cache_file_and_reuses_it(tin, xr, tmp_path):
    year, month, days = 2021, 3, [1]
    calls = {"n": 0}

    def make_ds():
        calls["n"] += 1
        month_start = np.datetime64("2021-03-01")
        times = np.arange(
            month_start, month_start + np.timedelta64(1, "D"), np.timedelta64(1, "h")
        )
        data = np.zeros((len(times), 2, 2))
        da = xr.DataArray(
            data,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": times,
                "latitude": np.array([10.0, 0.0]),
                "longitude": np.array([0.0, 1.0]),
            },
            name="test_var",
        )
        return xr.Dataset({"test_var": da})

    var_info = {
        "cds_name": "test_var",
        "accumulated": False,
        "daily_statistic": "daily_mean",
    }
    clients = {"arco": make_ds, "cds": lambda: (_ for _ in ()).throw(AssertionError("CDS should not be used"))}
    world_cache_dir = str(tmp_path / "world")

    path1, source1 = tin.fetch_month(
        clients, "test_var", var_info, year, month, days,
        [10.0, 0.0, 0.0, 1.0], str(tmp_path), True, True, world_cache_dir,
    )
    assert source1 == "arco"
    assert path1.endswith("_arco.nc")
    assert calls["n"] == 1

    # second call for the same month must hit the per-run cache, not
    # fetch again (nor even re-crop the world cache).
    path2, source2 = tin.fetch_month(
        clients, "test_var", var_info, year, month, days,
        [10.0, 0.0, 0.0, 1.0], str(tmp_path), True, True, world_cache_dir,
    )
    assert path2 == path1
    assert source2 == "arco"
    assert calls["n"] == 1


def test_fetch_arco_reuses_world_cache_across_different_areas(tin, xr, tmp_path):
    """The whole point of the world cache: a second request for the SAME
    (variable, year, month) but a DIFFERENT area must never touch
    ds_full again -- it should be served entirely from the local,
    permanent world-cache file fetch_arco_world_month() already wrote."""
    year, month, days = 2021, 3, [1]
    calls = {"n": 0}

    def make_ds():
        calls["n"] += 1
        month_start = np.datetime64("2021-03-01")
        times = np.arange(
            month_start, month_start + np.timedelta64(1, "D"), np.timedelta64(1, "h")
        )
        # distinct values per grid cell, so cropping different areas is
        # actually verifiable (not just "some number came back").
        lats = np.array([10.0, 0.0])
        lons = np.array([0.0, 1.0])
        data = np.zeros((len(times), 2, 2))
        data[:, 0, 0] = 111.0  # (lat=10, lon=0)
        data[:, 1, 1] = 222.0  # (lat=0, lon=1)
        da = xr.DataArray(
            data,
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": lats, "longitude": lons},
            name="test_var",
        )
        return xr.Dataset({"test_var": da})

    var_info = {
        "cds_name": "test_var",
        "accumulated": False,
        "daily_statistic": "daily_mean",
    }
    world_cache_dir = str(tmp_path / "world")

    out_nc_a = tmp_path / "area_a.nc"
    tin.fetch_arco(make_ds(), var_info, year, month, days, [10.0, 0.0, 10.0, 0.0], str(out_nc_a), world_cache_dir)
    assert calls["n"] == 1

    # a completely different area, same (variable, year, month): the
    # world cache file already exists, so ds_full must never be touched
    # again -- pass an object that raises on any access to prove it.
    class Poisoned:
        def __getitem__(self, key):
            raise AssertionError("ds_full accessed again -- world cache was not reused")

    out_nc_b = tmp_path / "area_b.nc"
    tin.fetch_arco(
        Poisoned(), var_info, year, month, days, [0.0, 1.0, 0.0, 1.0], str(out_nc_b),
        world_cache_dir,
    )
    assert calls["n"] == 1  # unchanged: ds_full was never called again

    result_a = xr.open_dataset(out_nc_a)["test_var"]
    result_b = xr.open_dataset(out_nc_b)["test_var"]
    np.testing.assert_allclose(result_a.values, 111.0)
    np.testing.assert_allclose(result_b.values, 222.0)
    result_a.close()
    result_b.close()


def test_arco_availability_final_era5t_and_unavailable(tin, np):
    month_end_excl = np.datetime64("2026-05-01")  # i.e. April 2026

    class Attrs(dict):
        pass

    ds_final = type("DS", (), {"attrs": {"valid_time_stop": "2026-04-30", "valid_time_stop_era5t": "2026-08-17"}})()
    assert tin._arco_availability(ds_final, month_end_excl) == "final"

    ds_era5t = type("DS", (), {"attrs": {"valid_time_stop": "2026-03-31", "valid_time_stop_era5t": "2026-08-17"}})()
    assert tin._arco_availability(ds_era5t, month_end_excl) == "era5t"

    ds_unavailable = type("DS", (), {"attrs": {"valid_time_stop": "2026-03-31", "valid_time_stop_era5t": "2026-04-15"}})()
    assert tin._arco_availability(ds_unavailable, month_end_excl) == "unavailable"

    ds_no_attrs = type("DS", (), {"attrs": {}})()
    assert tin._arco_availability(ds_no_attrs, month_end_excl) == "final"


def test_fetch_arco_world_month_tags_era5t_separately_from_final(tin, xr, np, tmp_path):
    year, month, days = 2026, 4, [1]
    month_start = np.datetime64("2026-04-01")
    times = np.arange(month_start, month_start + np.timedelta64(1, "D"), np.timedelta64(1, "h"))
    data = np.full((len(times), 2, 2), 5.0)
    da = xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": np.array([10.0, 0.0]), "longitude": np.array([0.0, 1.0])},
        name="test_var",
    )
    ds_full = xr.Dataset({"test_var": da})
    ds_full.attrs["valid_time_stop"] = "2026-03-31"  # April is beyond final coverage...
    ds_full.attrs["valid_time_stop_era5t"] = "2026-08-17"  # ...but within ERA5T coverage

    var_info = {"cds_name": "test_var", "accumulated": False, "daily_statistic": "daily_mean"}
    world_cache_dir = str(tmp_path / "world")

    world_nc = tin.fetch_arco_world_month(ds_full, var_info, year, month, days, world_cache_dir)
    assert world_nc.endswith("_era5t.nc")
    assert not os.path.exists(world_nc.replace("_era5t.nc", ".nc"))


def test_fetch_arco_world_month_raises_when_beyond_era5t_coverage(tin, xr, np, tmp_path):
    year, month, days = 2027, 1, [1]  # far beyond any plausible coverage
    ds_full = xr.Dataset({"test_var": xr.DataArray([0.0], dims=("time",), coords={"time": [np.datetime64("2027-01-01")]})})
    ds_full.attrs["valid_time_stop"] = "2026-03-31"
    ds_full.attrs["valid_time_stop_era5t"] = "2026-08-17"

    var_info = {"cds_name": "test_var", "accumulated": False, "daily_statistic": "daily_mean"}
    world_cache_dir = str(tmp_path / "world")

    with pytest.raises(tin.ArcoDataUnavailable):
        tin.fetch_arco_world_month(ds_full, var_info, year, month, days, world_cache_dir)


def test_fetch_month_falls_back_to_cds_and_caches_separately_when_arco_unavailable(
    tin, xr, np, tmp_path, monkeypatch
):
    year, month, days = 2027, 1, [1]
    calls = {"arco": 0, "cds": 0}

    def make_unavailable_ds():
        calls["arco"] += 1
        ds = xr.Dataset(
            {"test_var": xr.DataArray([0.0], dims=("time",), coords={"time": [np.datetime64("2027-01-01")]})}
        )
        ds.attrs["valid_time_stop"] = "2026-03-31"
        ds.attrs["valid_time_stop_era5t"] = "2026-08-17"
        return ds

    class FakeCdsClient:
        pass

    def make_cds_client():
        calls["cds"] += 1
        return FakeCdsClient()

    def fake_fetch_era5(client, var_info, year, month, days, area, out_nc):
        assert isinstance(client, FakeCdsClient)
        xr.Dataset(
            {
                "test_var": xr.DataArray(
                    np.full((1, 2, 2), 42.0),
                    dims=("time", "latitude", "longitude"),
                    coords={
                        "time": [np.datetime64("2027-01-01")],
                        "latitude": [10.0, 0.0],
                        "longitude": [0.0, 1.0],
                    },
                )
            }
        ).to_netcdf(out_nc)

    monkeypatch.setattr(tin, "fetch_era5", fake_fetch_era5)  # CDS path unrelated to this test

    var_info = {"cds_name": "test_var", "accumulated": False, "daily_statistic": "daily_mean"}
    clients = {"arco": make_unavailable_ds, "cds": make_cds_client}
    world_cache_dir = str(tmp_path / "world")

    path1, source1 = tin.fetch_month(
        clients, "test_var", var_info, year, month, days,
        [10.0, 0.0, 0.0, 1.0], str(tmp_path), True, True, world_cache_dir,
    )
    assert source1 == "arco_gap_cds"
    assert path1.endswith("_arco_gap_cds.nc")
    assert calls["arco"] == 1
    assert calls["cds"] == 1
    assert not os.path.exists(os.path.join(str(tmp_path), "test_var_202701_arco.nc"))

    # a second call must reuse the gap-fill cache, touching neither
    # ARCO nor CDS again.
    path2, source2 = tin.fetch_month(
        clients, "test_var", var_info, year, month, days,
        [10.0, 0.0, 0.0, 1.0], str(tmp_path), True, True, world_cache_dir,
    )
    assert path2 == path1
    assert source2 == "arco_gap_cds"
    assert calls["arco"] == 1
    assert calls["cds"] == 1


def test_fetch_arco_world_month_promotes_era5t_to_final_and_deletes_stale_file(
    tin, xr, np, tmp_path
):
    """Once the final release lands for a month previously cached as
    ERA5T, a later call must fetch+cache the final version and delete
    the now-superseded provisional file -- not keep serving it forever."""
    year, month, days = 2026, 6, [1]
    month_start = np.datetime64("2026-06-01")
    times = np.arange(month_start, month_start + np.timedelta64(1, "D"), np.timedelta64(1, "h"))
    var_info = {"cds_name": "test_var", "accumulated": False, "daily_statistic": "daily_mean"}
    world_cache_dir = str(tmp_path / "world")

    def make_ds(value, valid_time_stop):
        data = np.full((len(times), 2, 2), value)
        da = xr.DataArray(
            data,
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": np.array([10.0, 0.0]), "longitude": np.array([0.0, 1.0])},
            name="test_var",
        )
        ds = xr.Dataset({"test_var": da})
        ds.attrs["valid_time_stop"] = valid_time_stop
        ds.attrs["valid_time_stop_era5t"] = "2026-08-17"
        return ds

    # First run: June 2026 is only ERA5T-available (valid_time_stop is
    # still in March).
    era5t_ds = make_ds(1.0, "2026-03-31")
    world_nc_1 = tin.fetch_arco_world_month(era5t_ds, var_info, year, month, days, world_cache_dir)
    assert world_nc_1.endswith("_era5t.nc")
    assert os.path.exists(world_nc_1)

    # Second run: the final release has now landed for June (a later
    # ARCO snapshot reports valid_time_stop past the end of June), and
    # the final value differs from the old provisional one.
    final_ds = make_ds(2.0, "2026-06-30")
    world_nc_2 = tin.fetch_arco_world_month(final_ds, var_info, year, month, days, world_cache_dir)
    assert world_nc_2.endswith(".nc") and not world_nc_2.endswith("_era5t.nc")
    assert os.path.exists(world_nc_2)
    assert not os.path.exists(world_nc_1)  # stale provisional file removed

    result = xr.open_dataset(world_nc_2)["test_var"]
    np.testing.assert_allclose(result.values, 2.0)  # the final value, not the old provisional one
    result.close()

    # Third run: a final cache hit is trusted forever, no re-check.
    class Poisoned:
        def __getitem__(self, key):
            raise AssertionError("ds_full accessed again on a final cache hit")

        @property
        def attrs(self):
            raise AssertionError("ds_full.attrs accessed again on a final cache hit")

    world_nc_3 = tin.fetch_arco_world_month(Poisoned(), var_info, year, month, days, world_cache_dir)
    assert world_nc_3 == world_nc_2
