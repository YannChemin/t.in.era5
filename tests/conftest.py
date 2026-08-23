"""
pytest fixtures for t.in.era5's tests -- discovers the GRASS Python path via
`grass --config python_path` (no `grass --exec` wrapper needed) and loads
t.in.era5.py (whose filename isn't a valid Python module name, hence
importlib rather than a normal import). No GRASS session is spun up: the
functions under test here (the ARCO-ERA5 source) are pure xarray/numpy
computation and never call into GRASS itself.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GRASS_BIN = shutil.which("grass")
if not GRASS_BIN:
    collect_ignore_glob = ["*"]
else:
    _python_path = subprocess.run(
        [GRASS_BIN, "--config", "python_path"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if _python_path not in sys.path:
        sys.path.insert(0, _python_path)


@pytest.fixture(scope="session")
def tin():
    """The t.in.era5.py module, loaded fresh via importlib."""
    module_path = Path(__file__).resolve().parent.parent / "t.in.era5.py"
    spec = importlib.util.spec_from_file_location("t_in_era5", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def np():
    import numpy

    return numpy


@pytest.fixture(scope="session")
def xr():
    import xarray

    return xarray
