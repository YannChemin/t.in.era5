PGM = t.in.era5

include $(MODULE_TOPDIR)/include/Make/Script.make

default: script venv

# A dedicated virtualenv holding t.in.era5's extra Python dependencies
# (cdsapi, xarray, gcsfs, zarr, netCDF4 -- see requirements.txt), built
# under $(ETC)/$(PGM) so Script.make's own `install:` rule (which already
# copies that whole directory to $(INST_DIR)/etc/$(PGM)) carries it along
# for free. t.in.era5.py finds it at runtime relative to its own
# installed location (../etc/$(PGM)/venv) and prepends its site-packages
# to sys.path -- GRASS always runs addon scripts with GRASS's own Python,
# which has none of these, and a plain system-wide `pip install` into
# that Python is blocked outright on modern Debian/Ubuntu by PEP 668's
# externally-managed-environment guard.
VENV_DIR = $(ETC)/$(PGM)/venv
VENV_STAMP = $(VENV_DIR)/.requirements-installed

venv: $(VENV_STAMP)

$(VENV_STAMP): requirements.txt
	@mkdir -p $(ETC)/$(PGM)
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --quiet --upgrade pip
	$(VENV_DIR)/bin/pip install --quiet -r requirements.txt
	touch $@

.PHONY: venv
