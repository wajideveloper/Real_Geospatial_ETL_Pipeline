"""ETL package for the state geospatial dataset.

Modules:

* ``extract``   -- WFS extraction (with an opt-in local mock mode)
* ``transform`` -- validation, cleaning, CRS reprojection (EPSG:25832)
* ``load``      -- idempotent PostGIS loading with upsert + QA checks
* ``utils``     -- configuration, logging, exceptions
"""

__version__ = "1.0.0"
