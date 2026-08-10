# ---------------------------------------------------------------------------
# State ETL - container image
#
# python:3.12-slim keeps the image clean. GeoPandas/pyogrio/shapely ship
# self-contained wheels (GDAL/GEOS bundled), and psycopg2-binary bundles
# libpq, so no system-level GIS packages are required.
#
# No credentials are baked into this image; everything comes from the
# environment at runtime.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source and the development-only mock fixture.
COPY src ./src
COPY test_data ./test_data

# Runtime log directory (bind-mounted from the host in docker-compose).
RUN mkdir -p /app/logs

# Run the ETL pipeline. Configuration must be supplied via the environment.
CMD ["python", "src/main.py"]
