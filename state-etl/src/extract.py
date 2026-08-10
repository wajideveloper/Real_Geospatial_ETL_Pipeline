"""Extraction stage: retrieve source geospatial features.

Primary implementation fetches features from a WFS endpoint using ``requests``
with configurable paging (WFS 2.0 ``count``/``startIndex``).

A clearly separated *mock* extractor reads a local GeoJSON fixture so the full
pipeline can be exercised locally without any client infrastructure. The mock
mode is opt-in via ``USE_MOCK_WFS=true`` and is documented as development-only.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import Config, ExtractionError

logger = logging.getLogger("etl.extract")

# Feature collection JSON keys accepted by the parser.
_FEATURE_KEYS = ("features", "FeatureCollection")
_GEOJSON_CRS84 = "urn:ogc:def:crs:OGC:1.3:CRS84"

_WFS1_PARAM = "typeName"   # WFS 1.0.0 / 1.1.0
_WFS2_PARAM = "typeNames"  # WFS 2.0.0


# ---------------------------------------------------------------------------
# WFS request construction
# ---------------------------------------------------------------------------


def _typename_param(version: str) -> str:
    """Return the correct typename parameter for the WFS version."""
    if str(version).startswith("2"):
        return _WFS2_PARAM
    return _WFS1_PARAM


def build_get_feature_params(
    config: Config,
    page_size: int,
    start_index: int,
) -> dict:
    """Build the URL parameters for one ``GetFeature`` request."""
    params: dict = {
        "service": "WFS",
        "version": config.wfs_version,
        "request": "GetFeature",
        _typename_param(config.wfs_version): config.wfs_layer,
    }

    output_format = (config.wfs_output_format or "json").strip().lower()
    if output_format in {"json", "geojson", "geo+json"}:
        params["outputFormat"] = "application/json"

    if config.wfs_max_features and config.wfs_max_features > 0:
        params["count"] = min(page_size, config.wfs_max_features)
    else:
        params["count"] = page_size

    if config.wfs_enable_paging:
        params["startIndex"] = start_index

    return params


# ---------------------------------------------------------------------------
# Response validation and parsing
# ---------------------------------------------------------------------------


def validate_wfs_response(status_code: int, content: bytes) -> None:
    """Raise a descriptive :class:`ExtractionError` for bad WFS responses."""
    if status_code == 404:
        raise ExtractionError(
            f"WFS endpoint returned HTTP 404 (not found). Check WFS_URL "
            f"({status_code})."
        )
    if status_code == 401 or status_code == 403:
        raise ExtractionError(
            f"WFS endpoint denied access with HTTP {status_code}. "
            f"Check authentication / permissions."
        )
    if status_code == 500 or status_code == 502 or status_code == 503:
        raise ExtractionError(
            f"WFS server error with HTTP {status_code}. The layer may be "
            f"temporarily unavailable; retry later."
        )
    if status_code >= 400:
        raise ExtractionError(
            f"WFS request failed with HTTP {status_code}."
        )
    if not content:
        raise ExtractionError("WFS response body was empty.")


def parse_geojson_features(content: bytes, fallback_crs: str) -> gpd.GeoDataFrame:
    """Parse a GeoJSON ``FeatureCollection`` body into a GeoDataFrame."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExtractionError(
            "WFS response is not valid GeoJSON JSON."
        ) from exc

    if "features" not in payload:
        raise ExtractionError(
            "WFS GeoJSON response is missing a 'features' member; cannot "
            "build a feature collection."
        )

    features = payload.get("features") or []
    if not features:
        return gpd.GeoDataFrame({"geometry": []}, crs=fallback_crs)

    try:
        gdf = gpd.GeoDataFrame.from_features(features, crs=fallback_crs)
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise ExtractionError(
            f"Failed to parse WFS GeoJSON features into a GeoDataFrame: {exc}"
        ) from exc
    return gdf


def parse_gml_features(content: bytes) -> gpd.GeoDataFrame:
    """Parse a GML/XML WFS response via GDAL (pyogrio)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".gml", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        return gpd.read_file(tmp_path)
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise ExtractionError(
            f"Failed to parse GML WFS response with GDAL: {exc}. "
            f"If the server does not support GML, request GeoJSON via "
            f"WFS_OUTPUT_FORMAT=json."
        ) from exc
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:  # pragma: no cover - cleanup best-effort
            pass


def parse_wfs_response(
    content: bytes,
    content_type: str,
    output_format: str,
    fallback_crs: str,
) -> gpd.GeoDataFrame:
    """Route a WFS response body to the correct parser.

    GeoJSON output is the preferred, most portable format. Anything else is
    treated as GML and parsed with GDAL.
    """
    fmt = (output_format or "").lower()
    if "json" in fmt or (content_type and "json" in content_type.lower()):
        gdf = parse_geojson_features(content, fallback_crs)
        if gdf.crs is None:
            # GeoJSON has no EPSG metadata; CRS84 (WGS84) is the JSON default.
            gdf = gdf.set_crs(_GEOJSON_CRS84)
        return gdf
    return parse_gml_features(content)


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


def _make_session(timeout: int) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=4, pool_maxsize=4)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _http_get(session: requests.Session, url: str, params: dict, timeout: int) -> requests.Response:
    try:
        return session.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
    except requests.exceptions.Timeout as exc:
        raise ExtractionError(
            f"WFS request timed out after {timeout}s for layer "
            f"{params.get('typeNames') or params.get('typeName')}."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise ExtractionError(
            f"Could not connect to WFS endpoint at {url}. Check WFS_URL."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ExtractionError(f"WFS request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main WFS fetch
# ---------------------------------------------------------------------------


def fetch_wfs(config: Config, session: Optional[requests.Session] = None) -> gpd.GeoDataFrame:
    """Fetch every feature from the configured WFS layer.

    Uses ``count`` + ``startIndex`` paging (WFS 2.0) when enabled so large
    datasets are not held in a single HTTP response. The maximum number of
    features can be capped with ``WFS_MAX_FEATURES``.
    """
    own_session = session is None
    session = session or _make_session(config.wfs_timeout)

    page_size = max(1, config.wfs_page_size)
    start_index = 0
    pages: list[gpd.GeoDataFrame] = []

    try:
        while True:
            params = build_get_feature_params(config, page_size, start_index)
            logger.info(
                "WFS GetFeature request: layer=%s page_start=%s count=%s",
                config.wfs_layer, start_index, params["count"],
            )
            response = _http_get(session, config.wfs_url, params, config.wfs_timeout)
            logger.info("WFS HTTP response status: %s", response.status_code)
            validate_wfs_response(response.status_code, response.content)

            gdf = parse_wfs_response(
                response.content,
                response.headers.get("Content-Type", ""),
                config.wfs_output_format,
                _GEOJSON_CRS84,
            )
            logger.info("WFS page returned %s features", len(gdf))
            pages.append(gdf)

            count_this_page = len(gdf)
            start_index += count_this_page

            reached_total = count_this_page < page_size
            hit_cap = bool(config.wfs_max_features) and start_index >= config.wfs_max_features
            if reached_total or hit_cap or not config.wfs_enable_paging:
                break

        if not pages:
            return gpd.GeoDataFrame({"geometry": []}, crs=_GEOJSON_CRS84)

        combined = pages[0]
        for page in pages[1:]:
            combined = _concat_gdfs(combined, page)
        if config.wfs_max_features and config.wfs_max_features > 0:
            combined = combined.head(config.wfs_max_features)
        return combined
    finally:
        if own_session:
            session.close()


def _concat_gdfs(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Concatenate two GeoDataFrames without losing the geometry column."""
    import pandas as pd

    if left.empty:
        return right
    if right.empty:
        return left
    combined = gpd.GeoDataFrame(pd.concat([left, right], ignore_index=True), crs=left.crs)
    return combined


# ---------------------------------------------------------------------------
# Mock / local test source
# ---------------------------------------------------------------------------


def fetch_mock_data(config: Config) -> gpd.GeoDataFrame:
    """Read the local development fixture (NEVER used in production).

    The mock fixture is clearly labelled as test data and is used only when
    ``USE_MOCK_WFS=true`` so the pipeline can be validated without a real WFS.
    """
    path = Path(config.mock_data_path).expanduser()
    if not path.is_file():
        raise ExtractionError(
            f"MOCK_DATA_PATH file not found: {path}. Provide a GeoJSON "
            f"fixture for local testing."
        )

    logger.info("Loading mock source dataset from %s", path)
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise ExtractionError(f"Failed to read mock dataset {path}: {exc}") from exc

    if gdf.crs is None:
        gdf = gdf.set_crs(config.mock_crs or "EPSG:4326")

    logger.warning(
        "Using MOCK source data (USE_MOCK_WFS=true). This is a local "
        "development fixture, NOT the client's production data."
    )
    return gdf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_source_data(config: Config, session: Optional[requests.Session] = None) -> gpd.GeoDataFrame:
    """Load the source dataset using the configured mode.

    * ``USE_MOCK_WFS=true``  -> local fixture
    * otherwise              -> real WFS GetFeature requests
    """
    if config.use_mock_wfs:
        gdf = fetch_mock_data(config)
    else:
        gdf = fetch_wfs(config, session)

    if len(gdf) == 0:
        logger.warning("Extraction produced 0 features.")
    else:
        logger.info("Extraction complete: %s features retrieved", len(gdf))
    return gdf
