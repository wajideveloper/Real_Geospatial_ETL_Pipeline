"""Transformation stage: validate, clean and reproject source features.

Key rules:

* The source CRS must be known. If the source GeoDataFrame has no CRS, it is
  NEVER silently guessed -- ``SOURCE_CRS`` must be configured explicitly.
* All final geometries are reprojected to the configured target CRS
  (default ``EPSG:25832``).
* Features with missing/empty/invalid geometries or a missing unique
  identifier are excluded from loading and reported (not silently dropped).
* Source attributes are preserved unless the unique-identifier handling
  requires a string cast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import shapely.geometry

from .utils import Config, TransformationError

logger = logging.getLogger("etl.transform")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class TransformationReport:
    """Counters describing what happened during transformation."""

    source_features: int = 0
    missing_crs: bool = False
    source_crs_unknown: bool = False
    source_crs_assigned_from_config: bool = False
    invalid_geometries: int = 0
    empty_geometries: int = 0
    missing_ids: int = 0
    duplicate_ids: int = 0
    transformed_features: int = 0
    final_crs: str = ""
    prepared_for_load: int = 0


# ---------------------------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------------------------


def _has_geometry_column(gdf: gpd.GeoDataFrame) -> bool:
    return "geometry" in gdf.columns


def validate_geometries(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, TransformationReport]:
    """Validate the geometry column and remove unusable features.

    Returns the filtered GeoDataFrame and a report of the counts involved.
    Features with a ``None`` geometry, an empty geometry, or an invalid
    geometry are excluded and counted. Missing geometries are **not** assumed
    to be anything other than missing.
    """
    report = TransformationReport()
    report.source_features = len(gdf)

    if not _has_geometry_column(gdf):
        raise TransformationError(
            "Source data has no 'geometry' column. The WFS layer may not "
            "return geometries, or the response format may be unsupported."
        )

    if len(gdf) == 0:
        report.transformed_features = 0
        report.final_crs = str(gdf.crs) if gdf.crs else ""
        return gdf, report

    geoms = gdf["geometry"]

    missing_mask = geoms.isna()
    empty_mask = geoms.map(
        lambda g: bool(g is not None and g.is_empty)
    )
    invalid_mask = geoms.map(
        lambda g: bool(g is not None and not g.is_empty and not g.is_valid)
    )

    report.empty_geometries = int(empty_mask.sum())
    report.invalid_geometries = int(invalid_mask.sum())

    drop_mask = missing_mask | empty_mask | invalid_mask
    dropped = int(drop_mask.sum())
    if dropped:
        logger.warning(
            "Removing %s feature(s) with unusable geometries "
            "(missing=%s, empty=%s, invalid=%s)",
            dropped, int(missing_mask.sum()), report.empty_geometries,
            report.invalid_geometries,
        )

    filtered = gdf.loc[~drop_mask].copy()
    report.transformed_features = len(filtered)
    return filtered, report


# ---------------------------------------------------------------------------
# Attribute cleaning
# ---------------------------------------------------------------------------


def clean_attributes(
    gdf: gpd.GeoDataFrame,
    unique_id_field: str,
) -> tuple[gpd.GeoDataFrame, TransformationReport]:
    """Normalise attributes without arbitrarily altering source values.

    * Ensures the unique identifier exists and is a non-empty string.
    * Removes duplicated source identifiers, keeping the first occurrence.
    * Strips surrounding whitespace from string columns.
    """
    report = TransformationReport()

    if unique_id_field not in gdf.columns:
        raise TransformationError(
            f"Unique identifier field '{unique_id_field}' is not present in "
            f"the source data. Set UNIQUE_ID_FIELD to a field that exists."
        )

    gdf = gdf.copy()
    gdf[unique_id_field] = gdf[unique_id_field].astype(str).str.strip()

    missing_ids = gdf[unique_id_field].isin(["", "None", "nan"]).sum()
    report.missing_ids = int(missing_ids)
    if missing_ids:
        logger.warning(
            "Dropping %s feature(s) with a missing unique identifier.",
            missing_ids,
        )
        gdf = gdf[~gdf[unique_id_field].isin(["", "None", "nan"])].copy()

    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=[unique_id_field], keep="first")
    report.duplicate_ids = before - len(gdf)
    if report.duplicate_ids:
        logger.warning(
            "Removed %s duplicate identifier(s), keeping the first occurrence.",
            report.duplicate_ids,
        )

    for column in gdf.columns:
        if column == "geometry":
            continue
        if gdf[column].dtype == object:
            gdf[column] = gdf[column].map(
                lambda v: v.strip() if isinstance(v, str) else v
            )

    return gdf, report


# ---------------------------------------------------------------------------
# CRS handling
# ---------------------------------------------------------------------------


def resolve_source_crs(gdf: gpd.GeoDataFrame, config: Config) -> tuple[gpd.GeoDataFrame, TransformationReport]:
    """Resolve the CRS of the source GeoDataFrame.

    If the GeoDataFrame already carries a CRS it is used as-is. Otherwise the
    CRS must be supplied via ``SOURCE_CRS``; a missing CRS is an error, never
    an assumption.
    """
    report = TransformationReport()

    if gdf.crs is not None:
        return gdf, report

    report.missing_crs = True
    if not config.source_crs:
        raise TransformationError(
            "Source CRS is missing. Set SOURCE_CRS in .env before "
            "transformation. The pipeline will not guess a CRS."
        )

    report.source_crs_unknown = True
    report.source_crs_assigned_from_config = True
    logger.info(
        "Assigning configured source CRS %s to features that had no CRS.",
        config.source_crs,
    )
    gdf = gdf.set_crs(config.source_crs)
    return gdf, report


def reproject_geometries(
    gdf: gpd.GeoDataFrame,
    target_crs: str,
) -> gpd.GeoDataFrame:
    """Reproject all geometries to the target CRS (default EPSG:25832)."""
    current = str(gdf.crs)
    target = str(target_crs)

    if current == target:
        logger.info("Source CRS already matches target CRS %s; no reprojection needed.", target)
        return gdf

    logger.info("Reprojecting geometries from %s to %s", current, target)
    try:
        gdf = gdf.to_crs(target)
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise TransformationError(
            f"Failed to reproject geometries from {current} to {target}: {exc}"
        ) from exc
    return gdf


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def transform_data(
    gdf: gpd.GeoDataFrame,
    config: Config,
) -> tuple[gpd.GeoDataFrame, TransformationReport]:
    """Run the full transformation pipeline on the extracted features.

    Order of operations: attribute cleaning -> geometry validation -> CRS
    resolution -> reprojection. Returns the ready-to-load GeoDataFrame and a
    report with all counts required for logging/QA.
    """
    logger.info("Transformation start.")
    logger.info("Source CRS: %s", gdf.crs if gdf.crs is not None else "<unknown>")
    logger.info("Target CRS: %s", config.target_crs)

    combined = TransformationReport()

    gdf, clean_report = clean_attributes(gdf, config.unique_id_field)
    combined.missing_ids = clean_report.missing_ids
    combined.duplicate_ids = clean_report.duplicate_ids

    gdf, geom_report = validate_geometries(gdf)
    combined.source_features = geom_report.source_features
    combined.invalid_geometries = geom_report.invalid_geometries
    combined.empty_geometries = geom_report.empty_geometries

    gdf, crs_report = resolve_source_crs(gdf, config)
    combined.missing_crs = crs_report.missing_crs
    combined.source_crs_unknown = crs_report.source_crs_unknown
    combined.source_crs_assigned_from_config = crs_report.source_crs_assigned_from_config

    gdf = reproject_geometries(gdf, config.target_crs)

    combined.transformed_features = len(gdf)
    combined.final_crs = str(gdf.crs)
    combined.prepared_for_load = len(gdf)

    if combined.final_crs != config.target_crs:
        raise TransformationError(
            f"Final CRS is {combined.final_crs}, expected {config.target_crs}. "
            f"Reprojection did not apply correctly."
        )

    logger.info(
        "Transformation complete: source=%s invalid=%s empty=%s "
        "transformed=%s final_crs=%s prepared_for_load=%s",
        combined.source_features,
        combined.invalid_geometries,
        combined.empty_geometries,
        combined.transformed_features,
        combined.final_crs,
        combined.prepared_for_load,
    )
    return gdf, combined
