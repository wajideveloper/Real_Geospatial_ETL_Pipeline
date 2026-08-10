"""Load stage: write transformed features into PostgreSQL/PostGIS.

* Connectivity through SQLAlchemy (psycopg2 driver).
* Geometries are inserted as real PostGIS geometries via
  ``ST_GeomFromWKB(decode(..., 'hex'), srid)`` -- never as plain WKT text.
* Idempotency is enforced with ``INSERT ... ON CONFLICT (unique_id) DO UPDATE``.
  Repeated runs update existing rows instead of inserting duplicates.
* Commits happen per batch so a partial failure is restart-safe: already
  committed batches persist, and the next run upserts the remainder.
* In local test mode the target schema/table can be created automatically.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import geopandas as gpd
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .utils import Config, LoadError

logger = logging.getLogger("etl.load")

GEOMETRY_COLUMN = "geometry"
UPDATED_AT_COLUMN = "updated_at"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def build_db_url(config: Config) -> str:
    """Build a SQLAlchemy URL. The password lives only in this string."""
    return (
        f"postgresql+psycopg2://{config.db_user}:{config.db_password}"
        f"@{config.db_host}:{config.db_port}/{config.db_name}"
    )


def get_engine(config: Config) -> Engine:
    """Create a SQLAlchemy engine with connection pre-ping enabled."""
    logger.info(
        "Connecting to PostgreSQL %s:%s/%s (schema=%s, table=%s)",
        config.db_host, config.db_port, config.db_name,
        config.target_schema, config.target_table,
    )
    return sa.create_engine(
        build_db_url(config),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
        connect_args={"connect_timeout": 30},
    )


def ensure_postgis(engine: Engine) -> str:
    """Verify the PostGIS extension is available; return its version."""
    with engine.connect() as conn:
        try:
            result = conn.execute(sa.text("SELECT PostGIS_Version()")).scalar()
        except sa.exc.OperationalError as exc:
            raise LoadError(
                "Could not connect to the PostgreSQL database. Check DB_HOST, "
                "DB_PORT, DB_NAME, DB_USER and DB_PASSWORD."
            ) from exc
        except sa.exc.ProgrammingError as exc:
            raise LoadError(
                "PostGIS is not installed in this database. Run: "
                "CREATE EXTENSION IF NOT EXISTS postgis;"
            ) from exc
    if not result:
        raise LoadError("PostGIS returned no version; extension missing.")
    logger.info("PostGIS available: %s", result)
    return result


# ---------------------------------------------------------------------------
# Local test table creation (dev mode only)
# ---------------------------------------------------------------------------


def _sql_type_for_series(series: pd.Series) -> str:
    """Map a pandas dtype to a sensible PostgreSQL column type."""
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMPTZ"
    return "TEXT"


def create_local_table(engine: Engine, config: Config, gdf: gpd.GeoDataFrame) -> None:
    """Create the schema/table used for local development only.

    This is intentionally guarded by ``LOCAL_TEST_MODE`` and
    ``AUTO_CREATE_*`` flags so it can never touch a production schema by
    accident. In production the DBA owns the target table definition.

    Column types are derived from the transformed data so the helper works
    with any source fields; the unique identifier is always ``TEXT PRIMARY
    KEY`` to enforce idempotency.
    """
    if not (config.local_test_mode and config.auto_create_table):
        return

    if config.auto_create_schema:
        with engine.begin() as conn:
            conn.execute(sa.text(
                f'CREATE SCHEMA IF NOT EXISTS "{config.target_schema}"'
            ))
        logger.info("Ensured schema %s exists.", config.target_schema)

    srid = _srid_from_crs(config.target_crs)
    columns = [f'"{config.unique_id_field}" TEXT PRIMARY KEY']

    for column in _attribute_columns(gdf, config):
        if column in {"updated_at", "etl_loaded_at"}:
            continue
        columns.append(f'"{column}" {_sql_type_for_series(gdf[column])}')

    columns.append(f'"{GEOMETRY_COLUMN}" geometry(Geometry, {srid})')
    columns.append(f'"{UPDATED_AT_COLUMN}" TIMESTAMPTZ DEFAULT NOW()')
    columns.append('"etl_loaded_at" TIMESTAMPTZ DEFAULT NOW()')

    ddl = (
        f'CREATE TABLE IF NOT EXISTS "{config.target_schema}"."{config.target_table}" (\n'
        + ",\n".join(f"  {c}" for c in columns)
        + "\n)"
    )
    with engine.begin() as conn:
        conn.execute(sa.text(ddl))
    logger.info(
        "Created/verified local test table %s.%s",
        config.target_schema, config.target_table,
    )


def _srid_from_crs(crs: str) -> int:
    """Extract the EPSG SRID from a CRS string like 'EPSG:25832'."""
    part = str(crs).split(":", 1)[-1].strip()
    if not part.isdigit():
        raise LoadError(f"Cannot derive SRID from target CRS {crs!r}.")
    return int(part)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def _attribute_columns(gdf: gpd.GeoDataFrame, config: Config) -> list[str]:
    """Attribute columns to load (everything except geometry and the id)."""
    excluded = {GEOMETRY_COLUMN, config.unique_id_field}
    return [c for c in gdf.columns if c not in excluded]


def _build_upsert_sql(config: Config, attribute_cols: list[str], srid: int) -> sa.text:
    """Build the parameterised upsert statement.

    ``ON CONFLICT (unique_id) DO UPDATE`` provides idempotency: a second run
    updates the existing row instead of inserting a copy. Fields listed in
    ``LOAD_UPDATE_EXCLUDE`` are not overwritten.
    """
    schema = f'"{config.target_schema}"'
    table = f'"{config.target_table}"'
    ident = f'"{config.unique_id_field}"'

    insert_cols = [ident, *[f'"{c}"' for c in attribute_cols], GEOMETRY_COLUMN, UPDATED_AT_COLUMN]
    param_placeholders = [f":{_param_name(c)}" for c in attribute_cols]

    sql = (
        f"INSERT INTO {schema}.{table} ({', '.join(insert_cols)})\n"
        f"VALUES (:{config.unique_id_field}, {', '.join(param_placeholders)}, "
        f"ST_GeomFromWKB(decode(:geom_hex, 'hex'), {srid}), NOW())\n"
        f"ON CONFLICT ({ident}) DO UPDATE SET\n"
    )

    sets = []
    for col in attribute_cols:
        if col in config.load_update_exclude:
            # Config-excluded fields (e.g. immutable keys) are not overwritten.
            continue
        sets.append(f'"{col}" = EXCLUDED."{col}"')
    sets.append(f'{GEOMETRY_COLUMN} = EXCLUDED.{GEOMETRY_COLUMN}')
    sets.append(f'{UPDATED_AT_COLUMN} = NOW()')

    sql += ",\n".join(sets)
    sql += "\nRETURNING (xmax = 0) AS inserted;"
    return sa.text(sql)


def _param_name(column: str) -> str:
    """Normalise a column name into a safe SQLAlchemy bind parameter name."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in column)
    return f"p_{safe}"


def _row_params(
    row: pd.Series,
    attribute_cols: list[str],
    unique_id_field: str,
) -> dict[str, Any]:
    """Convert one row into bind parameters for the upsert statement."""
    params: dict[str, Any] = {}
    ident_value = row[unique_id_field]
    if isinstance(ident_value, bytes):
        ident_value = ident_value.decode("utf-8", errors="replace")
    params[unique_id_field] = str(ident_value)
    for col in attribute_cols:
        value = row[col]
        if pd.isna(value):
            params[_param_name(col)] = None
        elif isinstance(value, bytes):
            params[_param_name(col)] = value.decode("utf-8", errors="replace")
        else:
            params[_param_name(col)] = value

    geom = row[GEOMETRY_COLUMN]
    if geom is None:
        params["geom_hex"] = None
    else:
        params["geom_hex"] = geom.wkb_hex
    return params


# def upsert_batch(
#     engine: Engine,
#     gdf: gpd.GeoDataFrame,
#     config: Config,
#     batch: pd.DataFrame,
#     sql: sa.text,
#     attribute_cols: list[str],
# ) -> tuple[int, int]:
#     """Insert/update one batch; returns (inserted, updated)."""
#     params = [
#         _row_params(row, attribute_cols, config.unique_id_field)
#         for _, row in batch.iterrows()
#     ]
#     with engine.begin() as conn:
#         result = conn.execute(sql, params)
#         rows = result.fetchall()

#     inserted = 0
#     updated = 0
#     for (was_inserted,) in rows:
#         if was_inserted:
#             inserted += 1
#         else:
#             updated += 1
#     return inserted, updated

def upsert_batch(
    engine: Engine,
    gdf: gpd.GeoDataFrame,
    config: Config,
    batch: pd.DataFrame,
    sql: sa.text,
    attribute_cols: list[str],
) -> tuple[int, int]:
    """Insert/update one batch; returns (inserted, updated).

    Each record is executed individually inside a single database
    transaction so PostgreSQL RETURNING can reliably report whether
    each row was inserted or updated.
    """
    inserted = 0
    updated = 0

    with engine.begin() as conn:
        for _, row in batch.iterrows():
            params = _row_params(
                row,
                attribute_cols,
                config.unique_id_field,
            )

            result = conn.execute(sql, params)
            was_inserted = result.scalar_one()

            if was_inserted:
                inserted += 1
            else:
                updated += 1

    return inserted, updated

def load_data(
    engine: Engine,
    gdf: gpd.GeoDataFrame,
    config: Config,
) -> dict[str, int]:
    """Load the transformed GeoDataFrame into the target table.

    Commits one batch at a time. If a batch fails, it is rolled back, counted
    as a failure, and processing continues with the next batch so earlier work
    is never lost. A non-zero failure count raises at the end.
    """
    if len(gdf) == 0:
        logger.warning("No records to load.")
        return {"total": 0, "inserted": 0, "updated": 0, "failed": 0}

    attribute_cols = _attribute_columns(gdf, config)
    srid = _srid_from_crs(config.target_crs)
    sql = _build_upsert_sql(config, attribute_cols, srid)

    total_inserted = 0
    total_updated = 0
    failed = 0
    batch_number = 0

    for start in range(0, len(gdf), config.batch_size):
        batch_number += 1
        batch = gdf.iloc[start : start + config.batch_size]
        logger.info(
            "Loading batch %s (%s-%s of %s)",
            batch_number, start + 1, min(start + len(batch), len(gdf)), len(gdf),
        )
        try:
            inserted, updated = upsert_batch(engine, gdf, config, batch, sql, attribute_cols)
            total_inserted += inserted
            total_updated += updated
            logger.info(
                "Batch %s committed: inserted=%s updated=%s",
                batch_number, inserted, updated,
            )
        except LoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - per-batch error isolation
            failed += len(batch)
            logger.error(
                "Batch %s failed (%s records) and was rolled back: %s",
                batch_number, len(batch), exc,
            )

        if config.qa_fail_batch and batch_number >= config.qa_fail_batch:
            raise LoadError(
                f"QA_FAIL_BATCH hook triggered after {batch_number} committed "
                f"batch(es). This simulates an interrupted run for restart "
                f"safety testing."
            )

    summary = {
        "total": len(gdf),
        "inserted": total_inserted,
        "updated": total_updated,
        "failed": failed,
    }
    logger.info(
        "Load complete: total=%s inserted=%s updated=%s failed=%s",
        total_inserted + total_updated, total_inserted, total_updated, failed,
    )
    if failed:
        raise LoadError(
            f"Loading finished with {failed} failed record(s). "
            f"Re-running the ETL will retry them via upsert."
        )
    return summary


# ---------------------------------------------------------------------------
# QA checks
# ---------------------------------------------------------------------------


def run_qa_check(engine: Engine, config: Config) -> dict[str, Any]:
    """Run integrity checks against the target table for QA evidence."""
    schema = f'"{config.target_schema}"'
    table = f'"{config.target_table}"'
    ident = f'"{config.unique_id_field}"'
    full = f"{schema}.{table}"

    sql = sa.text(
        f"""
        SELECT
            COUNT(*)                                              AS total_records,
            COUNT(DISTINCT {ident})                               AS distinct_ids,
            COUNT(*) - COUNT(DISTINCT {ident})                    AS duplicate_ids,
            COUNT(*) FILTER (WHERE "{GEOMETRY_COLUMN}" IS NULL)   AS null_geometries,
            COUNT(*) FILTER (WHERE "{GEOMETRY_COLUMN}" IS NOT NULL
                             AND ST_IsEmpty("{GEOMETRY_COLUMN}"))  AS empty_geometries,
            COUNT(*) FILTER (WHERE "{GEOMETRY_COLUMN}" IS NOT NULL
                             AND ST_IsEmpty("{GEOMETRY_COLUMN}") IS NOT TRUE
                             AND NOT ST_IsValid("{GEOMETRY_COLUMN}")) AS invalid_geometries,
            COUNT(DISTINCT ST_SRID("{GEOMETRY_COLUMN}"))          AS distinct_srids,
            MIN(ST_SRID("{GEOMETRY_COLUMN}"))                     AS min_srid
        FROM {full}
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()

    qa = {
        "total_records": int(row.total_records),
        "distinct_ids": int(row.distinct_ids),
        "duplicate_ids": int(row.duplicate_ids),
        "null_geometries": int(row.null_geometries),
        "empty_geometries": int(row.empty_geometries),
        "invalid_geometries": int(row.invalid_geometries),
        "distinct_srids": int(row.distinct_srids),
        "min_srid": int(row.min_srid) if row.min_srid is not None else None,
    }

    if config.state_name:
        with engine.connect() as conn:
            state_sql = sa.text(
                f"SELECT COUNT(*) FILTER (WHERE state = :state), COUNT(*) "
                f"FROM {full}"
            )
            result = conn.execute(state_sql, {"state": config.state_name}).fetchone()
        qa["state_matching_records"] = int(result[0])
        qa["state_total_records"] = int(result[1])
        qa["state_isolation"] = qa["state_matching_records"] == qa["total_records"]

    logger.info(
        "QA check: total=%s distinct_ids=%s duplicate_ids=%s "
        "null_geom=%s empty_geom=%s invalid_geom=%s srids=%s",
        qa["total_records"], qa["distinct_ids"], qa["duplicate_ids"],
        qa["null_geometries"], qa["empty_geometries"], qa["invalid_geometries"],
        qa.get("min_srid"),
    )
    return qa
