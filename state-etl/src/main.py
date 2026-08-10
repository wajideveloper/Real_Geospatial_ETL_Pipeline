"""ETL entry point: extract -> transform -> load -> QA, with logging.

Run from the project root:

    python src/main.py

Exit codes: 0 = success, 1 = a pipeline stage failed.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any

from .extract import load_source_data
from .load import (
    create_local_table,
    ensure_postgis,
    get_engine,
    load_data,
    run_qa_check,
)
from .transform import transform_data
from .utils import (
    ETLError,
    Config,
    get_config,
    log_config_summary,
    setup_logging,
)
from .load import get_engine, ensure_postgis, create_local_table, load_data, run_qa_check

logger = logging.getLogger("etl")


def run_etl(config: Config) -> dict[str, Any]:
    """Execute one complete ETL pass and return a result summary.

    This function is importable so the QA runner can execute the pipeline
    multiple times within a single process.
    """
    started = time.perf_counter()
    logger.info("=" * 70)
    logger.info("ETL start")
    logger.info("=" * 70)
    log_config_summary(config)

    # --- EXTRACT ---------------------------------------------------------
    logger.info("Stage 1/3: EXTRACT")
    source_gdf = load_source_data(config)
    logger.info("Extracted %s source feature(s).", len(source_gdf))

    # --- TRANSFORM -------------------------------------------------------
    logger.info("Stage 2/3: TRANSFORM")
    transformed_gdf, report = transform_data(source_gdf, config)
    logger.info("Transform report: %s", report)

    # --- LOAD ------------------------------------------------------------
    logger.info("Stage 3/3: LOAD")
    engine = get_engine(config)
    ensure_postgis(engine)
    create_local_table(engine, config, transformed_gdf)
    load_summary = load_data(engine, transformed_gdf, config)

    # --- QA --------------------------------------------------------------
    qa = run_qa_check(engine, config)
    engine.dispose()

    elapsed = time.perf_counter() - started
    logger.info("=" * 70)
    logger.info("ETL completion")
    logger.info("Elapsed time: %.2f seconds", elapsed)
    logger.info("=" * 70)

    return {
        "source_features": int(len(source_gdf)),
        "transformed_features": int(len(transformed_gdf)),
        "transform_report": report,
        "load": load_summary,
        "qa": qa,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> int:
    """CLI entry point."""
    try:
        config = get_config()
    except ETLError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    setup_logging(config.log_level)
    try:
        run_etl(config)
    except ETLError as exc:
        logger.error("ETL failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        logger.exception("Unexpected ETL failure: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
