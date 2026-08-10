"""Shared configuration, logging and helper utilities for the state ETL.

This module owns:

* Loading configuration from environment variables / a local ``.env`` file.
* A frozen :class:`Config` dataclass used by every pipeline stage.
* Logging setup (console + rotating file handler in ``logs/etl.log``).
* The exception hierarchy used across the pipeline.

No credentials or secrets are ever logged here.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE: Path = PROJECT_ROOT / ".env"
DEFAULT_LOG_DIR: Path = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE: Path = DEFAULT_LOG_DIR / "etl.log"

DEFAULT_TARGET_CRS: str = "EPSG:25832"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ETLError(Exception):
    """Base class for all errors raised by the ETL pipeline."""


class ConfigError(ETLError):
    """Raised when required configuration is missing or invalid."""


class ExtractionError(ETLError):
    """Raised when the extraction stage fails."""


class TransformationError(ETLError):
    """Raised when the transformation stage fails."""


class LoadError(ETLError):
    """Raised when the load stage fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration, populated from environment variables.

    Every value is sourced from ``.env`` / process environment so the same
    codebase works for local testing and for the client's production IRIS
    infrastructure without modification.
    """

    # WFS source
    wfs_url: str = ""
    wfs_layer: str = ""
    wfs_version: str = "2.0.0"
    wfs_output_format: str = "json"
    wfs_enable_paging: bool = True
    wfs_page_size: int = 5000
    wfs_max_features: int = 0  # 0 == no explicit limit
    wfs_timeout: int = 60

    # PostgreSQL / PostGIS
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""

    # Target table
    target_schema: str = "iris_core"
    target_table: str = ""
    unique_id_field: str = "source_id"
    load_update_exclude: tuple[str, ...] = field(default_factory=tuple)

    # CRS handling
    source_crs: str = ""
    target_crs: str = DEFAULT_TARGET_CRS

    # Dataset context
    state_name: str = ""

    # Execution control
    batch_size: int = 1000
    log_level: str = "INFO"

    # Local / test mode
    local_test_mode: bool = False
    auto_create_schema: bool = False
    auto_create_table: bool = False
    use_mock_wfs: bool = False
    mock_data_path: str = ""
    mock_crs: str = "EPSG:4326"

    # Testing-only hook: raise a LoadError after this many committed batches.
    # 0 disables the hook. Used by tests/run_qa.py to prove restart safety.
    qa_fail_batch: int = 0

    # ------------------------------------------------------------------
    def config_summary(self) -> str:
        """Return a human-readable summary that NEVER contains the password.

        Also masks any other value that could be considered sensitive.
        """
        return (
            f"wfs_url={self.wfs_url or '<unset>'} | "
            f"wfs_layer={self.wfs_layer or '<unset>'} | "
            f"wfs_version={self.wfs_version} | "
            f"use_mock_wfs={self.use_mock_wfs} | "
            f"mock_data_path={self.mock_data_path or '<unset>'} | "
            f"db_host={self.db_host} | db_port={self.db_port} | "
            f"db_name={self.db_name or '<unset>'} | "
            f"db_user={self.db_user or '<unset>'} | "
            f"db_password={'***' if self.db_password else '<unset>'} | "
            f"target_schema={self.target_schema} | "
            f"target_table={self.target_table or '<unset>'} | "
            f"unique_id_field={self.unique_id_field} | "
            f"state_name={self.state_name or '<unset>'} | "
            f"source_crs={self.source_crs or '<unknown>' } | "
            f"target_crs={self.target_crs} | "
            f"batch_size={self.batch_size} | "
            f"local_test_mode={self.local_test_mode}"
        )

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the configuration is unusable."""
        if not self.use_mock_wfs and not self.wfs_url:
            raise ConfigError(
                "WFS_URL is not set. Either provide a WFS endpoint or set "
                "USE_MOCK_WFS=true for local testing."
            )
        if not self.use_mock_wfs and not self.wfs_layer:
            raise ConfigError(
                "WFS_LAYER is not set. Provide the layer/typename to extract."
            )
        if not self.db_name:
            raise ConfigError("DB_NAME is not set.")
        if not self.db_user:
            raise ConfigError("DB_USER is not set.")
        if not self.target_table:
            raise ConfigError("TARGET_TABLE is not set.")
        if not self.unique_id_field:
            raise ConfigError("UNIQUE_ID_FIELD is not set.")
        if self.batch_size < 1:
            raise ConfigError(f"BATCH_SIZE must be >= 1, got {self.batch_size}.")
        if self.local_test_mode and not self.mock_data_path:
            raise ConfigError(
                "LOCAL_TEST_MODE is enabled but MOCK_DATA_PATH is not set."
            )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(
            f"Invalid integer value for {name}: {raw!r}"
        ) from None


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_env(env_file: Path = DEFAULT_ENV_FILE) -> None:
    """Load ``.env`` (if present) into the process environment.

    Existing process environment variables always win over ``.env`` values,
    which is the standard ``python-dotenv`` override behaviour.
    """
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file, override=False)


def get_config(env_file: Path = DEFAULT_ENV_FILE) -> Config:
    """Build and validate the :class:`Config` from the environment."""
    load_env(env_file)
    config = Config(
        wfs_url=os.getenv("WFS_URL", "").strip(),
        wfs_layer=os.getenv("WFS_LAYER", "").strip(),
        wfs_version=os.getenv("WFS_VERSION", "2.0.0").strip(),
        wfs_output_format=os.getenv("WFS_OUTPUT_FORMAT", "json").strip(),
        wfs_enable_paging=_env_bool("WFS_ENABLE_PAGING", True),
        wfs_page_size=_env_int("WFS_PAGE_SIZE", 5000),
        wfs_max_features=_env_int("WFS_MAX_FEATURES", 0),
        wfs_timeout=_env_int("WFS_TIMEOUT", 60),
        db_host=os.getenv("DB_HOST", "localhost").strip(),
        db_port=_env_int("DB_PORT", 5432),
        db_name=os.getenv("DB_NAME", "").strip(),
        db_user=os.getenv("DB_USER", "").strip(),
        db_password=os.getenv("DB_PASSWORD", ""),
        target_schema=os.getenv("TARGET_SCHEMA", "iris_core").strip(),
        target_table=os.getenv("TARGET_TABLE", "").strip(),
        unique_id_field=os.getenv("UNIQUE_ID_FIELD", "source_id").strip(),
        load_update_exclude=_env_list("LOAD_UPDATE_EXCLUDE"),
        source_crs=os.getenv("SOURCE_CRS", "").strip(),
        target_crs=os.getenv("TARGET_CRS", DEFAULT_TARGET_CRS).strip(),
        state_name=os.getenv("STATE_NAME", "").strip(),
        batch_size=_env_int("BATCH_SIZE", 1000),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        local_test_mode=_env_bool("LOCAL_TEST_MODE", False),
        auto_create_schema=_env_bool("AUTO_CREATE_SCHEMA", False),
        auto_create_table=_env_bool("AUTO_CREATE_TABLE", False),
        use_mock_wfs=_env_bool("USE_MOCK_WFS", False),
        mock_data_path=os.getenv("MOCK_DATA_PATH", "").strip(),
        mock_crs=os.getenv("MOCK_CRS", "EPSG:4326").strip(),
        qa_fail_batch=_env_int("QA_FAIL_BATCH", 0),
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    log_level: str = "INFO",
    log_file: Path = DEFAULT_LOG_FILE,
) -> logging.Logger:
    """Configure the root ETL logger with console + file output.

    Idempotent: calling this repeatedly (e.g. from the QA runner) does not
    add duplicate handlers.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("etl")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_config_summary(config: Config) -> None:
    """Log the sanitised configuration summary (never the password)."""
    logging.getLogger("etl").info("Configuration: %s", config.config_summary())


def safe_str(value: Any) -> str:
    """Best-effort string conversion for values that may be exotic types."""
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive only
        return "<unprintable>"
