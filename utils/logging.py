"""
utils/logging.py
----------------
Production-grade logging for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
1. On first call to get_logger(module_name), we:
   a. Load log settings from config.yaml
   b. Create the /app/logs directory if it doesn't exist
   c. Attach TWO handlers to the logger:
      - Console handler  → prints to terminal (useful during dev + Docker logs)
      - File handler     → writes to a rotating log file per module
   d. Register the logger in a local dict so the same instance
      is reused if get_logger is called again (singleton pattern)

2. Rotating file logs:
   - Each log file has a max size (e.g. 10 MB)
   - When it hits that limit, it rotates: scanner.log → scanner.log.1
   - We keep up to 5 backups then delete the oldest
   - This prevents logs from filling up disk over time

3. Pre-defined logger functions (get_fetcher_logger, etc.):
   - Convenience wrappers so each module imports its own named logger
   - All write to module-specific log files for easy debugging
   - e.g. all engine errors go to engines.log, fetch errors to fetcher.log

WHY SINGLETON PATTERN:
   Without it, calling get_logger("fetcher") twice would add duplicate
   handlers — every log line would print twice. The registry prevents this.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import yaml


# =============================================================================
# CONFIG LOADER
# Reads log settings from config.yaml.
# Falls back to safe defaults if config is missing
# (e.g. during unit testing outside Docker).
# =============================================================================

def _load_config() -> dict:
    """Load logging configuration from config.yaml."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Safe fallback defaults — won't crash if config is missing
        return {
            "logging": {
                "level": "INFO",
                "log_dir": "/app/logs",
                "max_file_size_mb": 10,
                "backup_count": 5,
            }
        }


# =============================================================================
# LOG FORMAT
# Every log line looks like:
# 2026-06-01 16:35:12 | INFO     | fetcher                   | Batch 1/12 started
# This gives you: when, severity, which module, what happened
# =============================================================================

LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# LOGGER REGISTRY
# Stores already-created loggers so we never attach duplicate handlers.
# Key   = module_name string
# Value = logging.Logger instance
# =============================================================================

_logger_registry: dict = {}


# =============================================================================
# MAIN FACTORY FUNCTION
# This is the function every module calls to get its logger.
# =============================================================================

def get_logger(module_name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get or create a production-grade logger for a module.

    FLOW:
    1. Check registry → if logger already exists, return it immediately
    2. Load config settings (level, dir, size limit, backup count)
    3. Ensure log directory exists (create if needed)
    4. Create the logger and set its level
    5. Attach console handler (stdout)
    6. Attach rotating file handler
    7. Register in _logger_registry
    8. Return logger

    Args:
        module_name : Name of the calling module e.g. 'fetcher', 'smc_engine'
        log_file    : Optional custom log filename. Defaults to <module_name>.log

    Returns:
        Configured logging.Logger instance

    Usage:
        from utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Scan started")
        logger.error("Fetch failed", exc_info=True)  # exc_info=True adds traceback
    """

    # ── Step 1: Return existing logger if already registered ────────────────
    if module_name in _logger_registry:
        return _logger_registry[module_name]

    # ── Step 2: Load config ──────────────────────────────────────────────────
    config      = _load_config()
    log_cfg     = config.get("logging", {})

    log_level_str = log_cfg.get("level", "INFO").upper()
    log_level     = getattr(logging, log_level_str, logging.INFO)
    log_dir       = Path(log_cfg.get("log_dir", "/app/logs"))
    max_bytes     = log_cfg.get("max_file_size_mb", 10) * 1024 * 1024  # convert MB → bytes
    backup_count  = log_cfg.get("backup_count", 5)

    # ── Step 3: Ensure log directory exists ──────────────────────────────────
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 4: Create logger ────────────────────────────────────────────────
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Prevent log messages bubbling up to the root logger
    # (avoids duplicate output if root logger also has handlers)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Step 5: Console handler ──────────────────────────────────────────────
    # Prints logs to stdout — visible in terminal and in Docker logs
    # Guard ensures we don't add a second console handler if called again
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ── Step 6: Rotating file handler ───────────────────────────────────────
    # Writes logs to /app/logs/<module_name>.log
    # Rotates when file exceeds max_bytes, keeps backup_count old files
    filename    = log_file or f"{module_name.replace('.', '_')}.log"
    log_filepath = log_dir / filename

    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler(
            filename     = log_filepath,
            maxBytes     = max_bytes,
            backupCount  = backup_count,
            encoding     = "utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # ── Step 7: Register logger ──────────────────────────────────────────────
    _logger_registry[module_name] = logger

    logger.info(f"Logger initialised → {log_filepath}")
    return logger


# =============================================================================
# PRE-DEFINED MODULE LOGGERS
# Each module imports its own named logger directly.
# All engine logs share engines.log for easy consolidated engine debugging.
# All other modules have their own dedicated log file.
# =============================================================================

def get_fetcher_logger()   -> logging.Logger:
    return get_logger("fetcher",       "fetcher.log")

def get_database_logger()  -> logging.Logger:
    return get_logger("database",      "database.log")

def get_linreg_logger()    -> logging.Logger:
    return get_logger("linreg_engine", "engines.log")

def get_smc_logger()       -> logging.Logger:
    return get_logger("smc_engine",    "engines.log")

def get_volume_logger()    -> logging.Logger:
    return get_logger("volume_engine", "engines.log")

def get_scanner_logger()   -> logging.Logger:
    return get_logger("scanner",       "scanner.log")

def get_sentiment_logger() -> logging.Logger:
    return get_logger("sentiment",     "sentiment.log")

def get_ml_logger()        -> logging.Logger:
    return get_logger("ml",            "ml.log")

def get_dashboard_logger() -> logging.Logger:
    return get_logger("dashboard",     "dashboard.log")

def get_airflow_logger()   -> logging.Logger:
    return get_logger("airflow_dag",   "airflow.log")