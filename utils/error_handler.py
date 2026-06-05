"""
utils/error_handler.py
----------------------
Centralised error handling for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This module provides THREE layers of protection:

LAYER 1 — @retry decorator:
   Wraps network-dependent functions (yfinance calls, sentiment fetches).
   If a function fails, it waits and tries again up to N times.
   Uses exponential backoff — each retry waits longer than the last.
   e.g. attempt 1 fails → wait 5s → attempt 2 fails → wait 10s → attempt 3
   If all attempts fail, it either re-raises the exception or returns None.

LAYER 2 — @graceful decorator:
   Wraps per-ticker functions so one bad ticker never crashes the whole pipeline.
   If the function fails, it logs the error and returns a safe default value.
   The pipeline then just skips that ticker and moves on.
   e.g. SMC engine fails on TSLA → logs warning → returns None → TSLA skipped

LAYER 3 — handle_critical_error():
   For genuine fatal failures (database down, config missing, etc.)
   Logs the full stack trace at CRITICAL level then re-raises.
   Used in Airflow DAG tasks where we WANT the task to fail visibly.

CUSTOM EXCEPTIONS:
   Each pipeline layer has its own exception class.
   This makes it easy to catch specific layer failures without catching everything.
   e.g. except DataFetchError catches only fetch problems, not database problems.
"""

import time
import functools
import traceback
from typing import Callable, Any, Optional, Tuple, Type
from utils.logging import get_logger

logger = get_logger("error_handler", "error_handler.log")


# =============================================================================
# CUSTOM EXCEPTION CLASSES
# One exception class per pipeline layer.
# Inherit from StockScannerBaseError so you can also catch ALL scanner errors
# with a single except StockScannerBaseError clause if needed.
# =============================================================================

class StockScannerBaseError(Exception):
    """Base class — catch this to handle ANY scanner error."""
    pass

class DataFetchError(StockScannerBaseError):
    """Raised when yfinance data download fails."""
    pass

class DataValidationError(StockScannerBaseError):
    """Raised when downloaded data fails quality checks."""
    pass

class DatabaseError(StockScannerBaseError):
    """Raised when SQLite read or write operations fail."""
    pass

class EngineError(StockScannerBaseError):
    """Raised when LinReg, SMC or Volume engine calculations fail."""
    pass

class ScannerError(StockScannerBaseError):
    """Raised when the top-down waterfall scanner logic fails."""
    pass

class SentimentError(StockScannerBaseError):
    """Raised when Put/Call or Short Interest data fetch fails."""
    pass

class MLError(StockScannerBaseError):
    """Raised when ML model training or scoring fails."""
    pass

class ConfigError(StockScannerBaseError):
    """Raised when config.yaml is missing or has invalid values."""
    pass


# =============================================================================
# LAYER 1: @retry DECORATOR
# Use this on any function that makes a network call.
# Network calls fail intermittently — retrying silently fixes most issues.
# =============================================================================

def retry(
    attempts       : int                          = 3,
    delay_seconds  : float                        = 5.0,
    backoff_factor : float                        = 2.0,
    exceptions     : Tuple[Type[Exception], ...]  = (Exception,),
    raise_on_exhausted: bool                      = True,
):
    """
    Retry decorator with exponential backoff.

    HOW IT WORKS:
    1. Call the wrapped function
    2. If it raises one of the specified exceptions → log warning, wait, retry
    3. Each retry waits longer: delay * backoff_factor^attempt
       e.g. 5s → 10s → 20s for backoff_factor=2.0
    4. If all attempts fail:
       - raise_on_exhausted=True  → re-raise the last exception (DAG task fails)
       - raise_on_exhausted=False → return None silently (graceful skip)

    Args:
        attempts          : Maximum number of attempts (including first try)
        delay_seconds     : Initial wait time between retries in seconds
        backoff_factor    : Multiply delay by this after each failed attempt
        exceptions        : Which exception types to catch and retry on
        raise_on_exhausted: Whether to raise after all attempts are exhausted

    Usage:
        @retry(attempts=3, delay_seconds=5, exceptions=(DataFetchError,))
        def fetch_ticker(ticker: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay   = delay_seconds
            last_exception  = None

            for attempt in range(1, attempts + 1):
                try:
                    # ── Happy path: function succeeds ────────────────────────
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    logger.warning(
                        f"[RETRY] {func.__name__} | "
                        f"Attempt {attempt}/{attempts} failed | "
                        f"{type(e).__name__}: {e}"
                    )

                    if attempt < attempts:
                        # ── Wait before next attempt ─────────────────────────
                        logger.info(
                            f"[RETRY] Waiting {current_delay:.1f}s before retry..."
                        )
                        time.sleep(current_delay)
                        # ── Increase delay for next retry (exponential backoff)
                        current_delay *= backoff_factor
                    else:
                        # ── Final attempt also failed ────────────────────────
                        logger.error(
                            f"[RETRY] {func.__name__} | "
                            f"All {attempts} attempts exhausted | "
                            f"Final error: {type(e).__name__}: {e}"
                        )

            # ── All retries exhausted ────────────────────────────────────────
            if raise_on_exhausted and last_exception:
                raise last_exception
            return None  # Return None if raise_on_exhausted=False

        return wrapper
    return decorator


# =============================================================================
# LAYER 2: @graceful DECORATOR
# Use this on per-ticker functions so one failure doesn't kill the pipeline.
# The function returns a safe default instead of raising an exception.
# =============================================================================

def graceful(
    default_return : Any                          = None,
    exceptions     : Tuple[Type[Exception], ...]  = (Exception,),
    log_level      : str                          = "warning",
):
    """
    Graceful degradation decorator.

    HOW IT WORKS:
    1. Call the wrapped function normally
    2. If it raises → catch it, log the error, return default_return
    3. The pipeline continues with the next ticker — nothing crashes

    This is the key to scanning 2000 tickers without one bad ticker
    stopping the whole run. AAPL might fail, but NVDA still gets processed.

    Args:
        default_return : Value returned when function fails (usually None)
        exceptions     : Exception types to catch
        log_level      : 'warning' for expected failures, 'error' for unexpected

    Usage:
        @graceful(default_return=None, exceptions=(EngineError,))
        def compute_linreg(df: pd.DataFrame, ticker: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # ── Happy path ───────────────────────────────────────────────
                return func(*args, **kwargs)

            except exceptions as e:
                # ── Log the failure but don't crash ──────────────────────────
                log_fn = logger.error if log_level == "error" else logger.warning
                log_fn(
                    f"[GRACEFUL] {func.__name__} failed | "
                    f"{type(e).__name__}: {e} | "
                    f"Returning default: {default_return}"
                )
                # Log full traceback at DEBUG level for deep investigation
                logger.debug(
                    f"[GRACEFUL] Full traceback:\n{traceback.format_exc()}"
                )
                return default_return

        return wrapper
    return decorator


# =============================================================================
# LAYER 3: handle_critical_error()
# For genuine fatal failures — logs everything then re-raises.
# Used in Airflow tasks where we WANT a clear visible failure.
# =============================================================================

def handle_critical_error(
    error   : Exception,
    context : str,
    reraise : bool = True,
) -> None:
    """
    Handle a critical pipeline failure.

    HOW IT WORKS:
    1. Logs the error at CRITICAL level with full stack trace
    2. If reraise=True → re-raises the exception
       (Airflow marks the task as failed, sends alert)
    3. If reraise=False → just logs (use when you want to record but continue)

    Args:
        error   : The exception that was caught
        context : Human-readable description of what failed
        reraise : Whether to re-raise after logging

    Usage:
        try:
            run_full_pipeline()
        except Exception as e:
            handle_critical_error(e, context="DAG Task 1: fetch_data")
    """
    logger.critical(
        f"[CRITICAL] {context} | "
        f"{type(error).__name__}: {error}\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
    if reraise:
        raise error


# =============================================================================
# DATA VALIDATION HELPER
# Called in fetcher.py after every yfinance download.
# Returns True/False so the fetcher can decide whether to keep or discard data.
# =============================================================================

def validate_dataframe(
    df               : any,   # pandas DataFrame
    ticker           : str,
    required_columns : list,
) -> bool:
    """
    Validate a DataFrame returned from yfinance.

    CHECKS PERFORMED (in order):
    1. DataFrame is not None and not empty
    2. All required columns are present
    3. No required column is entirely null
    4. At least 200 rows of data exist
       (LinReg needs 200 candles to compute meaningfully)

    Args:
        df              : pandas DataFrame to validate
        ticker          : Ticker symbol (used only for logging)
        required_columns: List of column names that must be present

    Returns:
        True if all checks pass, False if any check fails
    """

    # ── Check 1: Not None or empty ───────────────────────────────────────────
    if df is None or df.empty:
        logger.warning(f"[VALIDATION] {ticker} | Empty or None DataFrame")
        return False

    # ── Check 2: All required columns present ────────────────────────────────
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"[VALIDATION] {ticker} | Missing column: '{col}'")
            return False

    # ── Check 3: No column is entirely null ──────────────────────────────────
        if df[col].isnull().all():
            logger.warning(f"[VALIDATION] {ticker} | Column '{col}' is all nulls")
            return False

    # ── Check 4: Sufficient rows for LinReg ──────────────────────────────────
    if len(df) < 200:
        logger.warning(
            f"[VALIDATION] {ticker} | Only {len(df)} rows — need 200+ for LinReg"
        )
        return False

    logger.debug(f"[VALIDATION] {ticker} | Passed all checks | Rows: {len(df)}")
    return True


# =============================================================================
# AIRFLOW FAILURE CALLBACK
# Passed to each Airflow task as on_failure_callback.
# Airflow calls this automatically when a task fails.
# Ensures all DAG failures are captured in our structured log files,
# not just Airflow's internal logs.
# =============================================================================

def airflow_failure_callback(context: dict) -> None:
    """
    Airflow on_failure_callback.

    Airflow injects a context dict containing task details.
    We extract what's useful and log it in our pipeline log format.

    Usage in DAG definition:
        fetch_task = PythonOperator(
            task_id='fetch_data',
            python_callable=run_data_pipeline,
            on_failure_callback=airflow_failure_callback,
        )
    """
    task_id        = context.get("task_instance").task_id
    dag_id         = context.get("dag").dag_id
    execution_date = context.get("execution_date")
    exception      = context.get("exception")

    logger.error(
        f"[AIRFLOW FAILURE] "
        f"DAG: {dag_id} | "
        f"Task: {task_id} | "
        f"Execution date: {execution_date} | "
        f"Error: {exception}"
    )