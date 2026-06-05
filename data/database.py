"""
data/database.py
----------------
SQLite database layer for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This module handles ALL database interactions. No other module
touches SQLite directly — everything goes through these functions.

ON STARTUP:
   initialise_database() is called once by the Airflow DAG.
   It creates all 6 tables if they don't already exist.
   Safe to call every startup — uses IF NOT EXISTS.

WRITE FLOW (daily pipeline):
   1. Airflow fetches OHLCV data        → write_raw_prices()
   2. Stage 1 filter runs               → write_filtered_universe()
   3. Indicator engines complete        → write_indicator_results()
   4. Sentiment fetch completes         → write_sentiment_data()
   5. Scanner waterfall completes       → write_scan_results()
   6. ML models train/score             → write_model_metrics()

READ FLOW (Streamlit dashboard):
   Dashboard reads ONLY — never writes.
   - read_latest_scan_results()   → populates the long/short results tables
   - read_latest_model_metrics()  → populates the model health section
   - read_raw_prices()            → feeds the SPY/QQQ/DIA Plotly charts

UPSERT PATTERN:
   All write functions use INSERT OR IGNORE or INSERT OR REPLACE.
   This means re-running the pipeline on the same day is always safe —
   duplicate rows are silently skipped or overwritten, never duplicated.

CONNECTION MANAGEMENT:
   All connections go through the get_connection() context manager.
   This guarantees connections are always closed, even if an error occurs.
   WAL journal mode is enabled for better concurrent read/write performance
   (Streamlit reading while Airflow is writing doesn't cause locks).
"""

import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import yaml

from utils.logging import get_database_logger
from utils.error_handler import DatabaseError, handle_critical_error

logger = get_database_logger()


# =============================================================================
# CONFIG
# Read the database path from config.yaml.
# We resolve this once at module load time so every function can use DB_PATH.
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config  = _load_config()
DB_PATH = Path(config["database"]["path"])


# =============================================================================
# CONNECTION CONTEXT MANAGER
# Every database operation uses this instead of opening connections manually.
#
# HOW IT WORKS:
# 1. Ensure the database directory exists (creates /app/data/ if needed)
# 2. Open a SQLite connection with a 30-second timeout
#    (timeout prevents crashes if Airflow and Streamlit access simultaneously)
# 3. Enable WAL mode — allows concurrent reads while writing
# 4. Enable foreign keys — enforces data integrity
# 5. Yield the connection to the caller (the actual SQL runs here)
# 6. On success → commit the transaction
# 7. On error   → rollback, wrap in DatabaseError, re-raise
# 8. Always     → close the connection (finally block)
# =============================================================================

@contextmanager
def get_connection():
    """
    Context manager for safe SQLite connections.

    Usage:
        with get_connection() as conn:
            df = pd.read_sql("SELECT * FROM raw_prices", conn)
            # Connection auto-commits on exit, auto-closes always
    """
    # Ensure directory exists before trying to open the database file
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)

        # WAL = Write-Ahead Logging
        # Allows Streamlit to read while Airflow is writing — no blocking
        conn.execute("PRAGMA journal_mode=WAL")

        # Enforce referential integrity
        conn.execute("PRAGMA foreign_keys=ON")

        yield conn          # ← caller does their work here
        conn.commit()       # ← commit on clean exit

    except sqlite3.Error as e:
        if conn:
            conn.rollback()  # ← undo any partial writes on error
        raise DatabaseError(f"SQLite error: {e}") from e

    finally:
        if conn:
            conn.close()    # ← always close, even if exception occurred


# =============================================================================
# TABLE CREATION
# Called once on pipeline startup.
# Creates 6 tables + 4 indexes for query performance.
#
# TABLE PURPOSES:
# raw_prices          → stores every OHLCV candle for every ticker
# filtered_universe   → tickers that passed Stage 1 filter on each scan date
# indicator_results   → LinReg values, SMC structure, volume signal per ticker
# sentiment_data      → Put/Call Ratio + Short Interest per ticker
# scan_results        → final long/short candidates ranked by ML score
# model_metrics       → Precision + AUC-ROC history for both ML models
# =============================================================================

def initialise_database() -> None:
    """
    Create all pipeline tables and indexes if they don't exist.
    Safe to call on every startup — uses IF NOT EXISTS throughout.
    """
    logger.info(f"Initialising database at: {DB_PATH}")

    ddl_statements = [

        # ── Table 1: Raw OHLCV prices ──────────────────────────────────────
        # One row per ticker per trading day.
        # UNIQUE(ticker, date) prevents duplicate candles on re-runs.
        """
        CREATE TABLE IF NOT EXISTS raw_prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      INTEGER NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now')),
            UNIQUE(ticker, date)
        )
        """,

        # ── Table 2: Filtered universe ─────────────────────────────────────
        # Records which tickers passed Stage 1 filter on each scan date.
        # Lets us audit why a stock appeared or disappeared from results.
        """
        CREATE TABLE IF NOT EXISTS filtered_universe (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            avg_volume  REAL    NOT NULL,
            last_close  REAL    NOT NULL,
            scan_date   TEXT    NOT NULL,
            created_at  TEXT    DEFAULT (datetime('now')),
            UNIQUE(ticker, scan_date)
        )
        """,

        # ── Table 3: Indicator results ─────────────────────────────────────
        # Stores all engine outputs for each ticker on each date.
        # linreg_slope_up  : 1 = uptrend, 0 = downtrend
        # price_sd_position: e.g. -1.8 means price is 1.8 SDs below LinReg
        # smc_structure    : 'bullish', 'bearish', or 'broken'
        # choch_detected   : 1 = CHoCH confirmed today, 0 = no CHoCH
        # volume_signal    : 'accumulation', 'distribution', or 'neutral'
        """
        CREATE TABLE IF NOT EXISTS indicator_results (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker            TEXT NOT NULL,
            date              TEXT NOT NULL,
            linreg_value      REAL,
            linreg_slope      REAL,
            linreg_slope_up   INTEGER,
            sd1_upper         REAL,
            sd1_lower         REAL,
            sd2_upper         REAL,
            sd2_lower         REAL,
            sd3_upper         REAL,
            sd3_lower         REAL,
            price_sd_position REAL,
            smc_structure     TEXT,
            choch_detected    INTEGER,
            volume_signal     TEXT,
            created_at        TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date)
        )
        """,

        # ── Table 4: Sentiment data ────────────────────────────────────────
        # Put/Call Ratio and Short Interest per ticker per date.
        # These feed directly into the ML feature vectors.
        """
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker             TEXT NOT NULL,
            date               TEXT NOT NULL,
            put_call_ratio     REAL,
            short_interest_pct REAL,
            created_at         TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date)
        )
        """,

        # ── Table 5: Final scan results ────────────────────────────────────
        # What the Streamlit dashboard reads.
        # One row per candidate per direction per scan date.
        # ml_rank = 1 is the highest probability candidate of the day.
        """
        CREATE TABLE IF NOT EXISTS scan_results (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date          TEXT NOT NULL,
            ticker             TEXT NOT NULL,
            direction          TEXT NOT NULL,       -- 'long' or 'short'
            sector             TEXT,
            sd_position        REAL,
            volume_signal      TEXT,
            put_call_ratio     REAL,
            short_interest_pct REAL,
            ml_score           REAL,
            ml_rank            INTEGER,
            created_at         TEXT DEFAULT (datetime('now')),
            UNIQUE(scan_date, ticker, direction)
        )
        """,

        # ── Table 6: ML model metrics ──────────────────────────────────────
        # Tracks model performance over time.
        # The dashboard reads the most recent row per model_name.
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT NOT NULL,          -- 'volume_classifier' or 'signal_ranker'
            train_date      TEXT NOT NULL,
            precision_score REAL,
            auc_roc_score   REAL,
            n_samples       INTEGER,
            created_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(model_name, train_date)
        )
        """,

        # ── Indexes for query performance ──────────────────────────────────
        # Without indexes, querying 365 days × 2000 tickers = slow full scans.
        # With indexes, lookups by ticker+date are near instant.
        "CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_date       ON raw_prices(ticker, date)",
        "CREATE INDEX IF NOT EXISTS idx_indicator_results_ticker_date ON indicator_results(ticker, date)",
        "CREATE INDEX IF NOT EXISTS idx_scan_results_scan_date        ON scan_results(scan_date)",
        "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date         ON sentiment_data(ticker, date)",
    ]

    try:
        with get_connection() as conn:
            for stmt in ddl_statements:
                conn.execute(stmt)
        logger.info("Database initialised successfully — all tables and indexes ready")

    except DatabaseError as e:
        handle_critical_error(e, context="database.initialise_database", reraise=True)


# =============================================================================
# WRITE OPERATIONS
# All writers use upsert patterns (INSERT OR IGNORE / INSERT OR REPLACE).
# This makes every write operation idempotent — safe to run multiple times
# on the same data without creating duplicate rows.
# =============================================================================

def write_raw_prices(df: pd.DataFrame) -> int:
    """
    Write raw OHLCV data to the raw_prices table.

    FLOW:
    1. Validate input — return early if empty
    2. Write to a temporary staging table
    3. INSERT OR IGNORE from staging → raw_prices
       (skips any ticker+date combos already in the table)
    4. Drop the staging table
    5. Return count of rows actually inserted

    Args:
        df: DataFrame with columns [ticker, date, open, high, low, close, volume]

    Returns:
        Number of new rows inserted (0 if all were duplicates)
    """
    if df.empty:
        logger.warning("write_raw_prices: Empty DataFrame — nothing to write")
        return 0

    # Keep only the columns we need — drop any extras
    required = ["ticker", "date", "open", "high", "low", "close", "volume"]
    df = df[required].copy()

    try:
        with get_connection() as conn:
            # Write to staging first — faster than row-by-row inserts
            df.to_sql("raw_prices_staging", conn, if_exists="replace", index=False)

            # Move from staging → main table, skipping duplicates
            cursor = conn.execute("""
                INSERT OR IGNORE INTO raw_prices
                    (ticker, date, open, high, low, close, volume)
                SELECT ticker, date, open, high, low, close, volume
                FROM raw_prices_staging
            """)
            rows_inserted = cursor.rowcount

            # Clean up staging table
            conn.execute("DROP TABLE IF EXISTS raw_prices_staging")

        logger.info(f"write_raw_prices: {rows_inserted} new rows inserted")
        return rows_inserted

    except DatabaseError as e:
        logger.error(f"write_raw_prices failed: {e}")
        raise


def write_filtered_universe(df: pd.DataFrame, scan_date: str) -> int:
    """
    Write Stage 1 filtered tickers for today's scan date.

    Args:
        df       : DataFrame with columns [ticker, avg_volume, last_close]
        scan_date: Today's date string YYYY-MM-DD

    Returns:
        Number of rows inserted
    """
    if df.empty:
        logger.warning("write_filtered_universe: Empty DataFrame")
        return 0

    df = df.copy()
    df["scan_date"] = scan_date  # Tag every row with today's date

    try:
        with get_connection() as conn:
            df[["ticker", "avg_volume", "last_close", "scan_date"]].to_sql(
                "filtered_universe_staging", conn, if_exists="replace", index=False
            )
            cursor = conn.execute("""
                INSERT OR IGNORE INTO filtered_universe
                    (ticker, avg_volume, last_close, scan_date)
                SELECT ticker, avg_volume, last_close, scan_date
                FROM filtered_universe_staging
            """)
            rows_inserted = cursor.rowcount
            conn.execute("DROP TABLE IF EXISTS filtered_universe_staging")

        logger.info(
            f"write_filtered_universe: {rows_inserted} tickers written for {scan_date}"
        )
        return rows_inserted

    except DatabaseError as e:
        logger.error(f"write_filtered_universe failed: {e}")
        raise


def write_indicator_results(df: pd.DataFrame) -> int:
    """
    Write indicator engine outputs for all tickers.
    Uses INSERT OR REPLACE — re-running engines overwrites previous results
    for the same ticker+date (useful if you need to reprocess a day).

    Args:
        df: DataFrame containing all indicator columns

    Returns:
        Number of rows written
    """
    if df.empty:
        logger.warning("write_indicator_results: Empty DataFrame")
        return 0

    try:
        with get_connection() as conn:
            df.to_sql("indicator_staging", conn, if_exists="replace", index=False)
            cursor = conn.execute("""
                INSERT OR REPLACE INTO indicator_results (
                    ticker, date,
                    linreg_value, linreg_slope, linreg_slope_up,
                    sd1_upper, sd1_lower,
                    sd2_upper, sd2_lower,
                    sd3_upper, sd3_lower,
                    price_sd_position,
                    smc_structure, choch_detected, volume_signal
                )
                SELECT
                    ticker, date,
                    linreg_value, linreg_slope, linreg_slope_up,
                    sd1_upper, sd1_lower,
                    sd2_upper, sd2_lower,
                    sd3_upper, sd3_lower,
                    price_sd_position,
                    smc_structure, choch_detected, volume_signal
                FROM indicator_staging
            """)
            rows_written = cursor.rowcount
            conn.execute("DROP TABLE IF EXISTS indicator_staging")

        logger.info(f"write_indicator_results: {rows_written} rows written")
        return rows_written

    except DatabaseError as e:
        logger.error(f"write_indicator_results failed: {e}")
        raise


def write_sentiment_data(df: pd.DataFrame) -> int:
    """
    Write Put/Call Ratio and Short Interest data.

    Args:
        df: DataFrame with columns [ticker, date, put_call_ratio, short_interest_pct]

    Returns:
        Number of rows written
    """
    if df.empty:
        logger.warning("write_sentiment_data: Empty DataFrame")
        return 0

    try:
        with get_connection() as conn:
            df[["ticker", "date", "put_call_ratio", "short_interest_pct"]].to_sql(
                "sentiment_staging", conn, if_exists="replace", index=False
            )
            cursor = conn.execute("""
                INSERT OR REPLACE INTO sentiment_data
                    (ticker, date, put_call_ratio, short_interest_pct)
                SELECT ticker, date, put_call_ratio, short_interest_pct
                FROM sentiment_staging
            """)
            rows_written = cursor.rowcount
            conn.execute("DROP TABLE IF EXISTS sentiment_staging")

        logger.info(f"write_sentiment_data: {rows_written} rows written")
        return rows_written

    except DatabaseError as e:
        logger.error(f"write_sentiment_data failed: {e}")
        raise


def write_scan_results(df: pd.DataFrame, scan_date: str) -> int:
    """
    Write final scanner results — the rows that appear in the dashboard.

    Args:
        df       : DataFrame with all scan result columns
        scan_date: Today's date string YYYY-MM-DD

    Returns:
        Number of rows written
    """
    if df.empty:
        logger.warning(f"write_scan_results: No candidates found for {scan_date}")
        return 0

    df = df.copy()
    df["scan_date"] = scan_date

    try:
        with get_connection() as conn:
            df.to_sql("scan_results_staging", conn, if_exists="replace", index=False)
            cursor = conn.execute("""
                INSERT OR REPLACE INTO scan_results (
                    scan_date, ticker, direction, sector,
                    sd_position, volume_signal,
                    put_call_ratio, short_interest_pct,
                    ml_score, ml_rank
                )
                SELECT
                    scan_date, ticker, direction, sector,
                    sd_position, volume_signal,
                    put_call_ratio, short_interest_pct,
                    ml_score, ml_rank
                FROM scan_results_staging
            """)
            rows_written = cursor.rowcount
            conn.execute("DROP TABLE IF EXISTS scan_results_staging")

        logger.info(
            f"write_scan_results: {rows_written} candidates written for {scan_date}"
        )
        return rows_written

    except DatabaseError as e:
        logger.error(f"write_scan_results failed: {e}")
        raise


def write_model_metrics(
    model_name : str,
    train_date : str,
    precision  : float,
    auc_roc    : float,
    n_samples  : int,
) -> None:
    """
    Write ML model performance metrics after each training run.

    Args:
        model_name : 'volume_classifier' or 'signal_ranker'
        train_date : Date string YYYY-MM-DD
        precision  : Precision score from evaluation
        auc_roc    : AUC-ROC score from evaluation
        n_samples  : Number of training samples used
    """
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO model_metrics
                    (model_name, train_date, precision_score, auc_roc_score, n_samples)
                VALUES (?, ?, ?, ?, ?)
            """, (model_name, train_date, precision, auc_roc, n_samples))

        logger.info(
            f"write_model_metrics: {model_name} | "
            f"Precision: {precision:.4f} | "
            f"AUC-ROC: {auc_roc:.4f} | "
            f"Samples: {n_samples}"
        )

    except DatabaseError as e:
        logger.error(f"write_model_metrics failed: {e}")
        raise


# =============================================================================
# READ OPERATIONS
# Used by the Streamlit dashboard and ML feature engineering.
# All reads are non-destructive — no writes happen here.
# =============================================================================

def read_raw_prices(ticker: str, days: Optional[int] = None) -> pd.DataFrame:
    """
    Read OHLCV data for a single ticker.

    Args:
        ticker : Ticker symbol e.g. 'AAPL'
        days   : If provided, returns only the most recent N days.
                 If None, returns all available history.

    Returns:
        DataFrame sorted by date ascending (oldest first)
    """
    try:
        with get_connection() as conn:
            if days:
                # Fetch last N days then re-sort ascending for indicator engines
                query = """
                    SELECT * FROM raw_prices
                    WHERE ticker = ?
                    ORDER BY date DESC
                    LIMIT ?
                """
                df = pd.read_sql(query, conn, params=(ticker, days))
            else:
                query = """
                    SELECT * FROM raw_prices
                    WHERE ticker = ?
                    ORDER BY date ASC
                """
                df = pd.read_sql(query, conn, params=(ticker,))

        # Always return in ascending date order for indicator engines
        return df.sort_values("date").reset_index(drop=True)

    except DatabaseError as e:
        logger.error(f"read_raw_prices failed for {ticker}: {e}")
        return pd.DataFrame()


def read_filtered_universe(scan_date: str) -> pd.DataFrame:
    """
    Read the list of tickers that passed Stage 1 filter on a given date.
    The indicator engines iterate over this list — not the full universe.

    Args:
        scan_date: Date string YYYY-MM-DD

    Returns:
        DataFrame of filtered tickers
    """
    try:
        with get_connection() as conn:
            df = pd.read_sql(
                "SELECT * FROM filtered_universe WHERE scan_date = ?",
                conn, params=(scan_date,)
            )
        logger.info(f"read_filtered_universe: {len(df)} tickers for {scan_date}")
        return df

    except DatabaseError as e:
        logger.error(f"read_filtered_universe failed: {e}")
        return pd.DataFrame()


def read_latest_scan_results(direction: Optional[str] = None) -> pd.DataFrame:
    """
    Read scan results for the most recent scan date.
    This is what the Streamlit dashboard displays.

    FLOW:
    1. Find the MAX scan_date in the table (most recent run)
    2. Return all results for that date
    3. Filter by direction if specified
    4. Sort by ml_score descending (highest probability first)

    Args:
        direction: 'long', 'short', or None (returns both)

    Returns:
        DataFrame sorted by ml_score descending
    """
    try:
        with get_connection() as conn:
            if direction:
                query = """
                    SELECT * FROM scan_results
                    WHERE scan_date = (SELECT MAX(scan_date) FROM scan_results)
                    AND direction = ?
                    ORDER BY ml_score DESC
                """
                df = pd.read_sql(query, conn, params=(direction,))
            else:
                query = """
                    SELECT * FROM scan_results
                    WHERE scan_date = (SELECT MAX(scan_date) FROM scan_results)
                    ORDER BY ml_score DESC
                """
                df = pd.read_sql(query, conn)

        logger.info(f"read_latest_scan_results: {len(df)} results returned")
        return df

    except DatabaseError as e:
        logger.error(f"read_latest_scan_results failed: {e}")
        return pd.DataFrame()


def read_latest_model_metrics() -> pd.DataFrame:
    """
    Read the most recent performance metrics for all ML models.
    Uses a self-join to find the latest train_date per model_name.

    Returns:
        DataFrame with one row per model (volume_classifier + signal_ranker)
    """
    try:
        with get_connection() as conn:
            query = """
                SELECT m1.*
                FROM model_metrics m1
                INNER JOIN (
                    SELECT model_name, MAX(train_date) AS latest_date
                    FROM model_metrics
                    GROUP BY model_name
                ) m2
                ON m1.model_name = m2.model_name
                AND m1.train_date = m2.latest_date
            """
            df = pd.read_sql(query, conn)
        return df

    except DatabaseError as e:
        logger.error(f"read_latest_model_metrics failed: {e}")
        return pd.DataFrame()


def read_indicator_results(ticker: str, date: str) -> pd.DataFrame:
    """
    Read indicator results for a specific ticker on a specific date.
    Used by ML feature engineering to assemble feature vectors.

    Args:
        ticker: Ticker symbol
        date  : Date string YYYY-MM-DD

    Returns:
        Single-row DataFrame with all indicator columns
    """
    try:
        with get_connection() as conn:
            df = pd.read_sql(
                "SELECT * FROM indicator_results WHERE ticker = ? AND date = ?",
                conn, params=(ticker, date)
            )
        return df

    except DatabaseError as e:
        logger.error(f"read_indicator_results failed for {ticker} on {date}: {e}")
        return pd.DataFrame()


def get_last_fetch_date(ticker: str) -> Optional[str]:
    """
    Find the most recent date we have OHLCV data for a ticker.

    Used by the fetcher to decide:
    - Returns None     → ticker not in DB → do full 365-day fetch
    - Returns a date   → ticker exists   → do incremental fetch only

    Args:
        ticker: Ticker symbol

    Returns:
        Most recent date string YYYY-MM-DD, or None if ticker not found
    """
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(date) FROM raw_prices WHERE ticker = ?",
                (ticker,)
            )
            result = cursor.fetchone()[0]  # Returns None if ticker has no rows
        return result

    except DatabaseError as e:
        logger.error(f"get_last_fetch_date failed for {ticker}: {e}")
        return None