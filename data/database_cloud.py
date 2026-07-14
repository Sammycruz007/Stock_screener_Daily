
"""
data/database_cloud.py
-----------------------
PostgreSQL version of database.py for cloud deployment.

Used by:
- run_pipeline_cloud.py  (GitHub Actions)
- dashboard/app_cloud.py (Streamlit Cloud)

DIFFERENCE FROM database.py:
- Uses psycopg2 (PostgreSQL) instead of sqlite3
- Reads connection string from SUPABASE_DB_URL environment variable
- Only stores RESULT tables (no raw_prices — too large for Supabase free tier)
- raw_prices is computed in-memory during each pipeline run and discarded

TABLES STORED IN SUPABASE:
- indicator_results   (engine outputs per ticker per day)
- scan_results        (ranked candidates)
- filtered_universe   (tickers that passed Stage 1)
- ticker_metadata     (sector classifications)
- model_metrics       (ML model performance)
"""

import os
import psycopg2
import psycopg2.extras
import pandas as pd
from contextlib import contextmanager
from typing import Optional
from utils.logging import get_database_logger
from utils.error_handler import DatabaseError

logger = get_database_logger()


# =============================================================================
# CONNECTION
# =============================================================================

def _get_db_url() -> str:
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise DatabaseError(
            "SUPABASE_DB_URL environment variable not set. "
            "Add it to GitHub Secrets or .streamlit/secrets.toml"
        )
    return url


@contextmanager
def get_connection():
    """Context manager for PostgreSQL connections."""
    conn = None
    try:
        conn = psycopg2.connect(_get_db_url(), sslmode="require", connect_timeout=10)
        yield conn
        conn.commit()
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        raise DatabaseError(f"PostgreSQL error: {e}") from e
    finally:
        if conn:
            conn.close()


# =============================================================================
# INITIALISE TABLES
# =============================================================================

def initialise_database() -> None:
    """
    Create all result tables in Supabase if they don't exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    tables = [
        """
        CREATE TABLE IF NOT EXISTS indicator_results (
            id                SERIAL PRIMARY KEY,
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
            has_valid_zone    INTEGER DEFAULT 0,
            volume_signal     TEXT,
            created_at        TIMESTAMP DEFAULT NOW(),
            UNIQUE(ticker, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_results (
            id             SERIAL PRIMARY KEY,
            ticker         TEXT NOT NULL,
            scan_date      TEXT NOT NULL,
            direction      TEXT NOT NULL,
            sector         TEXT,
            sd_position    REAL,
            volume_signal  TEXT,
            has_valid_zone INTEGER DEFAULT 0,
            ml_score       REAL,
            ml_rank        INTEGER,
            created_at     TIMESTAMP DEFAULT NOW(),
            UNIQUE(ticker, scan_date, direction)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS filtered_universe (
            id          SERIAL PRIMARY KEY,
            ticker      TEXT NOT NULL,
            scan_date   TEXT NOT NULL,
            avg_volume  REAL,
            last_close  REAL,
            protected   BOOLEAN DEFAULT FALSE,
            created_at  TIMESTAMP DEFAULT NOW(),
            UNIQUE(ticker, scan_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ticker_metadata (
            id           SERIAL PRIMARY KEY,
            ticker       TEXT NOT NULL UNIQUE,
            sector_name  TEXT,
            sector_etf   TEXT,
            fetch_date   TEXT,
            created_at   TIMESTAMP DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id              SERIAL PRIMARY KEY,
            model_name      TEXT NOT NULL,
            train_date      TEXT NOT NULL,
            precision_score REAL,
            recall_score    REAL,
            pr_auc_score    REAL,
            auc_roc_score   REAL,
            n_samples       INTEGER,
            created_at      TIMESTAMP DEFAULT NOW(),
            UNIQUE(model_name, train_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fetch_tracker (
            ticker          TEXT PRIMARY KEY,
            last_fetch_date DATE NOT NULL,
            updated_at      TIMESTAMP DEFAULT NOW()
        )
        """,
    ]

    with get_connection() as conn:
        cursor = conn.cursor()
        for ddl in tables:
            cursor.execute(ddl)

    logger.info("Supabase database initialised — all tables ready")


# =============================================================================
# WRITE FUNCTIONS
# =============================================================================

def write_indicator_results(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    records = df[[
        "ticker", "date", "linreg_value", "linreg_slope", "linreg_slope_up",
        "sd1_upper", "sd1_lower", "sd2_upper", "sd2_lower", "sd3_upper", "sd3_lower",
        "price_sd_position", "smc_structure", "has_valid_zone", "volume_signal"
    ]].to_dict("records")

    sql = """
        INSERT INTO indicator_results (
            ticker, date, linreg_value, linreg_slope, linreg_slope_up,
            sd1_upper, sd1_lower, sd2_upper, sd2_lower, sd3_upper, sd3_lower,
            price_sd_position, smc_structure, has_valid_zone, volume_signal
        ) VALUES (
            %(ticker)s, %(date)s, %(linreg_value)s, %(linreg_slope)s, %(linreg_slope_up)s,
            %(sd1_upper)s, %(sd1_lower)s, %(sd2_upper)s, %(sd2_lower)s,
            %(sd3_upper)s, %(sd3_lower)s, %(price_sd_position)s,
            %(smc_structure)s, %(has_valid_zone)s, %(volume_signal)s
        )
        ON CONFLICT (ticker, date) DO UPDATE SET
            linreg_value      = EXCLUDED.linreg_value,
            linreg_slope      = EXCLUDED.linreg_slope,
            linreg_slope_up   = EXCLUDED.linreg_slope_up,
            sd1_upper         = EXCLUDED.sd1_upper,
            sd1_lower         = EXCLUDED.sd1_lower,
            sd2_upper         = EXCLUDED.sd2_upper,
            sd2_lower         = EXCLUDED.sd2_lower,
            sd3_upper         = EXCLUDED.sd3_upper,
            sd3_lower         = EXCLUDED.sd3_lower,
            price_sd_position = EXCLUDED.price_sd_position,
            smc_structure     = EXCLUDED.smc_structure,
            has_valid_zone    = EXCLUDED.has_valid_zone,
            volume_signal     = EXCLUDED.volume_signal
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)
        rows = cursor.rowcount

    logger.info(f"write_indicator_results: {len(records)} rows upserted")
    return len(records)


def write_scan_results(df: pd.DataFrame, scan_date: str) -> int:
    if df.empty:
        return 0

    records = []
    for _, row in df.iterrows():
        records.append({
            "ticker"        : row["ticker"],
            "scan_date"     : scan_date,
            "direction"     : row["direction"],
            "sector"        : row.get("sector"),
            "sd_position"   : float(row.get("sd_position", 0)),
            "volume_signal" : row.get("volume_signal"),
            "has_valid_zone": int(row.get("has_valid_zone", 0)),
            "ml_score"      : float(row.get("ml_score", 0)),
            "ml_rank"       : int(row.get("ml_rank", 0)),
        })

    sql = """
        INSERT INTO scan_results (
            ticker, scan_date, direction, sector, sd_position,
            volume_signal, has_valid_zone, ml_score, ml_rank
        ) VALUES (
            %(ticker)s, %(scan_date)s, %(direction)s, %(sector)s, %(sd_position)s,
            %(volume_signal)s, %(has_valid_zone)s, %(ml_score)s, %(ml_rank)s
        )
        ON CONFLICT (ticker, scan_date, direction) DO UPDATE SET
            ml_score      = EXCLUDED.ml_score,
            ml_rank       = EXCLUDED.ml_rank,
            volume_signal = EXCLUDED.volume_signal,
            has_valid_zone = EXCLUDED.has_valid_zone
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_scan_results: {len(records)} candidates written for {scan_date}")
    return len(records)


def write_filtered_universe(df: pd.DataFrame, scan_date: str) -> int:
    if df.empty:
        return 0

    records = [
        {
            "ticker"    : row["ticker"],
            "scan_date" : scan_date,
            "avg_volume": float(row.get("avg_volume", 0)),
            "last_close": float(row.get("last_close", 0)),
            "protected" : bool(row.get("protected", False)),
        }
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO filtered_universe (ticker, scan_date, avg_volume, last_close, protected)
        VALUES (%(ticker)s, %(scan_date)s, %(avg_volume)s, %(last_close)s, %(protected)s)
        ON CONFLICT (ticker, scan_date) DO NOTHING
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_filtered_universe: {len(records)} tickers written for {scan_date}")
    return len(records)


def write_sector_metadata(df: pd.DataFrame, fetch_date: str) -> int:
    if df.empty:
        return 0

    records = [
        {
            "ticker"     : row["ticker"],
            "sector_name": row.get("sector_name"),
            "sector_etf" : row.get("sector_etf"),
            "fetch_date" : fetch_date,
        }
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO ticker_metadata (ticker, sector_name, sector_etf, fetch_date)
        VALUES (%(ticker)s, %(sector_name)s, %(sector_etf)s, %(fetch_date)s)
        ON CONFLICT (ticker) DO UPDATE SET
            sector_name = EXCLUDED.sector_name,
            sector_etf  = EXCLUDED.sector_etf,
            fetch_date  = EXCLUDED.fetch_date
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_sector_metadata: {len(records)} tickers written")
    return len(records)


def write_model_metrics(
    model_name : str,
    train_date : str,
    precision  : float,
    auc_roc    : float,
    n_samples  : int,
    recall     : float = 0.0,
    pr_auc     : float = 0.0,
) -> None:
    sql = """
        INSERT INTO model_metrics
            (model_name, train_date, precision_score, recall_score, pr_auc_score, auc_roc_score, n_samples)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (model_name, train_date) DO UPDATE SET
            precision_score = EXCLUDED.precision_score,
            recall_score    = EXCLUDED.recall_score,
            pr_auc_score    = EXCLUDED.pr_auc_score,
            auc_roc_score   = EXCLUDED.auc_roc_score,
            n_samples       = EXCLUDED.n_samples
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, (model_name, train_date, precision, recall, pr_auc, auc_roc, n_samples))

    logger.info(f"write_model_metrics: {model_name} written")


# =============================================================================
# READ FUNCTIONS (used by Streamlit Cloud dashboard)
# =============================================================================

def read_latest_indicator_results() -> pd.DataFrame:
    sql = """
        SELECT * FROM indicator_results
        WHERE date = (SELECT MAX(date) FROM indicator_results)
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def read_latest_scan_results(direction: Optional[str] = None) -> pd.DataFrame:
    if direction:
        sql = """
            SELECT * FROM scan_results
            WHERE scan_date = (SELECT MAX(scan_date) FROM scan_results)
              AND direction = %s
            ORDER BY ml_rank ASC
        """
        with get_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(sql, (direction,))
            rows = cursor.fetchall()
            return pd.DataFrame(rows)
    else:
        sql = """
            SELECT * FROM scan_results
            WHERE scan_date = (SELECT MAX(scan_date) FROM scan_results)
            ORDER BY ml_rank ASC
        """
        with get_connection() as conn:
            return pd.read_sql(sql, conn)


def read_latest_model_metrics() -> pd.DataFrame:
    sql = """
        SELECT DISTINCT ON (model_name) *
        FROM model_metrics
        ORDER BY model_name, train_date DESC
    """
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def read_sector_metadata() -> pd.DataFrame:
    sql = "SELECT * FROM ticker_metadata"
    with get_connection() as conn:
        return pd.read_sql(sql, conn)


def read_filtered_universe(scan_date: str) -> pd.DataFrame:
    sql = """
        SELECT * FROM filtered_universe
        WHERE scan_date = %s
    """
    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute(sql, (scan_date,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows)


def get_last_fetch_dates_bulk() -> dict:
    """
    Read every tracked ticker's last-fetched date in ONE query,
    instead of one query per ticker. Used by fetcher.py's smart_fetch()
    to decide full vs. incremental per ticker for the whole universe
    at once.

    Unlike the 15m project (which had no meaningful "last date" to
    track since intraday history only spans ~60 days), daily bars
    make genuine incremental fetching practical in the cloud — this
    table is what makes that possible.

    Returns:
        Dict mapping ticker -> last_fetch_date (as string 'YYYY-MM-DD')
    """
    sql = "SELECT ticker, last_fetch_date FROM fetch_tracker"

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

    return {ticker: str(date) for ticker, date in rows}


def write_last_fetch_dates(ticker_dates: dict) -> int:
    """
    Update the fetch tracker after a successful fetch.

    Args:
        ticker_dates: Dict mapping ticker -> the MAX date actually
                      fetched for that ticker (not just "today" —
                      a ticker might have gaps, be delisted, or fail
                      partway through a run)

    Returns:
        Number of tickers updated
    """
    if not ticker_dates:
        return 0

    records = [
        {"ticker": t, "last_fetch_date": d}
        for t, d in ticker_dates.items()
    ]

    sql = """
        INSERT INTO fetch_tracker (ticker, last_fetch_date, updated_at)
        VALUES (%(ticker)s, %(last_fetch_date)s, NOW())
        ON CONFLICT (ticker) DO UPDATE SET
            last_fetch_date = EXCLUDED.last_fetch_date,
            updated_at      = NOW()
    """

    with get_connection() as conn:
        cursor = conn.cursor()
        psycopg2.extras.execute_batch(cursor, sql, records, page_size=500)

    logger.info(f"write_last_fetch_dates: {len(records)} tickers updated")
    return len(records)


def prune_old_prices(max_days: int = 65) -> int:
    """No-op for cloud — raw_prices not stored in Supabase."""
    return 0
