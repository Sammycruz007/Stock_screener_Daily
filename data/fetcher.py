"""
data/fetcher.py
---------------
Data fetching layer for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
UNIVERSE SOURCING (updated):
   Instead of scraping Wikipedia for S&P 500 + NASDAQ 100 (~520 tickers),
   we now pull the COMPLETE list of NYSE + NASDAQ listed stocks from
   NASDAQ's official FTP directory — two files updated daily by NASDAQ:

   File 1: nasdaqlisted.txt  → all NASDAQ-listed stocks
   File 2: otherlisted.txt   → all NYSE, NYSE American, and other exchange stocks

   Combined: ~6,000 raw tickers
   After Stage 1 filter: ~1,500 quality actionable stocks

   This gives us genuine full-market coverage instead of just the
   top 520 stocks in two indices.

SMART FETCH:
   First run  → full 365-day history for every ticker
   Daily runs → incremental 5-day fetch for tickers already in DB

PARALLEL BATCHING:
   Downloads happen in parallel batches of 50 tickers × 10 workers
   to keep total fetch time within the 4:30 PM pipeline window.

STAGE 1 FILTER:
   Applied after fetching to remove:
   - Price < $10
   - Average daily volume < 500,000
   Indices and sector ETFs always bypass this filter.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time as _time
from io import StringIO
from datetime import datetime, time , timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import yaml

from data.database import (
    write_raw_prices,
    get_last_fetch_date,
    write_filtered_universe,
)
from utils.logging import get_fetcher_logger
from utils.error_handler import (
    retry,
    graceful,
    validate_dataframe,
    DataFetchError,
    DataValidationError,
)

from utils.yf_session import YF_SESSION

from curl_cffi import requests as curl_requests

# Shared session with browser impersonation — fixes YFTzMissingError

_YF_SESSION = curl_requests.Session(impersonate="chrome")
logger = get_fetcher_logger()


# =============================================================================
# CONFIG
# =============================================================================


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
FETCHER_CFG  = config["fetcher"]
FILTER_CFG   = config["filters"]
UNIVERSE_CFG = config["universe"]

BATCH_SIZE       = FETCHER_CFG["batch_size"]
MAX_WORKERS      = FETCHER_CFG["max_workers"]
RETRY_ATTEMPTS   = FETCHER_CFG["retry_attempts"]
RETRY_DELAY      = FETCHER_CFG["retry_delay_seconds"]
HISTORICAL_DAYS  = FETCHER_CFG["historical_days"]
INCREMENTAL_DAYS = FETCHER_CFG["incremental_days"]

MIN_PRICE        = FILTER_CFG["min_price"]
MIN_AVG_VOLUME   = FILTER_CFG["min_avg_volume"]


# =============================================================================
# NASDAQ FTP URLs
# NASDAQ maintains two files covering ALL US-listed stocks.
# These are updated daily and are free to access.
#
# nasdaqlisted.txt → stocks listed on NASDAQ exchange
# otherlisted.txt  → stocks listed on NYSE, NYSE American, ARCA, BATS etc.
#
# Together they cover the entire US equity market (~6,000 tickers).
# =============================================================================

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"


# =============================================================================
# TICKER CLEANING
# Raw tickers from NASDAQ FTP need cleaning before yfinance can use them.
# =============================================================================

def _clean_ticker(ticker) -> Optional[str]:
    """
    Clean and validate a raw ticker symbol from NASDAQ FTP files.

    CLEANING RULES (applied in this specific order):
    1. Guard against NaN — pandas reads empty cells as float NaN
    2. Ensure input is a string
    3. Strip whitespace
    4. Skip empty strings
    5. Replace '/' with '-' BEFORE length check
       (BRK/B → BRK-B must happen before we measure length)
    6. Skip tickers with $ (test symbols, special securities)
    7. Skip tickers longer than 5 characters (warrants, units, rights)

    Args:
        ticker: Raw ticker value from NASDAQ FTP (may be str or float NaN)

    Returns:
        Cleaned ticker string or None if ticker should be excluded
    """
    # ── Guard 1: Handle NaN from pandas empty cells ───────────────────────────
    if ticker is None:
        return None

    # ── Guard 2: Must be a string ─────────────────────────────────────────────
    if not isinstance(ticker, str):
        return None

    # ── Guard 3: Skip NaN string representation ───────────────────────────────
    if ticker.lower() == "nan":
        return None

    # ── Step 1: Strip whitespace ──────────────────────────────────────────────
    ticker = ticker.strip()

    # ── Step 2: Skip empty after strip ───────────────────────────────────────
    if not ticker:
        return None

    # ── Step 3: Replace slash BEFORE length check ─────────────────────────────
    # Must happen first — BRK/B is 5 chars after replacement (BRK-B)
    # If we check length before this, BRK/B (5 chars) would still pass
    # but the slash would remain — yfinance needs BRK-B not BRK/B
    ticker = ticker.replace("/", "-")

    # ── Step 4: Skip special securities ──────────────────────────────────────
    if "$" in ticker:
        return None

    # ── Step 5: Skip warrants, units, rights ─────────────────────────────────
    # These typically have suffixes making them longer than 5 characters
    # e.g. AAPLW (warrant), AAPLU (unit), AAPL+ (right)
    if len(ticker) > 5:
        return None

    return ticker


@retry(attempts=3, delay_seconds=5, exceptions=(DataFetchError, Exception))
def _fetch_nasdaq_listed() -> list[str]:
    """
    Fetch all NASDAQ-listed stock tickers from NASDAQ FTP.

    FILE FORMAT (pipe-delimited):
    Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares

    We filter out:
    - Test issues (Y in column 4)
    - ETFs (Y in column 7) — we add our own sector ETFs separately
    - Non-normal financial status

    Returns:
        List of clean NASDAQ ticker symbols
    """
    logger.info("Fetching NASDAQ listed tickers from NASDAQ FTP...")

    try:
        response = requests.get(NASDAQ_LISTED, timeout=30)
        response.raise_for_status()

        # Parse pipe-delimited file
        df = pd.read_csv(
            StringIO(response.text),
            sep="|",
            dtype=str,
        )

        # Drop the last row — NASDAQ files end with a file creation timestamp row
        df = df[:-1]

        # Filter out test issues
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] == "N"]

        # Filter out ETFs — we manage our own sector ETF list separately
        if "ETF" in df.columns:
            df = df[df["ETF"] == "N"]

        # Filter out non-normal financial status
        if "Financial Status" in df.columns:
            df = df[df["Financial Status"] == "N"]

        # Extract and clean tickers
        tickers = []
        # Cast to string first — pandas reads empty cells as float NaN
        for raw in df["Symbol"].astype(str).tolist():
            # Skip pandas NaN string representation
            if raw.lower() == "nan":
                continue
            cleaned = _clean_ticker(raw)
            if cleaned:
                tickers.append(cleaned)

        logger.info(f"NASDAQ listed: {len(tickers)} tickers fetched")
        return tickers

    except Exception as e:
        raise DataFetchError(f"Failed to fetch NASDAQ listed tickers: {e}") from e


@retry(attempts=3, delay_seconds=5, exceptions=(DataFetchError, Exception))
def _fetch_other_listed() -> list[str]:
    """
    Fetch all NYSE + other exchange tickers from NASDAQ FTP.

    FILE FORMAT (pipe-delimited):
    ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol

    Covers NYSE, NYSE American (AMEX), NYSE Arca, BATS, and other exchanges.

    We filter out:
    - Test issues
    - ETFs

    Returns:
        List of clean NYSE/other exchange ticker symbols
    """
    logger.info("Fetching NYSE/other listed tickers from NASDAQ FTP...")

    try:
        response = requests.get(OTHER_LISTED, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text),
            sep="|",
            dtype=str,
        )

        # Drop timestamp last row
        df = df[:-1]

        # Filter out test issues
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] == "N"]

        # Filter out ETFs
        if "ETF" in df.columns:
            df = df[df["ETF"] == "N"]

        # The ticker column is called "ACT Symbol" in otherlisted.txt
        symbol_col = "ACT Symbol" if "ACT Symbol" in df.columns else "Symbol"

        tickers = []
        # Cast to string first — pandas reads empty cells as float NaN
        for raw in df[symbol_col].astype(str).tolist():
            # Skip pandas NaN string representation
            if raw.lower() == "nan":
                continue
            cleaned = _clean_ticker(raw)
            if cleaned:
                tickers.append(cleaned)

        logger.info(f"NYSE/other listed: {len(tickers)} tickers fetched")
        return tickers

    except Exception as e:
        raise DataFetchError(f"Failed to fetch NYSE/other listed tickers: {e}") from e


def get_full_universe() -> list[str]:
    """
    Build the complete US stock universe from NASDAQ FTP files.

    FLOW:
    1. Fetch NASDAQ listed tickers (~3,500 after filtering)
    2. Fetch NYSE + other exchange tickers (~2,500 after filtering)
    3. Add our fixed indices (SPY, QQQ, DIA)
    4. Add our fixed sector ETFs (XLK, XLF, etc.)
    5. Deduplicate — some tickers appear on multiple exchanges
    6. Sort alphabetically for consistent ordering
    7. Return final universe

    Returns:
        Sorted list of unique tickers covering the full US market
    """
    logger.info("Building full US stock universe from NASDAQ FTP...")

    nasdaq_tickers = _fetch_nasdaq_listed()
    other_tickers  = _fetch_other_listed()
    indices        = UNIVERSE_CFG["indices"]
    sectors        = UNIVERSE_CFG["sectors"]

    # Combine all sources and deduplicate
    all_tickers = list(set(
        nasdaq_tickers +
        other_tickers  +
        indices        +
        sectors
    ))
    all_tickers.sort()

    logger.info(
        f"Full universe built | "
        f"NASDAQ: {len(nasdaq_tickers)} | "
        f"NYSE/Other: {len(other_tickers)} | "
        f"Indices: {len(indices)} | "
        f"Sectors: {len(sectors)} | "
        f"Total unique: {len(all_tickers)}"
    )

    return all_tickers


# =============================================================================
# SECTOR METADATA FETCHER
# Fetches sector information for each ticker dynamically from yfinance.
# Replaces the old hardcoded SECTOR_MAP in screener.py.
# =============================================================================

# Maps yfinance sector name strings → our sector ETF symbols
SECTOR_NAME_TO_ETF = {
    "Technology"              : "XLK",
    "Financial Services"      : "XLF",
    "Energy"                  : "XLE",
    "Healthcare"              : "XLV",
    "Industrials"             : "XLI",
    "Consumer Cyclical"       : "XLY",
    "Consumer Defensive"      : "XLP",
    "Utilities"               : "XLU",
    "Basic Materials"         : "XLB",
    "Real Estate"             : "XLRE",
    "Communication Services"  : "XLC",
}


@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def _fetch_ticker_sector(ticker: str) -> Optional[dict]:
    """
    Fetch sector information for a single ticker from yfinance.

    FLOW:
    1. Call yf.Ticker(ticker).info
    2. Extract 'sector' field
    3. Map sector name → sector ETF using SECTOR_NAME_TO_ETF
    4. Return dict with ticker, sector_name, sector_etf

    If sector not found:
    - sector_name = 'Unclassified'
    - sector_etf  = None
    - Stock still appears in results but tagged as Unclassified
    - Dashboard shows ⚠️ warning for these stocks

    Args:
        ticker: Ticker symbol

    Returns:
        Dict with ticker, sector_name, sector_etf or None on failure
    """
    tk   = yf.Ticker(ticker)
    info = tk.info

    if not info:
        logger.debug(f"{ticker} | No info returned from yfinance")
        return {
            "ticker"     : ticker,
            "sector_name": "Unclassified",
            "sector_etf" : None,
        }

    sector_name = info.get("sector", None)

    if not sector_name:
        logger.debug(f"{ticker} | No sector field in yfinance info")
        return {
            "ticker"     : ticker,
            "sector_name": "Unclassified",
            "sector_etf" : None,
        }

    # Map sector name to ETF
    sector_etf = SECTOR_NAME_TO_ETF.get(sector_name, None)

    if not sector_etf:
        logger.debug(
            f"{ticker} | Sector '{sector_name}' not in mapping — "
            f"tagging as Unclassified"
        )

    return {
        "ticker"     : ticker,
        "sector_name": sector_name if sector_name else "Unclassified",
        "sector_etf" : sector_etf,
    }


def fetch_sector_metadata(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch sector metadata for all tickers in parallel.

    FLOW:
    1. Exclude indices and sector ETFs — they don't have sector info
    2. Fetch sector for each stock ticker in parallel
    3. Collect results — every ticker gets a row even if Unclassified
    4. Return DataFrame ready for ticker_metadata table

    Args:
        tickers: Full universe ticker list

    Returns:
        DataFrame with columns [ticker, sector_name, sector_etf]
    """
    excluded = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])
    stock_tickers = [t for t in tickers if t not in excluded]

    logger.info(
        f"Fetching sector metadata | "
        f"{len(stock_tickers)} stock tickers | "
        f"Excluded: {len(excluded)} ETFs/indices"
    )

    results      = []
    unclassified = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_sector, ticker): ticker
            for ticker in stock_tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    if result["sector_etf"] is None:
                        unclassified += 1
                else:
                    # Even on total failure add an unclassified row
                    results.append({
                        "ticker"     : ticker,
                        "sector_name": "Unclassified",
                        "sector_etf" : None,
                    })
                    unclassified += 1
            except Exception as e:
                logger.warning(f"{ticker} | Sector fetch failed: {e}")
                results.append({
                    "ticker"     : ticker,
                    "sector_name": "Unclassified",
                    "sector_etf" : None,
                })
                unclassified += 1

    logger.info(
        f"Sector metadata complete | "
        f"Total: {len(results)} | "
        f"Unclassified: {unclassified} | "
        f"Classified: {len(results) - unclassified}"
    )

    return pd.DataFrame(results)


# =============================================================================
# SINGLE TICKER OHLCV FETCH
# =============================================================================

@retry(
    attempts   = RETRY_ATTEMPTS,
    delay_seconds = RETRY_DELAY,
    exceptions = (DataFetchError, Exception),
)
def fetch_single_ticker(
    ticker     : str,
    start_date : str,
    end_date   : str,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker via yfinance.

    FLOW:
    1. Download OHLCV from yfinance
    2. Flatten MultiIndex columns if present
    3. Standardise column names to lowercase
    4. Add ticker and date columns
    5. Validate data quality
    6. Drop nulls and bad rows
    7. Return clean DataFrame

    Args:
        ticker    : Ticker symbol
        start_date: Start date string YYYY-MM-DD
        end_date  : End date string YYYY-MM-DD

    Returns:
        Clean OHLCV DataFrame or None if fetch/validation fails
    """
    try:
        raw = yf.download(
            ticker,
            start      = start_date,
            end        = end_date,
            interval   = "1d",
            auto_adjust= True,
            progress   = False,
            threads    = False,
            session    = _YF_SESSION,
        )

        if raw.empty:
            logger.warning(f"{ticker} | yfinance returned empty DataFrame")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardise column names
        raw.columns = [c.lower() for c in raw.columns]

        # Keep only OHLCV columns
        required = ["open", "high", "low", "close", "volume"]
        missing  = [c for c in required if c not in raw.columns]
        if missing:
            logger.warning(f"{ticker} | Missing columns: {missing}")
            return None

        raw         = raw[required].copy()
        raw["ticker"] = ticker
        raw["date"]   = raw["date"] = raw.index.strftime("%Y-%m-%d")
        raw           = raw.reset_index(drop=True)

        # Validate
        if not validate_dataframe(raw, ticker, required):
            return None

        # Drop rows with null OHLCV or zero/negative prices
        raw = raw.dropna(subset=required)
        raw = raw[(raw["close"] > 0) & (raw["volume"] >= 0)]

        logger.debug(
            f"{ticker} | {len(raw)} rows fetched | "
            f"{start_date} to {end_date}"
        )
        return raw

    except Exception as e:
        raise DataFetchError(f"{ticker} fetch failed: {e}") from e


# =============================================================================
# BATCH PARALLEL FETCH
# =============================================================================

def fetch_batch(
    tickers    : list[str],
    start_date : str,
    end_date   : str,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a batch of tickers in parallel.

    FLOW:
    1. Submit all tickers to ThreadPoolExecutor simultaneously
    2. Collect results as they complete
    3. Log failed tickers
    4. Combine successful results into single DataFrame

    Args:
        tickers   : List of ticker symbols
        start_date: Start date string YYYY-MM-DD
        end_date  : End date string YYYY-MM-DD

    Returns:
        Combined DataFrame for all successful tickers in batch
    """
    results = []
    failed  = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_single_ticker, ticker, start_date, end_date): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results.append(df)
                else:
                    failed.append(ticker)
            except Exception as e:
                logger.warning(f"{ticker} | Batch fetch failed: {e}")
                failed.append(ticker)

    if failed:
        logger.warning(f"Batch: {len(failed)} tickers failed: {failed[:10]}...")

    if not results:
        logger.warning("Batch returned no data")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# =============================================================================
# FULL UNIVERSE FETCH — orchestrates batching
# =============================================================================

def fetch_universe(
    tickers    : list[str],
    start_date : str,
    end_date   : str,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for the entire universe in batches.

    FLOW:
    1. Split full ticker list into batches of BATCH_SIZE
    2. Process each batch sequentially
       (parallelism happens WITHIN each batch via ThreadPoolExecutor)
    3. Combine all batch results
    4. Log summary statistics

    Args:
        tickers   : Full list of tickers
        start_date: Start date YYYY-MM-DD
        end_date  : End date YYYY-MM-DD

    Returns:
        Combined DataFrame for all tickers
    """
  

    total   = len(tickers)
    batches = [
        tickers[i : i + BATCH_SIZE]
        for i in range(0, total, BATCH_SIZE)
    ]
    all_data = []

    logger.info(
        f"Fetching {total} tickers | "
        f"{len(batches)} batches of {BATCH_SIZE} | "
        f"Period: {start_date} to {end_date}"
    )

    for i, batch in enumerate(batches, 1):
        logger.info(f"Batch {i}/{len(batches)} | {len(batch)} tickers")
        batch_df = fetch_batch(batch, start_date, end_date)

        if not batch_df.empty:
            all_data.append(batch_df)

        # Pause between batches to avoid Yahoo Finance rate limiting
        # Without this, Yahoo blocks requests after ~20-30 batches
        if i < len(batches):
            _time.sleep(3)

    if not all_data:
        logger.error("fetch_universe: No data returned for any ticker")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"fetch_universe complete | "
        f"Total rows: {len(combined)} | "
        f"Tickers with data: {combined['ticker'].nunique()}"
    )
    return combined


# =============================================================================
# SMART FETCH — full vs incremental per ticker
# =============================================================================

def smart_fetch(tickers: list[str]) -> pd.DataFrame:
    """
    Intelligently decide full vs incremental fetch per ticker.

    DECISION LOGIC per ticker:
    - Not in database → full HISTORICAL_DAYS fetch (365 days)
    - Already in database → incremental INCREMENTAL_DAYS fetch (5 days)

    This means:
    - First ever run: downloads 365 days for all ~6,000 tickers
      (heavy one-time cost — expect 20-40 minutes)
    - All subsequent daily runs: only fetches 5 days per ticker
      (fast — expect 5-10 minutes)

    Args:
        tickers: Full universe ticker list

    Returns:
        Combined DataFrame of all new data fetched
    """
    today    = datetime.today()
    end_date = today.strftime("%Y-%m-%d")

    full_tickers        = []
    incremental_tickers = []

    # Check database for each ticker's last fetch date
    for ticker in tickers:
        last_date = get_last_fetch_date(ticker)
        if last_date is None:
            full_tickers.append(ticker)
        else:
            incremental_tickers.append(ticker)

    logger.info(
        f"smart_fetch | "
        f"Full fetch needed: {len(full_tickers)} tickers | "
        f"Incremental: {len(incremental_tickers)} tickers"
    )

    all_data = []

    # ── Full historical fetch ─────────────────────────────────────────────────
    if full_tickers:
        start_full = (
            today - timedelta(days=HISTORICAL_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Full fetch: {start_full} to {end_date}")
        df_full = fetch_universe(full_tickers, start_full, end_date)

        if not df_full.empty:
            all_data.append(df_full)

    # ── Incremental fetch ─────────────────────────────────────────────────────
    if incremental_tickers:
        start_incr = (
            today - timedelta(days=INCREMENTAL_DAYS)
        ).strftime("%Y-%m-%d")

        logger.info(f"Incremental fetch: {start_incr} to {end_date}")
        df_incr = fetch_universe(incremental_tickers, start_incr, end_date)

        if not df_incr.empty:
            all_data.append(df_incr)

    if not all_data:
        logger.warning("smart_fetch: No new data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"smart_fetch complete | Total rows: {len(combined)}")
    return combined


# =============================================================================
# STAGE 1 FILTER
# =============================================================================

def apply_stage1_filter(
    df        : pd.DataFrame,
    scan_date : str,
) -> pd.DataFrame:
    """
    Apply Stage 1 universe filter:
    - Last close price > $10
    - 20-day average volume > 500,000

    Indices and sector ETFs always bypass this filter —
    we always want their indicator data regardless of price/volume.

    FLOW:
    1. For each ticker group in the DataFrame
    2. Compute last close and 20-day average volume
    3. Check against thresholds
    4. Protected tickers (indices/sectors) always pass
    5. Return DataFrame of passing tickers

    Args:
        df       : Full OHLCV DataFrame for all tickers
        scan_date: Today's date string YYYY-MM-DD

    Returns:
        DataFrame with columns [ticker, avg_volume, last_close, protected]
        Only tickers that passed the filter
    """
    protected = set(
        UNIVERSE_CFG["indices"] +
        UNIVERSE_CFG["sectors"]
    )

    results = []

    for ticker, group in df.groupby("ticker"):
        group      = group.sort_values("date")
        last_close = group["close"].iloc[-1]
        avg_volume = group["volume"].tail(20).mean() # 20 daily candles = 1 month avg

        # Indices and sector ETFs always pass
        if ticker in protected:
            results.append({
                "ticker"    : ticker,
                "avg_volume": avg_volume,
                "last_close": last_close,
                "protected" : True,
            })
            continue

        # Apply Stage 1 filter for regular stocks
        if last_close >= MIN_PRICE and avg_volume >= MIN_AVG_VOLUME:
            results.append({
                "ticker"    : ticker,
                "avg_volume": avg_volume,
                "last_close": last_close,
                "protected" : False,
            })

    filtered_df = pd.DataFrame(results)
    total       = df["ticker"].nunique()
    passed      = len(filtered_df)

    logger.info(
        f"Stage 1 filter | "
        f"Input: {total} tickers | "
        f"Passed: {passed} | "
        f"Filtered out: {total - passed}"
    )

    return filtered_df


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# Called by Airflow DAG Tasks 1 and 2
# =============================================================================

def run_data_pipeline() -> dict:
    """
    Main entry point for the data pipeline.
    Called by Airflow DAG at 4:30 PM EST daily.

    FULL FLOW:
    1. Fetch full universe from NASDAQ FTP (~6,000 tickers)
    2. Smart fetch OHLCV data (full or incremental per ticker)
    3. Write raw prices to SQLite
    4. Apply Stage 1 filter (~1,500 tickers pass)
    5. Write filtered universe to SQLite
    6. Fetch sector metadata for all stock tickers
    7. Write sector metadata to SQLite

    Note on Step 6:
    Sector metadata is only re-fetched if the ticker_metadata
    table is empty or it has been more than 7 days since last fetch.
    Sector classifications rarely change so daily re-fetching is wasteful.

    Returns:
        Summary dict with counts for Airflow logging and monitoring
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE STARTED")
    logger.info("=" * 60)

    today   = datetime.today().strftime("%Y-%m-%d")
    summary = {}

    try:
        # ── Step 1: Get full universe ─────────────────────────────────────────
        tickers                  = get_full_universe()
        summary["universe_size"] = len(tickers)

        # ── Step 2: Smart fetch OHLCV ─────────────────────────────────────────
        df                    = smart_fetch(tickers)
        summary["rows_fetched"] = len(df)

        if df.empty:
            logger.error("Data pipeline: No data fetched. Aborting.")
            return summary

        # ── Step 3: Write raw prices ──────────────────────────────────────────
        rows_written            = write_raw_prices(df)
        summary["rows_written"] = rows_written

        # ── Step 4: Stage 1 filter ────────────────────────────────────────────
        filtered_df                        = apply_stage1_filter(df, today)
        summary["tickers_passed_filter"]   = len(filtered_df)

        # ── Step 5: Write filtered universe ───────────────────────────────────
        write_filtered_universe(filtered_df, today)

        # ── Step 6: Fetch and write sector metadata ───────────────────────────
        # Only fetch for stock tickers that passed Stage 1
        # (no point fetching sectors for filtered-out stocks)
        passed_tickers = filtered_df["ticker"].tolist()
        sector_df      = fetch_sector_metadata(passed_tickers)
        summary["sector_metadata_fetched"] = len(sector_df)

        # write_sector_metadata is added to database.py — see below
        from data.database import write_sector_metadata
        write_sector_metadata(sector_df, today)

        logger.info("=" * 60)
        logger.info(f"DATA PIPELINE COMPLETE | Summary: {summary}")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.critical(f"Data pipeline failed: {e}", exc_info=True)
        raise