"""
data/fetcher.py
---------------
Data fetching layer for the Stock Scanner pipeline.

Responsibilities:
- Download S&P 500 + NASDAQ 100 tickers (full universe)
- First run: fetch 365 days of OHLCV history
- Daily runs: incremental fetch (last 5 days only)
- Parallel batch downloading via concurrent.futures
- Retry logic via error_handler decorator
- Data validation before writing to SQLite
- Fetch indices (SPY, QQQ, DIA) and sector ETFs separately
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import requests
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

logger = get_fetcher_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = _load_config()
FETCHER_CFG = config["fetcher"]
FILTER_CFG  = config["filters"]
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
# UNIVERSE — fetch S&P 500 + NASDAQ 100 tickers
# =============================================================================

def get_sp500_tickers() -> list[str]:
    """
    Scrape S&P 500 tickers from Wikipedia.

    Returns:
        List of ticker strings
    """
    logger.info("Fetching S&P 500 tickers from Wikipedia")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].tolist()
        # Clean tickers — Wikipedia uses '.' but yfinance uses '-'
        tickers = [t.replace(".", "-") for t in tickers]
        logger.info(f"S&P 500: {len(tickers)} tickers fetched")
        return tickers
    except Exception as e:
        raise DataFetchError(f"Failed to fetch S&P 500 tickers: {e}") from e


def get_nasdaq100_tickers() -> list[str]:
    """
    Scrape NASDAQ 100 tickers from Wikipedia.

    Returns:
        List of ticker strings
    """
    logger.info("Fetching NASDAQ 100 tickers from Wikipedia")
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # NASDAQ 100 table is usually the 4th table on the page
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].tolist()
                logger.info(f"NASDAQ 100: {len(tickers)} tickers fetched")
                return tickers
        raise DataFetchError("Could not find ticker column in NASDAQ 100 Wikipedia page")
    except Exception as e:
        raise DataFetchError(f"Failed to fetch NASDAQ 100 tickers: {e}") from e


def get_full_universe() -> list[str]:
    """
    Combine S&P 500 + NASDAQ 100 + indices + sectors into
    a deduplicated universe.

    Returns:
        Sorted list of unique tickers
    """
    sp500   = get_sp500_tickers()
    nasdaq  = get_nasdaq100_tickers()
    indices = UNIVERSE_CFG["indices"]
    sectors = UNIVERSE_CFG["sectors"]

    universe = list(set(sp500 + nasdaq + indices + sectors))
    universe.sort()

    logger.info(f"Full universe: {len(universe)} unique tickers")
    return universe


# =============================================================================
# SINGLE TICKER FETCH
# =============================================================================

@retry(
    attempts=RETRY_ATTEMPTS,
    delay_seconds=RETRY_DELAY,
    exceptions=(DataFetchError, Exception),
)
def fetch_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker via yfinance.

    Args:
        ticker:     Ticker symbol
        start_date: Start date string YYYY-MM-DD
        end_date:   End date string YYYY-MM-DD

    Returns:
        Cleaned DataFrame or None if fetch/validation fails
    """
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,   # Adjust for splits and dividends
            progress=False,
            threads=False,      # We handle threading ourselves
        )

        if raw.empty:
            logger.warning(f"{ticker} | yfinance returned empty DataFrame")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardise column names to lowercase
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"adj close": "close"}) if "adj close" in raw.columns else raw

        # Keep only OHLCV
        raw = raw[["open", "high", "low", "close", "volume"]].copy()

        # Add ticker column
        raw["ticker"] = ticker
        raw["date"]   = raw.index.strftime("%Y-%m-%d")
        raw = raw.reset_index(drop=True)

        # Validate
        required = ["open", "high", "low", "close", "volume"]
        if not validate_dataframe(raw, ticker, required):
            return None

        # Drop rows with null OHLCV
        raw = raw.dropna(subset=required)

        # Drop rows with zero or negative prices
        raw = raw[(raw["close"] > 0) & (raw["volume"] >= 0)]

        logger.debug(f"{ticker} | {len(raw)} rows fetched from {start_date} to {end_date}")
        return raw

    except Exception as e:
        raise DataFetchError(f"{ticker} fetch failed: {e}") from e


# =============================================================================
# BATCH PARALLEL FETCH
# =============================================================================

def fetch_batch(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a batch of tickers in parallel.

    Args:
        tickers:    List of ticker symbols
        start_date: Start date string YYYY-MM-DD
        end_date:   End date string YYYY-MM-DD

    Returns:
        Combined DataFrame for all tickers in batch
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
        logger.warning(f"Batch: {len(failed)} tickers failed: {failed}")

    if not results:
        logger.warning("Batch returned no data")
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# =============================================================================
# FULL UNIVERSE FETCH — orchestrates batching
# =============================================================================

def fetch_universe(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for the entire universe in batches.

    Args:
        tickers:    Full list of tickers
        start_date: Start date string YYYY-MM-DD
        end_date:   End date string YYYY-MM-DD

    Returns:
        Combined DataFrame for all tickers
    """
    total    = len(tickers)
    batches  = [tickers[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    all_data = []

    logger.info(
        f"Fetching {total} tickers in {len(batches)} batches of {BATCH_SIZE} | "
        f"Period: {start_date} to {end_date}"
    )

    for i, batch in enumerate(batches, 1):
        logger.info(f"Processing batch {i}/{len(batches)} ({len(batch)} tickers)")
        batch_df = fetch_batch(batch, start_date, end_date)

        if not batch_df.empty:
            all_data.append(batch_df)

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
# SMART FETCH — decides full vs incremental per ticker
# =============================================================================

def smart_fetch(tickers: list[str]) -> pd.DataFrame:
    """
    Intelligently decide whether to do a full historical fetch
    or incremental fetch per ticker based on what's already in SQLite.

    - Ticker not in DB → full 365-day fetch
    - Ticker in DB     → incremental (last 5 days to fill any gaps)

    Args:
        tickers: Full universe ticker list

    Returns:
        Combined DataFrame of all new data fetched
    """
    today      = datetime.today()
    end_date   = today.strftime("%Y-%m-%d")

    full_tickers        = []
    incremental_tickers = []

    for ticker in tickers:
        last_date = get_last_fetch_date(ticker)
        if last_date is None:
            full_tickers.append(ticker)
        else:
            incremental_tickers.append(ticker)

    logger.info(
        f"smart_fetch | Full fetch: {len(full_tickers)} tickers | "
        f"Incremental: {len(incremental_tickers)} tickers"
    )

    all_data = []

    # Full historical fetch
    if full_tickers:
        start_full = (today - timedelta(days=HISTORICAL_DAYS)).strftime("%Y-%m-%d")
        logger.info(f"Full fetch: {start_full} to {end_date}")
        df_full = fetch_universe(full_tickers, start_full, end_date)
        if not df_full.empty:
            all_data.append(df_full)

    # Incremental fetch
    if incremental_tickers:
        start_incr = (today - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y-%m-%d")
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
# STAGE 1 FILTER — price > $10 and avg volume > 500K
# =============================================================================

def apply_stage1_filter(
    df: pd.DataFrame,
    scan_date: str,
) -> pd.DataFrame:
    """
    Apply Stage 1 universe filter:
    - Last close price > $10
    - 20-day average volume > 500,000

    Excludes indices and sector ETFs from filtering
    (they always pass through).

    Args:
        df:        Full OHLCV DataFrame for all tickers
        scan_date: Today's date string YYYY-MM-DD

    Returns:
        DataFrame of tickers that passed Stage 1 with
        columns [ticker, avg_volume, last_close]
    """
    protected = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])

    results = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date")

        last_close = group["close"].iloc[-1]
        avg_volume = group["volume"].tail(20).mean()

        # Always pass indices and sector ETFs through
        if ticker in protected:
            results.append({
                "ticker":     ticker,
                "avg_volume": avg_volume,
                "last_close": last_close,
                "protected":  True,
            })
            continue

        # Apply filters
        if last_close >= MIN_PRICE and avg_volume >= MIN_AVG_VOLUME:
            results.append({
                "ticker":     ticker,
                "avg_volume": avg_volume,
                "last_close": last_close,
                "protected":  False,
            })

    filtered_df = pd.DataFrame(results)
    passed      = len(filtered_df)
    total       = df["ticker"].nunique()

    logger.info(
        f"Stage 1 filter | "
        f"Input: {total} tickers | "
        f"Passed: {passed} | "
        f"Filtered out: {total - passed}"
    )

    return filtered_df


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_data_pipeline() -> dict:
    """
    Main entry point called by Airflow DAG Task 1 and Task 2.

    Steps:
    1. Get full universe
    2. Smart fetch (full or incremental per ticker)
    3. Write raw prices to SQLite
    4. Apply Stage 1 filter
    5. Write filtered universe to SQLite

    Returns:
        Summary dict with counts for logging/monitoring
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE STARTED")
    logger.info("=" * 60)

    today     = datetime.today().strftime("%Y-%m-%d")
    summary   = {}

    try:
        # Step 1: Get universe
        tickers = get_full_universe()
        summary["universe_size"] = len(tickers)

        # Step 2: Smart fetch
        df = smart_fetch(tickers)
        summary["rows_fetched"] = len(df)

        if df.empty:
            logger.error("Data pipeline: No data fetched. Aborting.")
            return summary

        # Step 3: Write raw prices
        rows_written = write_raw_prices(df)
        summary["rows_written"] = rows_written

        # Step 4: Stage 1 filter
        filtered_df = apply_stage1_filter(df, today)
        summary["tickers_passed_filter"] = len(filtered_df)

        # Step 5: Write filtered universe
        write_filtered_universe(filtered_df, today)

        logger.info("=" * 60)
        logger.info(f"DATA PIPELINE COMPLETE | Summary: {summary}")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.critical(f"Data pipeline failed: {e}", exc_info=True)
        raise