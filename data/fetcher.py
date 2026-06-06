"""
data/fetcher.py
---------------
Data fetching layer for the Stock Scanner pipeline.

CHANGES FROM PREVIOUS VERSION:
- Universe now pulled from NASDAQ FTP listings (nasdaqlisted.txt + otherlisted.txt)
  instead of Wikipedia scraping S&P500 + NASDAQ100 only.
- This gives us ~6,000 raw tickers across NYSE + NASDAQ
- Stage 1 filter (price > $10, vol > 500K) reduces this to ~1,500 quality names
- Sector data fetched dynamically from yfinance ticker.info
- Sector stored in ticker_metadata table (no hardcoded SECTOR_MAP)
- Unclassified stocks (no sector from yfinance) flagged but not excluded

LOGICAL FLOW:
─────────────
STEP 1 — Get full universe from NASDAQ FTP:
   Download nasdaqlisted.txt  → all NASDAQ listed stocks
   Download otherlisted.txt   → all NYSE + NYSE American listed stocks
   Combine, deduplicate, clean → ~6,000 raw tickers

STEP 2 — Filter out non-stocks:
   Remove warrants (ticker ends with W)
   Remove rights (ticker ends with R)
   Remove units (ticker ends with U)
   Remove preferred shares (ticker contains -)
   Remove test issues (flagged in the file)
   Remove ETFs (flagged in the file) EXCEPT our 11 sector ETFs + 3 indices
   → Reduces to ~4,000 clean stock tickers

STEP 3 — Smart fetch OHLCV data:
   First run  → full 365-day history
   Daily runs → incremental last 5 days only
   Parallel batches of 50 tickers, 10 workers

STEP 4 — Stage 1 filter:
   Price > $10 AND average volume > 500K
   → Reduces to ~1,500 actionable tickers

STEP 5 — Fetch sector metadata:
   For each ticker that passed Stage 1
   Fetch ticker.info from yfinance
   Extract sector name → map to sector ETF
   Store in ticker_metadata table
   Tag unclassified stocks with sector = 'Unclassified'
"""

import ftplib
import io
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import yaml

from data.database import (
    write_raw_prices,
    write_filtered_universe,
    write_ticker_metadata,
    get_last_fetch_date,
)
from utils.logging import get_fetcher_logger
from utils.error_handler import (
    retry,
    graceful,
    validate_dataframe,
    DataFetchError,
)

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

MIN_PRICE      = FILTER_CFG["min_price"]
MIN_AVG_VOLUME = FILTER_CFG["min_avg_volume"]

# NASDAQ FTP details
FTP_HOST    = "ftp.nasdaqtrader.com"
FTP_DIR     = "SymbolDirectory"
NASDAQ_FILE = "nasdaqlisted.txt"
OTHER_FILE  = "otherlisted.txt"

# Sector name → ETF mapping
# yfinance returns sector as a plain English string
# We map that to the corresponding sector ETF ticker
SECTOR_NAME_TO_ETF = {
    "Technology"            : "XLK",
    "Financial Services"    : "XLF",
    "Energy"                : "XLE",
    "Healthcare"            : "XLV",
    "Industrials"           : "XLI",
    "Consumer Cyclical"     : "XLY",
    "Consumer Defensive"    : "XLP",
    "Utilities"             : "XLU",
    "Basic Materials"       : "XLB",
    "Real Estate"           : "XLRE",
    "Communication Services": "XLC",
}


# =============================================================================
# STEP 1 — FULL UNIVERSE FROM NASDAQ FTP
# =============================================================================

def _download_ftp_file(filename: str) -> pd.DataFrame:
    """
    Download a single file from the NASDAQ FTP server.

    FLOW:
    1. Connect to ftp.nasdaqtrader.com anonymously
    2. Navigate to SymbolDirectory folder
    3. Download the file into memory (no disk write needed)
    4. Parse as pipe-delimited text into a DataFrame
    5. Return DataFrame

    Args:
        filename: 'nasdaqlisted.txt' or 'otherlisted.txt'

    Returns:
        DataFrame with raw ticker listing data
    """
    logger.info(f"Downloading {filename} from NASDAQ FTP")

    try:
        # ── Connect anonymously ───────────────────────────────────────────────
        ftp = ftplib.FTP(FTP_HOST)
        ftp.login("anonymous", "")       # Anonymous login — no credentials needed
        ftp.cwd(FTP_DIR)                 # Navigate to SymbolDirectory

        # ── Download file into memory buffer ──────────────────────────────────
        buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {filename}", buffer.write)
        ftp.quit()

        # ── Parse pipe-delimited content ──────────────────────────────────────
        buffer.seek(0)
        content = buffer.read().decode("utf-8")

        # Files use | as delimiter, last line is a file creation timestamp — skip it
        lines = [l for l in content.strip().split("\n") if "File Creation Time" not in l]
        clean_content = "\n".join(lines)

        df = pd.read_csv(io.StringIO(clean_content), sep="|")

        logger.info(f"{filename}: {len(df)} rows downloaded")
        return df

    except Exception as e:
        raise DataFetchError(f"FTP download failed for {filename}: {e}") from e


def _parse_nasdaq_listed(df: pd.DataFrame) -> list[str]:
    """
    Extract clean stock tickers from nasdaqlisted.txt.

    FILE COLUMNS:
    Symbol | Security Name | Market Category | Test Issue |
    Financial Status | Round Lot Size | ETF | NextShares

    FILTERS APPLIED:
    - Test Issue == 'N'     (exclude test symbols)
    - ETF == 'N'            (exclude ETFs — except our protected ones)
    - Symbol has no special characters except - (preferred shares use -)
    - Remove warrants (W suffix), rights (R suffix), units (U suffix)

    Args:
        df: Raw DataFrame from nasdaqlisted.txt

    Returns:
        List of clean ticker strings
    """
    protected = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])

    # Exclude test issues
    df = df[df["Test Issue"] == "N"]

    # Exclude ETFs (but keep our protected sector ETFs and indices)
    df = df[
        (df["ETF"] == "N") |
        (df["Symbol"].isin(protected))
    ]

    tickers = df["Symbol"].tolist()
    return tickers


def _parse_other_listed(df: pd.DataFrame) -> list[str]:
    """
    Extract clean stock tickers from otherlisted.txt.

    FILE COLUMNS:
    ACT Symbol | Security Name | Exchange | CQS Symbol |
    ETF | Round Lot Size | Test Issue | NASDAQ Symbol

    FILTERS APPLIED:
    - Test Issue == 'N'
    - ETF == 'N' (except protected)

    Args:
        df: Raw DataFrame from otherlisted.txt

    Returns:
        List of clean ticker strings
    """
    protected = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])

    df = df[df["Test Issue"] == "N"]
    df = df[
        (df["ETF"] == "N") |
        (df["ACT Symbol"].isin(protected))
    ]

    tickers = df["ACT Symbol"].tolist()
    return tickers


def _clean_tickers(tickers: list[str]) -> list[str]:
    """
    Clean and filter raw ticker list.

    REMOVES:
    - Warrants  : ticker ending in W  (e.g. SPCEQ → skip, AACLW → skip)
    - Rights    : ticker ending in R  (e.g. AACBR → skip)
    - Units     : ticker ending in U  (e.g. AACBU → skip)
    - Preferred : ticker containing $ (e.g. BAC-PK → skip on $ variant)
    - Too long  : ticker > 5 chars (usually special instruments)
    - Empty     : any null/empty strings

    KEEPS:
    - Clean common stock tickers (1-5 alphanumeric chars)
    - Our protected indices and sector ETFs always kept

    Args:
        tickers: Raw list of ticker strings

    Returns:
        Cleaned, deduplicated, sorted list of tickers
    """
    protected = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])
    cleaned   = []

    for ticker in tickers:
        # Always keep protected tickers regardless of format
        if ticker in protected:
            cleaned.append(ticker)
            continue

        # Skip empty or null
        if not ticker or pd.isna(ticker):
            continue

        ticker = str(ticker).strip().upper()

        # Skip warrants, rights, units by suffix
        if ticker.endswith(("W", "R", "U")):
            continue

        # Skip preferred shares (contain -)
        if "-" in ticker:
            continue

        # Skip anything longer than 5 characters
        if len(ticker) > 5:
            continue

        # Skip non-alphanumeric tickers
        if not ticker.isalpha():
            continue

        cleaned.append(ticker)

    # Deduplicate and sort
    result = sorted(list(set(cleaned)))
    return result


def get_full_universe() -> list[str]:
    """
    Download and combine the full US stock universe from NASDAQ FTP.

    FLOW:
    1. Download nasdaqlisted.txt (NASDAQ stocks)
    2. Download otherlisted.txt (NYSE + NYSE American stocks)
    3. Parse each file to extract tickers
    4. Combine and clean
    5. Add our protected indices and sector ETFs
    6. Return deduplicated sorted list

    Returns:
        Sorted list of ~4,000 clean US stock tickers
        (before Stage 1 volume/price filter)
    """
    logger.info("Building full US stock universe from NASDAQ FTP")

    # ── Download both files ───────────────────────────────────────────────────
    nasdaq_df = _download_ftp_file(NASDAQ_FILE)
    other_df  = _download_ftp_file(OTHER_FILE)

    # ── Parse each file ───────────────────────────────────────────────────────
    nasdaq_tickers = _parse_nasdaq_listed(nasdaq_df)
    other_tickers  = _parse_other_listed(other_df)

    # ── Combine + clean ───────────────────────────────────────────────────────
    all_tickers = nasdaq_tickers + other_tickers
    cleaned     = _clean_tickers(all_tickers)

    logger.info(
        f"Universe built | "
        f"NASDAQ: {len(nasdaq_tickers)} | "
        f"Other exchanges: {len(other_tickers)} | "
        f"After cleaning: {len(cleaned)}"
    )

    return cleaned


# =============================================================================
# STEP 2 — SINGLE TICKER OHLCV FETCH
# =============================================================================

@retry(
    attempts      = RETRY_ATTEMPTS,
    delay_seconds = RETRY_DELAY,
    exceptions    = (DataFetchError, Exception),
)
def fetch_single_ticker(
    ticker     : str,
    start_date : str,
    end_date   : str,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single ticker via yfinance.

    FLOW:
    1. Download via yf.download()
    2. Flatten MultiIndex columns if present
    3. Standardise column names to lowercase
    4. Add ticker and date columns
    5. Validate (not empty, has required columns, enough rows)
    6. Drop null and zero-price rows
    7. Return clean DataFrame

    Args:
        ticker    : Ticker symbol
        start_date: Start date YYYY-MM-DD
        end_date  : End date YYYY-MM-DD

    Returns:
        Clean OHLCV DataFrame or None if fetch/validation fails
    """
    try:
        raw = yf.download(
            ticker,
            start      = start_date,
            end        = end_date,
            auto_adjust= True,      # Adjust for splits and dividends
            progress   = False,
            threads    = False,     # We handle threading ourselves
        )

        if raw.empty:
            logger.warning(f"{ticker} | Empty DataFrame from yfinance")
            return None

        # ── Flatten MultiIndex columns ────────────────────────────────────────
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # ── Standardise column names ──────────────────────────────────────────
        raw.columns = [c.lower() for c in raw.columns]

        # ── Keep only OHLCV columns ───────────────────────────────────────────
        required = ["open", "high", "low", "close", "volume"]
        if not validate_dataframe(raw, ticker, required):
            return None

        raw          = raw[required].copy()
        raw["ticker"]= ticker
        raw["date"]  = raw.index.strftime("%Y-%m-%d")
        raw          = raw.reset_index(drop=True)

        # ── Drop bad rows ─────────────────────────────────────────────────────
        raw = raw.dropna(subset=required)
        raw = raw[(raw["close"] > 0) & (raw["volume"] >= 0)]

        logger.debug(f"{ticker} | {len(raw)} rows fetched")
        return raw

    except Exception as e:
        raise DataFetchError(f"{ticker} fetch failed: {e}") from e


# =============================================================================
# STEP 3 — BATCH PARALLEL FETCH
# =============================================================================

def fetch_batch(
    tickers    : list[str],
    start_date : str,
    end_date   : str,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a batch of tickers in parallel using ThreadPoolExecutor.

    FLOW:
    1. Submit all tickers to thread pool simultaneously
    2. Collect results as each future completes
    3. Track failed tickers for logging
    4. Concatenate successful results

    Args:
        tickers   : List of ticker symbols for this batch
        start_date: Start date YYYY-MM-DD
        end_date  : End date YYYY-MM-DD

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
                logger.warning(f"{ticker} | Batch fetch error: {e}")
                failed.append(ticker)

    if failed:
        logger.warning(f"Batch: {len(failed)} tickers failed: {failed[:10]}...")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def fetch_universe(
    tickers    : list[str],
    start_date : str,
    end_date   : str,
) -> pd.DataFrame:
    """
    Fetch OHLCV for the entire universe in sequential batches.

    FLOW:
    1. Split tickers into batches of BATCH_SIZE (50)
    2. Process each batch in parallel
    3. Collect and combine all results
    4. Log summary statistics

    Args:
        tickers   : Full list of tickers to fetch
        start_date: Start date YYYY-MM-DD
        end_date  : End date YYYY-MM-DD

    Returns:
        Combined DataFrame for all tickers
    """
    total   = len(tickers)
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    all_data= []

    logger.info(
        f"Fetching {total} tickers | "
        f"{len(batches)} batches of {BATCH_SIZE} | "
        f"{start_date} to {end_date}"
    )

    for i, batch in enumerate(batches, 1):
        logger.info(f"Batch {i}/{len(batches)} | {len(batch)} tickers")
        batch_df = fetch_batch(batch, start_date, end_date)
        if not batch_df.empty:
            all_data.append(batch_df)

    if not all_data:
        logger.error("fetch_universe: No data returned")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"Fetch complete | "
        f"Rows: {len(combined)} | "
        f"Tickers with data: {combined['ticker'].nunique()}"
    )
    return combined


# =============================================================================
# STEP 3B — SMART FETCH (full vs incremental per ticker)
# =============================================================================

def smart_fetch(tickers: list[str]) -> pd.DataFrame:
    """
    Decide full historical vs incremental fetch per ticker.

    LOGIC:
    - Ticker NOT in database → full 365-day fetch
    - Ticker already in database → incremental last 5 days only

    This means on first run everything is fetched from scratch.
    On every subsequent daily run only the latest candles are added.
    Much faster daily runs after the first.

    Args:
        tickers: Full universe ticker list

    Returns:
        Combined DataFrame of all newly fetched data
    """
    today    = datetime.today()
    end_date = today.strftime("%Y-%m-%d")

    full_tickers        = []
    incremental_tickers = []

    # ── Categorise each ticker ────────────────────────────────────────────────
    for ticker in tickers:
        last_date = get_last_fetch_date(ticker)
        if last_date is None:
            full_tickers.append(ticker)
        else:
            incremental_tickers.append(ticker)

    logger.info(
        f"Smart fetch | "
        f"Full: {len(full_tickers)} | "
        f"Incremental: {len(incremental_tickers)}"
    )

    all_data = []

    # ── Full historical fetch ─────────────────────────────────────────────────
    if full_tickers:
        start_full = (today - timedelta(days=HISTORICAL_DAYS)).strftime("%Y-%m-%d")
        df_full    = fetch_universe(full_tickers, start_full, end_date)
        if not df_full.empty:
            all_data.append(df_full)

    # ── Incremental fetch ─────────────────────────────────────────────────────
    if incremental_tickers:
        start_incr = (today - timedelta(days=INCREMENTAL_DAYS)).strftime("%Y-%m-%d")
        df_incr    = fetch_universe(incremental_tickers, start_incr, end_date)
        if not df_incr.empty:
            all_data.append(df_incr)

    if not all_data:
        logger.warning("smart_fetch: No new data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"smart_fetch complete | Total rows: {len(combined)}")
    return combined


# =============================================================================
# STEP 4 — STAGE 1 FILTER
# =============================================================================

def apply_stage1_filter(
    df        : pd.DataFrame,
    scan_date : str,
) -> pd.DataFrame:
    """
    Filter universe to actionable stocks only.

    CRITERIA:
    - Last close price >= $10  (no penny stocks)
    - 20-day average volume >= 500,000  (adequate liquidity)

    Protected tickers (indices + sector ETFs) always pass through
    regardless of price or volume — we always need them for the
    market and sector health checks.

    Args:
        df       : Full OHLCV DataFrame for all tickers
        scan_date: Today's date YYYY-MM-DD

    Returns:
        DataFrame with columns [ticker, avg_volume, last_close]
        for all tickers that passed Stage 1
    """
    protected = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])
    results   = []

    for ticker, group in df.groupby("ticker"):
        group      = group.sort_values("date")
        last_close = group["close"].iloc[-1]
        avg_volume = group["volume"].tail(20).mean()

        # Always pass indices and sector ETFs through
        if ticker in protected:
            results.append({
                "ticker"    : ticker,
                "avg_volume": avg_volume,
                "last_close": last_close,
                "protected" : True,
            })
            continue

        # Apply Stage 1 filter to regular stocks
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
        f"Input: {total} | "
        f"Passed: {passed} | "
        f"Filtered out: {total - passed}"
    )

    return filtered_df


# =============================================================================
# STEP 5 — DYNAMIC SECTOR METADATA FETCH
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def _fetch_ticker_sector(ticker: str) -> Optional[dict]:
    """
    Fetch sector information for a single ticker from yfinance.

    FLOW:
    1. Fetch ticker.info dictionary from yfinance
    2. Extract 'sector' field (plain English e.g. 'Technology')
    3. Map sector name to sector ETF using SECTOR_NAME_TO_ETF
    4. Return dict with ticker, sector_name, sector_etf
    5. If sector unavailable → tag as 'Unclassified'

    Args:
        ticker: Ticker symbol

    Returns:
        Dict with ticker, sector_name, sector_etf or None on failure
    """
    protected = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])

    # Protected tickers don't need sector lookup
    if ticker in protected:
        return {
            "ticker"     : ticker,
            "sector_name": "Index/ETF",
            "sector_etf" : ticker,   # Maps to itself
        }

    tk          = yf.Ticker(ticker)
    info        = tk.info
    sector_name = info.get("sector", None)

    if sector_name:
        sector_etf = SECTOR_NAME_TO_ETF.get(sector_name, None)
    else:
        sector_name = "Unclassified"
        sector_etf  = None

    if sector_etf is None and sector_name != "Unclassified":
        # Sector name exists but not in our map — still flag it
        logger.warning(
            f"{ticker} | Unmapped sector: '{sector_name}' "
            f"— tagging as Unclassified"
        )
        sector_name = "Unclassified"

    logger.debug(
        f"{ticker} | Sector: {sector_name} | ETF: {sector_etf}"
    )

    return {
        "ticker"     : ticker,
        "sector_name": sector_name,
        "sector_etf" : sector_etf,
    }


def fetch_sector_metadata(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch sector metadata for all tickers in parallel.

    FLOW:
    1. Submit all tickers to ThreadPoolExecutor
    2. Collect results — None results replaced with Unclassified
    3. Return DataFrame for write to ticker_metadata table

    Args:
        tickers: List of tickers that passed Stage 1

    Returns:
        DataFrame with columns [ticker, sector_name, sector_etf]
    """
    logger.info(f"Fetching sector metadata for {len(tickers)} tickers")

    results         = []
    unclassified    = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_sector, ticker): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result is None:
                    # Graceful decorator returned None — tag as unclassified
                    result = {
                        "ticker"     : ticker,
                        "sector_name": "Unclassified",
                        "sector_etf" : None,
                    }
                    unclassified += 1
                elif result["sector_name"] == "Unclassified":
                    unclassified += 1

                results.append(result)

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
        f"Unclassified: {unclassified}"
    )

    return pd.DataFrame(results)


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# Called by Airflow DAG Tasks 1 and 2
# =============================================================================

def run_data_pipeline() -> dict:
    """
    Main entry point for the full data pipeline.

    FLOW:
    1. Get full universe from NASDAQ FTP (~4,000 tickers)
    2. Smart fetch OHLCV data (full or incremental per ticker)
    3. Write raw prices to SQLite
    4. Apply Stage 1 filter (~1,500 tickers pass)
    5. Write filtered universe to SQLite
    6. Fetch sector metadata for filtered tickers
    7. Write sector metadata to SQLite

    Returns:
        Summary dict with counts for Airflow monitoring
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE STARTED")
    logger.info("=" * 60)

    today   = datetime.today().strftime("%Y-%m-%d")
    summary = {}

    try:
        # ── Step 1: Get universe ──────────────────────────────────────────────
        tickers = get_full_universe()
        summary["universe_size"] = len(tickers)

        # ── Step 2: Smart fetch OHLCV ─────────────────────────────────────────
        df = smart_fetch(tickers)
        summary["rows_fetched"] = len(df)

        if df.empty:
            logger.error("Data pipeline: No data fetched. Aborting.")
            return summary

        # ── Step 3: Write raw prices ──────────────────────────────────────────
        rows_written = write_raw_prices(df)
        summary["rows_written"] = rows_written

        # ── Step 4: Stage 1 filter ────────────────────────────────────────────
        filtered_df = apply_stage1_filter(df, today)
        summary["tickers_passed_filter"] = len(filtered_df)

        # ── Step 5: Write filtered universe ───────────────────────────────────
        write_filtered_universe(filtered_df, today)

        # ── Step 6: Fetch sector metadata ─────────────────────────────────────
        filtered_tickers = filtered_df["ticker"].tolist()
        sector_df        = fetch_sector_metadata(filtered_tickers)
        summary["sector_metadata_fetched"] = len(sector_df)

        # ── Step 7: Write sector metadata ─────────────────────────────────────
        write_ticker_metadata(sector_df)

        logger.info("=" * 60)
        logger.info(f"DATA PIPELINE COMPLETE | {summary}")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.critical(f"Data pipeline failed: {e}", exc_info=True)
        raise