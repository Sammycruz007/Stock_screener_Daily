"""
sentiment/sentiment.py
----------------------
Sentiment data fetcher for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This module fetches two sentiment signals per stock:

SIGNAL 1 — Put/Call Ratio:
   Options traders are generally more informed than equity traders.
   The Put/Call ratio tells us what options traders are betting on.

   HOW WE FETCH IT:
   yfinance exposes options chain data per ticker.
   We fetch the nearest expiry options chain and compute:
   Put/Call Ratio = Total Put Open Interest / Total Call Open Interest

   INTERPRETATION:
   - Below 0.7  → More calls than puts → Bullish sentiment
   - Above 1.2  → More puts than calls → Bearish sentiment
   - Between    → Neutral

SIGNAL 2 — Short Interest %:
   Short interest tells us what % of the float is currently sold short.
   High short interest on a long setup = potential short squeeze fuel.
   If a stock with 20% short interest starts moving up, shorts are
   forced to cover (buy) which accelerates the move.

   HOW WE FETCH IT:
   yfinance exposes short interest via ticker.info dictionary.
   We compute: Short Interest % = shortPercentOfFloat * 100

   INTERPRETATION:
   - Below 5%  → Low short interest → Normal
   - Above 15% → High short interest → Potential squeeze fuel on longs

FAILURE HANDLING:
   Not all tickers have options data.
   Not all tickers have short interest data.
   The @graceful decorator ensures missing data returns None
   rather than crashing the pipeline.
   The pipeline continues with NULL sentiment values for those tickers.
   The ML model handles NULLs via imputation in feature engineering.

BATCHING:
   We fetch sentiment in parallel using ThreadPoolExecutor
   same pattern as the data fetcher for speed.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import yaml

from utils.logging import get_sentiment_logger
from utils.error_handler import (
    graceful,
    retry,
    SentimentError,
)

logger = get_sentiment_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
FETCHER_CFG  = config["fetcher"]
SENTIMENT_CFG = config["sentiment"]

MAX_WORKERS  = FETCHER_CFG["max_workers"]

PUT_CALL_BULLISH  = SENTIMENT_CFG["put_call_bullish_threshold"]   # 0.7
PUT_CALL_BEARISH  = SENTIMENT_CFG["put_call_bearish_threshold"]   # 1.2
SHORT_INT_HIGH    = SENTIMENT_CFG["short_interest_high_threshold"] # 15.0


# =============================================================================
# PUT/CALL RATIO FETCHER
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def _fetch_put_call_ratio(ticker: str) -> Optional[float]:
    """
    Fetch Put/Call Ratio for a single ticker from yfinance options chain.

    FLOW:
    1. Get available options expiry dates for the ticker
    2. Pick the nearest expiry (most liquid, most relevant)
    3. Fetch the options chain for that expiry
    4. Sum all put open interest and all call open interest
    5. Compute ratio = total_put_oi / total_call_oi
    6. Return ratio or None if data unavailable

    Args:
        ticker: Ticker symbol e.g. 'AAPL'

    Returns:
        Float put/call ratio or None if unavailable
    """
    tk = yf.Ticker(ticker)

    # ── Step 1: Get available expiry dates ────────────────────────────────────
    expiry_dates = tk.options

    if not expiry_dates:
        logger.debug(f"{ticker} | No options data available")
        return None

    # ── Step 2: Pick nearest expiry ───────────────────────────────────────────
    # First expiry = most liquid, most representative of current sentiment
    nearest_expiry = expiry_dates[0]

    # ── Step 3: Fetch options chain ───────────────────────────────────────────
    chain = tk.option_chain(nearest_expiry)

    calls = chain.calls
    puts  = chain.puts

    if calls.empty or puts.empty:
        logger.debug(f"{ticker} | Empty options chain for expiry {nearest_expiry}")
        return None

    # ── Step 4: Sum open interest ─────────────────────────────────────────────
    total_call_oi = calls["openInterest"].sum()
    total_put_oi  = puts["openInterest"].sum()

    # ── Step 5: Avoid division by zero ───────────────────────────────────────
    if total_call_oi == 0:
        logger.debug(f"{ticker} | Zero call open interest — cannot compute ratio")
        return None

    # ── Step 6: Compute ratio ─────────────────────────────────────────────────
    ratio = round(total_put_oi / total_call_oi, 4)

    logger.debug(
        f"{ticker} | Put/Call Ratio: {ratio} | "
        f"Puts OI: {total_put_oi:,.0f} | Calls OI: {total_call_oi:,.0f}"
    )

    return ratio


# =============================================================================
# SHORT INTEREST FETCHER
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def _fetch_short_interest(ticker: str) -> Optional[float]:
    """
    Fetch Short Interest % of float for a single ticker via yfinance.

    FLOW:
    1. Fetch ticker.info dictionary from yfinance
    2. Extract shortPercentOfFloat field
    3. Convert to percentage (yfinance returns as decimal e.g. 0.032 = 3.2%)
    4. Return percentage or None if unavailable

    Args:
        ticker: Ticker symbol

    Returns:
        Float short interest as percentage (e.g. 3.2 for 3.2%) or None
    """
    tk   = yf.Ticker(ticker)
    info = tk.info

    if not info:
        logger.debug(f"{ticker} | No info data returned")
        return None

    # yfinance returns shortPercentOfFloat as a decimal (0.032 = 3.2%)
    short_pct = info.get("shortPercentOfFloat", None)

    if short_pct is None:
        logger.debug(f"{ticker} | shortPercentOfFloat not available")
        return None

    # Convert decimal to percentage
    short_pct_formatted = round(float(short_pct) * 100, 2)

    logger.debug(f"{ticker} | Short Interest: {short_pct_formatted}%")

    return short_pct_formatted


# =============================================================================
# SINGLE TICKER SENTIMENT FETCH
# Combines both signals into one call per ticker
# =============================================================================

def _fetch_ticker_sentiment(ticker: str, date: str) -> dict:
    """
    Fetch all sentiment data for a single ticker.

    FLOW:
    1. Fetch Put/Call Ratio (returns None if unavailable)
    2. Fetch Short Interest % (returns None if unavailable)
    3. Package into dict with ticker and date
    4. Return dict regardless of whether data was available
       (NULL values are handled downstream in ML feature engineering)

    Args:
        ticker: Ticker symbol
        date:   Today's date string YYYY-MM-DD

    Returns:
        Dict with ticker, date, put_call_ratio, short_interest_pct
        Values may be None if data unavailable for that ticker
    """
    put_call_ratio     = _fetch_put_call_ratio(ticker)
    short_interest_pct = _fetch_short_interest(ticker)

    return {
        "ticker"            : ticker,
        "date"              : date,
        "put_call_ratio"    : put_call_ratio,
        "short_interest_pct": short_interest_pct,
    }


# =============================================================================
# BATCH SENTIMENT FETCHER
# Runs all tickers in parallel for speed
# =============================================================================

def fetch_sentiment_batch(
    tickers : list[str],
    date    : str,
) -> pd.DataFrame:
    """
    Fetch sentiment data for all tickers in parallel.

    FLOW:
    1. Submit all tickers to ThreadPoolExecutor simultaneously
    2. Collect results as they complete
    3. Log any tickers that had no sentiment data available
    4. Return combined DataFrame for database write

    Args:
        tickers : List of ticker symbols to fetch sentiment for
        date    : Today's date string YYYY-MM-DD

    Returns:
        DataFrame with columns [ticker, date, put_call_ratio, short_interest_pct]
        Rows with unavailable data will have None/NaN in sentiment columns
    """
    logger.info(
        f"Sentiment fetch starting | "
        f"{len(tickers)} tickers | Date: {date}"
    )

    results          = []
    no_options_count = 0
    no_short_count   = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(_fetch_ticker_sentiment, ticker, date): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results.append(result)

                # Track how many had missing data for logging
                if result["put_call_ratio"] is None:
                    no_options_count += 1
                if result["short_interest_pct"] is None:
                    no_short_count += 1

            except Exception as e:
                logger.warning(f"{ticker} | Sentiment fetch failed: {e}")
                # Still add a row with None values so the ticker isn't dropped
                results.append({
                    "ticker"            : ticker,
                    "date"              : date,
                    "put_call_ratio"    : None,
                    "short_interest_pct": None,
                })

    logger.info(
        f"Sentiment fetch complete | "
        f"Total: {len(results)} | "
        f"No options data: {no_options_count} | "
        f"No short interest: {no_short_count}"
    )

    return pd.DataFrame(results)


# =============================================================================
# SENTIMENT SIGNAL INTERPRETER
# Converts raw numbers into human-readable signals for the dashboard
# =============================================================================

def interpret_put_call(ratio: Optional[float]) -> str:
    """
    Convert a raw Put/Call ratio into a readable sentiment label.

    Args:
        ratio: Put/Call ratio float or None

    Returns:
        'bullish', 'bearish', 'neutral', or 'unavailable'
    """
    if ratio is None or np.isnan(ratio):
        return "unavailable"
    if ratio < PUT_CALL_BULLISH:
        return "bullish"
    if ratio > PUT_CALL_BEARISH:
        return "bearish"
    return "neutral"


def interpret_short_interest(pct: Optional[float]) -> str:
    """
    Convert a raw Short Interest % into a readable label.

    Args:
        pct: Short interest percentage float or None

    Returns:
        'high', 'normal', or 'unavailable'
    """
    if pct is None or np.isnan(pct):
        return "unavailable"
    if pct >= SHORT_INT_HIGH:
        return "high"
    return "normal"


# =============================================================================
# MAIN ENTRY POINT
# Called by Airflow DAG Task 6
# =============================================================================

def run_sentiment_pipeline(
    tickers : list[str],
    date    : str,
) -> pd.DataFrame:
    """
    Main entry point for the sentiment pipeline.
    Called by Airflow after the indicator engines complete.

    FLOW:
    1. Exclude indices and sector ETFs from sentiment fetch
       (they don't have meaningful options/short interest data)
    2. Fetch sentiment for all stock tickers in parallel
    3. Return DataFrame ready for database write

    Args:
        tickers : Full list of tickers from filtered universe
        date    : Today's date string YYYY-MM-DD

    Returns:
        DataFrame with sentiment data for all stock tickers
    """
    config       = _load_config()
    excluded     = set(
        config["universe"]["indices"] +
        config["universe"]["sectors"]
    )

    # Only fetch sentiment for actual stocks — not ETFs or indices
    stock_tickers = [t for t in tickers if t not in excluded]

    logger.info(
        f"Sentiment pipeline | "
        f"Stock tickers: {len(stock_tickers)} | "
        f"Excluded ETFs/indices: {len(tickers) - len(stock_tickers)}"
    )

    return fetch_sentiment_batch(stock_tickers, date)