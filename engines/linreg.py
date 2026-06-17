"""
engines/linreg.py
-----------------
Linear Regression engine for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This engine runs on every ticker that passed Stage 1 filter.
For each ticker it does the following:

STEP 1 — Fit the Linear Regression line:
   Using the last 200 closing prices, we fit a straight line
   through the data using least squares regression (scipy).
   This line represents the "fair value" trend of the stock.
   The slope of this line tells us the trend direction.

STEP 2 — Compute Standard Deviation bands:
   We measure how far prices typically deviate from the LinReg line.
   Using the residuals (actual price minus LinReg value at each point),
   we compute the standard deviation of those residuals.
   Then we add/subtract 1x, 2x, 3x that standard deviation
   to get the upper and lower bands around the LinReg line.

STEP 3 — Determine slope direction:
   If the LinReg slope is positive → uptrend → long bias
   If the LinReg slope is negative → downtrend → short bias
   Slope is normalised by price level so it's comparable across stocks.

STEP 4 — Determine price SD position:
   We calculate exactly WHERE the current price sits relative
   to the bands. e.g. -1.8 means price is 1.8 standard deviations
   BELOW the LinReg line. +2.3 means 2.3 SDs ABOVE.
   This single number tells the scanner exactly which zone price is in.

STEP 5 — Package results:
   Return a clean dict with all computed values ready to be
   written to the indicator_results table in SQLite.

WHY SCIPY OVER PANDAS-TA:
   pandas-ta has a linreg function but it only returns the line values.
   We need the residuals to compute our own SD bands exactly as
   we want them. scipy gives us full control over the regression.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_linreg_logger
from utils.error_handler import graceful, EngineError

logger = get_linreg_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config     = _load_config()
LINREG_CFG = config["linreg"]

PERIOD   = LINREG_CFG["period"]       # 150 candles
STD_DEV_PERIOD = LINREG_CFG["std_dev_period"]  # 21 candles
STD_DEVS = LINREG_CFG["std_devs"]     # [1, 2, 3]


# =============================================================================
# CORE LINREG CALCULATION
# This is the mathematical heart of the engine.
# =============================================================================

def _compute_linreg(closes: np.ndarray) -> dict:
    """
    Fit a linear regression line through an array of closing prices
    and compute standard deviation bands around it.

    MATHS:
    - x = [0, 1, 2, ..., N-1] — time indices
    - y = closing prices
    - We fit: y = slope * x + intercept  (least squares)
    - residuals = y - y_fitted  (how far each price deviates from the line)
    - std_dev = standard deviation of residuals
    - bands = y_fitted ± (n * std_dev) for n in [1, 2, 3]

    Args:
        closes: numpy array of closing prices, oldest first

    Returns:
        Dict with linreg_value, slope, std_dev, and all band values
        for the MOST RECENT candle (last value in the array)
    """

    n = len(closes)
    x = np.arange(n)  # Time indices: 0, 1, 2, ..., 199

    # ── Step 1: Fit linear regression ────────────────────────────────────────
    # scipy.stats.linregress returns slope, intercept, r_value, p_value, stderr
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, closes)

    # ── Step 2: Compute fitted values (the LinReg line itself) ───────────────
    # y_fitted[i] = slope * i + intercept  for each time index i
    y_fitted = slope * x + intercept

    # ── Step 3: Compute residuals ─────────────────────────────────────────────
    # How far each actual price deviates from the fitted line
    residuals = closes - y_fitted

    # ── Step 4: Compute standard deviation of residuals ───────────────────────
    # This is the "width" of typical price deviation from the LinReg line
    std_dev = np.std(residuals[-STD_DEV_PERIOD:], ddof=1)  # ddof=1 for sample std dev

    # ── Step 5: Get values for the MOST RECENT candle ─────────────────────────
    # We only care about where the line and bands sit TODAY (last candle)
    current_linreg = y_fitted[-1]
    current_close  = closes[-1]

    # ── Step 6: Compute SD bands for the current candle ───────────────────────
    bands = {}
    for sd in STD_DEVS:
        bands[f"sd{sd}_upper"] = current_linreg + (sd * std_dev)
        bands[f"sd{sd}_lower"] = current_linreg - (sd * std_dev)

    # ── Step 7: Compute where current price sits (SD position) ────────────────
    # Positive = above LinReg, Negative = below LinReg
    # e.g. -1.8 means price is 1.8 SDs below the LinReg line
    if std_dev > 0:
        price_sd_position = (current_close - current_linreg) / std_dev
    else:
        price_sd_position = 0.0  # Avoid division by zero on flat price

    # ── Step 8: Normalise slope for comparability across stocks ───────────────
    # Raw slope is in price units (e.g. $0.05/day for a $50 stock)
    # Dividing by current price gives a percentage slope
    # This lets us compare slope steepness between a $10 and $500 stock
    normalised_slope = slope / current_close if current_close > 0 else slope

    return {
        "linreg_value"      : round(current_linreg, 4),
        "linreg_slope"      : round(normalised_slope, 6),
        "linreg_slope_up"   : 1 if slope > 0 else 0,
        "std_dev"           : round(std_dev, 4),
        "price_sd_position" : round(price_sd_position, 4),
        **{k: round(v, 4) for k, v in bands.items()},  # sd1_upper, sd1_lower, etc.
    }


# =============================================================================
# FULL SERIES LINREG LINE
# Used for charting the SPY/QQQ/DIA Plotly charts on the dashboard.
# Returns the entire LinReg line and bands for all candles — not just the last.
# =============================================================================

def compute_linreg_series(closes: np.ndarray) -> pd.DataFrame:
    """
    Compute the full LinReg line and SD bands for all candles.
    Used exclusively for dashboard charting of SPY, QQQ, DIA.

    For the scanner (individual stocks), we only need the latest
    candle values — use compute_linreg_latest() instead.

    Args:
        closes: numpy array of closing prices, oldest first

    Returns:
        DataFrame with columns:
        [linreg, sd1_upper, sd1_lower, sd2_upper, sd2_lower, sd3_upper, sd3_lower]
        One row per candle in the input array.
    """
    n = len(closes)
    x = np.arange(n)

    slope, intercept, _, _, _ = stats.linregress(x, closes)
    y_fitted  = slope * x + intercept
    residuals = closes - y_fitted
    std_dev   = np.std(residuals, ddof=1)

    # Build a DataFrame with the full line and all bands
    result = pd.DataFrame({"linreg": y_fitted})

    for sd in STD_DEVS:
        result[f"sd{sd}_upper"] = y_fitted + (sd * std_dev)
        result[f"sd{sd}_lower"] = y_fitted - (sd * std_dev)

    return result.round(4)


# =============================================================================
# PER-TICKER ENTRY POINT
# Called by the scanner for each ticker in the filtered universe.
# Wrapped with @graceful so one bad ticker never crashes the pipeline.
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def compute_linreg_latest(
    ticker : str,
    df     : pd.DataFrame,
    date   : str,
) -> Optional[dict]:
    """
    Compute LinReg values for the most recent candle of a single ticker.
    This is what the Airflow pipeline calls per ticker.

    FLOW:
    1. Validate we have enough data (need at least PERIOD candles)
    2. Extract the last PERIOD closing prices
    3. Run _compute_linreg() to get all values
    4. Package into a flat dict ready for database insertion
    5. Return dict (or None if anything fails — @graceful handles that)

    Args:
        ticker : Ticker symbol e.g. 'AAPL'
        df     : Full OHLCV DataFrame for this ticker, sorted date ascending
        date   : Today's date string YYYY-MM-DD (for database keying)

    Returns:
        Dict with all LinReg values for this ticker today, or None on failure
    """

    # ── Step 1: Check we have enough candles ─────────────────────────────────
    if len(df) < PERIOD:
        logger.warning(
            f"{ticker} | Insufficient data: {len(df)} rows, need {PERIOD}"
        )
        return None

    # ── Step 2: Extract the last PERIOD closing prices as numpy array ─────────
    closes = df["close"].values[-PERIOD:]

    # ── Step 3: Run the core LinReg calculation ───────────────────────────────
    linreg_values = _compute_linreg(closes)

    # ── Step 4: Add ticker and date for database insertion ────────────────────
    result = {
        "ticker" : ticker,
        "date"   : date,
        **linreg_values,
    }

    logger.debug(
        f"{ticker} | LinReg: {result['linreg_value']} | "
        f"Slope: {'UP' if result['linreg_slope_up'] else 'DOWN'} | "
        f"SD Position: {result['price_sd_position']}"
    )

    return result


# =============================================================================
# BATCH RUNNER
# Runs the LinReg engine across all tickers in the filtered universe.
# Returns a DataFrame ready to be written to indicator_results table.
# =============================================================================

def run_linreg_engine(
    tickers_data : dict[str, pd.DataFrame],
    date         : str,
) -> pd.DataFrame:
    """
    Run LinReg computation for all tickers in the filtered universe.

    FLOW:
    1. Iterate over each ticker and its OHLCV DataFrame
    2. Call compute_linreg_latest() for each
    3. Collect results — skipping any tickers that returned None
    4. Return combined DataFrame for bulk database write

    Args:
        tickers_data : Dict mapping ticker → OHLCV DataFrame
                       e.g. {"AAPL": df_aapl, "TSLA": df_tsla, ...}
        date         : Today's date string YYYY-MM-DD

    Returns:
        DataFrame with one row per ticker containing all LinReg columns
    """
    logger.info(f"LinReg engine starting | {len(tickers_data)} tickers | Date: {date}")

    results = []
    failed  = 0

    for ticker, df in tickers_data.items():
        result = compute_linreg_latest(ticker, df, date)

        if result is not None:
            results.append(result)
        else:
            failed += 1

    logger.info(
        f"LinReg engine complete | "
        f"Computed: {len(results)} | "
        f"Skipped: {failed}"
    )

    if not results:
        logger.warning("LinReg engine: No results produced")
        return pd.DataFrame()

    return pd.DataFrame(results)