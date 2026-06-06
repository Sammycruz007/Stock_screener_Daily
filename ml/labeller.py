"""
ml/labeller.py
--------------
Programmatic label generation for both ML models.

LOGICAL FLOW:
─────────────
We never manually label data. Instead we let historical price
action tell us whether a setup succeeded or failed.

THE LABELLING LOGIC:

For LONG setups:
   At time T, price is between -1 and -3 SD below LinReg.
   We look forward 15 days from T.
   If price CLOSES ABOVE the LinReg line at any point in those 15 days:
   → Label = 1 (success — price mean reverted as expected)
   If price never reaches the LinReg line within 15 days:
   → Label = 0 (failure — setup didn't play out)

For SHORT setups:
   At time T, price is between +1 and +3 SD above LinReg.
   We look forward 15 days from T.
   If price CLOSES BELOW the LinReg line at any point in those 15 days:
   → Label = 1 (success)
   If price never drops to LinReg within 15 days:
   → Label = 0 (failure)

WHY 15 DAYS:
   15 trading days = 3 calendar weeks.
   Long enough for a genuine mean reversion to play out.
   Short enough to avoid labelling setups that "eventually" worked
   but were actually failures in practice (too slow = not useful).
   Changed from 10 to give setups more room to develop.

VOLUME CLASSIFIER LABELS:
   Same logic applied specifically to volume patterns.
   We label a volume pattern as ACCUMULATION if the stock
   rallied to the LinReg mean within 15 days.
   We label it DISTRIBUTION if the stock dropped to LinReg
   from above within 15 days.

DATA REQUIREMENT:
   We need at least 200 candles of history for LinReg calculation
   PLUS 15 forward days for labelling.
   So minimum data per ticker = 215 candles.
   With 365 days of history we have plenty.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_ml_logger
from utils.error_handler import graceful, MLError

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = _load_config()
ML_CFG = config["ml"]

FORWARD_DAYS  = ML_CFG["label_forward_days"]    # 15 days
LINREG_PERIOD = config["linreg"]["period"]       # 200 candles
LONG_SD_MIN   = config["scanner"]["long_entry_sd_min"]   # -1
LONG_SD_MAX   = config["scanner"]["long_entry_sd_max"]   # -3
SHORT_SD_MIN  = config["scanner"]["short_entry_sd_min"]  # +1
SHORT_SD_MAX  = config["scanner"]["short_entry_sd_max"]  # +3


# =============================================================================
# LINREG HELPER
# We recompute LinReg values inline here for labelling purposes.
# This avoids importing the full engine and keeps the labeller self-contained.
# =============================================================================

def _compute_linreg_value(closes: np.ndarray) -> float:
    """
    Compute the LinReg value for the last candle in a price series.
    Used to check if forward prices crossed the LinReg line.

    Args:
        closes: numpy array of closing prices

    Returns:
        LinReg value at the last candle
    """
    n           = len(closes)
    x           = np.arange(n)
    slope, intercept = np.polyfit(x, closes, 1)
    return slope * (n - 1) + intercept


# =============================================================================
# CORE LABELLING FUNCTION — PER TICKER
# =============================================================================

@graceful(default_return=pd.DataFrame(), exceptions=(Exception,), log_level="warning")
def label_ticker(
    ticker    : str,
    df        : pd.DataFrame,
    direction : str = "long",
) -> pd.DataFrame:
    """
    Generate historical labels for a single ticker.

    FLOW:
    1. Ensure enough data exists (LINREG_PERIOD + FORWARD_DAYS minimum)
    2. Slide a window across all valid historical candles
    3. At each candle T:
       a. Compute LinReg value using last 200 closes up to T
       b. Compute SD position of price at T
       c. Check if price is in the entry zone (long or short)
       d. If yes → look forward 15 days
       e. Check if price crossed the LinReg line in those 15 days
       f. Label = 1 if crossed, 0 if not
    4. Return DataFrame of all labelled observations

    Args:
        ticker   : Ticker symbol
        df       : Full OHLCV DataFrame sorted date ascending
        direction: 'long' or 'short'

    Returns:
        DataFrame with columns:
        [ticker, date, direction, sd_position, volume_signal, label]
        One row per valid historical setup found
    """
    min_required = LINREG_PERIOD + FORWARD_DAYS
    if len(df) < min_required:
        logger.debug(
            f"{ticker} | Insufficient data for labelling: "
            f"{len(df)} rows, need {min_required}"
        )
        return pd.DataFrame()

    closes  = df["close"].values
    dates   = df["date"].values
    n       = len(closes)
    labels  = []

    # ── Slide window across all valid positions ───────────────────────────────
    # Start at LINREG_PERIOD (need 200 candles to compute LinReg)
    # Stop at n - FORWARD_DAYS (need 15 forward candles for labelling)
    for t in range(LINREG_PERIOD, n - FORWARD_DAYS):

        # ── Step 1: Compute LinReg value at time T ────────────────────────────
        window_closes  = closes[t - LINREG_PERIOD : t]
        linreg_value   = _compute_linreg_value(window_closes)

        # ── Step 2: Compute standard deviation of residuals ───────────────────
        x              = np.arange(LINREG_PERIOD)
        slope, intcpt  = np.polyfit(x, window_closes, 1)
        fitted         = slope * x + intcpt
        residuals      = window_closes - fitted
        std_dev        = np.std(residuals, ddof=1)

        if std_dev == 0:
            continue

        # ── Step 3: Compute SD position at time T ─────────────────────────────
        current_close  = closes[t]
        sd_position    = (current_close - linreg_value) / std_dev

        # ── Step 4: Check if price is in the entry zone ───────────────────────
        if direction == "long":
            # Long zone: between -1 and -3 SD (price below LinReg)
            in_zone = LONG_SD_MAX <= sd_position <= LONG_SD_MIN

        else:  # short
            # Short zone: between +1 and +3 SD (price above LinReg)
            in_zone = SHORT_SD_MIN <= sd_position <= SHORT_SD_MAX

        if not in_zone:
            continue   # Price not in entry zone at time T — skip

        # ── Step 5: Look forward 15 days ──────────────────────────────────────
        forward_closes = closes[t + 1 : t + 1 + FORWARD_DAYS]

        if direction == "long":
            # Success: price closes ABOVE LinReg at any point in 15 days
            # We use the LinReg value at time T as the target
            # (slightly conservative — LinReg will have moved forward
            #  but this keeps labelling consistent and leakage-free)
            label = int(np.any(forward_closes >= linreg_value))

        else:  # short
            # Success: price closes BELOW LinReg at any point in 15 days
            label = int(np.any(forward_closes <= linreg_value))

        # ── Step 6: Store the labelled observation ────────────────────────────
        labels.append({
            "ticker"     : ticker,
            "date"       : dates[t],
            "direction"  : direction,
            "sd_position": round(sd_position, 4),
            "label"      : label,
        })

    if not labels:
        logger.debug(f"{ticker} | No valid setups found for labelling")
        return pd.DataFrame()

    result = pd.DataFrame(labels)

    success_rate = result["label"].mean() * 100
    logger.debug(
        f"{ticker} | {direction.upper()} | "
        f"Labelled: {len(result)} | "
        f"Success rate: {success_rate:.1f}%"
    )

    return result


# =============================================================================
# VOLUME PATTERN LABELLER
# Specifically for the Volume Classifier model.
# Labels each historical volume pattern as accumulation or distribution
# based on whether price subsequently mean reverted.
# =============================================================================

@graceful(default_return=pd.DataFrame(), exceptions=(Exception,), log_level="warning")
def label_volume_patterns(
    ticker : str,
    df     : pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate volume pattern labels for the Volume Classifier.

    LOGIC:
    At each candle T where price is in either the long or short zone:
    1. Extract volume features over the lookback window
    2. Look forward 15 days
    3. If price reached LinReg mean → label = 'accumulation' (for longs)
       or 'distribution' (for shorts)
    4. Otherwise → label = 'neutral'

    This gives the Volume Classifier a ground truth to learn from:
    "Which volume patterns actually preceded successful mean reversions?"

    Args:
        ticker: Ticker symbol
        df    : Full OHLCV DataFrame sorted date ascending

    Returns:
        DataFrame with volume features and labels
        [ticker, date, direction, vol_slope, vol_ratio,
         green_red_ratio, label]
    """
    from engines.volume import (
        _is_volume_declining_on_down_days,
        _is_shakeout_present,
        _is_volume_expanding_on_green_days,
        _is_volume_dryup_present,
        _is_volume_rising_on_up_days,
        _is_volume_expanding_on_red_days,
    )

    AVG_VOL_PERIOD = config["volume"]["avg_volume_period"]   # 20
    LOOKBACK       = config["volume"]["lookback_days"]        # 10
    min_required   = LINREG_PERIOD + FORWARD_DAYS + AVG_VOL_PERIOD

    if len(df) < min_required:
        return pd.DataFrame()

    closes = df["close"].values
    n      = len(df)
    labels = []

    for t in range(LINREG_PERIOD, n - FORWARD_DAYS):

        # ── Compute LinReg at T ───────────────────────────────────────────────
        window_closes = closes[t - LINREG_PERIOD : t]
        linreg_value  = _compute_linreg_value(window_closes)

        x             = np.arange(LINREG_PERIOD)
        slope, intcpt = np.polyfit(x, window_closes, 1)
        fitted        = slope * x + intcpt
        residuals     = window_closes - fitted
        std_dev       = np.std(residuals, ddof=1)

        if std_dev == 0:
            continue

        current_close = closes[t]
        sd_position   = (current_close - linreg_value) / std_dev

        # ── Determine direction ───────────────────────────────────────────────
        in_long_zone  = LONG_SD_MAX  <= sd_position <= LONG_SD_MIN
        in_short_zone = SHORT_SD_MIN <= sd_position <= SHORT_SD_MAX

        if not (in_long_zone or in_short_zone):
            continue

        direction = "long" if in_long_zone else "short"

        # ── Extract volume features at T ──────────────────────────────────────
        # Use a slice of the DataFrame up to time T
        df_slice   = df.iloc[max(0, t - AVG_VOL_PERIOD - LOOKBACK) : t].copy()
        avg_volume = df_slice["volume"].tail(AVG_VOL_PERIOD).mean()

        if avg_volume == 0:
            continue

        # Compute all volume condition booleans
        cond1 = _is_volume_declining_on_down_days(df_slice, avg_volume)
        cond2 = _is_shakeout_present(df_slice, avg_volume)
        cond3 = _is_volume_expanding_on_green_days(df_slice, avg_volume)
        cond4 = _is_volume_dryup_present(df_slice, avg_volume)
        dist1 = _is_volume_rising_on_up_days(df_slice, avg_volume)
        dist2 = _is_volume_expanding_on_red_days(df_slice, avg_volume)

        # ── Label based on forward price action ───────────────────────────────
        forward_closes = closes[t + 1 : t + 1 + FORWARD_DAYS]

        if direction == "long":
            label = int(np.any(forward_closes >= linreg_value))
        else:
            label = int(np.any(forward_closes <= linreg_value))

        labels.append({
            "ticker"    : ticker,
            "date"      : df["date"].values[t],
            "direction" : direction,
            "sd_position": round(sd_position, 4),
            "cond1"     : int(cond1),
            "cond2"     : int(cond2),
            "cond3"     : int(cond3),
            "cond4"     : int(cond4),
            "dist1"     : int(dist1),
            "dist2"     : int(dist2),
            "avg_volume": round(avg_volume, 0),
            "label"     : label,
        })

    if not labels:
        return pd.DataFrame()

    return pd.DataFrame(labels)


# =============================================================================
# BATCH LABELLER
# Runs labelling across all tickers and combines results
# =============================================================================

def run_labeller(
    tickers_data : dict[str, pd.DataFrame],
    direction    : str = "both",
) -> pd.DataFrame:
    """
    Run labelling for all tickers and combine results.

    FLOW:
    1. Iterate over all tickers
    2. Run label_ticker() for long and/or short direction
    3. Collect all labelled observations
    4. Check minimum sample threshold
    5. Return combined DataFrame

    Args:
        tickers_data: Dict mapping ticker → OHLCV DataFrame
        direction   : 'long', 'short', or 'both'

    Returns:
        Combined labelled DataFrame ready for feature engineering
    """
    logger.info(
        f"Labeller starting | "
        f"{len(tickers_data)} tickers | "
        f"Direction: {direction} | "
        f"Forward days: {FORWARD_DAYS}"
    )

    all_labels  = []
    directions  = ["long", "short"] if direction == "both" else [direction]

    for ticker, df in tickers_data.items():
        for d in directions:
            result = label_ticker(ticker, df, d)
            if not result.empty:
                all_labels.append(result)

    if not all_labels:
        logger.warning("Labeller: No labels generated")
        return pd.DataFrame()

    combined     = pd.concat(all_labels, ignore_index=True)
    total        = len(combined)
    success_rate = combined["label"].mean() * 100
    min_samples  = ML_CFG["min_training_samples"]   # 500

    logger.info(
        f"Labeller complete | "
        f"Total observations: {total} | "
        f"Overall success rate: {success_rate:.1f}% | "
        f"Min required: {min_samples}"
    )

    if total < min_samples:
        logger.warning(
            f"Labeller: Only {total} samples — "
            f"below minimum of {min_samples}. "
            f"Models will not train until more data is accumulated."
        )

    return combined


# =============================================================================
# VOLUME PATTERN BATCH LABELLER
# =============================================================================

def run_volume_labeller(
    tickers_data : dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Run volume pattern labelling for all tickers.

    Args:
        tickers_data: Dict mapping ticker → OHLCV DataFrame

    Returns:
        Combined volume pattern labelled DataFrame
    """
    logger.info(
        f"Volume labeller starting | "
        f"{len(tickers_data)} tickers | "
        f"Forward days: {FORWARD_DAYS}"
    )

    all_labels = []

    for ticker, df in tickers_data.items():
        result = label_volume_patterns(ticker, df)
        if not result.empty:
            all_labels.append(result)

    if not all_labels:
        logger.warning("Volume labeller: No labels generated")
        return pd.DataFrame()

    combined     = pd.concat(all_labels, ignore_index=True)
    success_rate = combined["label"].mean() * 100

    logger.info(
        f"Volume labeller complete | "
        f"Total: {len(combined)} | "
        f"Success rate: {success_rate:.1f}%"
    )

    return combined