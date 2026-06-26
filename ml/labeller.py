"""
ml/labeller.py
--------------
Programmatic label generation for both ML models.

LOGICAL FLOW:
─────────────
We never manually label data. Instead we let historical
price action tell us the answer by looking FORWARD in time.

LABEL 1 — Volume Classifier labels:
   For every historical instance where price was in the
   accumulation/distribution zone (between ±1 and ±3 SD),
   we look forward N days and ask:

   "Did price reverse toward the LinReg mean?"

   If price was below LinReg (potential long) and moved UP
   toward the mean → label = 1 (true accumulation)
   If price continued DOWN away from the mean → label = 0

   If price was above LinReg (potential short) and moved DOWN
   toward the mean → label = 1 (true distribution)
   If price continued UP away from the mean → label = 0

LABEL 2 — Signal Ranker labels:
   For every historical scanner hit (stock that met ALL
   scanner criteria on a given day), we look forward N days
   and ask:

   "Did price reach the LinReg mean within N days?"

   Yes → label = 1 (successful setup)
   No  → label = 0 (failed setup)

   This is what the Signal Ranker learns to predict —
   given all the features of a setup, what is the
   probability it succeeds?

WHY PROGRAMMATIC LABELLING WORKS:
   We have years of daily data for ~1,500 stocks.
   Each stock has hundreds of trading days.
   Many of those days had price in the SD zones.
   That gives us tens of thousands of labelled examples
   without any manual annotation.
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

FORWARD_DAYS    = ML_CFG["label_forward_periods"]   # 1 trading period to check
SCANNER_CFG     = config["scanner"]
LONG_SD_MIN     = SCANNER_CFG["long_entry_sd_min"]   # -1
LONG_SD_MAX     = SCANNER_CFG["long_entry_sd_max"]   # -3
SHORT_SD_MIN    = SCANNER_CFG["short_entry_sd_min"]  # +1
SHORT_SD_MAX    = SCANNER_CFG["short_entry_sd_max"]  # +3


# =============================================================================
# VOLUME CLASSIFIER LABELLER
# Labels historical volume patterns as true accumulation or distribution
# =============================================================================

def label_volume_patterns(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate labels for the Volume Classifier.

    FLOW:
    1. For each ticker, get its full price and indicator history
    2. Find all dates where price was in the entry zone (±1 to ±3 SD)
    3. For each such date, look forward FORWARD_DAYS candles
    4. Check if price moved toward the LinReg mean
    5. Assign label 1 (moved toward mean) or 0 (moved away)
    6. Return DataFrame of (ticker, date, label, direction)

    WHAT "MOVED TOWARD MEAN" MEANS:
    For a long zone (price below LinReg):
    - We measure price distance from LinReg on entry day
    - We measure price distance from LinReg N days later
    - If distance DECREASED → price moved toward mean → label = 1

    For a short zone (price above LinReg):
    - Same logic but price should move DOWN toward mean
    - If distance DECREASED → label = 1

    Args:
        prices_df    : Full OHLCV DataFrame with columns
                       [ticker, date, open, high, low, close, volume]
        indicators_df: Full indicator results DataFrame with columns
                       [ticker, date, linreg_value, price_sd_position, ...]

    Returns:
        DataFrame with columns [ticker, date, direction, label]
    """
    logger.info("Generating Volume Classifier labels...")

    all_labels = []

    for ticker in indicators_df["ticker"].unique():

        # ── Get this ticker's indicator and price history ──────────────────
        ind = indicators_df[indicators_df["ticker"] == ticker].sort_values("date")
        px  = prices_df[prices_df["ticker"] == ticker].sort_values("date")

        if len(ind) < FORWARD_DAYS + 1:
            continue

        # ── Merge price and indicator data on date ────────────────────────
        merged = pd.merge(
            ind[["date", "linreg_value", "price_sd_position"]],
            px[["date", "close"]],
            on="date",
            how="inner"
        ).sort_values("date").reset_index(drop=True)

        if len(merged) < FORWARD_DAYS + 1:
            continue

        # ── Find all dates where price was in a valid entry zone ──────────
        for i in range(len(merged) - FORWARD_DAYS):
            row        = merged.iloc[i]
            sd_pos     = row["price_sd_position"]
            linreg_val = row["linreg_value"]
            close      = row["close"]
            date       = row["date"]

            # Determine if this is a long or short zone
            in_long_zone  = LONG_SD_MAX  <= sd_pos <= LONG_SD_MIN   # -3 to -1
            in_short_zone = SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX  # +1 to +3

            if not in_long_zone and not in_short_zone:
                continue

            direction = "long" if in_long_zone else "short"

            # ── Measure distance from LinReg on entry day ─────────────────
            entry_distance = abs(close - linreg_val)

            # ── Look forward FORWARD_DAYS candles ─────────────────────────
            future_rows    = merged.iloc[i + 1 : i + FORWARD_DAYS + 1]
            future_closes  = future_rows["close"].values
            future_linregs = future_rows["linreg_value"].values

            # ── Measure minimum distance from LinReg in forward window ─────
            # Min distance = best case price got closest to mean
            future_distances = np.abs(future_closes - future_linregs)
            min_future_dist  = future_distances.min()

            # ── Label: did price move closer to LinReg? ───────────────────
            # Label 1 = price moved toward mean (true accumulation/distribution)
            # Label 0 = price moved further away (false signal)
            label = 1 if min_future_dist < entry_distance else 0

            all_labels.append({
                "ticker"    : ticker,
                "date"      : date,
                "direction" : direction,
                "label"     : label,
            })

    result = pd.DataFrame(all_labels)

    if result.empty:
        logger.warning("Volume Classifier labeller: No labels generated")
        return result

    # Log label balance
    pos = (result["label"] == 1).sum()
    neg = (result["label"] == 0).sum()
    logger.info(
        f"Volume Classifier labels generated | "
        f"Total: {len(result)} | "
        f"Positive (1): {pos} ({pos/len(result)*100:.1f}%) | "
        f"Negative (0): {neg} ({neg/len(result)*100:.1f}%)"
    )

    return result


# =============================================================================
# SIGNAL RANKER LABELLER
# Labels historical scanner hits as successful or failed setups
# =============================================================================

def label_scanner_hits(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
    scan_hits_df : pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate labels for the Signal Ranker.

    FLOW:
    1. For each historical scanner hit (ticker + date + direction)
    2. Get the LinReg value on that date
    3. Look forward FORWARD_DAYS candles
    4. For longs: check if any candle's HIGH reached the LinReg value
       For shorts: check if any candle's LOW reached the LinReg value
    5. Label 1 = price reached LinReg mean, Label 0 = did not reach

    WHY HIGH/LOW INSTEAD OF CLOSE:
    We use high for longs because price reaching the LinReg on an
    intraday basis counts as the target being hit — even if it
    closed below. This is realistic for a take-profit scenario.

    Args:
        prices_df    : Full OHLCV DataFrame
        indicators_df: Full indicator results DataFrame
        scan_hits_df : Historical scanner results with columns
                       [ticker, date, direction]
                       (from scan_results table in SQLite)

    Returns:
        DataFrame with columns [ticker, date, direction, label]
    """
    logger.info("Generating Signal Ranker labels...")

    all_labels = []

    for _, hit in scan_hits_df.iterrows():
        ticker    = hit["ticker"]
        date      = hit["date"] if "date" in hit else hit["scan_date"]
        direction = hit["direction"]

        # ── Get LinReg value on the signal date ───────────────────────────
        ind_row = indicators_df[
            (indicators_df["ticker"] == ticker) &
            (indicators_df["date"]   == date)
        ]

        if ind_row.empty:
            continue

        linreg_val = ind_row.iloc[0]["linreg_value"]

        # ── Get price data from signal date forward ────────────────────────
        px = prices_df[
            (prices_df["ticker"] == ticker) &
            (prices_df["date"]   >  date)
        ].sort_values("date").head(FORWARD_DAYS)

        if len(px) < 1:
            continue

        # ── Check if price reached LinReg within forward window ───────────
        if direction == "long":
            # For longs: price needs to move UP to reach LinReg
            # Check if any candle's HIGH reached or exceeded LinReg value
            reached = bool((px["high"] >= linreg_val).any())
        else:
            # For shorts: price needs to move DOWN to reach LinReg
            # Check if any candle's LOW reached or went below LinReg value
            reached = bool((px["low"] <= linreg_val).any())

        label = 1 if reached else 0

        all_labels.append({
            "ticker"    : ticker,
            "date"      : date,
            "direction" : direction,
            "label"     : label,
        })

    result = pd.DataFrame(all_labels)

    if result.empty:
        logger.warning("Signal Ranker labeller: No labels generated")
        return result

    pos = (result["label"] == 1).sum()
    neg = (result["label"] == 0).sum()
    logger.info(
        f"Signal Ranker labels generated | "
        f"Total: {len(result)} | "
        f"Positive (1): {pos} ({pos/len(result)*100:.1f}%) | "
        f"Negative (0): {neg} ({neg/len(result)*100:.1f}%)"
    )

    return result