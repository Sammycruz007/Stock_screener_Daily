"""
ml/labeller.py
--------------
Programmatic label generation for both ML models.

LOGICAL FLOW:
─────────────
We never manually label data. Instead we derive labels from
historical price/volume patterns.

LABEL 1 — Volume Classifier labels (OPTION A):
   Labels are based on VOLUME PATTERN STRUCTURE, not price outcome.
   We ask: "Does this volume pattern show true accumulation/distribution
   characteristics?" using the same features the model will be trained on.

   TRUE ACCUMULATION (label=1) requires ALL of:
   - Volume declining on red candles (sellers exhausted)
   - Volume expanding on green candles (buyers stepping in)
   - At least one volume dry-up candle (capitulation)
   - Price recovery on high-volume down days (absorption)

   TRUE DISTRIBUTION (label=1) requires ALL of:
   - Volume declining on green candles (buyers exhausted)
   - Volume expanding on red candles (sellers stepping in)
   - No dry-up candle (sustained selling pressure)
   - Poor price recovery on high-volume down days

   EVERYTHING ELSE → label=0

   This directly matches what the Volume Classifier is supposed
   to learn — not price outcomes, but volume pattern quality.

LABEL 2 — Signal Ranker labels:
   For every historical scanner hit (stock that met scanner
   criteria on a given day), we look forward FORWARD_PERIODS candles
   and ask: "Did price CLOSE at or beyond the LinReg mean?"

   Uses CLOSE price (not high/low intraday touch) against a
   FIXED LinReg value (snapshot at signal date, not rolling future).
   This prevents the near-universal positive labelling problem.

   Yes → label = 1 (successful setup)
   No  → label = 0 (failed setup)

WHY SEPARATE LABEL LOGIC:
   Volume Classifier: learns volume pattern quality → structural signal
   Signal Ranker:     learns setup outcome → probability of success
   These are different questions requiring different labelling approaches.
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

# label_forward_periods takes priority over legacy label_forward_days
FORWARD_PERIODS = ML_CFG.get(
    "label_forward_periods",
    ML_CFG.get("label_forward_days", 26)
)

SCANNER_CFG  = config["scanner"]
LONG_SD_MIN  = SCANNER_CFG["long_entry_sd_min"]   # -1
LONG_SD_MAX  = SCANNER_CFG["long_entry_sd_max"]   # -3
SHORT_SD_MIN = SCANNER_CFG["short_entry_sd_min"]  # +1
SHORT_SD_MAX = SCANNER_CFG["short_entry_sd_max"]  # +3

VOLUME_CFG      = config["volume"]
AVG_VOL_PERIOD  = VOLUME_CFG.get("avg_volume_period", 20)  # 20 candles baseline
LOOKBACK        = VOLUME_CFG.get("lookback_days", 26)       # pattern window


# =============================================================================
# VOLUME CLASSIFIER LABELLER — OPTION A
# Labels based on volume pattern structure, NOT price outcome
# =============================================================================

def _is_accumulation(recent: pd.DataFrame, avg_volume: float) -> bool:
    """
    Determine if a window of candles shows true accumulation.

    TRUE ACCUMULATION requires ALL of:
    1. Volume DECLINING on red candles  (sellers exhausted)
    2. Volume EXPANDING on green candles (buyers stepping in)
    3. At least one volume dry-up candle (capitulation signal)
    4. Price recovery > 0.4 on at least one high-volume down day
       (institutions absorbing selling pressure)

    Args:
        recent    : Recent OHLCV candles (lookback window)
        avg_volume: Baseline average volume for this ticker

    Returns:
        True if pattern qualifies as true accumulation
    """
    green = recent[recent["close"] >= recent["open"]]
    red   = recent[recent["close"] <  recent["open"]]

    # Need enough candles of each type to measure slopes
    if len(red) < 2 or len(green) < 2:
        return False

    # ── Condition 1: Volume declining on red candles ──────────────────────────
    x_red     = np.arange(len(red))
    slope_red, _ = np.polyfit(x_red, red["volume"].values, 1)
    vol_declining_on_red = slope_red < 0   # negative slope = declining

    # ── Condition 2: Volume expanding on green candles ────────────────────────
    x_green   = np.arange(len(green))
    slope_green, _ = np.polyfit(x_green, green["volume"].values, 1)
    vol_expanding_on_green = slope_green > 0   # positive slope = expanding

    # ── Condition 3: At least one dry-up candle ───────────────────────────────
    dryup_threshold = avg_volume * 0.5
    has_dryup       = bool((recent["volume"] < dryup_threshold).any())

    # ── Condition 4: Price recovery on high-volume down days ──────────────────
    high_vol_red = red[red["volume"] > avg_volume * 1.5]
    if len(high_vol_red) > 0:
        row          = high_vol_red.loc[high_vol_red["volume"].idxmax()]
        candle_range = row["high"] - row["low"]
        recovery     = (row["close"] - row["low"]) / candle_range if candle_range > 0 else 0
        good_recovery = recovery > 0.40   # price recovered more than 40% of range
    else:
        # No high-volume down day — neutral, don't penalise
        good_recovery = True

    return bool(
        vol_declining_on_red and
        vol_expanding_on_green and
        has_dryup and
        good_recovery
    )


def _is_distribution(recent: pd.DataFrame, avg_volume: float) -> bool:
    """
    Determine if a window of candles shows true distribution.

    TRUE DISTRIBUTION requires ALL of:
    1. Volume DECLINING on green candles (buyers exhausted)
    2. Volume EXPANDING on red candles  (sellers stepping in)
    3. NO dry-up candle present         (sustained selling pressure)
    4. Poor price recovery (< 0.35) on high-volume down days
       (no absorption — sellers in control)

    Args:
        recent    : Recent OHLCV candles (lookback window)
        avg_volume: Baseline average volume for this ticker

    Returns:
        True if pattern qualifies as true distribution
    """
    green = recent[recent["close"] >= recent["open"]]
    red   = recent[recent["close"] <  recent["open"]]

    if len(red) < 2 or len(green) < 2:
        return False

    # ── Condition 1: Volume declining on green candles ────────────────────────
    x_green   = np.arange(len(green))
    slope_green, _ = np.polyfit(x_green, green["volume"].values, 1)
    vol_declining_on_green = slope_green < 0

    # ── Condition 2: Volume expanding on red candles ──────────────────────────
    x_red     = np.arange(len(red))
    slope_red, _ = np.polyfit(x_red, red["volume"].values, 1)
    vol_expanding_on_red = slope_red > 0

    # ── Condition 3: No dry-up (sustained selling, not capitulation) ──────────
    dryup_threshold = avg_volume * 0.5
    no_dryup        = not bool((recent["volume"] < dryup_threshold).any())

    # ── Condition 4: Poor price recovery on high-volume down days ─────────────
    high_vol_red = red[red["volume"] > avg_volume * 1.5]
    if len(high_vol_red) > 0:
        row          = high_vol_red.loc[high_vol_red["volume"].idxmax()]
        candle_range = row["high"] - row["low"]
        recovery     = (row["close"] - row["low"]) / candle_range if candle_range > 0 else 0
        poor_recovery = recovery < 0.35   # price barely recovered
    else:
        # No high-volume down day — neutral, don't penalise
        poor_recovery = True

    return bool(
        vol_declining_on_green and
        vol_expanding_on_red and
        no_dryup and
        poor_recovery
    )


def label_volume_patterns(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate labels for the Volume Classifier using Option A:
    volume pattern structure, not price outcome.

    FLOW:
    1. For each ticker, get its full price and indicator history
    2. Find all dates where price was in the entry zone (±1 to ±3 SD)
    3. For each such date, extract the lookback window of candles
    4. Apply _is_accumulation() or _is_distribution() based on direction
    5. Assign label 1 (true pattern) or 0 (false/unclear pattern)
    6. Return DataFrame of (ticker, date, direction, label)

    DIRECTION:
    - In long zone (-1 to -3 SD) → check for accumulation
    - In short zone (+1 to +3 SD) → check for distribution

    Args:
        prices_df    : Full OHLCV DataFrame [ticker, date, open, high, low, close, volume]
        indicators_df: Indicator results [ticker, date, linreg_value, price_sd_position, ...]

    Returns:
        DataFrame with columns [ticker, date, direction, label]
    """
    logger.info("Generating Volume Classifier labels (Option A — pattern structure)...")

    all_labels = []

    for ticker in indicators_df["ticker"].unique():

        ind = indicators_df[
            indicators_df["ticker"] == ticker
        ].sort_values("date").reset_index(drop=True)

        px = prices_df[
            prices_df["ticker"] == ticker
        ].sort_values("date").reset_index(drop=True)

        if len(ind) < 1 or len(px) < AVG_VOL_PERIOD + LOOKBACK:
            continue

        for _, row in ind.iterrows():
            sd_pos = float(row.get("price_sd_position", 0))
            date   = str(row["date"])

            in_long_zone  = LONG_SD_MAX  <= sd_pos <= LONG_SD_MIN   # -3 to -1
            in_short_zone = SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX  # +1 to +3

            if not in_long_zone and not in_short_zone:
                continue

            direction = "long" if in_long_zone else "short"

            # ── Get price data UP TO signal date only (no lookahead) ──────────
            past_px = px[px["date"] <= date].tail(AVG_VOL_PERIOD + LOOKBACK)

            if len(past_px) < AVG_VOL_PERIOD:
                continue

            # ── Baseline average volume ────────────────────────────────────────
            avg_volume = past_px["volume"].tail(AVG_VOL_PERIOD).mean()
            if avg_volume == 0:
                continue

            # ── Lookback window for pattern detection ──────────────────────────
            recent = past_px.tail(LOOKBACK)

            # ── Apply structural pattern check ────────────────────────────────
            if direction == "long":
                label = 1 if _is_accumulation(recent, avg_volume) else 0
            else:
                label = 1 if _is_distribution(recent, avg_volume) else 0

            all_labels.append({
                "ticker"   : ticker,
                "date"     : date,
                "direction": direction,
                "label"    : label,
            })

    result = pd.DataFrame(all_labels)

    if result.empty:
        logger.warning("Volume Classifier labeller: No labels generated")
        return result

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
# Labels historical scanner hits as successful or failed setups.
# Uses FIXED LinReg snapshot + CLOSE price to prevent universal positives.
# =============================================================================

def label_scanner_hits(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
    scan_hits_df : pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate labels for the Signal Ranker. (OPTIMIZED FOR SPEED)
    """
    logger.info("Generating Signal Ranker labels...")

    if scan_hits_df.empty:
        logger.warning("Signal Ranker labeller: No scan hits provided")
        return pd.DataFrame()

    # 1. BULK LOOKUP: Merge indicators to get the fixed LinReg value in one shot
    # This replaces the slow pandas filtering inside the loop
    hits_with_linreg = pd.merge(
        scan_hits_df,
        indicators_df[['ticker', 'date', 'linreg_value']],
        on=['ticker', 'date'],
        how='inner'
    )

    if hits_with_linreg.empty:
        return pd.DataFrame()

    # 2. PRE-PROCESS PRICES: Group by ticker into fast NumPy arrays
    # This prevents scanning the entire prices_df for every single hit
    logger.info("Pre-processing price data for fast lookup...")
    price_dict = {}
    
    # Ensure prices are sorted once globally
    sorted_prices = prices_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    for ticker, group in sorted_prices.groupby("ticker"):
        price_dict[ticker] = {
            "dates": group["date"].values,
            "closes": group["close"].values
        }

    all_labels = []

    # 3. FAST ITERATION: Use itertuples (faster than iterrows) and np.searchsorted
    logger.info("Evaluating forward returns...")
    for row in hits_with_linreg.itertuples():
        ticker = row.ticker
        date = str(row.date)
        direction = row.direction
        linreg_at_signal = float(row.linreg_value)

        if ticker not in price_dict:
            continue

        t_dates  = price_dict[ticker]["dates"]
        t_closes = price_dict[ticker]["closes"]

        # O(log n) lookup for the date index
        idx = np.searchsorted(t_dates, date, side="right")

        # Slice the next FORWARD_PERIODS directly from the NumPy array
        future_closes = t_closes[idx : idx + FORWARD_PERIODS]

        if len(future_closes) < 1:
            continue

        # Check if CLOSE reached the FIXED LinReg target using vectorized np.any
        if direction == "long":
            reached = bool(np.any(future_closes >= linreg_at_signal))
        else:
            reached = bool(np.any(future_closes <= linreg_at_signal))

        all_labels.append({
            "ticker"   : ticker,
            "date"     : date,
            "direction": direction,
            "label"    : 1 if reached else 0,
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
