"""
engines/volume.py
-----------------
Volume analysis engine for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This engine runs AFTER the LinReg engine so it knows which SD zone
each ticker is currently in. It analyses volume behaviour to determine
whether price is being accumulated (quietly bought) or distributed
(quietly sold) near the LinReg bands.

The engine checks FOUR conditions and combines them into a final signal:

CONDITION 1 — Volume declining on down days (drying up):
   We look at red candles (close < open) over the last N days.
   If the volume on those down days is TRENDING LOWER, selling
   pressure is exhausting itself. Nobody left to sell = price ready to bounce.
   This is the most important accumulation signal.

CONDITION 2 — Shakeout detection (high volume down day + price holds):
   A single very high volume down day (> 1.5x average) where price
   DOESN'T close near the lows (close > midpoint of candle range).
   This is the classic "shakeout" — big money selling to retail,
   then buying all that supply back. Price holds = demand absorbed the selling.

CONDITION 3 — Volume expanding on green days:
   We look at green candles (close > open) near the SD zone.
   If volume is HIGHER on up days than on down days, buyers are
   more aggressive than sellers = accumulation signature.

CONDITION 4 — Volume dry-up candle:
   At least one candle in the lookback window with volume that is
   significantly BELOW average (< 0.5x average volume).
   This "quiet" candle means the selling has completely stopped.
   Often the last candle before a bounce.

FINAL SIGNAL LOGIC:
   ACCUMULATION : Conditions 1 + 3 both true, OR conditions 2 + 4 both true
   DISTRIBUTION : Mirror conditions — rising volume on down days, etc.
   NEUTRAL      : Neither accumulation nor distribution clearly present

The @graceful decorator ensures one failed ticker never crashes the pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_volume_logger
from utils.error_handler import graceful, EngineError

logger = get_volume_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config     = _load_config()
VOLUME_CFG = config["volume"]

LOOKBACK_DAYS      = VOLUME_CFG["lookback_days"]        # 10 candles
AVG_VOLUME_PERIOD  = VOLUME_CFG["avg_volume_period"]    # 20 candles

# Volume thresholds
HIGH_VOLUME_MULTIPLIER = 1.5   # Volume > 1.5x average = high volume candle
LOW_VOLUME_MULTIPLIER  = 0.5   # Volume < 0.5x average = dry-up candle
GREEN_RED_RATIO        = 1.2   # Green day avg volume must be 1.2x red day avg
                                # for the volume expansion condition to trigger


# =============================================================================
# INDIVIDUAL CONDITIONS
# Each condition returns True/False.
# They are combined in the main engine function.
# =============================================================================

def _is_volume_declining_on_down_days(
    df          : pd.DataFrame,
    avg_volume  : float,
) -> bool:
    """
    CONDITION 1: Is volume trending lower on red (down) candles?

    LOGIC:
    1. Filter to only red candles (close < open) in the lookback window
    2. Need at least 3 red candles to assess a trend
    3. Fit a simple linear trend through the volumes of those red candles
    4. If the slope is negative (volumes getting smaller) → True

    This catches the "selling pressure drying up" signature.

    Args:
        df         : Recent OHLCV DataFrame (last LOOKBACK_DAYS candles)
        avg_volume : 20-day average volume baseline

    Returns:
        True if volume is declining on down days
    """
    recent = df.tail(LOOKBACK_DAYS).copy()

    # Identify red candles (close below open = down day)
    red_candles = recent[recent["close"] < recent["open"]]

    # Need at least 3 red candles to measure a trend
    if len(red_candles) < 3:
        return False

    # Fit a linear trend through red candle volumes
    # x = sequential index, y = volume
    x = np.arange(len(red_candles))
    y = red_candles["volume"].values

    # Use numpy polyfit (degree 1 = linear)
    slope, _ = np.polyfit(x, y, 1)

    # Negative slope = volumes are declining = selling exhausting
    return bool(slope < 0)


def _is_shakeout_present(
    df          : pd.DataFrame,
    avg_volume  : float,
) -> bool:
    """
    CONDITION 2: Has a shakeout candle occurred in the lookback window?

    A SHAKEOUT candle has ALL THREE of:
    1. Volume > HIGH_VOLUME_MULTIPLIER x average (unusually high selling)
    2. It is a red candle (close < open)
    3. Close is above the midpoint of the candle's range
       (price recovered within the candle = demand absorbed the selling)

    The combination of high volume + red candle + recovery suggests
    big money sold aggressively but buyers absorbed all that supply.
    Price holding = bullish absorption.

    Args:
        df         : Recent OHLCV DataFrame
        avg_volume : 20-day average volume baseline

    Returns:
        True if at least one shakeout candle exists in the lookback window
    """
    recent = df.tail(LOOKBACK_DAYS).copy()

    for _, row in recent.iterrows():
        candle_range = row["high"] - row["low"]

        # Skip candles with no meaningful range (doji-like)
        if candle_range < 0.001:
            continue

        candle_midpoint = row["low"] + (candle_range * 0.5)

        # Check all three shakeout conditions simultaneously
        high_volume   = row["volume"] > (avg_volume * HIGH_VOLUME_MULTIPLIER)
        is_red        = row["close"] < row["open"]
        price_held    = row["close"] > candle_midpoint

        if high_volume and is_red and price_held:
            logger.debug(
                f"Shakeout candle detected | "
                f"Volume: {row['volume']:,.0f} vs avg {avg_volume:,.0f} | "
                f"Close: {row['close']:.4f} | Mid: {candle_midpoint:.4f}"
            )
            return True

    return False


def _is_volume_expanding_on_green_days(
    df         : pd.DataFrame,
    avg_volume : float,
) -> bool:
    """
    CONDITION 3: Is volume higher on green (up) days than red (down) days?

    LOGIC:
    1. Split recent candles into green (close > open) and red (close < open)
    2. Compare the AVERAGE volume of green vs red days
    3. If green avg volume > GREEN_RED_RATIO x red avg volume → True

    Higher buying volume than selling volume = buyers more aggressive = accumulation.

    Args:
        df         : Recent OHLCV DataFrame
        avg_volume : 20-day average volume baseline (used as fallback)

    Returns:
        True if green day average volume meaningfully exceeds red day average
    """
    recent = df.tail(LOOKBACK_DAYS).copy()

    green_candles = recent[recent["close"] > recent["open"]]
    red_candles   = recent[recent["close"] < recent["open"]]

    # Need at least 2 of each type to make a fair comparison
    if len(green_candles) < 2 or len(red_candles) < 2:
        return False

    avg_green_volume = green_candles["volume"].mean()
    avg_red_volume   = red_candles["volume"].mean()

    # Avoid division by zero
    if avg_red_volume == 0:
        return False

    return bool(avg_green_volume > avg_red_volume * GREEN_RED_RATIO)


def _is_volume_dryup_present(
    df         : pd.DataFrame,
    avg_volume : float,
) -> bool:
    """
    CONDITION 4: Has at least one volume dry-up candle occurred recently?

    A dry-up candle has volume significantly BELOW average:
    volume < LOW_VOLUME_MULTIPLIER x average_volume

    This is the "nobody left to sell" signal — complete exhaustion of selling.
    Often appears as the final quiet candle right before a reversal.

    Args:
        df         : Recent OHLCV DataFrame
        avg_volume : 20-day average volume baseline

    Returns:
        True if at least one dry-up candle found in lookback window
    """
    recent         = df.tail(LOOKBACK_DAYS)
    threshold      = avg_volume * LOW_VOLUME_MULTIPLIER
    dryup_candles  = recent[recent["volume"] < threshold]

    return len(dryup_candles) > 0


# =============================================================================
# DISTRIBUTION CONDITIONS (mirror of accumulation for shorts)
# =============================================================================

def _is_volume_rising_on_up_days(df: pd.DataFrame, avg_volume: float) -> bool:
    """
    DISTRIBUTION mirror of Condition 1.
    Rising volume on green candles = selling pressure building = distribution.
    """
    recent       = df.tail(LOOKBACK_DAYS).copy()
    green_candles = recent[recent["close"] > recent["open"]]

    if len(green_candles) < 3:
        return False

    x = np.arange(len(green_candles))
    y = green_candles["volume"].values
    slope, _ = np.polyfit(x, y, 1)

    # Positive slope = volumes increasing on up days = distribution
    return bool(slope > 0)


def _is_volume_expanding_on_red_days(df: pd.DataFrame, avg_volume: float) -> bool:
    """
    DISTRIBUTION mirror of Condition 3.
    Higher selling volume than buying volume = distribution.
    """
    recent        = df.tail(LOOKBACK_DAYS).copy()
    green_candles = recent[recent["close"] > recent["open"]]
    red_candles   = recent[recent["close"] < recent["open"]]

    if len(green_candles) < 2 or len(red_candles) < 2:
        return False

    avg_green_volume = green_candles["volume"].mean()
    avg_red_volume   = red_candles["volume"].mean()

    if avg_green_volume == 0:
        return False

    return bool(avg_red_volume > avg_green_volume * GREEN_RED_RATIO)


# =============================================================================
# FINAL SIGNAL COMBINER
# Combines all condition results into a single ACCUMULATION/DISTRIBUTION/NEUTRAL
# =============================================================================

def _determine_volume_signal(
    cond1_declining_down : bool,
    cond2_shakeout       : bool,
    cond3_expanding_up   : bool,
    cond4_dryup          : bool,
    dist1_rising_up      : bool,
    dist2_expanding_down : bool,
) -> str:
    """
    Combine all volume conditions into a single signal.

    ACCUMULATION requires (either):
    - Path A: Volume declining on down days AND expanding on green days
              (systematic drying up of sellers + growing buyer aggression)
    - Path B: Shakeout present AND dry-up candle present
              (absorption event followed by complete selling exhaustion)

    DISTRIBUTION requires (either):
    - Path A: Volume rising on up days AND expanding on red days
              (systematic increase in sellers + growing seller aggression)

    NEUTRAL:
    - Neither accumulation nor distribution criteria met

    Args:
        All boolean condition results

    Returns:
        'accumulation', 'distribution', or 'neutral'
    """

    # ── Check accumulation paths ──────────────────────────────────────────────
    accumulation_path_a = cond1_declining_down and cond3_expanding_up
    accumulation_path_b = cond2_shakeout       and cond4_dryup

    if accumulation_path_a or accumulation_path_b:
        return "accumulation"

    # ── Check distribution paths ──────────────────────────────────────────────
    distribution_path_a = dist1_rising_up and dist2_expanding_down

    if distribution_path_a:
        return "distribution"

    # ── Neither clearly present ───────────────────────────────────────────────
    return "neutral"


# =============================================================================
# PER-TICKER ENTRY POINT
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def compute_volume_signal(
    ticker : str,
    df     : pd.DataFrame,
    date   : str,
) -> Optional[dict]:
    """
    Run the full volume engine for a single ticker.

    FLOW:
    1. Validate sufficient data
    2. Compute 20-day average volume baseline
    3. Run all individual conditions
    4. Combine into final signal
    5. Package for database

    Args:
        ticker : Ticker symbol
        df     : Full OHLCV DataFrame, sorted date ascending
        date   : Today's date string YYYY-MM-DD

    Returns:
        Dict with volume_signal, or None on failure
    """

    # Need enough data for both lookback and average volume period
    min_required = AVG_VOLUME_PERIOD + LOOKBACK_DAYS
    if len(df) < min_required:
        logger.warning(
            f"{ticker} | Insufficient data: {len(df)} rows, need {min_required}"
        )
        return None

    # ── Step 1: Compute baseline average volume ───────────────────────────────
    avg_volume = df["volume"].tail(AVG_VOLUME_PERIOD).mean()

    # Skip tickers with zero average volume (data issue)
    if avg_volume == 0:
        logger.warning(f"{ticker} | Zero average volume — skipping")
        return None

    # ── Step 2: Run all accumulation conditions ───────────────────────────────
    cond1 = _is_volume_declining_on_down_days(df, avg_volume)
    cond2 = _is_shakeout_present(df, avg_volume)
    cond3 = _is_volume_expanding_on_green_days(df, avg_volume)
    cond4 = _is_volume_dryup_present(df, avg_volume)

    # ── Step 3: Run all distribution conditions ───────────────────────────────
    dist1 = _is_volume_rising_on_up_days(df, avg_volume)
    dist2 = _is_volume_expanding_on_red_days(df, avg_volume)

    # ── Step 4: Combine into final signal ────────────────────────────────────
    signal = _determine_volume_signal(cond1, cond2, cond3, cond4, dist1, dist2)

    # ── Step 5: Package result ────────────────────────────────────────────────
    result = {
        "ticker"       : ticker,
        "date"         : date,
        "volume_signal": signal,
    }

    logger.debug(
        f"{ticker} | Volume: {signal.upper()} | "
        f"C1:{cond1} C2:{cond2} C3:{cond3} C4:{cond4} | "
        f"D1:{dist1} D2:{dist2}"
    )

    return result


# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_volume_engine(
    tickers_data : dict[str, pd.DataFrame],
    date         : str,
) -> pd.DataFrame:
    """
    Run volume signal computation for all tickers in the filtered universe.

    Args:
        tickers_data : Dict mapping ticker → OHLCV DataFrame
        date         : Today's date string YYYY-MM-DD

    Returns:
        DataFrame with one row per ticker containing volume_signal column
    """
    logger.info(f"Volume engine starting | {len(tickers_data)} tickers | Date: {date}")

    results = []
    failed  = 0

    for ticker, df in tickers_data.items():
        result = compute_volume_signal(ticker, df, date)

        if result is not None:
            results.append(result)
        else:
            failed += 1

    logger.info(
        f"Volume engine complete | "
        f"Computed: {len(results)} | "
        f"Skipped: {failed}"
    )

    if not results:
        logger.warning("Volume engine: No results produced")
        return pd.DataFrame()

    return pd.DataFrame(results)