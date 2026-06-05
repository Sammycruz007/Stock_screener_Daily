"""
engines/smc.py
--------------
Smart Money Concepts engine for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This engine detects two things for each ticker:

THING 1 — Swing Highs and Swing Lows (structural pivots):
   A swing high is a candle whose HIGH is higher than all candles
   within a minimum window of N candles on BOTH left and right sides.
   A swing low is the mirror: lowest LOW within N candles on both sides.

   We use an ADAPTIVE approach — no fixed lookback cap:
   - Minimum 5 candles each side (filters out noise)
   - No maximum — the engine scans back until it finds the most
     recent confirmed pivot, no matter how far back that is
   - This handles both choppy markets (frequent pivots close together)
     and trending markets (pivots far apart)

THING 2 — Change of Character (CHoCH):
   In an UPTREND:
   - We track the series of Higher Lows (HL)
   - A CHoCH occurs when price CLOSES BELOW the most recent HL
   - This means the bullish structure is broken

   In a DOWNTREND:
   - We track the series of Lower Highs (LH)
   - A CHoCH occurs when price CLOSES ABOVE the most recent LH
   - This means the bearish structure is broken

   We use CANDLE CLOSE (not wicks) for confirmation.
   This is more conservative and filters out fakeout wicks.

OUTPUT per ticker:
   - smc_structure  : 'bullish', 'bearish', or 'broken'
   - choch_detected : True/False (did CHoCH happen in last N candles?)
   - swing_highs    : list of (index, price) tuples
   - swing_lows     : list of (index, price) tuples

WHY THIS MATTERS FOR THE SCANNER:
   The market/sector filter requires:
   - LinReg sloping UP + smc_structure == 'bullish' → LONG filter passes
   - LinReg sloping DOWN + smc_structure == 'bearish' → SHORT filter passes
   - choch_detected == True → filter FAILS regardless of LinReg slope
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_smc_logger
from utils.error_handler import graceful, EngineError

logger = get_smc_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config  = _load_config()
SMC_CFG = config["smc"]

MIN_PIVOT_CANDLES = SMC_CFG["min_pivot_candles"]   # 5 candles each side minimum


# =============================================================================
# SWING HIGH / LOW DETECTION
# Adaptive pivot detection — finds the most recent confirmed pivot
# with at least MIN_PIVOT_CANDLES on each side.
# =============================================================================

def _find_swing_highs(highs: np.ndarray) -> list[tuple[int, float]]:
    """
    Find all confirmed swing highs in a price series.

    A candle at index i is a swing HIGH if:
    - Its high is the highest value in the window:
      [i - MIN_PIVOT_CANDLES  ...  i  ...  i + MIN_PIVOT_CANDLES]
    - Both the left and right sides have at least MIN_PIVOT_CANDLES candles
      (so we skip the first and last MIN_PIVOT_CANDLES candles)

    Args:
        highs: numpy array of candle high prices, oldest first

    Returns:
        List of (index, price) tuples for each confirmed swing high
        Ordered oldest to newest
    """
    swing_highs = []
    n = len(highs)

    # We need MIN_PIVOT_CANDLES candles on EACH side of the candidate pivot
    # So we can only check candles from index MIN_PIVOT_CANDLES
    # to index n - MIN_PIVOT_CANDLES - 1
    for i in range(MIN_PIVOT_CANDLES, n - MIN_PIVOT_CANDLES):

        # Define the full window around this candle
        window = highs[i - MIN_PIVOT_CANDLES : i + MIN_PIVOT_CANDLES + 1]

        # This candle is a swing high if its value is the MAX in the window
        if highs[i] == np.max(window):
            swing_highs.append((i, highs[i]))

    return swing_highs


def _find_swing_lows(lows: np.ndarray) -> list[tuple[int, float]]:
    """
    Find all confirmed swing lows in a price series.
    Mirror of _find_swing_highs but uses lows and finds minimums.

    Args:
        lows: numpy array of candle low prices, oldest first

    Returns:
        List of (index, price) tuples for each confirmed swing low
        Ordered oldest to newest
    """
    swing_lows = []
    n = len(lows)

    for i in range(MIN_PIVOT_CANDLES, n - MIN_PIVOT_CANDLES):
        window = lows[i - MIN_PIVOT_CANDLES : i + MIN_PIVOT_CANDLES + 1]

        # This candle is a swing low if its value is the MIN in the window
        if lows[i] == np.min(window):
            swing_lows.append((i, lows[i]))

    return swing_lows


# =============================================================================
# TREND STRUCTURE DETECTION
# Determines if the market is making Higher Highs + Higher Lows (bullish)
# or Lower Highs + Lower Lows (bearish) based on the swing points.
# =============================================================================

def _detect_structure(
    swing_highs : list[tuple[int, float]],
    swing_lows  : list[tuple[int, float]],
) -> str:
    """
    Determine market structure from the most recent swing points.

    LOGIC:
    We look at the last 2 confirmed swing highs and last 2 swing lows.

    BULLISH structure:
    - Latest swing high > previous swing high (Higher High)
    - Latest swing low  > previous swing low  (Higher Low)

    BEARISH structure:
    - Latest swing high < previous swing high (Lower High)
    - Latest swing low  < previous swing low  (Lower Low)

    BROKEN / MIXED:
    - The above conditions are not cleanly met
    - e.g. Higher High but Lower Low = choppy / consolidating

    Args:
        swing_highs: List of (index, price) tuples, oldest first
        swing_lows : List of (index, price) tuples, oldest first

    Returns:
        'bullish', 'bearish', or 'broken'
    """

    # Need at least 2 of each to compare consecutive pivots
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "broken"  # Not enough data to determine structure

    # Get the two most recent swing highs and lows
    prev_sh, last_sh = swing_highs[-2][1], swing_highs[-1][1]
    prev_sl, last_sl = swing_lows[-2][1],  swing_lows[-1][1]

    # ── Check for bullish structure ───────────────────────────────────────────
    higher_high = last_sh > prev_sh
    higher_low  = last_sl > prev_sl

    if higher_high and higher_low:
        return "bullish"

    # ── Check for bearish structure ───────────────────────────────────────────
    lower_high = last_sh < prev_sh
    lower_low  = last_sl < prev_sl

    if lower_high and lower_low:
        return "bearish"

    # ── Mixed / transitional ──────────────────────────────────────────────────
    return "broken"


# =============================================================================
# CHoCH DETECTION
# The most critical filter in the scanner.
# A CHoCH means the trend structure has CHANGED — we do not trade in
# the direction of the previous trend until structure is re-established.
# =============================================================================

def _detect_choch(
    closes      : np.ndarray,
    swing_highs : list[tuple[int, float]],
    swing_lows  : list[tuple[int, float]],
    structure   : str,
) -> bool:
    """
    Detect if a Change of Character (CHoCH) has occurred.

    LOGIC:

    For BULLISH structure:
    - The key level is the MOST RECENT confirmed swing low (the last HL)
    - If any recent candle CLOSES BELOW this level → CHoCH = True
    - The bullish structure is broken

    For BEARISH structure:
    - The key level is the MOST RECENT confirmed swing high (the last LH)
    - If any recent candle CLOSES ABOVE this level → CHoCH = True
    - The bearish structure is broken

    We check the last MIN_PIVOT_CANDLES * 2 candles for the close violation.
    This gives a reasonable recent window without going too far back.

    Args:
        closes      : numpy array of close prices, oldest first
        swing_highs : List of (index, price) swing high tuples
        swing_lows  : List of (index, price) swing low tuples
        structure   : Current structure ('bullish', 'bearish', 'broken')

    Returns:
        True if CHoCH detected, False otherwise
    """

    # If structure is already broken, CHoCH is irrelevant
    if structure == "broken":
        return False

    # Define the recent window to check for close violations
    # We look back 2x the minimum pivot candles
    lookback = MIN_PIVOT_CANDLES * 2
    recent_closes = closes[-lookback:]

    if structure == "bullish":
        # ── Bullish CHoCH: close below most recent swing low ─────────────────
        if not swing_lows:
            return False

        # The most recent swing low is the key level protecting bullish structure
        most_recent_swing_low = swing_lows[-1][1]

        # CHoCH confirmed if ANY recent candle closes below this level
        choch = bool(np.any(recent_closes < most_recent_swing_low))

        if choch:
            logger.debug(
                f"CHoCH detected (bullish→broken) | "
                f"Key level: {most_recent_swing_low} | "
                f"Min recent close: {recent_closes.min():.4f}"
            )
        return choch

    elif structure == "bearish":
        # ── Bearish CHoCH: close above most recent swing high ─────────────────
        if not swing_highs:
            return False

        # The most recent swing high is the key level protecting bearish structure
        most_recent_swing_high = swing_highs[-1][1]

        # CHoCH confirmed if ANY recent candle closes above this level
        choch = bool(np.any(recent_closes > most_recent_swing_high))

        if choch:
            logger.debug(
                f"CHoCH detected (bearish→broken) | "
                f"Key level: {most_recent_swing_high} | "
                f"Max recent close: {recent_closes.max():.4f}"
            )
        return choch

    return False


# =============================================================================
# PER-TICKER ENTRY POINT
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def compute_smc(
    ticker : str,
    df     : pd.DataFrame,
    date   : str,
) -> Optional[dict]:
    """
    Run the full SMC engine for a single ticker.

    FLOW:
    1. Extract numpy arrays for highs, lows, closes
    2. Find all swing highs and swing lows
    3. Determine trend structure (bullish/bearish/broken)
    4. Detect CHoCH
    5. Package results for database

    Args:
        ticker : Ticker symbol
        df     : Full OHLCV DataFrame, sorted date ascending
        date   : Today's date string YYYY-MM-DD

    Returns:
        Dict with smc_structure and choch_detected, or None on failure
    """

    # Need at least 3x MIN_PIVOT_CANDLES to have meaningful pivots
    min_required = MIN_PIVOT_CANDLES * 3
    if len(df) < min_required:
        logger.warning(
            f"{ticker} | Insufficient data for SMC: {len(df)} rows, need {min_required}"
        )
        return None

    # ── Step 1: Extract price arrays ──────────────────────────────────────────
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    # ── Step 2: Find swing highs and swing lows ───────────────────────────────
    swing_highs = _find_swing_highs(highs)
    swing_lows  = _find_swing_lows(lows)

    logger.debug(
        f"{ticker} | Swing highs: {len(swing_highs)} | "
        f"Swing lows: {len(swing_lows)}"
    )

    # ── Step 3: Determine market structure ────────────────────────────────────
    structure = _detect_structure(swing_highs, swing_lows)

    # ── Step 4: Detect CHoCH ──────────────────────────────────────────────────
    choch = _detect_choch(closes, swing_highs, swing_lows, structure)

    # ── Step 5: Package results ───────────────────────────────────────────────
    result = {
        "ticker"        : ticker,
        "date"          : date,
        "smc_structure" : structure,
        "choch_detected": 1 if choch else 0,
    }

    logger.debug(
        f"{ticker} | Structure: {structure} | "
        f"CHoCH: {'YES' if choch else 'no'}"
    )

    return result


# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_smc_engine(
    tickers_data : dict[str, pd.DataFrame],
    date         : str,
) -> pd.DataFrame:
    """
    Run SMC computation for all tickers in the filtered universe.

    FLOW:
    1. Iterate over each ticker and its OHLCV DataFrame
    2. Call compute_smc() for each
    3. Collect non-None results
    4. Return combined DataFrame for database write

    Args:
        tickers_data : Dict mapping ticker → OHLCV DataFrame
        date         : Today's date string YYYY-MM-DD

    Returns:
        DataFrame with one row per ticker containing SMC columns
    """
    logger.info(f"SMC engine starting | {len(tickers_data)} tickers | Date: {date}")

    results = []
    failed  = 0

    for ticker, df in tickers_data.items():
        result = compute_smc(ticker, df, date)

        if result is not None:
            results.append(result)
        else:
            failed += 1

    logger.info(
        f"SMC engine complete | "
        f"Computed: {len(results)} | "
        f"Skipped: {failed}"
    )

    if not results:
        logger.warning("SMC engine: No results produced")
        return pd.DataFrame()

    return pd.DataFrame(results)