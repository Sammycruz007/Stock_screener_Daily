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
# BOS DETECTION (Break of Structure — trend continuation, opposite of CHoCH)
# =============================================================================

def _find_bos(
    closes      : np.ndarray,
    swing_highs : list[tuple[int, float]],
    swing_lows  : list[tuple[int, float]],
    structure   : str,
) -> Optional[tuple[int, int]]:
    """
    Detect the most recent Break of Structure (BOS) in the direction
    of the current trend.

    BULLISH structure:
    - BOS = candle CLOSES ABOVE the most recent confirmed swing high
    - The swing LOW immediately preceding that swing high is the
      "base of the BOS swing" — this is where we look for the
      demand zone engulfing candle

    BEARISH structure:
    - BOS = candle CLOSES BELOW the most recent confirmed swing low
    - The swing HIGH immediately preceding that swing low is the
      "base of the BOS swing" — supply zone search location

    Args:
        closes      : numpy array of close prices, oldest first
        swing_highs : list of (index, price) swing highs
        swing_lows  : list of (index, price) swing lows
        structure   : 'bullish', 'bearish', or 'broken'

    Returns:
        Tuple of (bos_candle_index, base_swing_index) or None if no BOS found
        base_swing_index = index of the swing point at the base of the move
    """
    if structure == "broken":
        return None

    lookback = MIN_PIVOT_CANDLES * 2
    recent_indices = range(max(0, len(closes) - lookback), len(closes))

    if structure == "bullish":
        if not swing_highs or not swing_lows:
            return None

        # Most recent swing high = the level that must be broken for BOS
        bos_level = swing_highs[-1][1]

        # Find the swing low that occurred BEFORE this swing high
        # — that's the "base" of the impulsive move
        sh_index  = swing_highs[-1][0]
        base_candidates = [sl for sl in swing_lows if sl[0] < sh_index]

        if not base_candidates:
            return None

        base_swing_index = base_candidates[-1][0]

        # Check recent candles for a close above bos_level
        for i in recent_indices:
            if closes[i] > bos_level:
                return (i, base_swing_index)

        return None

    elif structure == "bearish":
        if not swing_highs or not swing_lows:
            return None

        bos_level = swing_lows[-1][1]
        sl_index  = swing_lows[-1][0]
        base_candidates = [sh for sh in swing_highs if sh[0] < sl_index]

        if not base_candidates:
            return None

        base_swing_index = base_candidates[-1][0]

        for i in recent_indices:
            if closes[i] < bos_level:
                return (i, base_swing_index)

        return None

    return None


# =============================================================================
# ENGULFING CANDLE DETECTION
# =============================================================================

def _is_bullish_engulfing(
    opens  : np.ndarray,
    closes : np.ndarray,
    idx    : int,
) -> bool:
    """
    Check if the candle at idx (and idx-1) forms a bullish engulfing pattern.

    BULLISH ENGULFING:
    - Candle idx-1 is red (close < open)
    - Candle idx   is green (close > open)
    - Candle idx's body fully engulfs candle idx-1's body:
      open[idx] <= close[idx-1] AND close[idx] >= open[idx-1]

    Args:
        opens  : numpy array of open prices
        closes : numpy array of close prices
        idx    : index of the second (engulfing) candle

    Returns:
        True if bullish engulfing pattern confirmed at idx
    """
    if idx < 1 or idx >= len(closes):
        return False

    prev_red    = closes[idx - 1] < opens[idx - 1]
    curr_green  = closes[idx]     > opens[idx]
    engulfs     = (opens[idx] <= closes[idx - 1]) and (closes[idx] >= opens[idx - 1])

    return bool(prev_red and curr_green and engulfs)


def _is_bearish_engulfing(
    opens  : np.ndarray,
    closes : np.ndarray,
    idx    : int,
) -> bool:
    """
    Check if the candle at idx (and idx-1) forms a bearish engulfing pattern.

    BEARISH ENGULFING:
    - Candle idx-1 is green (close > open)
    - Candle idx   is red (close < open)
    - Candle idx's body fully engulfs candle idx-1's body:
      open[idx] >= close[idx-1] AND close[idx] <= open[idx-1]

    Args:
        opens  : numpy array of open prices
        closes : numpy array of close prices
        idx    : index of the second (engulfing) candle

    Returns:
        True if bearish engulfing pattern confirmed at idx
    """
    if idx < 1 or idx >= len(closes):
        return False

    prev_green = closes[idx - 1] > opens[idx - 1]
    curr_red   = closes[idx]     < opens[idx]
    engulfs    = (opens[idx] >= closes[idx - 1]) and (closes[idx] <= opens[idx - 1])

    return bool(prev_green and curr_red and engulfs)


# =============================================================================
# DEMAND / SUPPLY ZONE DETECTION
# Combines BOS + engulfing candle + SD band overlap check
# =============================================================================

def _find_demand_supply_zone(
    df          : pd.DataFrame,
    swing_highs : list[tuple[int, float]],
    swing_lows  : list[tuple[int, float]],
    structure   : str,
    sd1         : float,
    sd3         : float,
) -> Optional[dict]:
    """
    Find a valid Demand Zone (bullish) or Supply Zone (bearish) that:
    1. Is formed by a BOS in the direction of the trend
    2. Has an engulfing candle at the base of the BOS swing
    3. Overlaps the -1 to -3 SD band (demand) or +1 to +3 SD band (supply)

    SEARCH WINDOW:
    The engulfing candle is searched for in a small window around the
    base swing index (±2 candles), since the exact engulfing candle
    may be the swing candle itself or one adjacent to it.

    Args:
        df          : Full OHLCV DataFrame, sorted date ascending
        swing_highs : list of (index, price) swing highs
        swing_lows  : list of (index, price) swing lows
        structure   : 'bullish', 'bearish', or 'broken'
        sd1         : Current SD1 band value (lower for demand, upper for supply)
        sd3         : Current SD3 band value (lower for demand, upper for supply)

    Returns:
        Dict with zone details if valid zone found, else None:
        {
            "zone_type"  : "demand" or "supply",
            "zone_high"  : float,
            "zone_low"   : float,
            "candle_idx" : int,
        }
    """
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    bos_result = _find_bos(closes, swing_highs, swing_lows, structure)
    if bos_result is None:
        return None

    _, base_idx = bos_result

    # Search window around the base swing for the engulfing candle
    search_start = max(1, base_idx - 2)
    search_end   = min(len(closes), base_idx + 3)

    for idx in range(search_start, search_end):

        if structure == "bullish":
            if _is_bullish_engulfing(opens, closes, idx):
                zone_high = max(opens[idx], closes[idx-1], highs[idx], highs[idx-1])
                zone_low  = min(opens[idx], closes[idx-1], lows[idx],  lows[idx-1])

                # Demand zone must overlap the -1 to -3 SD band
                # (sd1 and sd3 passed in as lower bounds, sd1 > sd3 numerically since both negative offset)
                band_lower = min(sd1, sd3)
                band_upper = max(sd1, sd3)

                overlaps = (zone_low <= band_upper) and (zone_high >= band_lower)

                if overlaps:
                    return {
                        "zone_type" : "demand",
                        "zone_high" : float(zone_high),
                        "zone_low"  : float(zone_low),
                        "candle_idx": idx,
                    }

        elif structure == "bearish":
            if _is_bearish_engulfing(opens, closes, idx):
                zone_high = max(opens[idx-1], closes[idx], highs[idx], highs[idx-1])
                zone_low  = min(opens[idx-1], closes[idx], lows[idx],  lows[idx-1])

                band_lower = min(sd1, sd3)
                band_upper = max(sd1, sd3)

                overlaps = (zone_low <= band_upper) and (zone_high >= band_lower)

                if overlaps:
                    return {
                        "zone_type" : "supply",
                        "zone_high" : float(zone_high),
                        "zone_low"  : float(zone_low),
                        "candle_idx": idx,
                    }

    return None


# =============================================================================
# PER-TICKER ENTRY POINT
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")

def compute_smc(
    ticker      : str,
    df          : pd.DataFrame,
    date        : str,
    sd1_lower   : Optional[float] = None,
    sd3_lower   : Optional[float] = None,
    sd1_upper   : Optional[float] = None,
    sd3_upper   : Optional[float] = None,
) -> Optional[dict]:
    """
    Run the full SMC engine for a single ticker.

    FLOW:
    1. Extract numpy arrays for highs, lows, closes
    2. Find all swing highs and swing lows
    3. Determine trend structure (bullish/bearish/broken)
    4. Detect CHoCH
    5. NEW: Find Demand/Supply zone (BOS + engulfing + SD band overlap)
    6. Package results for database

    Args:
        ticker    : Ticker symbol
        df        : Full OHLCV DataFrame, sorted date ascending
        date      : Today's date string YYYY-MM-DD
        sd1_lower : Current SD1 lower band value (for demand zone check)
        sd3_lower : Current SD3 lower band value (for demand zone check)
        sd1_upper : Current SD1 upper band value (for supply zone check)
        sd3_upper : Current SD3 upper band value (for supply zone check)

    Returns:
        Dict with smc_structure, choch_detected, has_valid_zone, or None on failure
    """
    min_required = MIN_PIVOT_CANDLES * 3
    if len(df) < min_required:
        logger.warning(
            f"{ticker} | Insufficient data for SMC: {len(df)} rows, need {min_required}"
        )
        return None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    swing_highs = _find_swing_highs(highs)
    swing_lows  = _find_swing_lows(lows)

    logger.debug(
        f"{ticker} | Swing highs: {len(swing_highs)} | "
        f"Swing lows: {len(swing_lows)}"
    )

    structure = _detect_structure(swing_highs, swing_lows)
    choch     = _detect_choch(closes, swing_highs, swing_lows, structure)

    # ── NEW: Demand/Supply zone detection ─────────────────────────────────────
    zone           = None
    has_valid_zone = 0

    if structure == "bullish" and sd1_lower is not None and sd3_lower is not None:
        zone = _find_demand_supply_zone(
            df, swing_highs, swing_lows, structure,
            sd1=sd1_lower, sd3=sd3_lower
        )
    elif structure == "bearish" and sd1_upper is not None and sd3_upper is not None:
        zone = _find_demand_supply_zone(
            df, swing_highs, swing_lows, structure,
            sd1=sd1_upper, sd3=sd3_upper
        )

    if zone is not None:
        has_valid_zone = 1

    result = {
        "ticker"        : ticker,
        "date"          : date,
        "smc_structure" : structure,
        "choch_detected": 1 if choch else 0,
        "has_valid_zone": has_valid_zone,
    }

    logger.debug(
        f"{ticker} | Structure: {structure} | "
        f"CHoCH: {'YES' if choch else 'no'} | "
        f"Valid Zone: {'YES' if has_valid_zone else 'no'}"
    )

    return result


# =============================================================================
# BATCH RUNNER
# =============================================================================

def run_smc_engine(
    tickers_data : dict[str, pd.DataFrame],
    date         : str,
    linreg_df    : Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run SMC computation for all tickers in the filtered universe.

    Args:
        tickers_data : Dict mapping ticker → OHLCV DataFrame
        date         : Today's date string YYYY-MM-DD
        linreg_df    : Output of run_linreg_engine() — provides SD band
                       values needed for demand/supply zone detection.
                       If None, zone detection is skipped (has_valid_zone=0).

    Returns:
        DataFrame with columns including smc_structure, choch_detected,
        has_valid_zone
    """
    logger.info(f"SMC engine starting | {len(tickers_data)} tickers | Date: {date}")

    # Build lookup of SD bands per ticker if linreg_df provided
    sd_lookup = {}
    if linreg_df is not None and not linreg_df.empty:
        for _, row in linreg_df.iterrows():
            sd_lookup[row["ticker"]] = {
                "sd1_lower": row.get("sd1_lower"),
                "sd3_lower": row.get("sd3_lower"),
                "sd1_upper": row.get("sd1_upper"),
                "sd3_upper": row.get("sd3_upper"),
            }

    results = []
    failed  = 0

    for ticker, df in tickers_data.items():
        sd = sd_lookup.get(ticker, {})

        result = compute_smc(
            ticker, df, date,
            sd1_lower = sd.get("sd1_lower"),
            sd3_lower = sd.get("sd3_lower"),
            sd1_upper = sd.get("sd1_upper"),
            sd3_upper = sd.get("sd3_upper"),
        )

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