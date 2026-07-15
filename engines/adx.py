"""
engines/adx.py
--------------
Average Directional Index (ADX) engine — measures trend STRENGTH,
independent of direction. Complements LinReg (which measures trend
DIRECTION via slope) and SMC (which measures trend STRUCTURE via
pivots/CHoCH). Added as a learned feature for the Signal Ranker, not
a hard filter — see project discussion for reasoning.

LOGICAL FLOW:
─────────────
1. True Range (TR) per candle:
   max(high-low, |high-prev_close|, |low-prev_close|)

2. Directional Movement:
   +DM = today's high - yesterday's high  (if positive AND > -DM, else 0)
   -DM = yesterday's low - today's low    (if positive AND > +DM, else 0)

3. Wilder's smoothing (an EMA-like running average, alpha = 1/period)
   applied to TR, +DM, -DM over PERIOD candles.

4. Directional Indicators:
   +DI = 100 * smoothed(+DM) / smoothed(TR)
   -DI = 100 * smoothed(-DM) / smoothed(TR)

5. Directional Index:
   DX = 100 * |+DI - -DI| / (+DI + -DI)

6. ADX = Wilder-smoothed DX over PERIOD candles.
   Higher ADX = stronger trend (either direction).
   Low ADX (~<20) = choppy/ranging market, low ADX = weak trend.
   High ADX (~>25) = strong trend, conventionally.

OUTPUT per ticker:
   - adx_value : the ADX reading itself (trend strength, 0-100+)
   - plus_di   : +DI (bullish directional pressure)
   - minus_di  : -DI (bearish directional pressure)

WHY THIS MATTERS FOR THE MODEL:
   ADX doesn't say WHICH way price is trending — LinReg slope already
   covers that. It says HOW STRONGLY. A setup with a clear LinReg
   slope but low ADX may be a weak, choppy trend not worth trusting
   as much as the same slope with high ADX behind it. Feeding this as
   a learned feature (not a hard filter) lets the model find the
   actual relationship rather than assuming a fixed "ADX > 25" cutoff
   that may not hold for this universe/timeframe.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_adx_logger
from utils.error_handler import graceful, EngineError

logger = get_adx_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config  = _load_config()
ADX_CFG = config["adx"]

PERIOD = ADX_CFG["period"]   # 14 candles — standard for daily bars


# =============================================================================
# CORE CALCULATION
# =============================================================================

def _wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's smoothing — the specific running-average variant ADX's
    original formula uses (similar to an EMA, but with alpha = 1/period
    instead of the standard 2/(period+1)).

    Args:
        series: Raw values to smooth (TR, +DM, or -DM)
        period: Smoothing period

    Returns:
        Smoothed array, same length as input (first `period` values
        are the seed — a simple average of the first `period` raw
        values — everything after follows Wilder's recursive formula)
    """
    smoothed = np.zeros_like(series, dtype=float)
    smoothed[period - 1] = series[:period].sum()

    for i in range(period, len(series)):
        smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + series[i]

    return smoothed


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> dict:
    """
    Core ADX calculation from raw OHLC arrays.

    Args:
        high, low, close: Arrays of equal length, chronologically ordered
        period           : Smoothing period (PERIOD from config)

    Returns:
        Dict with adx_value, plus_di, minus_di (for the LAST candle)
    """
    n = len(close)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]

    # ── True Range ───────────────────────────────────────────────────────────
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )

    # ── Directional Movement ─────────────────────────────────────────────────
    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # First candle has no prior candle to compare against — zero it out
    tr[0] = high[0] - low[0]
    plus_dm[0]  = 0.0
    minus_dm[0] = 0.0

    # ── Wilder smoothing ──────────────────────────────────────────────────────
    smoothed_tr    = _wilder_smooth(tr, period)
    smoothed_plus  = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)

    # ── Directional Indicators ───────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di  = np.where(smoothed_tr > 0, 100 * smoothed_plus  / smoothed_tr, 0.0)
        minus_di = np.where(smoothed_tr > 0, 100 * smoothed_minus / smoothed_tr, 0.0)

    # ── DX and ADX ────────────────────────────────────────────────────────────
    di_sum = plus_di + minus_di
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    # ADX is itself a Wilder-smoothed average of DX over `period`,
    # but only valid starting once DX itself has `period` valid values
    # (i.e. starting at index 2*period - 2). Simplify by taking a plain
    # rolling mean of the last `period` DX values for the latest reading —
    # equivalent in practice once past the warm-up window.
    if n >= 2 * period:
        adx_value = float(np.mean(dx[-period:]))
    else:
        adx_value = float(np.mean(dx[period - 1:])) if n > period else 0.0

    return {
        "adx_value": round(adx_value, 4),
        "plus_di"  : round(float(plus_di[-1]), 4),
        "minus_di" : round(float(minus_di[-1]), 4),
    }


# =============================================================================
# PUBLIC ENTRY POINT — matches linreg.py / smc.py's calling convention
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def compute_adx_latest(
    ticker : str,
    df     : pd.DataFrame,
    date   : str,
) -> Optional[dict]:
    """
    Compute ADX values for the most recent candle of a single ticker.
    Matches compute_linreg_latest / compute_smc's calling convention —
    called the same way from run_pipeline_cloud.py's engine step and
    train_models.py's backfill_ticker().

    Args:
        ticker : Ticker symbol e.g. 'AAPL'
        df     : Full OHLCV DataFrame for this ticker, sorted date ascending,
                 filtered to <= the target date by the caller
        date   : Signal date string YYYY-MM-DD (for database keying)

    Returns:
        Dict with ticker, date, adx_value, plus_di, minus_di —
        or None if insufficient data (@graceful handles exceptions)
    """
    # Need at least 2*PERIOD candles for a stable ADX reading (one
    # PERIOD to seed the smoothing, another for DX itself to stabilise)
    min_required = PERIOD * 2
    if len(df) < min_required:
        logger.warning(
            f"{ticker} | Only {len(df)} rows — need {min_required}+ for ADX"
        )
        return None

    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values

    adx_values = _compute_adx(high, low, close, PERIOD)

    result = {
        "ticker": ticker,
        "date"  : date,
        **adx_values,
    }

    logger.debug(
        f"{ticker} | ADX: {result['adx_value']} | "
        f"+DI: {result['plus_di']} | -DI: {result['minus_di']}"
    )

    return result
