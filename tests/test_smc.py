"""
Test SMC engine — swing detection, structure, CHoCH.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.smc import (
    _find_swing_highs,
    _find_swing_lows,
    _detect_structure,
    _detect_choch,
    compute_smc,
    run_smc_engine,
)

TODAY = "2026-06-06"


def make_bullish_df(n: int = 100) -> pd.DataFrame:
    """
    Generate a clear bullish structure:
    Higher Highs and Higher Lows in a staircase pattern.
    """
    closes = []
    highs  = []
    lows   = []
    base   = 100.0

    for i in range(n):
        # Create a staircase pattern — clear HH and HL
        cycle     = i % 20
        base_val  = 100 + (i // 20) * 5     # Steps up every 20 candles
        close     = base_val + np.sin(cycle / 20 * 2 * np.pi) * 3
        closes.append(close)
        highs.append(close + 0.5)
        lows.append(close - 0.5)

    return pd.DataFrame({
        "open"  : np.array(closes) * 0.99,
        "high"  : highs,
        "low"   : lows,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n),
        "date"  : pd.date_range(end=TODAY, periods=n).strftime("%Y-%m-%d"),
    })


def make_bearish_df(n: int = 100) -> pd.DataFrame:
    """Generate a clear bearish structure: Lower Highs and Lower Lows."""
    closes = []
    highs  = []
    lows   = []

    for i in range(n):
        cycle    = i % 20
        base_val = 100 - (i // 20) * 5     # Steps down every 20 candles
        close    = base_val + np.sin(cycle / 20 * 2 * np.pi) * 3
        closes.append(close)
        highs.append(close + 0.5)
        lows.append(close - 0.5)

    return pd.DataFrame({
        "open"  : np.array(closes) * 0.99,
        "high"  : highs,
        "low"   : lows,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n),
        "date"  : pd.date_range(end=TODAY, periods=n).strftime("%Y-%m-%d"),
    })


def test_swing_high_detection():
    """Swing highs should be detected at clear peak points."""
    # Build a longer array with clear peaks at known positions
    # Pattern: rise to peak, fall, rise to bigger peak, fall
    # Need enough candles on each side of peaks (min 5 each side)
    highs = np.array([
        1, 2, 3, 4, 5,          # rising
        10,                      # PEAK 1 — index 5
        5, 4, 3, 2, 1,          # falling (5 candles right of peak)
        2, 3, 4, 5,             # rising again
        12,                      # PEAK 2 — index 16
        5, 4, 3, 2, 1,          # falling (5 candles right of peak)
    ], dtype=float)

    swing_highs = _find_swing_highs(highs)

    assert len(swing_highs) > 0, "Should detect at least one swing high"
    prices = [p for _, p in swing_highs]
    assert any(p >= 10.0 for p in prices), f"Should detect peak values, got {prices}"
    print(f"✅ Swing high detection: {len(swing_highs)} highs found at {prices}")


def test_swing_low_detection():
    """Swing lows should be detected at clear trough points."""
    lows = np.array([
        10, 9, 8, 7, 6,         # falling
        1,                       # TROUGH 1 — index 5
        6, 7, 8, 9, 10,         # rising (5 candles right of trough)
        9, 8, 7, 6,             # falling again
        2,                       # TROUGH 2 — index 16
        6, 7, 8, 9, 10,         # rising (5 candles right of trough)
    ], dtype=float)

    swing_lows = _find_swing_lows(lows)

    assert len(swing_lows) > 0, "Should detect at least one swing low"
    prices = [p for _, p in swing_lows]
    assert any(p <= 2.0 for p in prices), f"Should detect trough values, got {prices}"
    print(f"✅ Swing low detection: {len(swing_lows)} lows found at {prices}")

def test_bullish_structure():
    """Higher Highs + Higher Lows should return bullish structure."""
    swing_highs = [(0, 10.0), (10, 12.0), (20, 14.0)]  # Each higher than last
    swing_lows  = [(5, 8.0),  (15, 9.0),  (25, 10.0)]  # Each higher than last
    structure   = _detect_structure(swing_highs, swing_lows)

    assert structure == "bullish"
    print("✅ Structure detection: bullish correctly identified (HH + HL)")


def test_bearish_structure():
    """Lower Highs + Lower Lows should return bearish structure."""
    swing_highs = [(0, 14.0), (10, 12.0), (20, 10.0)]  # Each lower than last
    swing_lows  = [(5, 10.0), (15, 9.0),  (25, 8.0)]   # Each lower than last
    structure   = _detect_structure(swing_highs, swing_lows)

    assert structure == "bearish"
    print("✅ Structure detection: bearish correctly identified (LH + LL)")


def test_broken_structure():
    """Mixed pivots should return broken structure."""
    swing_highs = [(0, 10.0), (10, 12.0)]   # Higher High
    swing_lows  = [(5, 8.0),  (15, 7.0)]    # Lower Low — mixed!
    structure   = _detect_structure(swing_highs, swing_lows)

    assert structure == "broken"
    print("✅ Structure detection: broken correctly identified (mixed pivots)")


def test_choch_bullish_broken():
    """CHoCH should be detected when close breaks below recent swing low."""
    # Swing low at price 100 — then price closes below it
    swing_lows  = [(50, 100.0)]
    swing_highs = [(40, 110.0)]

    # Recent closes that go below the swing low of 100
    closes      = np.array([105, 103, 101, 99, 97])  # Last values drop below 100

    choch = _detect_choch(closes, swing_highs, swing_lows, "bullish")
    assert choch == True
    print("✅ CHoCH detection: bullish CHoCH correctly detected (close below swing low)")


def test_choch_not_triggered():
    """CHoCH should NOT trigger if closes stay above swing low."""
    swing_lows  = [(50, 100.0)]
    swing_highs = [(40, 110.0)]

    # All closes remain above the swing low of 100
    closes      = np.array([105, 103, 102, 104, 106])

    choch = _detect_choch(closes, swing_highs, swing_lows, "bullish")
    assert choch == False
    print("✅ CHoCH detection: correctly NOT triggered (closes above swing low)")


def test_compute_smc_full():
    """Full SMC engine should return valid structure for a trending series."""
    df     = make_bullish_df(100)
    result = compute_smc("AAPL", df, TODAY)

    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["smc_structure"] in ["bullish", "bearish", "broken"]
    assert result["choch_detected"] in [0, 1]
    print(f"✅ compute_smc: structure={result['smc_structure']} | choch={result['choch_detected']}")


def test_run_smc_engine():
    """Batch runner should process all tickers."""
    tickers_data = {
        "AAPL": make_bullish_df(100),
        "TSLA": make_bearish_df(100),
        "MSFT": make_bullish_df(100),
    }
    result = run_smc_engine(tickers_data, TODAY)

    assert len(result) == 3
    assert "smc_structure" in result.columns
    assert "choch_detected" in result.columns
    print(f"✅ run_smc_engine: {len(result)} tickers processed")
    print(result[["ticker", "smc_structure", "choch_detected"]].to_string(index=False))


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING SMC ENGINE")
    print("=" * 50)
    test_swing_high_detection()
    test_swing_low_detection()
    test_bullish_structure()
    test_bearish_structure()
    test_broken_structure()
    test_choch_bullish_broken()
    test_choch_not_triggered()
    test_compute_smc_full()
    test_run_smc_engine()
    print("\n✅ All SMC engine tests passed")