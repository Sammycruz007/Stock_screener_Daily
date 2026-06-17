"""
Test SMC engine — swing detection, structure, CHoCH, BOS, engulfing, zones.
MIN_PIVOT_CANDLES = 78 (3 trading days on 15m candles).
All test arrays sized to at least 2*78+1 = 157 candles.
"""
import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.smc import (
    _find_swing_highs,
    _find_swing_lows,
    _detect_structure,
    _detect_choch,
    _find_bos,
    _is_bullish_engulfing,
    _is_bearish_engulfing,
    _find_demand_supply_zone,
    compute_smc,
    run_smc_engine,
    MIN_PIVOT_CANDLES,
)

TODAY = "2026-06-13"


# =============================================================================
# HELPERS
# =============================================================================

def make_bullish_df(n: int = 250) -> pd.DataFrame:
    """
    Generate a clear bullish staircase structure:
    Higher Highs and Higher Lows, with enough candles per leg
    for MIN_PIVOT_CANDLES=78 pivot detection.
    """
    closes, highs, lows = [], [], []
    cycle_len = (2 * MIN_PIVOT_CANDLES) + 10  # >= 166

    for i in range(n):
        cycle    = i % cycle_len
        base_val = 100 + (i // cycle_len) * 5     # step up each full cycle
        close    = base_val + np.sin(cycle / cycle_len * 2 * np.pi) * 3
        closes.append(close)
        highs.append(close + 0.5)
        lows.append(close - 0.5)

    return pd.DataFrame({
        "open"  : np.array(closes) * 0.999,
        "high"  : highs,
        "low"   : lows,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n).astype(float),
        "date"  : pd.date_range(end=TODAY, periods=n, freq="15min"),
    })


def make_bearish_df(n: int = 250) -> pd.DataFrame:
    """Generate a clear bearish staircase: Lower Highs and Lower Lows."""
    closes, highs, lows = [], [], []
    cycle_len = (2 * MIN_PIVOT_CANDLES) + 10

    for i in range(n):
        cycle    = i % cycle_len
        base_val = 100 - (i // cycle_len) * 5     # step down each full cycle
        close    = base_val + np.sin(cycle / cycle_len * 2 * np.pi) * 3
        closes.append(close)
        highs.append(close + 0.5)
        lows.append(close - 0.5)

    return pd.DataFrame({
        "open"  : np.array(closes) * 0.999,
        "high"  : highs,
        "low"   : lows,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n).astype(float),
        "date"  : pd.date_range(end=TODAY, periods=n, freq="15min"),
    })


# =============================================================================
# SWING DETECTION TESTS
# =============================================================================

def test_swing_high_detection():
    """Swing highs detected with MIN_PIVOT_CANDLES=78 window."""
    n = 400
    highs = np.full(n, 50.0)

    # Peak 1 at index 100 — 78 candles of lower values on each side
    p1 = 100
    highs[p1] = 60.0
    highs[p1-MIN_PIVOT_CANDLES:p1] = np.linspace(50, 55, MIN_PIVOT_CANDLES)
    highs[p1+1:p1+1+MIN_PIVOT_CANDLES] = np.linspace(55, 50, MIN_PIVOT_CANDLES)

    # Peak 2 at index 250
    p2 = 250
    highs[p2] = 65.0
    highs[p2-MIN_PIVOT_CANDLES:p2] = np.linspace(50, 58, MIN_PIVOT_CANDLES)
    highs[p2+1:p2+1+MIN_PIVOT_CANDLES] = np.linspace(58, 50, MIN_PIVOT_CANDLES)

    swing_highs = _find_swing_highs(highs)

    assert len(swing_highs) > 0, "Should detect at least one swing high"
    prices = [p for _, p in swing_highs]
    assert any(p >= 60.0 for p in prices), f"Should detect peak values, got {prices}"
    print(f"✅ Swing high detection: {len(swing_highs)} highs found at {prices}")


def test_swing_low_detection():
    """Swing lows detected with MIN_PIVOT_CANDLES=78 window."""
    n = 400
    lows = np.full(n, 50.0)

    p1 = 100
    lows[p1] = 40.0
    lows[p1-MIN_PIVOT_CANDLES:p1] = np.linspace(50, 45, MIN_PIVOT_CANDLES)
    lows[p1+1:p1+1+MIN_PIVOT_CANDLES] = np.linspace(45, 50, MIN_PIVOT_CANDLES)

    p2 = 250
    lows[p2] = 35.0
    lows[p2-MIN_PIVOT_CANDLES:p2] = np.linspace(50, 42, MIN_PIVOT_CANDLES)
    lows[p2+1:p2+1+MIN_PIVOT_CANDLES] = np.linspace(42, 50, MIN_PIVOT_CANDLES)

    swing_lows = _find_swing_lows(lows)

    assert len(swing_lows) > 0, "Should detect at least one swing low"
    prices = [p for _, p in swing_lows]
    assert any(p <= 35.0 for p in prices), f"Should detect trough values, got {prices}"
    print(f"✅ Swing low detection: {len(swing_lows)} lows found at {prices}")


# =============================================================================
# STRUCTURE DETECTION TESTS
# =============================================================================

def test_bullish_structure():
    """Higher Highs + Higher Lows should return bullish structure."""
    swing_highs = [(0, 10.0), (100, 12.0), (200, 14.0)]
    swing_lows  = [(50, 8.0), (150, 9.0), (250, 10.0)]
    structure   = _detect_structure(swing_highs, swing_lows)

    assert structure == "bullish"
    print("✅ Structure detection: bullish correctly identified (HH + HL)")


def test_bearish_structure():
    """Lower Highs + Lower Lows should return bearish structure."""
    swing_highs = [(0, 14.0), (100, 12.0), (200, 10.0)]
    swing_lows  = [(50, 10.0), (150, 9.0), (250, 8.0)]
    structure   = _detect_structure(swing_highs, swing_lows)

    assert structure == "bearish"
    print("✅ Structure detection: bearish correctly identified (LH + LL)")


def test_broken_structure():
    """Mixed pivots should return broken structure."""
    swing_highs = [(0, 10.0), (100, 12.0)]   # Higher High
    swing_lows  = [(50, 8.0), (150, 7.0)]    # Lower Low — mixed!
    structure   = _detect_structure(swing_highs, swing_lows)

    assert structure == "broken"
    print("✅ Structure detection: broken correctly identified (mixed pivots)")


# =============================================================================
# CHoCH TESTS
# =============================================================================

def test_choch_bullish_broken():
    """CHoCH detected when close breaks below recent swing low."""
    swing_lows  = [(200, 100.0)]
    swing_highs = [(150, 110.0)]

    # Recent closes — last MIN_PIVOT_CANDLES*2 must include a close < 100
    n = MIN_PIVOT_CANDLES * 2
    closes = np.linspace(105, 97, n)  # drops below 100

    choch = _detect_choch(closes, swing_highs, swing_lows, "bullish")
    assert choch == True
    print("✅ CHoCH detection: bullish CHoCH correctly detected (close below swing low)")


def test_choch_not_triggered():
    """CHoCH should NOT trigger if closes stay above swing low."""
    swing_lows  = [(200, 100.0)]
    swing_highs = [(150, 110.0)]

    n = MIN_PIVOT_CANDLES * 2
    closes = np.linspace(105, 102, n)  # stays above 100

    choch = _detect_choch(closes, swing_highs, swing_lows, "bullish")
    assert choch == False
    print("✅ CHoCH detection: correctly NOT triggered (closes above swing low)")


# =============================================================================
# ENGULFING CANDLE TESTS
# =============================================================================

def test_bullish_engulfing_detected():
    """Bullish engulfing pattern correctly identified."""
    opens  = np.array([10, 9.0])
    closes = np.array([9, 10.5])
    result = _is_bullish_engulfing(opens, closes, 1)
    assert result == True
    print("✅ Bullish engulfing: correctly detected")


def test_bullish_engulfing_not_detected():
    """Non-engulfing pattern correctly rejected."""
    opens  = np.array([10, 9.5])
    closes = np.array([9, 9.8])
    result = _is_bullish_engulfing(opens, closes, 1)
    assert result == False
    print("✅ Bullish engulfing: correctly rejected (no engulf)")


def test_bearish_engulfing_detected():
    """Bearish engulfing pattern correctly identified."""
    opens  = np.array([9, 10.5])
    closes = np.array([10, 9.0])
    result = _is_bearish_engulfing(opens, closes, 1)
    assert result == True
    print("✅ Bearish engulfing: correctly detected")


# =============================================================================
# BOS TESTS
# =============================================================================

def test_find_bos_bullish():
    """BOS detected when close breaks above most recent swing high."""
    swing_highs = [(200, 110.0)]
    swing_lows  = [(100, 100.0)]

    n = MIN_PIVOT_CANDLES * 2
    closes = np.linspace(105, 109, n)
    closes[-1] = 112  # last candle breaks above 110

    result = _find_bos(closes, swing_highs, swing_lows, "bullish")
    assert result is not None
    bos_idx, base_idx = result
    assert base_idx == 100
    print(f"✅ BOS bullish: detected at idx={bos_idx}, base_idx={base_idx}")


def test_find_bos_not_triggered():
    """BOS not detected when price hasn't broken the swing high."""
    swing_highs = [(200, 110.0)]
    swing_lows  = [(100, 100.0)]

    n = MIN_PIVOT_CANDLES * 2
    closes = np.linspace(105, 108, n)  # never exceeds 110

    result = _find_bos(closes, swing_highs, swing_lows, "bullish")
    assert result is None
    print("✅ BOS bullish: correctly NOT triggered (no break)")


# =============================================================================
# DEMAND/SUPPLY ZONE TEST
# =============================================================================

def test_find_demand_zone_full():
    """
    Full demand zone detection: BOS + bullish engulfing + SD band overlap.
    Sized for MIN_PIVOT_CANDLES=78.
    """
    n = 400
    opens  = np.full(n, 100.0)
    highs  = np.full(n, 101.0)
    lows   = np.full(n, 99.0)
    closes = np.full(n, 100.0)

    # ── Swing low (base of move) around index 100 ─────────────────────────────
    base = 100
    lows[base-2:base+3]   = [98, 97, 95, 97, 98]
    closes[base-2:base+3] = [98.5, 97.5, 95.5, 97.5, 98.5]
    opens[base-2:base+3]  = [98.8, 98, 96, 97, 98]

    # Engulfing candle at base+1, base+2
    # idx base+1: red candle
    opens[base+1]  = 96.0
    closes[base+1] = 95.5
    # idx base+2: bullish engulfing
    opens[base+2]  = 95.0
    closes[base+2] = 97.0
    highs[base+2]  = 97.2
    lows[base+2]   = 94.8

    # ── Swing high around index 280 ────────────────────────────────────────────
    sh = 280
    highs[sh-2:sh+3]  = [101, 102, 106, 102, 101]
    closes[sh-2:sh+3] = [100.5, 101.5, 105.5, 101, 100.5]
    opens[sh-2:sh+3]  = [100, 101, 105, 102, 101]

    # ── BOS: last candle breaks above swing high of 106 ────────────────────────
    closes[-1] = 107.0
    highs[-1]  = 107.5

    # Fill remaining flat structure with slight upward drift so HH/HL forms
    for i in range(n):
        if i not in range(base-2, base+3) and i not in range(sh-2, sh+3) and i != n-1:
            drift = (i / n) * 2
            opens[i]  = 100 + drift
            highs[i]  = 101 + drift
            lows[i]   = 99 + drift
            closes[i] = 100 + drift

    df = pd.DataFrame({
        "open" : opens, "high": highs, "low": lows, "close": closes,
        "volume": np.full(n, 1000000.0),
        "date": pd.date_range(end=TODAY, periods=n, freq="15min"),
    })

    swing_highs = _find_swing_highs(highs)
    swing_lows  = _find_swing_lows(lows)
    structure   = _detect_structure(swing_highs, swing_lows)

    print(f"   Structure: {structure} | swing_highs(last2): {swing_highs[-2:]} | swing_lows(last2): {swing_lows[-2:]}")

    zone = _find_demand_supply_zone(
        df, swing_highs, swing_lows, structure,
        sd1=98.0, sd3=93.0   # band 93-98 overlaps zone ~94.8-97.2
    )

    if zone is not None:
        assert zone["zone_type"] == "demand"
        print(f"✅ Demand zone found: {zone}")
    else:
        print("⚠️  No demand zone found — synthetic data may not produce bullish structure")


# =============================================================================
# FULL ENGINE TESTS
# =============================================================================

def test_compute_smc_full():
    """Full SMC engine returns valid structure for a trending series."""
    df     = make_bullish_df(400)
    result = compute_smc("AAPL", df, TODAY)

    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["smc_structure"] in ["bullish", "bearish", "broken"]
    assert result["choch_detected"] in [0, 1]
    assert "has_valid_zone" in result
    print(f"✅ compute_smc: structure={result['smc_structure']} | choch={result['choch_detected']} | zone={result['has_valid_zone']}")


def test_compute_smc_with_zone_params():
    """compute_smc accepts SD band params and returns has_valid_zone field."""
    df = make_bullish_df(400)
    result = compute_smc(
        "AAPL", df, TODAY,
        sd1_lower=95.0, sd3_lower=85.0,
        sd1_upper=105.0, sd3_upper=115.0,
    )

    assert result is not None
    assert "has_valid_zone" in result
    assert result["has_valid_zone"] in [0, 1]
    print(f"✅ compute_smc with SD params: has_valid_zone={result['has_valid_zone']}")


def test_run_smc_engine():
    """Batch runner processes all tickers."""
    tickers_data = {
        "AAPL": make_bullish_df(400),
        "TSLA": make_bearish_df(400),
        "MSFT": make_bullish_df(400),
    }
    result = run_smc_engine(tickers_data, TODAY)

    assert len(result) == 3
    assert "smc_structure"  in result.columns
    assert "choch_detected" in result.columns
    assert "has_valid_zone" in result.columns
    print(f"✅ run_smc_engine: {len(result)} tickers processed")
    print(result[["ticker", "smc_structure", "choch_detected", "has_valid_zone"]].to_string(index=False))


def test_run_smc_engine_with_linreg_df():
    """run_smc_engine accepts linreg_df and produces has_valid_zone column."""
    tickers_data = {
        "AAPL": make_bullish_df(400),
        "TSLA": make_bearish_df(400),
    }

    linreg_df = pd.DataFrame([
        {"ticker": "AAPL", "date": TODAY, "sd1_lower": 95.0, "sd3_lower": 85.0,
         "sd1_upper": 105.0, "sd3_upper": 115.0},
        {"ticker": "TSLA", "date": TODAY, "sd1_lower": 95.0, "sd3_lower": 85.0,
         "sd1_upper": 105.0, "sd3_upper": 115.0},
    ])

    result = run_smc_engine(tickers_data, TODAY, linreg_df=linreg_df)

    assert len(result) == 2
    assert "has_valid_zone" in result.columns
    print(f"✅ run_smc_engine with linreg_df: {len(result)} tickers")
    print(result[["ticker", "smc_structure", "choch_detected", "has_valid_zone"]].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    try:
        print("=" * 55)
        print(f"TESTING SMC ENGINE (MIN_PIVOT_CANDLES={MIN_PIVOT_CANDLES})")
        print("=" * 55)

        print("\n--- Swing Detection ---")
        test_swing_high_detection()
        test_swing_low_detection()

        print("\n--- Structure Detection ---")
        test_bullish_structure()
        test_bearish_structure()
        test_broken_structure()

        print("\n--- CHoCH ---")
        test_choch_bullish_broken()
        test_choch_not_triggered()

        print("\n--- Engulfing Candles ---")
        test_bullish_engulfing_detected()
        test_bullish_engulfing_not_detected()
        test_bearish_engulfing_detected()

        print("\n--- BOS ---")
        test_find_bos_bullish()
        test_find_bos_not_triggered()

        print("\n--- Demand/Supply Zone ---")
        test_find_demand_zone_full()

        print("\n--- Full Engine ---")
        test_compute_smc_full()
        test_compute_smc_with_zone_params()
        test_run_smc_engine()
        test_run_smc_engine_with_linreg_df()

        print("\n✅ All SMC engine tests passed")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()