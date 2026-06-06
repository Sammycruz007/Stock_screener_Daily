"""
Test Volume engine — all four accumulation conditions and final signal.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.volume import (
    _is_volume_declining_on_down_days,
    _is_volume_expanding_on_green_days,
    _is_shakeout_present,
    _is_volume_dryup_present,
    _determine_volume_signal,
    compute_volume_signal,
    run_volume_engine,
)

TODAY = "2026-06-06"


def make_volume_df(
    n               : int   = 50,
    declining_vol   : bool  = False,
    shakeout        : bool  = False,
    expanding_green : bool  = False,
    dryup           : bool  = False,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV with controllable volume conditions.
    Allows us to test each condition independently and in combination.
    """
    closes  = 100 + np.cumsum(np.random.randn(n) * 0.3)
    opens   = closes * 0.999
    highs   = closes * 1.005
    lows    = closes * 0.995
    volumes = np.random.randint(800000, 1200000, n).astype(float)

    # ── Inject declining volume on down days ──────────────────────────────────
    if declining_vol:
        # Make last 10 candles red with declining volumes
        for i in range(n - 10, n):
            opens[i]   = closes[i] * 1.002   # Open above close = red candle
            volumes[i] = 1000000 - (i - (n - 10)) * 80000  # Declining

    # ── Inject shakeout candle ────────────────────────────────────────────────
    if shakeout:
        idx           = n - 5
        opens[idx]    = closes[idx] * 1.005   # Red candle
        highs[idx]    = closes[idx] * 1.01
        lows[idx]     = closes[idx] * 0.985
        # Close near the HIGH of the candle (above midpoint = price held)
        closes[idx]   = lows[idx] + (highs[idx] - lows[idx]) * 0.75
        volumes[idx]  = 3500000               # Very high volume

    # ── Inject expanding volume on green days ─────────────────────────────────
    if expanding_green:
        for i in range(n - 10, n):
            if i % 2 == 0:
                # Green candle with high volume
                opens[i]   = closes[i] * 0.998
                volumes[i] = 2500000
            else:
                # Red candle with low volume
                opens[i]   = closes[i] * 1.002
                volumes[i] = 600000

    # ── Inject volume dry-up candle ───────────────────────────────────────────
    if dryup:
        volumes[n - 3] = 150000   # Very low volume candle

    return pd.DataFrame({
        "open"  : opens,
        "high"  : highs,
        "low"   : lows,
        "close" : closes,
        "volume": volumes,
        "date"  : pd.date_range(end=TODAY, periods=n).strftime("%Y-%m-%d"),
    })


def test_volume_declining_on_down_days():
    df         = make_volume_df(50, declining_vol=True)
    avg_volume = df["volume"].tail(20).mean()
    result     = _is_volume_declining_on_down_days(df, avg_volume)
    assert result == True
    print("✅ Condition 1: Volume declining on down days — detected correctly")


def test_shakeout_detection():
    df         = make_volume_df(50, shakeout=True)
    avg_volume = df["volume"].tail(20).mean()
    result     = _is_shakeout_present(df, avg_volume)
    assert result == True
    print("✅ Condition 2: Shakeout candle — detected correctly")


def test_volume_expanding_on_green():
    df         = make_volume_df(50, expanding_green=True)
    avg_volume = df["volume"].tail(20).mean()
    result     = _is_volume_expanding_on_green_days(df, avg_volume)
    assert result == True
    print("✅ Condition 3: Volume expanding on green days — detected correctly")


def test_volume_dryup():
    df         = make_volume_df(50, dryup=True)
    avg_volume = df["volume"].tail(20).mean()
    result     = _is_volume_dryup_present(df, avg_volume)
    assert result == True
    print("✅ Condition 4: Volume dry-up candle — detected correctly")


def test_accumulation_signal_path_a():
    """Path A: declining down volume + expanding green volume = accumulation."""
    signal = _determine_volume_signal(
        cond1_declining_down = True,
        cond2_shakeout       = False,
        cond3_expanding_up   = True,
        cond4_dryup          = False,
        dist1_rising_up      = False,
        dist2_expanding_down = False,
    )
    assert signal == "accumulation"
    print("✅ Signal Path A: accumulation correctly identified (C1 + C3)")


def test_accumulation_signal_path_b():
    """Path B: shakeout + dry-up = accumulation."""
    signal = _determine_volume_signal(
        cond1_declining_down = False,
        cond2_shakeout       = True,
        cond3_expanding_up   = False,
        cond4_dryup          = True,
        dist1_rising_up      = False,
        dist2_expanding_down = False,
    )
    assert signal == "accumulation"
    print("✅ Signal Path B: accumulation correctly identified (C2 + C4)")


def test_distribution_signal():
    """Rising up volume + expanding red volume = distribution."""
    signal = _determine_volume_signal(
        cond1_declining_down = False,
        cond2_shakeout       = False,
        cond3_expanding_up   = False,
        cond4_dryup          = False,
        dist1_rising_up      = True,
        dist2_expanding_down = True,
    )
    assert signal == "distribution"
    print("✅ Distribution signal correctly identified (D1 + D2)")


def test_neutral_signal():
    """No conditions met = neutral."""
    signal = _determine_volume_signal(
        cond1_declining_down = False,
        cond2_shakeout       = False,
        cond3_expanding_up   = False,
        cond4_dryup          = False,
        dist1_rising_up      = False,
        dist2_expanding_down = False,
    )
    assert signal == "neutral"
    print("✅ Neutral signal correctly identified (no conditions met)")


def test_compute_volume_signal_full():
    """Full engine should return a valid signal dict."""
    df     = make_volume_df(50, declining_vol=True, expanding_green=True)
    result = compute_volume_signal("AAPL", df, TODAY)

    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["volume_signal"] in ["accumulation", "distribution", "neutral"]
    print(f"✅ compute_volume_signal: signal = {result['volume_signal']}")


def test_run_volume_engine():
    """Batch runner should process all tickers."""
    tickers_data = {
        "AAPL": make_volume_df(50, declining_vol=True, expanding_green=True),
        "TSLA": make_volume_df(50, shakeout=True, dryup=True),
        "MSFT": make_volume_df(50),
    }
    result = run_volume_engine(tickers_data, TODAY)

    assert len(result) == 3
    assert "volume_signal" in result.columns
    print(f"✅ run_volume_engine: {len(result)} tickers processed")
    print(result[["ticker", "volume_signal"]].to_string(index=False))


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING VOLUME ENGINE")
    print("=" * 50)
    test_volume_declining_on_down_days()
    test_shakeout_detection()
    test_volume_expanding_on_green()
    test_volume_dryup()
    test_accumulation_signal_path_a()
    test_accumulation_signal_path_b()
    test_distribution_signal()
    test_neutral_signal()
    test_compute_volume_signal_full()
    test_run_volume_engine()
    print("\n✅ All Volume engine tests passed")