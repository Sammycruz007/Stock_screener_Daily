"""
Test LinReg engine — regression line, SD bands, slope detection, SD position.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.linreg import (
    _compute_linreg,
    compute_linreg_series,
    compute_linreg_latest,
    run_linreg_engine,
)

TODAY = "2026-06-06"


def make_trending_df(n: int = 250, trend: float = 0.1) -> pd.DataFrame:
    """Generate a clearly trending price series."""
    closes = 100 + np.arange(n) * trend + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open"  : closes * 0.99,
        "high"  : closes * 1.01,
        "low"   : closes * 0.98,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n),
        "date"  : pd.date_range(end=TODAY, periods=n).strftime("%Y-%m-%d"),
    })


def test_compute_linreg_uptrend():
    """LinReg on a rising series should have positive slope."""
    df     = make_trending_df(250, trend=0.1)
    closes = df["close"].values[-200:]
    result = _compute_linreg(closes)

    assert result["linreg_slope_up"] == 1, "Expected uptrend slope"
    assert result["linreg_slope"] > 0
    assert result["sd1_upper"] > result["linreg_value"]
    assert result["sd1_lower"] < result["linreg_value"]
    assert result["sd2_upper"] > result["sd1_upper"]
    assert result["sd3_upper"] > result["sd2_upper"]
    assert result["sd1_lower"] > result["sd2_lower"]
    assert result["sd2_lower"] > result["sd3_lower"]
    print("✅ LinReg uptrend: slope positive, bands ordered correctly")


def test_compute_linreg_downtrend():
    """LinReg on a falling series should have negative slope."""
    df     = make_trending_df(250, trend=-0.1)
    closes = df["close"].values[-200:]
    result = _compute_linreg(closes)

    assert result["linreg_slope_up"] == 0, "Expected downtrend slope"
    assert result["linreg_slope"] < 0
    print("✅ LinReg downtrend: slope negative correctly detected")


def test_sd_position_below():
    """Price below LinReg should give negative SD position."""
    closes       = np.ones(200) * 100   # Flat line at 100
    closes[-1]   = 85                   # Force last price well below
    result       = _compute_linreg(closes)

    assert result["price_sd_position"] < 0
    print(f"✅ SD position below LinReg: {result['price_sd_position']:.4f} (negative ✅)")


def test_sd_position_above():
    """Price above LinReg should give positive SD position."""
    closes       = np.ones(200) * 100
    closes[-1]   = 115                  # Force last price above
    result       = _compute_linreg(closes)

    assert result["price_sd_position"] > 0
    print(f"✅ SD position above LinReg: {result['price_sd_position']:.4f} (positive ✅)")


def test_compute_linreg_series():
    """Full series should return same number of rows as input."""
    df     = make_trending_df(250)
    closes = df["close"].values
    result = compute_linreg_series(closes)

    assert len(result) == 250
    assert "linreg" in result.columns
    assert "sd1_upper" in result.columns
    assert "sd3_lower" in result.columns
    print(f"✅ compute_linreg_series: {len(result)} rows returned with correct columns")


def test_insufficient_data():
    """Should return None when less than 200 rows provided."""
    df     = make_trending_df(150)  # Only 150 rows — not enough
    result = compute_linreg_latest("AAPL", df, TODAY)
    assert result is None
    print("✅ Insufficient data: correctly returned None for < 200 rows")


def test_run_linreg_engine():
    """Batch runner should process all tickers and return DataFrame."""
    tickers_data = {
        "AAPL": make_trending_df(250, trend=0.1),
        "TSLA": make_trending_df(250, trend=-0.05),
        "MSFT": make_trending_df(250, trend=0.08),
    }
    result = run_linreg_engine(tickers_data, TODAY)

    assert len(result) == 3
    assert "ticker" in result.columns
    assert "linreg_value" in result.columns
    assert "price_sd_position" in result.columns
    assert "linreg_slope_up" in result.columns

    print(f"✅ run_linreg_engine: {len(result)} tickers processed")
    print(result[["ticker", "linreg_value", "linreg_slope_up", "price_sd_position"]].to_string(index=False))


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING LINREG ENGINE")
    print("=" * 50)
    test_compute_linreg_uptrend()
    test_compute_linreg_downtrend()
    test_sd_position_below()
    test_sd_position_above()
    test_compute_linreg_series()
    test_insufficient_data()
    test_run_linreg_engine()
    print("\n✅ All LinReg engine tests passed")