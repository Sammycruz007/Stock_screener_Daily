"""
Test scanner waterfall — market health, sector health, stock filtering.
Uses synthetic indicator data so no database connection needed.
Updated to reflect dynamic sector lookup approach.
"""
import sys
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.screener import (
    _check_market_health,
    _check_sector_health,
    _is_long_candidate,
    _is_short_candidate,
    run_scanner,
)

TODAY = "2026-06-06"


# =============================================================================
# HELPERS
# =============================================================================

def make_indicator_row(
    ticker         : str,
    slope_up       : int   = 1,
    choch          : int   = 0,
    sd_position    : float = -1.8,
    volume_signal  : str   = "accumulation",
    has_valid_zone : int   = 1,        # ← new parameter, default 1 so existing tests still pass
) -> dict:
    """Helper to build a synthetic indicator result row."""
    return {
        "ticker"           : ticker,
        "date"             : TODAY,
        "linreg_value"     : 100.0,
        "linreg_slope"     : 0.002 if slope_up else -0.002,
        "linreg_slope_up"  : slope_up,
        "sd1_upper"        : 105.0,
        "sd1_lower"        : 95.0,
        "sd2_upper"        : 110.0,
        "sd2_lower"        : 90.0,
        "sd3_upper"        : 115.0,
        "sd3_lower"        : 85.0,
        "price_sd_position": sd_position,
        "smc_structure"    : "bullish" if slope_up else "bearish",
        "choch_detected"   : choch,
        "volume_signal"    : volume_signal,
        "has_valid_zone"   : has_valid_zone,
    }


def make_full_indicator_df() -> pd.DataFrame:
    """
    Build a complete indicator DataFrame with:
    - All 3 indices (bullish)
    - All 11 sector ETFs (bullish)
    - A few stocks with mix of valid and invalid setups
    """
    rows = []

    # ── Indices — all bullish ─────────────────────────────────────────────────
    for idx in ["SPY", "QQQ", "DIA"]:
        rows.append(make_indicator_row(
            idx, slope_up=1, choch=0, sd_position=0.2
        ))

    # ── Sector ETFs — all bullish ─────────────────────────────────────────────
    for sector in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC"]:
        rows.append(make_indicator_row(
            sector, slope_up=1, choch=0, sd_position=0.1
        ))

    # ── Stocks ────────────────────────────────────────────────────────────────
    # Valid long: uptrend + no CHoCH + in buy zone + accumulation + bullish sector
    rows.append(make_indicator_row(
        "AAPL", slope_up=1, choch=0, sd_position=-1.8, volume_signal="accumulation"
    ))
    # Invalid: in buy zone but wrong volume signal
    rows.append(make_indicator_row(
        "TSLA", slope_up=1, choch=0, sd_position=-2.1, volume_signal="neutral"
    ))
    # Invalid: CHoCH detected
    rows.append(make_indicator_row(
        "MSFT", slope_up=1, choch=1, sd_position=-1.5, volume_signal="accumulation"
    ))
    # Invalid: not in buy zone
    rows.append(make_indicator_row(
        "NVDA", slope_up=1, choch=0, sd_position=0.5, volume_signal="accumulation"
    ))

    return pd.DataFrame(rows)


def make_ticker_to_etf() -> dict:
    """
    Synthetic sector lookup dict — replaces database call in tests.
    Maps ticker → sector ETF.
    """
    return {
        "AAPL": "XLK",
        "TSLA": "XLY",
        "MSFT": "XLK",
        "NVDA": "XLK",
    }


def make_ticker_to_name() -> dict:
    """
    Synthetic sector name lookup dict — replaces database call in tests.
    Maps ticker → human readable sector name.
    """
    return {
        "AAPL": "Technology",
        "TSLA": "Consumer Discretionary",
        "MSFT": "Technology",
        "NVDA": "Technology",
    }


def make_sentiment_df() -> pd.DataFrame:
    """Synthetic sentiment data for test stocks."""
    return pd.DataFrame([
        {
            "ticker"            : "AAPL",
            "date"              : TODAY,
            "put_call_ratio"    : 0.45,
            "short_interest_pct": 3.2,
        },
        {
            "ticker"            : "TSLA",
            "date"              : TODAY,
            "put_call_ratio"    : 0.89,
            "short_interest_pct": 18.7,
        },
    ])


# =============================================================================
# TESTS
# =============================================================================

def test_market_health_bullish():
    """All bullish indices should return bullish market bias."""
    df     = make_full_indicator_df()
    result = _check_market_health(df)

    assert result["SPY"]         == "bullish"
    assert result["QQQ"]         == "bullish"
    assert result["DIA"]         == "bullish"
    assert result["market_bias"] == "bullish"
    print("✅ Market health: all bullish indices = bullish bias")


def test_market_health_with_choch():
    """CHoCH on one index should result in mixed market bias."""
    df = make_full_indicator_df()

    # Inject CHoCH + downtrend on SPY
    df.loc[df["ticker"] == "SPY", "choch_detected"]   = 1
    df.loc[df["ticker"] == "SPY", "linreg_slope_up"]  = 0

    result = _check_market_health(df)
    assert result["SPY"]         == "broken"
    assert result["market_bias"] == "mixed"
    print("✅ Market health: CHoCH on SPY correctly produces mixed bias")


def test_sector_health():
    """All bullish sector ETFs should return bullish status."""
    df     = make_full_indicator_df()
    result = _check_sector_health(df)

    assert result["XLK"]  == "bullish"
    assert result["XLE"]  == "bullish"
    assert result["XLRE"] == "bullish"
    assert result["XLF"]  == "bullish"
    print("✅ Sector health: all bullish sectors detected correctly")


def test_sector_health_with_choch():
    """CHoCH on a sector should mark it as broken."""
    df = make_full_indicator_df()

    # Inject CHoCH on XLK (Technology)
    df.loc[df["ticker"] == "XLK", "choch_detected"]  = 1
    df.loc[df["ticker"] == "XLK", "linreg_slope_up"] = 0

    result = _check_sector_health(df)
    assert result["XLK"] == "broken"
    print("✅ Sector health: CHoCH on XLK correctly marked as broken")


def test_long_candidate_valid():
    """
    AAPL with all conditions met should pass long check.
    Uses synthetic sector lookup — no database needed.
    """
    df            = make_full_indicator_df()
    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()
    aapl_row      = df[df["ticker"] == "AAPL"].iloc[0]

    result = _is_long_candidate(aapl_row, sector_health, ticker_to_etf)
    assert result == True
    print("✅ Long candidate: AAPL correctly identified as valid long")


def test_long_candidate_wrong_volume():
    """TSLA with neutral volume should fail long check."""
    df            = make_full_indicator_df()
    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()
    tsla_row      = df[df["ticker"] == "TSLA"].iloc[0]

    result = _is_long_candidate(tsla_row, sector_health, ticker_to_etf)
    assert result == False
    print("✅ Long candidate: TSLA correctly rejected (neutral volume)")


def test_long_candidate_choch():
    """MSFT with CHoCH should fail long check."""
    df            = make_full_indicator_df()
    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()
    msft_row      = df[df["ticker"] == "MSFT"].iloc[0]

    result = _is_long_candidate(msft_row, sector_health, ticker_to_etf)
    assert result == False
    print("✅ Long candidate: MSFT correctly rejected (CHoCH detected)")


def test_long_candidate_wrong_sd_position():
    """NVDA not in buy zone should fail long check."""
    df            = make_full_indicator_df()
    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()
    nvda_row      = df[df["ticker"] == "NVDA"].iloc[0]

    result = _is_long_candidate(nvda_row, sector_health, ticker_to_etf)
    assert result == False
    print("✅ Long candidate: NVDA correctly rejected (not in buy zone)")


def test_long_candidate_bad_sector():
    """
    AAPL should fail long check if its sector (XLK) is not bullish.
    """
    df = make_full_indicator_df()

    # Make XLK bearish
    df.loc[df["ticker"] == "XLK", "linreg_slope_up"] = 0

    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()
    aapl_row      = df[df["ticker"] == "AAPL"].iloc[0]

    result = _is_long_candidate(aapl_row, sector_health, ticker_to_etf)
    assert result == False
    print("✅ Long candidate: AAPL correctly rejected (sector XLK not bullish)")


def test_long_candidate_unclassified_sector():
    """
    Stock with no sector mapping should pass through with warning.
    Unclassified stocks are not blocked — they are flagged on dashboard.
    """
    df            = make_full_indicator_df()
    sector_health = _check_sector_health(df)

    # Empty sector lookup — AAPL has no sector mapping
    ticker_to_etf = {}

    aapl_row = df[df["ticker"] == "AAPL"].iloc[0]
    result   = _is_long_candidate(aapl_row, sector_health, ticker_to_etf)

    # Should still pass — unclassified stocks are not blocked
    assert result == True
    print("✅ Long candidate: unclassified sector passes through correctly")


def test_short_candidate_valid():
    """Stock with full bearish setup should pass short check."""
    df = make_full_indicator_df()

    # Make all sectors bearish
    for sector in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC"]:
        df.loc[df["ticker"] == sector, "linreg_slope_up"] = 0

    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()

    # Build a valid short candidate row
    short_row = pd.Series(make_indicator_row(
        "AAPL",
        slope_up      = 0,
        choch         = 0,
        sd_position   = 2.1,
        volume_signal = "distribution"
    ))

    result = _is_short_candidate(short_row, sector_health, ticker_to_etf)
    assert result == True
    print("✅ Short candidate: valid short setup correctly identified")


def test_short_candidate_wrong_volume():
    """Short candidate with accumulation volume should fail."""
    df = make_full_indicator_df()

    for sector in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC"]:
        df.loc[df["ticker"] == sector, "linreg_slope_up"] = 0

    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()

    short_row = pd.Series(make_indicator_row(
        "AAPL",
        slope_up      = 0,
        choch         = 0,
        sd_position   = 2.1,
        volume_signal = "accumulation"   # Wrong volume for short
    ))

    result = _is_short_candidate(short_row, sector_health, ticker_to_etf)
    assert result == False
    print("✅ Short candidate: correctly rejected (wrong volume signal)")


def test_run_scanner():
    """
    Full scanner should return at least one long candidate.
    Mocks _load_sector_lookup to avoid database dependency.
    """
    df           = make_full_indicator_df()
    sentiment_df = make_sentiment_df()

    # Mock the database sector lookup so no SQLite needed
    with patch(
        "scanner.screener._load_sector_lookup",
        return_value=(make_ticker_to_etf(), make_ticker_to_name())
    ):
        result = run_scanner(df, sentiment_df, TODAY)

    assert not result.empty,            "Scanner returned no candidates"
    assert "ticker"      in result.columns
    assert "direction"   in result.columns
    assert "sector"      in result.columns
    assert "ml_score"    in result.columns
    assert "ml_rank"     in result.columns

    longs  = result[result["direction"] == "long"]
    shorts = result[result["direction"] == "short"]

    assert len(longs) >= 1,             "Expected at least one long candidate"
    assert longs.iloc[0]["ticker"] == "AAPL"

    print(f"✅ run_scanner: {len(result)} total candidates")
    print(f"   Longs: {len(longs)} | Shorts: {len(shorts)}")
    print(
        result[[
            "ticker", "direction", "sector",
            "sd_position", "volume_signal", "ml_score"
        ]].to_string(index=False)
    )

def test_long_candidate_no_valid_zone():
    """Stock without a valid demand zone should fail long check."""
    df            = make_full_indicator_df()
    sector_health = _check_sector_health(df)
    ticker_to_etf = make_ticker_to_etf()

    # AAPL but with has_valid_zone = 0
    row_dict = make_indicator_row(
        "AAPL", slope_up=1, choch=0, sd_position=-1.8,
        volume_signal="accumulation", has_valid_zone=0
    )
    row = pd.Series(row_dict)

    result = _is_long_candidate(row, sector_health, ticker_to_etf)
    assert result == False
    print("✅ Long candidate: correctly rejected (no valid demand zone)")


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    try:
        print("=" * 50)
        print("TESTING SCANNER WATERFALL")
        print("=" * 50)
        test_market_health_bullish()
        test_market_health_with_choch()
        test_sector_health()
        test_sector_health_with_choch()
        test_long_candidate_valid()
        test_long_candidate_no_valid_zone()
        test_long_candidate_wrong_volume()
        test_long_candidate_choch()
        test_long_candidate_wrong_sd_position()
        test_long_candidate_bad_sector()
        test_long_candidate_unclassified_sector()
        test_short_candidate_valid()
        test_short_candidate_wrong_volume()
        test_run_scanner()
        print("\n✅ All scanner tests passed")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()