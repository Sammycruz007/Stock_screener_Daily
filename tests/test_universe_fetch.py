"""
Test that NASDAQ FTP URLs are reachable and return valid ticker data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetcher import (
    _fetch_nasdaq_listed,
    _fetch_other_listed,
    get_full_universe,
    _clean_ticker,
)


def test_clean_ticker():
    """Ticker cleaning rules work correctly."""
    assert _clean_ticker("AAPL")    == "AAPL"       # Normal ticker
    assert _clean_ticker("BRK/B")   == "BRK-B"      # Slash replaced
    assert _clean_ticker("AAPL$")   is None          # $ symbol excluded
    assert _clean_ticker("AAPLWW")  is None          # Too long excluded
    assert _clean_ticker("")        is None          # Empty excluded
    assert _clean_ticker("  MSFT ") == "MSFT"        # Whitespace stripped
    print("✅ _clean_ticker: all cleaning rules work correctly")


def test_fetch_nasdaq_listed():
    """NASDAQ listed file should return thousands of tickers."""
    tickers = _fetch_nasdaq_listed()

    assert len(tickers) > 1000, f"Expected 1000+, got {len(tickers)}"

    # Spot check some well known NASDAQ tickers
    assert "AAPL" in tickers, "AAPL should be in NASDAQ listed"
    assert "MSFT" in tickers, "MSFT should be in NASDAQ listed"
    assert "NVDA" in tickers, "NVDA should be in NASDAQ listed"

    # Confirm no dirty tickers slipped through
    for t in tickers:
        assert "$" not in t,    f"Found $ in ticker: {t}"
        assert len(t) <= 5,     f"Found ticker longer than 5 chars: {t}"
        assert t == t.strip(),  f"Found unstripped ticker: {t}"

    print(f"✅ NASDAQ listed: {len(tickers)} tickers fetched and validated")
    print(f"   Sample: {tickers[:5]}")


def test_fetch_other_listed():
    """NYSE/other listed file should return thousands of tickers."""
    tickers = _fetch_other_listed()

    assert len(tickers) > 500, f"Expected 500+, got {len(tickers)}"

    # Spot check some well known NYSE tickers
    assert "JPM"  in tickers, "JPM should be in NYSE listed"
    assert "BAC"  in tickers, "BAC should be in NYSE listed"

    print(f"✅ NYSE/other listed: {len(tickers)} tickers fetched and validated")
    print(f"   Sample: {tickers[:5]}")


def test_get_full_universe():
    """Full universe should combine both files with no duplicates."""
    universe = get_full_universe()

    # Should be well over 3000 unique tickers
    assert len(universe) > 3000, f"Expected 3000+, got {len(universe)}"

    # No duplicates
    assert len(universe) == len(set(universe)), "Duplicates found in universe"

    # Indices and sectors included
    for ticker in ["SPY", "QQQ", "DIA"]:
        assert ticker in universe, f"{ticker} missing from universe"

    for ticker in ["XLK", "XLF", "XLE", "XLV"]:
        assert ticker in universe, f"{ticker} missing from universe"

    # Should be sorted
    assert universe == sorted(universe), "Universe is not sorted"

    print(f"✅ Full universe: {len(universe)} unique tickers")
    print(f"   Sample first 5: {universe[:5]}")
    print(f"   Sample last 5:  {universe[-5:]}")


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING UNIVERSE FETCH")
    print("=" * 50)
    test_clean_ticker()
    print("\nFetching NASDAQ listed (requires internet)...")
    test_fetch_nasdaq_listed()
    print("\nFetching NYSE/other listed (requires internet)...")
    test_fetch_other_listed()
    print("\nBuilding full universe...")
    test_get_full_universe()
    print("\n✅ All universe fetch tests passed")