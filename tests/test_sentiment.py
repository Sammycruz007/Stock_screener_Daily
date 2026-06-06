"""
Test sentiment module — Put/Call ratio interpretation and Short Interest.
We mock yfinance calls so tests run without internet connection.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentiment.sentiment import (
    interpret_put_call,
    interpret_short_interest,
    fetch_sentiment_batch,
)


def test_interpret_put_call():
    """Put/Call ratio interpretation thresholds."""
    assert interpret_put_call(0.5)  == "bullish"     # Below 0.7
    assert interpret_put_call(0.95) == "neutral"     # Between 0.7 and 1.2
    assert interpret_put_call(1.5)  == "bearish"     # Above 1.2
    assert interpret_put_call(None) == "unavailable"
    print("✅ interpret_put_call: all thresholds correct")


def test_interpret_short_interest():
    """Short interest interpretation thresholds."""
    assert interpret_short_interest(3.0)  == "normal"
    assert interpret_short_interest(18.0) == "high"
    assert interpret_short_interest(None) == "unavailable"
    print("✅ interpret_short_interest: all thresholds correct")


def test_fetch_sentiment_batch_mocked():
    """
    Test batch fetcher with mocked yfinance calls.
    We mock yf.Ticker so no internet connection is needed.
    """
    # Build mock options chain
    mock_calls = MagicMock()
    mock_calls.empty = False
    mock_calls.__getitem__ = lambda self, key: pd.Series([1000, 2000, 1500])
    mock_calls["openInterest"] = pd.Series([1000, 2000, 1500])

    mock_puts = MagicMock()
    mock_puts.empty = False
    mock_puts["openInterest"] = pd.Series([800, 1200, 600])

    mock_chain        = MagicMock()
    mock_chain.calls  = pd.DataFrame({"openInterest": [1000, 2000, 1500]})
    mock_chain.puts   = pd.DataFrame({"openInterest": [800, 1200, 600]})

    mock_ticker             = MagicMock()
    mock_ticker.options     = ["2026-06-20"]
    mock_ticker.option_chain.return_value = mock_chain
    mock_ticker.info        = {"shortPercentOfFloat": 0.032}

    with patch("sentiment.sentiment.yf.Ticker", return_value=mock_ticker):
        result = fetch_sentiment_batch(["AAPL", "MSFT"], "2026-06-06")

    assert len(result) == 2
    assert "ticker"             in result.columns
    assert "put_call_ratio"     in result.columns
    assert "short_interest_pct" in result.columns
    print(f"✅ fetch_sentiment_batch: {len(result)} rows returned")
    print(result[["ticker", "put_call_ratio", "short_interest_pct"]].to_string(index=False))


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING SENTIMENT MODULE")
    print("=" * 50)
    test_interpret_put_call()
    test_interpret_short_interest()
    test_fetch_sentiment_batch_mocked()
    print("\n✅ All sentiment tests passed")