"""
Test SQLite database — table creation, writes and reads.
Uses a temporary test database so it never touches the real one.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Point database to a temp test file
import yaml
config_path = Path("config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Override to test database path
config["database"]["path"] = "data/test_scanner.db"
with open(config_path, "w") as f:
    yaml.dump(config, f)

import pandas as pd
import numpy as np
from datetime import datetime
from data.database import (
    initialise_database,
    write_raw_prices,
    write_filtered_universe,
    write_indicator_results,
    write_sentiment_data,
    write_scan_results,
    write_model_metrics,
    read_raw_prices,
    read_filtered_universe,
    read_latest_scan_results,
    read_latest_model_metrics,
    get_last_fetch_date,
)

TODAY = datetime.today().strftime("%Y-%m-%d")


def make_ohlcv(ticker: str, n: int = 250) -> pd.DataFrame:
    """Helper to generate fake OHLCV data."""
    closes = 50 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "ticker": ticker,
        "date"  : pd.date_range(end=TODAY, periods=n).strftime("%Y-%m-%d"),
        "open"  : closes * 0.99,
        "high"  : closes * 1.01,
        "low"   : closes * 0.98,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n),
    })


def test_initialise():
    initialise_database()
    db_path = Path("data/test_scanner.db")
    assert db_path.exists()
    print("✅ Database initialised and file created")


def test_write_read_raw_prices():
    df = make_ohlcv("AAPL", 250)
    rows = write_raw_prices(df)
    assert rows == 250
    print(f"✅ write_raw_prices: {rows} rows inserted")

    result = read_raw_prices("AAPL")
    assert len(result) == 250
    assert list(result.columns[:6]) == ["id", "ticker", "date", "open", "high", "low"]
    print(f"✅ read_raw_prices: {len(result)} rows retrieved")

    # Test idempotency — inserting same data again should insert 0 rows
    rows_again = write_raw_prices(df)
    assert rows_again == 0
    print("✅ write_raw_prices: duplicate insert correctly blocked (0 rows inserted)")


def test_write_read_filtered_universe():
    df = pd.DataFrame({
        "ticker"    : ["AAPL", "MSFT", "NVDA"],
        "avg_volume": [85000000, 25000000, 45000000],
        "last_close": [185.0, 420.0, 875.0],
    })
    rows = write_filtered_universe(df, TODAY)
    assert rows == 3
    print(f"✅ write_filtered_universe: {rows} tickers written")

    result = read_filtered_universe(TODAY)
    assert len(result) == 3
    print(f"✅ read_filtered_universe: {len(result)} tickers retrieved")


def test_write_read_indicator_results():
    df = pd.DataFrame([{
        "ticker"           : "AAPL",
        "date"             : TODAY,
        "linreg_value"     : 182.5,
        "linreg_slope"     : 0.0023,
        "linreg_slope_up"  : 1,
        "sd1_upper"        : 187.5,
        "sd1_lower"        : 177.5,
        "sd2_upper"        : 192.5,
        "sd2_lower"        : 172.5,
        "sd3_upper"        : 197.5,
        "sd3_lower"        : 167.5,
        "price_sd_position": -1.8,
        "smc_structure"    : "bullish",
        "choch_detected"   : 0,
        "volume_signal"    : "accumulation",
    }])
    rows = write_indicator_results(df)
    assert rows >= 1
    print(f"✅ write_indicator_results: {rows} rows written")


def test_write_read_sentiment():
    df = pd.DataFrame([{
        "ticker"            : "AAPL",
        "date"              : TODAY,
        "put_call_ratio"    : 0.45,
        "short_interest_pct": 3.2,
    }])
    rows = write_sentiment_data(df)
    assert rows >= 1
    print(f"✅ write_sentiment_data: {rows} rows written")


def test_write_read_scan_results():
    df = pd.DataFrame([{
        "ticker"            : "AAPL",
        "direction"         : "long",
        "sector"            : "Technology",
        "sd_position"       : -1.8,
        "volume_signal"     : "accumulation",
        "put_call_ratio"    : 0.45,
        "short_interest_pct": 3.2,
        "ml_score"          : 0.81,
        "ml_rank"           : 1,
    }])
    rows = write_scan_results(df, TODAY)
    assert rows >= 1
    print(f"✅ write_scan_results: {rows} rows written")

    result = read_latest_scan_results(direction="long")
    assert len(result) >= 1
    assert result.iloc[0]["ticker"] == "AAPL"
    print(f"✅ read_latest_scan_results: {len(result)} results retrieved")


def test_write_read_model_metrics():
    write_model_metrics("signal_ranker", TODAY, 0.64, 0.67, 1500)
    write_model_metrics("volume_classifier", TODAY, 0.61, 0.63, 1200)
    print("✅ write_model_metrics: both models written")

    result = read_latest_model_metrics()
    assert len(result) == 2
    print(f"✅ read_latest_model_metrics: {len(result)} models retrieved")
    print(result[["model_name", "precision_score", "auc_roc_score"]].to_string(index=False))


def test_get_last_fetch_date():
    result = get_last_fetch_date("AAPL")
    assert result is not None
    print(f"✅ get_last_fetch_date: AAPL last date = {result}")

    result_missing = get_last_fetch_date("FAKEXYZ")
    assert result_missing is None
    print("✅ get_last_fetch_date: unknown ticker correctly returns None")


def cleanup():
    """Remove test database after tests."""
    test_db = Path("data/test_scanner.db")
    if test_db.exists():
        test_db.unlink()
        print("\n🧹 Test database cleaned up")

    # Restore real database path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["database"]["path"] = "/app/data/scanner.db"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print("🧹 config.yaml restored to production path")


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING DATABASE")
    print("=" * 50)
    test_initialise()
    test_write_read_raw_prices()
    test_write_read_filtered_universe()
    test_write_read_indicator_results()
    test_write_read_sentiment()
    test_write_read_scan_results()
    test_write_read_model_metrics()
    test_get_last_fetch_date()
    cleanup()
    print("\n✅ All database tests passed")