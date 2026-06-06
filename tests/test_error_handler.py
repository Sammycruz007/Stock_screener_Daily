"""
Test retry decorator, graceful decorator and custom exceptions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.error_handler import (
    retry,
    graceful,
    validate_dataframe,
    DataFetchError,
    EngineError,
    StockScannerBaseError,
)
import pandas as pd
import numpy as np

# ─────────────────────────────────────
# Test 1: retry decorator
# ─────────────────────────────────────
def test_retry():
    attempt_count = {"n": 0}

    @retry(attempts=3, delay_seconds=0.1, exceptions=(DataFetchError,))
    def flaky_function():
        attempt_count["n"] += 1
        if attempt_count["n"] < 3:
            raise DataFetchError("Simulated fetch failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert attempt_count["n"] == 3
    print(f"✅ Retry decorator: succeeded on attempt {attempt_count['n']}")


# ─────────────────────────────────────
# Test 2: retry exhausted — should raise
# ─────────────────────────────────────
def test_retry_exhausted():
    @retry(attempts=2, delay_seconds=0.1, exceptions=(DataFetchError,), raise_on_exhausted=True)
    def always_fails():
        raise DataFetchError("Always fails")

    try:
        always_fails()
        assert False, "Should have raised"
    except DataFetchError:
        print("✅ Retry exhausted: correctly raised after all attempts")


# ─────────────────────────────────────
# Test 3: graceful decorator
# ─────────────────────────────────────
def test_graceful():
    @graceful(default_return="fallback", exceptions=(EngineError,))
    def broken_engine():
        raise EngineError("Engine exploded")

    result = broken_engine()
    assert result == "fallback"
    print("✅ Graceful decorator: returned fallback value instead of crashing")


# ─────────────────────────────────────
# Test 4: validate_dataframe
# ─────────────────────────────────────
def test_validate_dataframe():
    # Valid DataFrame — should pass
    df_valid = pd.DataFrame({
        "open"  : np.random.rand(250),
        "high"  : np.random.rand(250),
        "low"   : np.random.rand(250),
        "close" : np.random.rand(250),
        "volume": np.random.randint(100000, 1000000, 250),
    })
    assert validate_dataframe(df_valid, "AAPL", ["open", "high", "low", "close", "volume"])
    print("✅ validate_dataframe: valid DataFrame passed correctly")

    # Empty DataFrame — should fail
    df_empty = pd.DataFrame()
    assert not validate_dataframe(df_empty, "AAPL", ["close"])
    print("✅ validate_dataframe: empty DataFrame rejected correctly")

    # Missing column — should fail
    df_missing = df_valid.drop(columns=["volume"])
    assert not validate_dataframe(df_missing, "AAPL", ["open", "high", "low", "close", "volume"])
    print("✅ validate_dataframe: missing column rejected correctly")

    # Too few rows — should fail
    df_short = df_valid.head(100)
    assert not validate_dataframe(df_short, "AAPL", ["open", "high", "low", "close", "volume"])
    print("✅ validate_dataframe: insufficient rows rejected correctly")


# ─────────────────────────────────────
# Test 5: custom exception hierarchy
# ─────────────────────────────────────
def test_exception_hierarchy():
    try:
        raise DataFetchError("test")
    except StockScannerBaseError:
        print("✅ Exception hierarchy: DataFetchError caught as StockScannerBaseError")


if __name__ == "__main__":
    print("=" * 50)
    print("TESTING ERROR HANDLER")
    print("=" * 50)
    test_retry()
    test_retry_exhausted()
    test_graceful()
    test_validate_dataframe()
    test_exception_hierarchy()
    print("\n✅ All error handler tests passed")