"""
Test ML pipeline — labeller, features, volume classifier, signal ranker.
Uses synthetic data so no database or internet connection needed.
"""
import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.labeller import label_volume_patterns, label_scanner_hits
from ml.features import (
    compute_volume_features,
    compute_signal_features,
    build_volume_feature_matrix,
    VOLUME_FEATURE_COLS,
    SIGNAL_FEATURE_COLS,
)
from ml.volume_classifier import (
    train_volume_classifier,
    score_volume_signals,
    load_volume_classifier,
)
from ml.signal_ranker import (
    train_signal_ranker,
    score_candidates,
    load_signal_ranker,
)

TODAY  = datetime.today().strftime("%Y-%m-%d")
TICKER = "AAPL"


# =============================================================================
# SYNTHETIC DATA HELPERS
# =============================================================================

def make_price_df(
    ticker  : str   = "AAPL",
    n       : int   = 300,
    trend   : float = 0.05,
) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    dates  = pd.date_range(end=TODAY, periods=n)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5 + trend)
    closes = np.maximum(closes, 1.0)

    return pd.DataFrame({
        "ticker": ticker,
        "date"  : dates.strftime("%Y-%m-%d"),
        "open"  : closes * 0.999,
        "high"  : closes * 1.005,
        "low"   : closes * 0.995,
        "close" : closes,
        "volume": np.random.randint(500000, 5000000, n).astype(float),
    })


def make_indicator_df(
    ticker    : str   = "AAPL",
    n         : int   = 300,
    slope_up  : int   = 1,
    sd_pos    : float = -1.8,
) -> pd.DataFrame:
    """Generate synthetic indicator results DataFrame."""
    dates = pd.date_range(end=TODAY, periods=n)
    return pd.DataFrame({
        "ticker"           : ticker,
        "date"             : dates.strftime("%Y-%m-%d"),
        "linreg_value"     : 100.0 + np.arange(n) * 0.05,
        "linreg_slope"     : 0.002 if slope_up else -0.002,
        "linreg_slope_up"  : slope_up,
        "sd1_upper"        : 105.0,
        "sd1_lower"        : 95.0,
        "sd2_upper"        : 110.0,
        "sd2_lower"        : 90.0,
        "sd3_upper"        : 115.0,
        "sd3_lower"        : 85.0,
        "price_sd_position": sd_pos,
        "smc_structure"    : "bullish" if slope_up else "bearish",
        "choch_detected"   : 0,
        "volume_signal"    : "accumulation",
    })


def make_sentiment_df(ticker: str = "AAPL") -> pd.DataFrame:
    """Generate synthetic sentiment DataFrame."""
    dates = pd.date_range(end=TODAY, periods=300)
    return pd.DataFrame({
        "ticker"            : ticker,
        "date"              : dates.strftime("%Y-%m-%d"),
        "put_call_ratio"    : np.random.uniform(0.4, 1.5, 300),
        "short_interest_pct": np.random.uniform(1.0, 20.0, 300),
    })


def make_large_feature_matrix(n: int = 600) -> pd.DataFrame:
    """
    Generate a large synthetic feature matrix for model training.
    Needs at least MIN_SAMPLES (500) rows.
    """
    rows = []
    for _ in range(n):
        row = {col: np.random.randn() for col in VOLUME_FEATURE_COLS}
        row["dryup_candle_present"] = np.random.randint(0, 2)
        row["n_red_candles"]        = np.random.randint(0, 10)
        row["n_green_candles"]      = np.random.randint(0, 10)
        row["label"]                = np.random.randint(0, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def make_large_signal_matrix(n: int = 600) -> pd.DataFrame:
    """Generate a large synthetic signal feature matrix."""
    rows = []
    for _ in range(n):
        row = {col: np.random.randn() for col in SIGNAL_FEATURE_COLS}
        row["dryup_candle_present"] = np.random.randint(0, 2)
        row["n_red_candles"]        = np.random.randint(0, 10)
        row["n_green_candles"]      = np.random.randint(0, 10)
        row["market_choch_count"]   = np.random.randint(0, 3)
        row["direction_flag"]       = np.random.randint(0, 2)
        row["label"]                = np.random.randint(0, 2)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# TESTS
# =============================================================================

def test_volume_features():
    """Volume feature computation returns correct structure."""
    df       = make_price_df(n=60)
    features = compute_volume_features(df, TODAY)

    assert features is not None, "Volume features returned None"
    assert len(features) == len(VOLUME_FEATURE_COLS)

    for col in VOLUME_FEATURE_COLS:
        assert col in features, f"Missing feature: {col}"

    print(f"✅ compute_volume_features: {len(features)} features computed")
    for k, v in features.items():
        print(f"   {k}: {v}")


def test_volume_features_insufficient_data():
    """Should return None when insufficient data."""
    df     = make_price_df(n=10)   # Not enough data
    result = compute_volume_features(df, TODAY)
    assert result is None
    print("✅ compute_volume_features: correctly returns None for insufficient data")


def test_label_volume_patterns():
    """Volume pattern labeller generates labels correctly."""
    prices_df    = make_price_df(n=300)
    indicators_df = make_indicator_df(n=300, sd_pos=-1.8)

    labels = label_volume_patterns(prices_df, indicators_df)

    # With sd_pos=-1.8, all rows are in long zone — should generate labels
    assert isinstance(labels, pd.DataFrame)
    assert "label"  in labels.columns
    assert "ticker" in labels.columns
    assert "date"   in labels.columns

    print(
        f"✅ label_volume_patterns: {len(labels)} labels generated | "
        f"Positive: {(labels['label']==1).sum()} | "
        f"Negative: {(labels['label']==0).sum()}"
    )


def test_label_scanner_hits():
    """Signal ranker labeller generates labels from scanner hits."""
    prices_df     = make_price_df(n=300)
    indicators_df = make_indicator_df(n=300)

    # Simulate scanner hits — 5 historical dates
    dates = pd.date_range(end=TODAY, periods=300).strftime("%Y-%m-%d").tolist()
    scan_hits = pd.DataFrame([
        {"ticker": TICKER, "date": dates[i*30], "direction": "long"}
        for i in range(5)
    ])

    labels = label_scanner_hits(prices_df, indicators_df, scan_hits)

    assert isinstance(labels, pd.DataFrame)
    if len(labels) > 0:
        assert "label"     in labels.columns
        assert "direction" in labels.columns

    print(f"✅ label_scanner_hits: {len(labels)} labels generated")


def test_build_volume_feature_matrix():
    """Feature matrix builder produces correct shape."""
    prices_df     = make_price_df(n=300)
    indicators_df = make_indicator_df(n=300, sd_pos=-1.8)

    labels = label_volume_patterns(prices_df, indicators_df)

    if labels.empty:
        print("⚠️  No labels generated — skipping matrix build test")
        return

    matrix = build_volume_feature_matrix(prices_df, indicators_df, labels)

    assert not matrix.empty
    assert "label" in matrix.columns
    for col in VOLUME_FEATURE_COLS:
        assert col in matrix.columns, f"Missing column: {col}"

    print(
        f"✅ build_volume_feature_matrix: "
        f"{len(matrix)} rows × {len(matrix.columns)} columns"
    )


def test_train_volume_classifier():
    """Volume Classifier trains and saves successfully."""
    matrix   = make_large_feature_matrix(n=600)
    pipeline, metrics = train_volume_classifier(matrix)

    assert pipeline   is not None
    assert "precision" in metrics
    assert "auc_roc"   in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["auc_roc"]   <= 1.0

    print(
        f"✅ train_volume_classifier | "
        f"Precision: {metrics['precision']} | "
        f"AUC-ROC: {metrics['auc_roc']}"
    )


def test_load_volume_classifier():
    """Volume Classifier loads from disk correctly."""
    pipeline = load_volume_classifier()
    assert pipeline is not None
    print("✅ load_volume_classifier: model loaded from disk")


def test_score_volume_signals():
    """Volume scoring returns scores for all tickers."""
    tickers_data = {
        "AAPL": make_price_df("AAPL", n=60),
        "MSFT": make_price_df("MSFT", n=60),
        "TSLA": make_price_df("TSLA", n=60),
    }

    pipeline = load_volume_classifier()
    scores   = score_volume_signals(tickers_data, TODAY, pipeline)

    assert len(scores) == 3
    for ticker, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{ticker} score out of range: {score}"

    print(f"✅ score_volume_signals: {len(scores)} tickers scored")
    for ticker, score in scores.items():
        print(f"   {ticker}: {score:.4f}")


def test_train_signal_ranker():
    """Signal Ranker trains and saves successfully."""
    matrix   = make_large_signal_matrix(n=600)
    pipeline, metrics = train_signal_ranker(matrix)

    assert pipeline   is not None
    assert "precision" in metrics
    assert "auc_roc"   in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["auc_roc"]   <= 1.0

    print(
        f"✅ train_signal_ranker | "
        f"Precision: {metrics['precision']} | "
        f"AUC-ROC: {metrics['auc_roc']}"
    )


def test_load_signal_ranker():
    """Signal Ranker loads from disk correctly."""
    pipeline = load_signal_ranker()
    assert pipeline is not None
    print("✅ load_signal_ranker: model loaded from disk")


def test_score_candidates():
    """Score candidates returns ranked DataFrame."""
    # Build synthetic candidates
    candidates_df = pd.DataFrame([{
        "ticker"            : "AAPL",
        "direction"         : "long",
        "sector"            : "Technology",
        "sd_position"       : -1.8,
        "volume_signal"     : "accumulation",
        "put_call_ratio"    : 0.45,
        "short_interest_pct": 3.2,
        "ml_score"          : 0.0,
        "ml_rank"           : 0,
    }])

    prices_df     = make_price_df("AAPL", n=300)
    indicators_df = make_indicator_df("AAPL", n=300)
    sentiment_df  = make_sentiment_df("AAPL")
    market_ind_df = pd.concat([
        make_indicator_df("SPY", n=300),
        make_indicator_df("QQQ", n=300),
        make_indicator_df("DIA", n=300),
    ])
    vol_scores    = {"AAPL": 0.72}
    pipeline      = load_signal_ranker()

    result = score_candidates(
        candidates_df  = candidates_df,
        prices_df      = prices_df,
        indicators_df  = indicators_df,
        market_ind_df  = market_ind_df,
        vol_scores     = vol_scores,
        signal_date    = TODAY,
        pipeline       = pipeline,
    )

    assert not result.empty
    assert "ml_score" in result.columns
    assert "ml_rank"  in result.columns
    assert result.iloc[0]["ml_rank"] == 1

    print(
        f"✅ score_candidates: {len(result)} candidates scored | "
        f"Top score: {result.iloc[0]['ml_score']:.4f}"
    )


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    try:
        print("=" * 55)
        print("TESTING ML PIPELINE")
        print("=" * 55)

        print("\n--- Feature Engineering ---")
        test_volume_features()
        test_volume_features_insufficient_data()

        print("\n--- Labelling ---")
        test_label_volume_patterns()
        test_label_scanner_hits()
        test_build_volume_feature_matrix()

        print("\n--- Volume Classifier ---")
        test_train_volume_classifier()
        test_load_volume_classifier()
        test_score_volume_signals()

        print("\n--- Signal Ranker ---")
        test_train_signal_ranker()
        test_load_signal_ranker()
        test_score_candidates()

        print("\n✅ All ML tests passed")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()