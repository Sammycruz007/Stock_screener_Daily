"""
ml/features.py
--------------
Feature engineering for both ML models.

LOGICAL FLOW:
─────────────
This module takes raw indicator outputs, sentiment data and
labelled observations and assembles them into clean feature
vectors ready for model training and inference.

FEATURES FOR SIGNAL RANKER:
   Group 1 — LinReg features:
   - sd_position       : Where price sits relative to LinReg (-1.8, -2.3 etc)
   - linreg_slope      : Steepness of LinReg slope (normalised)
   - distance_to_mean  : How far price needs to travel to reach LinReg

   Group 2 — Volume features:
   - vol_classifier_score : Output of Volume Classifier (probability)
   - volume_signal_enc    : Encoded volume signal (accumulation=1, neutral=0, distribution=-1)

   Group 3 — Sentiment features:
   - put_call_ratio     : Options market sentiment
   - short_interest_pct : Short interest % of float

   Group 4 — Market/Sector context:
   - market_slope_mean  : Average LinReg slope across SPY, QQQ, DIA
   - sector_slope       : LinReg slope of the stock's sector ETF
   - market_bullish     : 1 if all 3 indices bullish, 0 otherwise

   Group 5 — Stock momentum:
   - rsi_14             : 14-period RSI at time of signal

FEATURES FOR VOLUME CLASSIFIER:
   - cond1 through cond4 : Boolean accumulation conditions
   - dist1, dist2        : Boolean distribution conditions
   - sd_position         : Where price is relative to LinReg
   - avg_volume          : Baseline volume level

MISSING VALUE HANDLING:
   Put/Call Ratio and Short Interest may be None for some tickers.
   We impute with the median of available values.
   This is safer than dropping rows (would lose valid setups).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_ml_logger
from utils.error_handler import graceful, MLError

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = _load_config()


# =============================================================================
# RSI CALCULATOR
# Needed as a momentum feature — not importing from engine
# to keep feature engineering self-contained.
# =============================================================================

def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """
    Compute RSI for the most recent candle.

    FORMULA:
    RSI = 100 - (100 / (1 + RS))
    RS  = Average Gain / Average Loss over last N periods

    Args:
        closes: numpy array of closing prices
        period: RSI lookback period (default 14)

    Returns:
        RSI value between 0 and 100
    """
    if len(closes) < period + 1:
        return 50.0  # Return neutral RSI if insufficient data

    # Compute price changes
    deltas = np.diff(closes[-period - 1:])
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0   # All gains — max RSI

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(float(rsi), 2)


# =============================================================================
# VOLUME SIGNAL ENCODER
# Converts categorical volume signal to numeric for the model
# =============================================================================

VOLUME_SIGNAL_ENCODING = {
    "accumulation": 1,
    "neutral"     : 0,
    "distribution": -1,
}

def _encode_volume_signal(signal: str) -> int:
    """Convert volume signal string to integer for model input."""
    return VOLUME_SIGNAL_ENCODING.get(signal, 0)


# =============================================================================
# FEATURE BUILDER — SIGNAL RANKER
# Builds one feature vector per scanner candidate
# =============================================================================

def build_signal_ranker_features(
    indicator_row  : pd.Series,
    sentiment_row  : Optional[pd.Series],
    market_health  : dict,
    sector_health  : dict,
    closes         : np.ndarray,
    vol_score      : float = 0.5,
) -> dict:
    """
    Build a feature vector for the Signal Ranker model.

    Called for each stock candidate after the scanner waterfall.
    Assembles all available signals into a flat feature dict.

    Args:
        indicator_row : Row from indicator_results for this ticker
        sentiment_row : Row from sentiment_data (may be None)
        market_health : Dict of SPY/QQQ/DIA health statuses
        sector_health : Dict of sector ETF health statuses
        closes        : Recent closing prices for RSI calculation
        vol_score     : Volume Classifier probability output (0-1)

    Returns:
        Dict of feature name → feature value
        All values are numeric (floats or ints)
    """

    # ── Group 1: LinReg features ──────────────────────────────────────────────
    sd_position      = float(indicator_row["price_sd_position"])
    linreg_value     = float(indicator_row["linreg_value"])
    linreg_slope     = float(indicator_row["linreg_slope"])
    current_close    = closes[-1] if len(closes) > 0 else linreg_value

    # Distance to mean = how far price needs to travel to reach LinReg
    # Expressed as a percentage of current price
    distance_to_mean = abs(current_close - linreg_value) / current_close * 100

    # ── Group 2: Volume features ──────────────────────────────────────────────
    volume_signal     = str(indicator_row.get("volume_signal", "neutral"))
    volume_signal_enc = _encode_volume_signal(volume_signal)

    # ── Group 3: Sentiment features ───────────────────────────────────────────
    # Use None as placeholder — imputed at the DataFrame level later
    if sentiment_row is not None and not sentiment_row.empty:
        put_call_ratio     = sentiment_row.get("put_call_ratio", None)
        short_interest_pct = sentiment_row.get("short_interest_pct", None)
    else:
        put_call_ratio     = None
        short_interest_pct = None

    # Convert to float safely
    put_call_ratio     = float(put_call_ratio)     if put_call_ratio     is not None else np.nan
    short_interest_pct = float(short_interest_pct) if short_interest_pct is not None else np.nan

    # ── Group 4: Market/Sector context ───────────────────────────────────────
    # Average slope across the 3 indices
    # 1 = all bullish, 0 = mixed, -1 = all bearish
    index_statuses   = [
        market_health.get("SPY", "broken"),
        market_health.get("QQQ", "broken"),
        market_health.get("DIA", "broken"),
    ]
    bullish_count    = sum(1 for s in index_statuses if s == "bullish")
    bearish_count    = sum(1 for s in index_statuses if s == "bearish")
    market_bullish   = 1 if bullish_count == 3 else (0 if bullish_count >= 2 else -1)

    # Sector health encoding
    ticker          = str(indicator_row["ticker"])
    sector_etf      = indicator_row.get("sector_etf", None)
    sector_status   = sector_health.get(sector_etf, "broken") if sector_etf else "broken"
    sector_bullish  = 1 if sector_status == "bullish" else (0 if sector_status == "bearish" else -1)

    # ── Group 5: Momentum ─────────────────────────────────────────────────────
    rsi_14 = _compute_rsi(closes, period=14)

    # ── Assemble feature vector ───────────────────────────────────────────────
    features = {
        # LinReg
        "sd_position"       : sd_position,
        "linreg_slope"      : linreg_slope,
        "distance_to_mean"  : round(distance_to_mean, 4),

        # Volume
        "vol_classifier_score" : vol_score,
        "volume_signal_enc"    : volume_signal_enc,

        # Sentiment
        "put_call_ratio"       : put_call_ratio,
        "short_interest_pct"   : short_interest_pct,

        # Market/Sector context
        "market_bullish"       : market_bullish,
        "sector_bullish"       : sector_bullish,

        # Momentum
        "rsi_14"               : rsi_14,
    }

    return features


# =============================================================================
# FEATURE BUILDER — VOLUME CLASSIFIER
# Builds feature vector from volume condition booleans
# =============================================================================

def build_volume_classifier_features(
    labelled_row : pd.Series,
) -> dict:
    """
    Build a feature vector for the Volume Classifier model.

    The volume classifier features are simpler than the signal ranker —
    they are the raw boolean condition outputs from the volume engine
    plus the SD position and average volume context.

    Args:
        labelled_row: Row from volume labeller output

    Returns:
        Dict of feature name → feature value
    """
    return {
        "cond1"       : int(labelled_row["cond1"]),   # Volume declining on down days
        "cond2"       : int(labelled_row["cond2"]),   # Shakeout present
        "cond3"       : int(labelled_row["cond3"]),   # Volume expanding on green days
        "cond4"       : int(labelled_row["cond4"]),   # Volume dry-up present
        "dist1"       : int(labelled_row["dist1"]),   # Volume rising on up days
        "dist2"       : int(labelled_row["dist2"]),   # Volume expanding on red days
        "sd_position" : float(labelled_row["sd_position"]),
        "avg_volume"  : float(labelled_row["avg_volume"]),
    }


# =============================================================================
# FULL FEATURE MATRIX BUILDERS
# Converts labelled DataFrames into X (features) and y (labels)
# ready for model training.
# =============================================================================

def build_signal_ranker_matrix(
    labelled_df    : pd.DataFrame,
    indicator_df   : pd.DataFrame,
    sentiment_df   : pd.DataFrame,
    market_health  : dict,
    sector_health  : dict,
    tickers_data   : dict[str, pd.DataFrame],
    vol_scores     : Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature matrix for Signal Ranker training.

    FLOW:
    1. Iterate over each labelled observation
    2. Look up indicator data for that ticker + date
    3. Look up sentiment data for that ticker + date
    4. Compute RSI from raw OHLCV
    5. Build feature vector
    6. Collect all vectors into X matrix
    7. Extract labels into y Series
    8. Impute missing values (Put/Call, Short Interest)

    Args:
        labelled_df  : Output from run_labeller()
        indicator_df : Historical indicator results from SQLite
        sentiment_df : Historical sentiment data from SQLite
        market_health: Market health dict for context features
        sector_health: Sector health dict for context features
        tickers_data : Dict of ticker → OHLCV DataFrame
        vol_scores   : Optional dict of ticker+date → vol classifier score

    Returns:
        Tuple of (X DataFrame, y Series)
    """
    if labelled_df.empty:
        logger.warning("build_signal_ranker_matrix: Empty labelled DataFrame")
        return pd.DataFrame(), pd.Series()

    feature_rows = []
    labels       = []

    for _, row in labelled_df.iterrows():
        ticker = row["ticker"]
        date   = row["date"]

        # ── Look up indicator data ────────────────────────────────────────────
        ind_rows = indicator_df[
            (indicator_df["ticker"] == ticker) &
            (indicator_df["date"]   == date)
        ]
        if ind_rows.empty:
            continue
        ind_row = ind_rows.iloc[0]

        # ── Look up sentiment data ────────────────────────────────────────────
        sent_rows = sentiment_df[
            (sentiment_df["ticker"] == ticker) &
            (sentiment_df["date"]   == date)
        ]
        sent_row = sent_rows.iloc[0] if not sent_rows.empty else None

        # ── Get closes for RSI ────────────────────────────────────────────────
        df_ticker = tickers_data.get(ticker, pd.DataFrame())
        if df_ticker.empty:
            continue

        # Get closes up to and including this date
        df_up_to_date = df_ticker[df_ticker["date"] <= date]
        closes        = df_up_to_date["close"].values[-50:]  # Last 50 for RSI

        # ── Get volume classifier score ───────────────────────────────────────
        vol_score = 0.5   # Default neutral score
        if vol_scores:
            vol_score = vol_scores.get(f"{ticker}_{date}", 0.5)

        # ── Build feature vector ──────────────────────────────────────────────
        features = build_signal_ranker_features(
            indicator_row = ind_row,
            sentiment_row = sent_row,
            market_health = market_health,
            sector_health = sector_health,
            closes        = closes,
            vol_score     = vol_score,
        )

        feature_rows.append(features)
        labels.append(int(row["label"]))

    if not feature_rows:
        logger.warning("build_signal_ranker_matrix: No feature rows built")
        return pd.DataFrame(), pd.Series()

    X = pd.DataFrame(feature_rows)
    y = pd.Series(labels, name="label")

    # ── Impute missing sentiment values with column median ────────────────────
    # Safe imputation: median is robust to outliers
    for col in ["put_call_ratio", "short_interest_pct"]:
        if col in X.columns:
            median_val = X[col].median()
            X[col]     = X[col].fillna(median_val)

    logger.info(
        f"Signal Ranker feature matrix | "
        f"Shape: {X.shape} | "
        f"Label balance: {y.mean():.2%} positive"
    )

    return X, y


def build_volume_classifier_matrix(
    labelled_df : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the full feature matrix for Volume Classifier training.

    Much simpler than the Signal Ranker matrix —
    all features are already in the labelled DataFrame.

    Args:
        labelled_df: Output from run_volume_labeller()

    Returns:
        Tuple of (X DataFrame, y Series)
    """
    if labelled_df.empty:
        logger.warning("build_volume_classifier_matrix: Empty DataFrame")
        return pd.DataFrame(), pd.Series()

    feature_cols = ["cond1", "cond2", "cond3", "cond4",
                    "dist1", "dist2", "sd_position", "avg_volume"]

    X = labelled_df[feature_cols].copy().astype(float)
    y = labelled_df["label"].astype(int)

    logger.info(
        f"Volume Classifier feature matrix | "
        f"Shape: {X.shape} | "
        f"Label balance: {y.mean():.2%} positive"
    )

    return X, y


# =============================================================================
# INFERENCE FEATURE BUILDER
# Used at scan time (not training time) to build features
# for today's candidates to score them with the trained model.
# =============================================================================

def build_inference_features(
    candidates_df  : pd.DataFrame,
    indicator_df   : pd.DataFrame,
    sentiment_df   : pd.DataFrame,
    market_health  : dict,
    sector_health  : dict,
    tickers_data   : dict[str, pd.DataFrame],
    vol_scores     : dict,
) -> pd.DataFrame:
    """
    Build feature vectors for today's scanner candidates.
    These are fed into the trained model to generate ML scores.

    Same feature engineering as training time — critical for
    avoiding train/inference mismatch.

    Args:
        candidates_df : Today's scanner candidates from screener.py
        indicator_df  : Today's indicator results
        sentiment_df  : Today's sentiment data
        market_health : Today's market health
        sector_health : Today's sector health
        tickers_data  : Dict of ticker → OHLCV DataFrame
        vol_scores    : Dict of ticker → vol classifier probability

    Returns:
        DataFrame of features — one row per candidate
        Same columns as training feature matrix
    """
    feature_rows = []
    tickers      = []

    for _, row in candidates_df.iterrows():
        ticker = row["ticker"]

        # ── Look up indicator data ────────────────────────────────────────────
        ind_rows = indicator_df[indicator_df["ticker"] == ticker]
        if ind_rows.empty:
            continue
        ind_row = ind_rows.iloc[0]

        # ── Look up sentiment data ────────────────────────────────────────────
        sent_rows = sentiment_df[sentiment_df["ticker"] == ticker]
        sent_row  = sent_rows.iloc[0] if not sent_rows.empty else None

        # ── Get closes for RSI ────────────────────────────────────────────────
        df_ticker = tickers_data.get(ticker, pd.DataFrame())
        closes    = df_ticker["close"].values[-50:] if not df_ticker.empty else np.array([])

        # ── Get volume classifier score ───────────────────────────────────────
        vol_score = vol_scores.get(ticker, 0.5)

        # ── Build feature vector ──────────────────────────────────────────────
        features = build_signal_ranker_features(
            indicator_row = ind_row,
            sentiment_row = sent_row,
            market_health = market_health,
            sector_health = sector_health,
            closes        = closes,
            vol_score     = vol_score,
        )

        feature_rows.append(features)
        tickers.append(ticker)

    if not feature_rows:
        logger.warning("build_inference_features: No features built")
        return pd.DataFrame()

    X            = pd.DataFrame(feature_rows)
    X["ticker"]  = tickers

    # ── Impute missing sentiment values ───────────────────────────────────────
    for col in ["put_call_ratio", "short_interest_pct"]:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    logger.info(f"Inference features built | {len(X)} candidates | {X.shape[1]} features")
    return X