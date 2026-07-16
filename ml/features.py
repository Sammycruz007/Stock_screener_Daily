"""
ml/features.py
--------------
Feature engineering for both ML models.

LOGICAL FLOW:
─────────────
Both models share a common feature set but with different focuses:

VOLUME CLASSIFIER FEATURES:
   Focused purely on volume behaviour patterns.
   These are the raw ingredients the model uses to learn
   what true accumulation/distribution looks like.

SIGNAL RANKER FEATURES:
   Broader set covering the full setup quality:
   - Price position (which SD band, how close to mean)
   - LinReg trend strength (slope steepness)
   - Volume classifier output (reuses volume model score)
   - Sentiment (Put/Call Ratio, Short Interest)
   - Market context (how strong is the overall market)
   - Sector context (how strong is the sector)

FEATURE ENGINEERING PRINCIPLES:
   1. All features are normalised where needed
      (volume ratios instead of raw volumes —
       allows comparison across different stocks)
   2. Missing values are median-imputed
      (some tickers lack options/sentiment data)
   3. No lookahead bias — we only use data available
      on the signal date, never future data
   4. Features are computed fresh for each prediction —
      the model sees exactly what was available that day
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_ml_logger
from utils.error_handler import graceful, MLError
from engines.adx import compute_adx_latest
from engines.relative_strength import compute_relative_strength_latest, BENCHMARK as RS_BENCHMARK

if bool(os.getenv("SUPABASE_DB_URL")):
    from data.database_cloud import read_sector_metadata
else:
    from data.database import read_sector_metadata

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config     = _load_config()
VOLUME_CFG = config["volume"]

LOOKBACK_DAYS     = VOLUME_CFG["lookback_days"]       # 10
AVG_VOLUME_PERIOD = VOLUME_CFG["avg_volume_period"]   # 20


# =============================================================================
# VOLUME FEATURES
# Used by Volume Classifier
# These capture the volume behaviour patterns we defined in volume.py
# =============================================================================

def compute_volume_features(
    df         : pd.DataFrame,
    signal_date: str,
) -> Optional[dict]:
    """
    Compute volume-based features for a single ticker on a signal date.

    FEATURES COMPUTED:
    1.  vol_slope_down_days    : Slope of volume on red candles (negative = declining)
    2.  vol_slope_up_days      : Slope of volume on green candles (positive = rising)
    3.  avg_vol_ratio          : Today's volume vs 20-day average (normalised)
    4.  green_red_vol_ratio    : Average green day volume / average red day volume
    5.  max_down_vol_ratio     : Highest single down day volume vs average
    6.  dryup_candle_present   : 1 if any candle < 0.5x average volume in lookback
    7.  n_red_candles          : Count of red candles in lookback window
    8.  n_green_candles        : Count of green candles in lookback window
    9.  vol_consistency        : Std dev of volume / mean volume (lower = more consistent)
    10. price_recovery_ratio   : On high vol down days, how much did price recover?

    All ratios are used instead of raw volumes so features are
    comparable across different stocks with very different trading volumes.

    Args:
        df         : Full OHLCV DataFrame for one ticker, sorted date ascending
        signal_date: The date of the signal YYYY-MM-DD

    Returns:
        Dict of feature name → value, or None if insufficient data
    """
    # Get data up to and including signal date (no lookahead)
    data = df[df["date"] <= signal_date].tail(AVG_VOLUME_PERIOD + LOOKBACK_DAYS)

    if len(data) < AVG_VOLUME_PERIOD:
        return None

    # ── Baseline average volume ───────────────────────────────────────────────
    avg_volume = data["volume"].tail(AVG_VOLUME_PERIOD).mean()
    if avg_volume == 0:
        return None

    recent = data.tail(LOOKBACK_DAYS)

    # ── Separate green and red candles ────────────────────────────────────────
    green = recent[recent["close"] >= recent["open"]]
    red   = recent[recent["close"] <  recent["open"]]

    # ── Feature 1: Volume slope on red candles ────────────────────────────────
    if len(red) >= 2:
        x = np.arange(len(red))
        slope_red, _ = np.polyfit(x, red["volume"].values, 1)
        # Normalise by avg_volume so it's comparable across stocks
        vol_slope_down = slope_red / avg_volume
    else:
        vol_slope_down = 0.0

    # ── Feature 2: Volume slope on green candles ──────────────────────────────
    if len(green) >= 2:
        x = np.arange(len(green))
        slope_green, _ = np.polyfit(x, green["volume"].values, 1)
        vol_slope_up = slope_green / avg_volume
    else:
        vol_slope_up = 0.0

    # ── Feature 3: Today's volume vs average ──────────────────────────────────
    today_vol      = data.iloc[-1]["volume"]
    avg_vol_ratio  = today_vol / avg_volume

    # ── Feature 4: Green vs red average volume ratio ──────────────────────────
    avg_green_vol  = green["volume"].mean() if len(green) > 0 else avg_volume
    avg_red_vol    = red["volume"].mean()   if len(red)   > 0 else avg_volume
    green_red_ratio = avg_green_vol / avg_red_vol if avg_red_vol > 0 else 1.0

    # ── Feature 5: Max down day volume ratio ──────────────────────────────────
    max_down_vol       = red["volume"].max() if len(red) > 0 else avg_volume
    max_down_vol_ratio = max_down_vol / avg_volume

    # ── Feature 6: Volume dry-up present ──────────────────────────────────────
    dryup_threshold    = avg_volume * 0.5
    dryup_present      = int((recent["volume"] < dryup_threshold).any())

    # ── Feature 7 & 8: Count of red and green candles ─────────────────────────
    n_red_candles   = len(red)
    n_green_candles = len(green)

    # ── Feature 9: Volume consistency ─────────────────────────────────────────
    # Lower = more consistent volume = more orderly market behaviour
    vol_std         = recent["volume"].std()
    vol_mean        = recent["volume"].mean()
    vol_consistency = vol_std / vol_mean if vol_mean > 0 else 1.0

    # ── Feature 10: Price recovery on high volume down days ───────────────────
    # Measures how much price recovered intraday on the biggest down volume day
    # High recovery = buyers absorbed the selling = bullish
    high_vol_down = red[red["volume"] > avg_volume * 1.5]
    if len(high_vol_down) > 0:
        # Recovery = (close - low) / (high - low) for the highest vol down day
        row           = high_vol_down.loc[high_vol_down["volume"].idxmax()]
        candle_range  = row["high"] - row["low"]
        recovery      = (row["close"] - row["low"]) / candle_range if candle_range > 0 else 0.5
    else:
        recovery = 0.5  # Neutral default when no high volume down days

    return {
        "vol_slope_down_days"  : round(vol_slope_down,   6),
        "vol_slope_up_days"    : round(vol_slope_up,     6),
        "avg_vol_ratio"        : round(avg_vol_ratio,    4),
        "green_red_vol_ratio"  : round(green_red_ratio,  4),
        "max_down_vol_ratio"   : round(max_down_vol_ratio, 4),
        "dryup_candle_present" : dryup_present,
        "n_red_candles"        : n_red_candles,
        "n_green_candles"      : n_green_candles,
        "vol_consistency"      : round(vol_consistency,  4),
        "price_recovery_ratio" : round(recovery,         4),
    }


# =============================================================================
# SIGNAL RANKER FEATURES
# Broader feature set covering the full setup quality
# =============================================================================

def build_sector_price_cache(prices_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Build the two lookups needed for the Sector Relative Strength feature:
    which sector ETF each ticker belongs to, and that ETF's own price
    series (pre-extracted once, not per ticker/date — sector ETFs are
    already part of the fetched universe, no new data source needed).

    Reuses read_sector_metadata() — the same source scanner/screener.py
    already relies on for sector health checks — so there's a single
    source of truth for ticker->sector-ETF mapping across the project.

    Args:
        prices_df: Full OHLCV data (must include the sector ETF tickers
                    alongside all scanned tickers)

    Returns:
        Tuple of:
        - ticker_to_etf: dict mapping ticker -> sector ETF symbol (e.g. 'XLK')
        - etf_price_cache: dict mapping ETF symbol -> its own price DataFrame
    """
    sector_df = read_sector_metadata()

    if sector_df.empty:
        logger.warning(
            "build_sector_price_cache: no sector metadata available — "
            "relative_strength_sector will be 0.0 for all rows"
        )
        return {}, {}

    ticker_to_etf = dict(zip(sector_df["ticker"], sector_df["sector_etf"]))

    etf_price_cache = {}
    for etf in sector_df["sector_etf"].dropna().unique():
        etf_prices = prices_df[prices_df["ticker"] == etf].sort_values("date")
        if not etf_prices.empty:
            etf_price_cache[etf] = etf_prices

    return ticker_to_etf, etf_price_cache


def compute_signal_features(
    ticker              : str,
    signal_date         : str,
    direction           : str,
    prices_df           : pd.DataFrame,
    indicators_df       : pd.DataFrame,
    market_ind_df       : pd.DataFrame,
    vol_clf_score       : Optional[float] = None,
    benchmark_prices_df : Optional[pd.DataFrame] = None,
    sector_prices_df    : Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Compute all features for the Signal Ranker for a single ticker/date.

    FEATURE GROUPS:

    GROUP 1 — Price position (3 features):
    - sd_position       : How deep in the band (-1.8 = 1.8 SDs below mean)
    - dist_to_mean      : Distance from current price to LinReg mean (normalised)
    - band_penetration  : How far through the band is price? (0 = at ±1SD, 1 = at ±3SD)

    GROUP 2 — Trend strength (2 features):
    - linreg_slope      : Steepness of LinReg slope (normalised by price)
    - days_in_trend     : How many consecutive days has slope been in same direction?

    GROUP 3 — Volume (from volume features):
    - All 10 volume features from compute_volume_features()
    - vol_clf_score     : Output of Volume Classifier (0-1 probability)

    GROUP 4 — Sentiment (2 features):
    - put_call_ratio    : Options sentiment (None → median imputed)
    - short_interest    : Float short % (None → median imputed)

    GROUP 5 — Market context (2 features):
    - market_slope_avg  : Average LinReg slope of SPY + QQQ + DIA
    - market_choch_count: Number of indices with CHoCH (0, 1, 2, or 3)

    GROUP 6 — ADX / trend strength (3 features, see engines/adx.py):
    - adx_value         : Trend strength, direction-independent (0-100+)
    - plus_di           : Bullish directional pressure
    - minus_di          : Bearish directional pressure

    GROUP 7 — Relative Strength (2 features, see engines/relative_strength.py):
    - relative_strength        : Outperformance vs benchmark (SPY) over the
                                  configured lookback window.
    - relative_strength_sector : Outperformance vs the ticker's OWN sector
                                  ETF (e.g. XLK for tech) over the same
                                  window — answers "is this a leader within
                                  its own sector," a different question than
                                  market-wide RS.

    Args:
        ticker             : Ticker symbol
        signal_date         : Date of signal YYYY-MM-DD
        direction           : 'long' or 'short'
        prices_df           : Full OHLCV data for this ticker
        indicators_df       : Full indicator results (all tickers)
        sentiment_df        : Sentiment data (all tickers)
        market_ind_df       : Indicator results for SPY, QQQ, DIA only
        vol_clf_score       : Output of Volume Classifier model (optional)
        benchmark_prices_df : Full OHLCV data for the RS benchmark (SPY),
                               used for the relative_strength feature.
                               Falls back to 0.0 if not provided.
        sector_prices_df    : Full OHLCV data for THIS ticker's own sector
                               ETF, used for relative_strength_sector.
                               Falls back to 0.0 if not provided.

    Returns:
        Dict of all features or None if insufficient data
    """

    # ── Get indicator row for this ticker on signal date ──────────────────────
    ind_row = indicators_df[
        (indicators_df["ticker"] == ticker) &
        (indicators_df["date"]   == signal_date)
    ]

    if ind_row.empty:
        return None

    ind = ind_row.iloc[0]

    # ── Get price data up to signal date ─────────────────────────────────────
    px = prices_df[
        (prices_df["ticker"] == ticker) &
        (prices_df["date"]   <= signal_date)
    ].sort_values("date")

    if len(px) < AVG_VOLUME_PERIOD:
        return None

    # ── GROUP 1: Price position features ────────────────────────────────────
    sd_position  = float(ind["price_sd_position"])
    linreg_val   = float(ind["linreg_value"])
    current_close = px.iloc[-1]["close"]

    # Distance from price to LinReg, normalised by price level
    dist_to_mean = abs(current_close - linreg_val) / current_close

    # How far through the band is price?
    # 0 = just entered the ±1 SD band, 1 = at the ±3 SD extreme
    abs_sd = abs(sd_position)
    band_penetration = np.clip((abs_sd - 1.0) / 2.0, 0.0, 1.0)

    # ── GROUP 2: Trend strength features ────────────────────────────────────
    linreg_slope = float(ind["linreg_slope"])

    # Count consecutive days with same slope direction
    ticker_ind = indicators_df[
        indicators_df["ticker"] == ticker
    ].sort_values("date")

    current_slope = int(ind["linreg_slope_up"])

    # Count consecutive trailing candles with the same slope direction
    days_in_trend = 0
    for val in reversed(ticker_ind["linreg_slope_up"].tolist()):
        if val == current_slope:
            days_in_trend += 1
        else:
            break

    # ── GROUP 3: Volume features ─────────────────────────────────────────────
    vol_features = compute_volume_features(px, signal_date)
    if vol_features is None:
        vol_features = {k: 0.0 for k in [
            "vol_slope_down_days", "vol_slope_up_days", "avg_vol_ratio",
            "green_red_vol_ratio", "max_down_vol_ratio", "dryup_candle_present",
            "n_red_candles", "n_green_candles", "vol_consistency",
            "price_recovery_ratio",
        ]}

    # ── ATR14 ─────────────────────────────────────────────────────────────────
    # Average True Range over last 14 candles
    # Used to normalise BOS strength and volume delta
    recent_14 = px.tail(14)
    if len(recent_14) >= 2:
        high_low   = recent_14["high"] - recent_14["low"]
        high_close = (recent_14["high"] - recent_14["close"].shift(1)).abs()
        low_close  = (recent_14["low"]  - recent_14["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr14      = true_range.mean()
    else:
        atr14 = float(px["close"].iloc[-1]) * 0.01   # 1% fallback

    # ── BOS strength (candle body / ATR14) ────────────────────────────────────
    # Measures how violent the Break of Structure was
    # High value = strong institutional move, not retail noise
    # Uses the most recent significant candle as BOS proxy
    last_candle    = px.iloc[-1]
    bos_body       = abs(last_candle["close"] - last_candle["open"])
    bos_strength   = bos_body / atr14 if atr14 > 0 else 0.0

    # ── GROUP 6: ADX (trend strength, direction-independent) ──────────────────
    # See engines/adx.py — falls back to 0.0 if insufficient history rather
    # than dropping the whole row, since ADX needs 2×PERIOD candles and we
    # don't want to lose training examples for tickers still ramping up
    # their history (e.g. recently listed, or early in the 8-year backfill).
    adx_result = compute_adx_latest(ticker, px, signal_date)
    if adx_result is not None:
        adx_value = adx_result["adx_value"]
        plus_di   = adx_result["plus_di"]
        minus_di  = adx_result["minus_di"]
    else:
        adx_value = 0.0
        plus_di   = 0.0
        minus_di  = 0.0

    # ── GROUP 7: Relative Strength vs SPY ──────────────────────────────────────
    # See engines/relative_strength.py — same fallback reasoning as ADX above.
    # benchmark_prices_df is None when the caller doesn't have SPY's price
    # series available (shouldn't normally happen — SPY is always part of
    # the fetched universe — but this keeps the function from crashing if
    # it's ever called without it).
    relative_strength = 0.0
    if benchmark_prices_df is not None and not benchmark_prices_df.empty:
        bench_slice = benchmark_prices_df[
            benchmark_prices_df["date"] <= signal_date
        ].sort_values("date")

        rs_result = compute_relative_strength_latest(
            ticker, px, bench_slice, signal_date
        )
        if rs_result is not None:
            relative_strength = rs_result["relative_strength"]

    # ── GROUP 7b: Relative Strength vs THIS ticker's own sector ETF ───────────
    # Same engine, different benchmark — answers "is this a leader within
    # its own sector," a distinct question from market-wide RS above.
    relative_strength_sector = 0.0
    if sector_prices_df is not None and not sector_prices_df.empty:
        sector_slice = sector_prices_df[
            sector_prices_df["date"] <= signal_date
        ].sort_values("date")

        sector_rs_result = compute_relative_strength_latest(
            ticker, px, sector_slice, signal_date
        )
        if sector_rs_result is not None:
            relative_strength_sector = sector_rs_result["relative_strength"]

    # ── GROUP 5: Market context features ────────────────────────────────────
    mkt = market_ind_df[market_ind_df["date"] == signal_date]

    if not mkt.empty:
        market_slope_avg   = mkt["linreg_slope"].mean()
        #market_choch_count = int(mkt["choch_detected"].sum())
    else:
        market_slope_avg   = 0.0
        #market_choch_count = 0

    # ── Direction encoding ────────────────────────────────────────────────────
    # 1 = long setup, 0 = short setup
    direction_flag = 1 if direction == "long" else 0

    # ── Volume signal encoded ─────────────────────────────────────────────────
    vol_signal_map    = {"accumulation": 2, "neutral": 1, "distribution": 0}
    vol_sig_str       = str(ind.get("volume_signal", "neutral"))
    volume_signal_enc = vol_signal_map.get(vol_sig_str, 1)

    # ── Zone flag ─────────────────────────────────────────────────────────────
    has_valid_zone = int(ind.get("has_valid_zone", 0))

    # ── Assemble all features ─────────────────────────────────────────────────
    features = {
        # Identifiers (not used in training — dropped before fit)
        "ticker"             : ticker,
        "date"               : signal_date,
        "direction"          : direction,

        # Group 1: Price position
        "sd_position"        : round(sd_position,     4),
        "dist_to_mean"       : round(dist_to_mean,    6),
        "band_penetration"   : round(band_penetration, 4),

        # Group 2: Trend strength
        "linreg_slope"       : round(linreg_slope,    6),
        "days_in_trend"      : days_in_trend,
        "bos_strength_vs_atr"  : round(bos_strength, 4),

        # Group 3: Volume
        **vol_features,
        "vol_clf_score"      : vol_clf_score if vol_clf_score is not None else np.nan,

        # Group 5: Market context
        "market_slope_avg"   : round(market_slope_avg,   6),
        #"market_choch_count" : market_choch_count,

        # Group 6: ADX (trend strength, direction-independent)
        "adx_value"          : adx_value,
        "plus_di"            : plus_di,
        "minus_di"           : minus_di,

        # Group 7: Relative Strength vs benchmark (SPY) and own sector ETF
        "relative_strength"        : relative_strength,
        "relative_strength_sector" : relative_strength_sector,

        # Direction
        "direction_flag"     : direction_flag,
        # Volume signal (encoded for ML)
        "volume_signal_encoded" : volume_signal_enc,
        # Zone flag
        "has_valid_zone"        : has_valid_zone,
    }

    return features


# =============================================================================
# FEATURE MATRIX BUILDER
# Builds the full feature matrix for model training
# =============================================================================

def build_volume_feature_matrix(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
    labels_df    : pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full feature matrix for Volume Classifier training.

    FLOW:
    1. For each labelled example (ticker, date, direction, label)
    2. Compute volume features for that ticker on that date
    3. Attach the label
    4. Return combined matrix ready for model training

    Args:
        prices_df    : Full OHLCV data
        indicators_df: Full indicator results
        labels_df    : Labels from label_volume_patterns()

    Returns:
        DataFrame with feature columns + 'label' column
        Ready for sklearn/XGBoost training
    """
    logger.info(
        f"Building Volume Classifier feature matrix | "
        f"{len(labels_df)} labelled examples"
    )

    rows   = []
    failed = 0

    for _, label_row in labels_df.iterrows():
        ticker = label_row["ticker"]
        date   = label_row["date"]
        label  = label_row["label"]

        # Get price data for this ticker
        px = prices_df[prices_df["ticker"] == ticker].sort_values("date")

        if px.empty:
            failed += 1
            continue

        # Compute volume features
        vol_features = compute_volume_features(px, date)

        if vol_features is None:
            failed += 1
            continue

        vol_features["label"] = label
        rows.append({"date" : date,
                     **vol_features,})

    result = pd.DataFrame(rows)

    # Set chronologically - required for walk-forward method
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop = True)

    logger.info(
        f"Volume feature matrix built | "
        f"Rows: {len(result)} | "
        f"Failed: {failed} | "
        f"Features: {len(result.columns) - 1}"
    )

    return result


def build_signal_feature_matrix(
    prices_df    : pd.DataFrame,
    indicators_df: pd.DataFrame,
    market_ind_df: pd.DataFrame,
    labels_df    : pd.DataFrame,
    vol_scores   : Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for Signal Ranker training.

    Args:
        prices_df    : Full OHLCV data (must include the RS benchmark
                        ticker, e.g. SPY, alongside all scanned tickers —
                        extracted once internally for relative_strength)
        indicators_df: Full indicator results (all tickers)
        market_ind_df: Indicator results for SPY, QQQ, DIA only
        labels_df    : Labels from label_scanner_hits()
        vol_scores   : Optional dict of (ticker, date) → vol_clf_score

    Returns:
        DataFrame with all feature columns + 'label' column
    """
    logger.info(
        f"Building Signal Ranker feature matrix | "
        f"{len(labels_df)} labelled examples"
    )

    # Extract the benchmark's (SPY) price series ONCE, not per labelled
    # example — used for the Relative Strength feature. If it's missing
    # from the universe entirely, relative_strength falls back to 0.0
    # for every row (compute_signal_features handles that gracefully).
    benchmark_prices_df = prices_df[
        prices_df["ticker"] == RS_BENCHMARK
    ].sort_values("date")

    if benchmark_prices_df.empty:
        logger.warning(
            f"{RS_BENCHMARK} not found in prices_df — relative_strength "
            f"will be 0.0 for all training examples"
        )

    # Build ticker -> sector ETF mapping + each ETF's price series ONCE,
    # for the relative_strength_sector feature. See build_sector_price_cache().
    ticker_to_etf, etf_price_cache = build_sector_price_cache(prices_df)

    rows   = []
    failed = 0

    for _, label_row in labels_df.iterrows():
        ticker    = label_row["ticker"]
        date      = label_row["date"]
        direction = label_row["direction"]
        label     = label_row["label"]

        # Get volume classifier score for this ticker/date if available
        vol_score = None
        if vol_scores:
            vol_score = vol_scores.get((ticker, date))

        # Get price data for this ticker
        px = prices_df[prices_df["ticker"] == ticker].sort_values("date")

        if px.empty:
            failed += 1
            continue

        sector_etf        = ticker_to_etf.get(ticker)
        sector_prices_df  = etf_price_cache.get(sector_etf) if sector_etf else None

        features = compute_signal_features(
            ticker              = ticker,
            signal_date         = date,
            direction           = direction,
            prices_df           = px,
            indicators_df       = indicators_df,
            market_ind_df       = market_ind_df,
            vol_clf_score       = vol_score,
            benchmark_prices_df = benchmark_prices_df,
            sector_prices_df    = sector_prices_df,
        )

        if features is None:
            failed += 1
            continue

        features["label"] = label
        rows.append(features)

    result = pd.DataFrame(rows)
    
    # Set chronologically - required for walk-forward method
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop = True)

    logger.info(
        f"Signal feature matrix built | "
        f"Rows: {len(result)} | "
        f"Failed: {failed}"
    )

    return result


# =============================================================================
# FEATURE COLUMNS — used by both models at train and inference time
# =============================================================================

VOLUME_FEATURE_COLS = [
    "vol_slope_down_days",
    "vol_slope_up_days",
    "avg_vol_ratio",
    "green_red_vol_ratio",
    "max_down_vol_ratio",
    "dryup_candle_present",
    "n_red_candles",
    "n_green_candles",
    "vol_consistency",
    "price_recovery_ratio",
]

SIGNAL_FEATURE_COLS = [
    # Price position
    "sd_position",
    "dist_to_mean",
    "band_penetration",
    # Trend
    "linreg_slope",
    "days_in_trend",
    # Volume
    "vol_slope_down_days",
    "vol_slope_up_days",
    "avg_vol_ratio",
    "green_red_vol_ratio",
    "max_down_vol_ratio",
    "dryup_candle_present",
    "n_red_candles",
    "n_green_candles",
    "vol_consistency",
    "price_recovery_ratio",
    "vol_clf_score",
    "bos_strength_vs_atr",      # BOS candle body / ATR14
    "market_slope_avg",
    # ADX — trend strength, direction-independent
    "adx_value",
    "plus_di",
    "minus_di",
    # Relative Strength vs benchmark (SPY) and own sector ETF
    "relative_strength",
    "relative_strength_sector",
    # Direction
    "direction_flag",
    "volume_signal_encoded",    # accumulation=2, neutral=1, distribution=0
    "has_valid_zone",           # 1 = valid demand/supply zone exists
]
