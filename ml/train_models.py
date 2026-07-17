"""
ml/train_models.py
-------------------
Backfills a rolling indicator history from raw_prices data,
generates labels, builds feature matrices, and trains both ML models.

KEY FIXES vs previous version:
1. Relaxed scan hit detection — drops has_valid_zone requirement
   so enough training examples are generated
2. min_training_samples lowered to 10 — Signal Ranker trains even
   with few examples; XGBoost handles small datasets gracefully
3. Class imbalance handled via scale_pos_weight in both models
4. Signal Ranker always ranks however many candidates exist (even 2)
5. CHoCH removed as a hard filter in scan hit detection
6. label_forward_periods config key used (not label_forward_days)
7. Volume Classifier skips retraining if .pkl already exists

"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
import pandas as pd
import numpy as np
import yaml

if os.getenv("SUPABASE_DB_URL"):
    from data.database_cloud import (
        initialise_database,
        read_filtered_universe,
        read_sector_metadata,
        write_model_metrics,
    )
    from data.storage_cloud import read_price_history
    _CLOUD_MODE = True
else:
    from data.database import (
        initialise_database,
        read_filtered_universe,
        read_raw_prices,
        read_sector_metadata,
        write_model_metrics,
    )
    _CLOUD_MODE = False
from engines.linreg import compute_linreg_latest, PERIOD as LINREG_PERIOD
from engines.smc    import compute_smc
from engines.volume import compute_volume_signal
from scanner.screener import _check_sector_health
from ml.labeller import label_volume_patterns, label_scanner_hits
from ml.features import (
    build_volume_feature_matrix,
    build_signal_feature_matrix,
    compute_volume_features,
    VOLUME_FEATURE_COLS,
)
from ml.volume_classifier import (
    train_volume_classifier,
    score_volume_signals,
    load_volume_classifier,
    MODEL_PATH as VOL_MODEL_PATH,
)
from ml.signal_ranker import train_signal_ranker
from utils.logging import get_ml_logger

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config       = _load_config()
ML_CFG       = config["ml"]
UNIVERSE_CFG = config["universe"]
SCANNER_CFG  = config["scanner"]

# Use label_forward_periods if available, fall back to label_forward_days
FORWARD = ML_CFG.get("label_forward_periods", ML_CFG.get("label_forward_days", 26))
STRIDE  = ML_CFG.get("backfill_stride", 4)

# Minimum samples — lowered so Signal Ranker trains even with limited history
# XGBoost handles small datasets; more data will accumulate over daily runs
MIN_SIGNAL_SAMPLES = 10   # was 500 — will grow as pipeline runs daily

# SD zone thresholds from scanner config
LONG_SD_MIN  = SCANNER_CFG["long_entry_sd_min"]   # -1
LONG_SD_MAX  = SCANNER_CFG["long_entry_sd_max"]   # -3
SHORT_SD_MIN = SCANNER_CFG["short_entry_sd_min"]  # +1
SHORT_SD_MAX = SCANNER_CFG["short_entry_sd_max"]  # +3

# Cap for quick test runs — set to None for full universe
MAX_TICKERS_FOR_TRAINING = None


# =============================================================================
# ROLLING BACKFILL
# Re-runs engines across history for one ticker to generate
# many (ticker, date, indicator) rows for training
# =============================================================================

def backfill_ticker(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run LinReg, SMC, Volume engines on rolling windows of df.

    For each step i (i = LINREG_PERIOD .. len(df)-1, stride STRIDE):
        window = df.iloc[:i+1]
        date   = df.iloc[i]["date"]
        → compute_linreg_latest, compute_smc, compute_volume_signal

    Args:
        ticker: Ticker symbol
        df    : Full OHLCV DataFrame sorted date ascending

    Returns:
        DataFrame with one row per backfilled (ticker, date)
    """
    rows = []
    n    = len(df)

    # Need LINREG_PERIOD candles for LinReg + FORWARD candles for labelling
    if n < LINREG_PERIOD + FORWARD + 1:
        logger.debug(
            f"{ticker} | Insufficient data for backfill: "
            f"{n} rows, need {LINREG_PERIOD + FORWARD + 1}"
        )
        return pd.DataFrame()

    for i in range(LINREG_PERIOD, n, STRIDE):
        window = df.iloc[: i + 1]
        date   = str(df.iloc[i]["date"])

        lr = compute_linreg_latest(ticker, window, date)
        if lr is None:
            continue

        smc = compute_smc(
            ticker, window, date,
            sd1_lower = lr.get("sd1_lower"),
            sd3_lower = lr.get("sd3_lower"),
            sd1_upper = lr.get("sd1_upper"),
            sd3_upper = lr.get("sd3_upper"),
        )
        if smc is None:
            continue

        vol = compute_volume_signal(ticker, window, date)
        if vol is None:
            continue

        rows.append({
            "ticker"        : ticker,
            "date"          : date,
            **{k: v for k, v in lr.items() if k not in ("ticker", "date")},
            "smc_structure" : smc["smc_structure"],
            #"choch_detected": smc["choch_detected"],   # kept for ML feature only
            "has_valid_zone": smc["has_valid_zone"],
            "volume_signal" : vol["volume_signal"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# RELAXED SCAN HIT DETECTION
# Drops has_valid_zone and CHoCH hard filters to maximise training examples.
# The model learns from these as features rather than hard gates.
# This is training only — live scanner still applies all conditions.
# =============================================================================

def _is_long_candidate_relaxed(row: pd.Series) -> bool:
    """
    Relaxed long candidate check for training data generation.
    Requires: slope up + price in -1 to -3 SD + accumulation volume.
    Does NOT require: has_valid_zone, CHoCH absence, sector health.
    These become ML features instead of hard filters.
    """
    if int(row.get("linreg_slope_up", 0)) != 1:
        return False

    sd_pos = float(row.get("price_sd_position", 0))
    if not (LONG_SD_MAX <= sd_pos <= LONG_SD_MIN):   # -3 <= sd <= -1
        return False

    if str(row.get("volume_signal", "neutral")) != "accumulation":
        return False

    return True


def _is_short_candidate_relaxed(row: pd.Series) -> bool:
    """
    Relaxed short candidate check for training data generation.
    Requires: slope down + price in +1 to +3 SD + distribution volume.
    """
    if int(row.get("linreg_slope_up", 1)) != 0:
        return False

    sd_pos = float(row.get("price_sd_position", 0))
    if not (SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX):   # +1 <= sd <= +3
        return False

    if str(row.get("volume_signal", "neutral")) != "distribution":
        return False

    return True


def build_historical_scan_hits(
    indicators_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find all historical (ticker, date) combinations that would have
    qualified as long or short candidates using the relaxed conditions.

    Uses relaxed checks (no has_valid_zone, no CHoCH, no sector filter)
    to maximise training examples. These dropped conditions become
    ML features that the Signal Ranker learns to weight appropriately.

    Args:
        indicators_history_df: Full backfilled indicator history

    Returns:
        DataFrame with columns [ticker, date, direction]
    """
    excluded = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])
    stock_rows = indicators_history_df[
        ~indicators_history_df["ticker"].isin(excluded)
    ]

    hits = []
    for _, row in stock_rows.iterrows():
        if _is_long_candidate_relaxed(row):
            hits.append({
                "ticker"   : row["ticker"],
                "date"     : row["date"],
                "direction": "long",
            })
        if _is_short_candidate_relaxed(row):
            hits.append({
                "ticker"   : row["ticker"],
                "date"     : row["date"],
                "direction": "short",
            })

    logger.info(f"build_historical_scan_hits: {len(hits)} historical setups found")
    return pd.DataFrame(hits) if hits else pd.DataFrame(
        columns=["ticker", "date", "direction"]
    )


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def run_training():
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    initialise_database()
    today = datetime.today().strftime("%Y-%m-%d")

    # Delete stale models to force clean retraining with new fixes
    from ml.volume_classifier import MODEL_PATH as VOL_PATH
    from ml.signal_ranker     import MODEL_PATH as SIG_PATH

    for model_path in [VOL_PATH, SIG_PATH]:
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Deleted stale model: {model_path}")
            

    # ── Step 1: Load filtered universe ────────────────────────────────────────
    filtered = read_filtered_universe(today)
    if filtered.empty:
        logger.error("No filtered universe found — run run_pipeline.py first")
        return

    tickers = filtered["ticker"].tolist()
    if MAX_TICKERS_FOR_TRAINING:
        tickers = tickers[:MAX_TICKERS_FOR_TRAINING]

    logger.info(
        f"Backfilling indicators for {len(tickers)} tickers "
        f"(stride={STRIDE}, linreg_period={LINREG_PERIOD})"
    )

    # ── Step 2: Rolling backfill per ticker ───────────────────────────────────
    indicator_history = []
    prices_all        = []
    backfill_failed   = 0

    if _CLOUD_MODE:
        # Load the full rolling window once from Supabase Storage, then
        # filter per ticker in memory — avoids re-downloading snapshots
        # on every loop iteration.
        #
        # NOTE: no days= argument — this project needs the FULL
        # accumulated history for regime coverage across all 8 years,
        # not a rolling window. The 15m project used days=60 here
        # because it only needed a recent window; that default would
        # silently gut this project's entire purpose if carried over.
        logger.info("Loading full price history from Supabase Storage...")
        full_history = read_price_history()
        logger.info(f"Loaded {len(full_history)} total rows across all snapshots")

        if full_history.empty:
            logger.error(
                "No price history snapshots available yet. "
                "Snapshots accumulate daily from run_pipeline_cloud.py — "
                "wait for more daily runs before training."
            )
            return

    for n, ticker in enumerate(tickers, 1):
        if _CLOUD_MODE:
            df = full_history[full_history["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        else:
            df = read_raw_prices(ticker)

        if df.empty:
            backfill_failed += 1
            continue

        prices_all.append(df)

        hist = backfill_ticker(ticker, df)
        if not hist.empty:
            indicator_history.append(hist)
        else:
            backfill_failed += 1

        if n % 50 == 0:
            logger.info(
                f"Backfill progress: {n}/{len(tickers)} tickers | "
                f"History rows so far: "
                f"{sum(len(h) for h in indicator_history)}"
            )

    if not indicator_history:
        logger.error(
            "No indicator history produced — "
            "tickers may not have enough candles yet. "
            "Run the pipeline for more days to accumulate data."
        )
        return

    indicators_history_df = pd.concat(indicator_history, ignore_index=True)
    prices_all_df          = pd.concat(prices_all, ignore_index=True)

    logger.info(
        f"Backfill complete | "
        f"History rows: {len(indicators_history_df)} | "
        f"Tickers with history: {indicators_history_df['ticker'].nunique()} | "
        f"Failed: {backfill_failed}"
    )

    # ── Step 3: Market indicator history for Signal Ranker features ───────────
    market_ind_df = indicators_history_df[
        indicators_history_df["ticker"].isin(UNIVERSE_CFG["indices"])
    ]
    logger.info(f"Market history rows (SPY/QQQ/DIA): {len(market_ind_df)}")

    # ── Step 4: Train Volume Classifier ───────────────────────────────────────
    # Skip if already trained and .pkl exists — saves time on re-runs
    vol_pipeline = None

    if VOL_MODEL_PATH.exists():
        logger.info(
            f"Volume Classifier already trained at {VOL_MODEL_PATH} — "
            f"loading existing model. Delete the .pkl to force retraining."
        )
        vol_pipeline = load_volume_classifier()
    else:
        logger.info("Training Volume Classifier...")
        vol_labels = label_volume_patterns(prices_all_df, indicators_history_df)

        if vol_labels.empty:
            logger.error("No volume labels generated — aborting Volume Classifier training")
        else:
            # Log class balance
            pos = (vol_labels["label"] == 1).sum()
            neg = (vol_labels["label"] == 0).sum()
            logger.info(
                f"Volume labels | Total: {len(vol_labels)} | "
                f"Positive: {pos} ({pos/len(vol_labels)*100:.1f}%) | "
                f"Negative: {neg} ({neg/len(vol_labels)*100:.1f}%)"
            )

            vol_matrix = build_volume_feature_matrix(
                prices_all_df, indicators_history_df, vol_labels
            )

            if not vol_matrix.empty and len(vol_matrix) >= 10:
                vol_pipeline, vol_metrics = train_volume_classifier(vol_matrix)
                write_model_metrics(
                    model_name = vol_metrics["model_name"],
                    train_date = vol_metrics["train_date"],
                    precision  = vol_metrics["precision"],
                    auc_roc    = vol_metrics["auc_roc"],
                    n_samples  = vol_metrics["n_train"] + vol_metrics["n_test"],
                    recall     = vol_metrics.get("recall", 0.0),
                    pr_auc     = vol_metrics.get("pr_auc", 0.0),
                )
            else:
                logger.warning(
                    f"Volume feature matrix too small: {len(vol_matrix)} rows. "
                    f"Skipping Volume Classifier training."
                )

    # ── Step 5: Build vol_clf_score lookup ────────────────────────────────────
    # Score every (ticker, date) in backfilled history using trained vol model.
    #
    # REWRITTEN FOR SPEED: the original version called score_volume_signals()
    # once per (ticker, date) pair — with ~1,800 tickers × ~250 dates each,
    # that's roughly 460,000 individual predict_proba() calls, each preceded
    # by an O(n) re-filter of that ticker's ENTIRE price history from scratch
    # (px[px["date"] <= date], repeated every iteration). That combination is
    # what caused multi-hour runtimes. Fixed two ways:
    #   1. np.searchsorted() for an O(log n) cutoff lookup instead of an
    #      O(n) boolean mask re-scan on every single date.
    #   2. Accumulate ALL feature rows first, then call predict_proba() ONCE
    #      on the full batch — XGBoost is vectorized, so one call over many
    #      rows is dramatically faster than many calls of one row each.
    vol_scores = {}

    if vol_pipeline is not None:
        logger.info("Scoring volume patterns across backfilled history...")

        feature_rows = []
        score_keys   = []

        for ticker, group in indicators_history_df.groupby("ticker"):
            px = prices_all_df[
                prices_all_df["ticker"] == ticker
            ].sort_values("date").reset_index(drop=True)

            px_dates = px["date"].values

            for date in group["date"].unique():
                # O(log n) cutoff instead of re-filtering the whole array
                cutoff = np.searchsorted(px_dates, date, side="right")
                if cutoff == 0:
                    continue

                px_slice = px.iloc[:cutoff]
                features = compute_volume_features(px_slice, str(date))

                if features is None:
                    continue

                feature_rows.append(features)
                score_keys.append((ticker, str(date)))

        if feature_rows:
            X_all = pd.DataFrame(feature_rows)[VOLUME_FEATURE_COLS]
            probs = vol_pipeline.predict_proba(X_all)[:, 1]

            for key, prob in zip(score_keys, probs):
                vol_scores[key] = round(float(prob), 4)

        logger.info(f"Volume scoring complete: {len(vol_scores)} (ticker, date) pairs scored")
    else:
        logger.warning(
            "Volume Classifier not available — "
            "vol_clf_score will be 0.5 (neutral) for all training examples"
        )

    # ── Step 6: Train Signal Ranker ───────────────────────────────────────────
    logger.info("Building historical scan hits for Signal Ranker training...")

    scan_hits = build_historical_scan_hits(indicators_history_df)

    if scan_hits.empty:
        logger.warning(
            "No historical scan hits found. "
            "This means no tickers had the right combination of: "
            "slope up/down + price in SD zone + accumulation/distribution volume. "
            "Run the pipeline for more days to accumulate diverse market conditions."
        )
        logger.info("=" * 60)
        logger.info("ML TRAINING PIPELINE COMPLETE (Signal Ranker skipped)")
        logger.info("=" * 60)
        return

    logger.info(
        f"Scan hits found: {len(scan_hits)} | "
        f"Longs: {(scan_hits['direction']=='long').sum()} | "
        f"Shorts: {(scan_hits['direction']=='short').sum()}"
    )

    # Generate labels for scan hits
    sig_labels = label_scanner_hits(prices_all_df, indicators_history_df, scan_hits)

    if sig_labels.empty:
        logger.warning("No signal labels generated — Signal Ranker skipped")
        return

    pos = (sig_labels["label"] == 1).sum()
    neg = (sig_labels["label"] == 0).sum()
    logger.info(
        f"Signal labels | Total: {len(sig_labels)} | "
        f"Positive: {pos} ({pos/len(sig_labels)*100:.1f}%) | "
        f"Negative: {neg} ({neg/len(sig_labels)*100:.1f}%)"
    )

    # Build feature matrix
    sig_matrix = build_signal_feature_matrix(
        prices_df     = prices_all_df,
        indicators_df = indicators_history_df,
        market_ind_df = market_ind_df,
        labels_df     = sig_labels,
        vol_scores    = vol_scores,
    )

    if sig_matrix.empty:
        logger.warning("Signal feature matrix is empty — Signal Ranker skipped")
        return

    logger.info(f"Signal feature matrix: {len(sig_matrix)} rows")

    # Train — allow even small sample sizes (model still ranks by probability)
    if len(sig_matrix) < MIN_SIGNAL_SAMPLES:
        logger.warning(
            f"Signal Ranker has only {len(sig_matrix)} training samples "
            f"(minimum is {MIN_SIGNAL_SAMPLES}). "
            f"Temporarily lowering threshold to train anyway."
        )
        # Monkey-patch the threshold for this run
        import ml.signal_ranker as sr_module
        original_min = sr_module.MIN_SAMPLES
        sr_module.MIN_SAMPLES = max(2, len(sig_matrix))

    try:
        sig_pipeline, sig_metrics = train_signal_ranker(sig_matrix)
        write_model_metrics(
            model_name = sig_metrics["model_name"],
            train_date = sig_metrics["train_date"],
            precision  = sig_metrics["precision"],
            auc_roc    = sig_metrics["auc_roc"],
            n_samples  = sig_metrics["n_train"] + sig_metrics["n_test"],
            recall     = sig_metrics.get("recall", 0.0),
            pr_auc     = sig_metrics.get("pr_auc", 0.0),
        )
        
        logger.info(
            f"Signal Ranker trained | "
            f"Precision: {sig_metrics['precision']:.4f} | "
            f"AUC-ROC: {sig_metrics['auc_roc']:.4f} | "
        )
    finally:
        # Restore original threshold
        if len(sig_matrix) < MIN_SIGNAL_SAMPLES:
            sr_module.MIN_SAMPLES = original_min

    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training()
