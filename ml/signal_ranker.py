"""
ml/signal_ranker.py
--------------------
Signal Ranker ML model for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This model takes every stock that passed the scanner waterfall
and assigns it a probability score: how likely is this setup
to succeed (price reaching the LinReg mean within N days)?

It is the final ranking layer before the dashboard display.

TRAINING FLOW:
1. Load 
led scanner hits from labeller.py
2. Build full feature matrix from features.py
   (includes volume features + vol_clf_score + sentiment + market)
3. Handle class imbalance
4. Train XGBoost with cross-validation
5. Evaluate (Precision + AUC-ROC)
6. Save model to disk
7. Write metrics to SQLite

INFERENCE FLOW (daily):
1. Scanner waterfall returns N candidates
2. For each candidate, compute full feature vector
3. Run predict_proba() → probability of success
4. Sort candidates by probability descending
5. Assign ml_rank (1 = highest probability)
6. Write ranked results to SQLite scan_results table
7. Dashboard reads and displays

RELATIONSHIP TO VOLUME CLASSIFIER:
The Signal Ranker uses the Volume Classifier's output score
(vol_clf_score) as ONE of its features. This creates a
two-stage model pipeline:
   Volume Classifier → scores volume pattern quality
   Signal Ranker     → combines all features including vol score
                       to rank overall setup probability
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
from sklearn import pipeline
import yaml

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score)
from sklearn.metrics import make_scorer, precision_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from ml.features import (
    SIGNAL_FEATURE_COLS,
    compute_signal_features,
)
from utils.logging import get_ml_logger
from utils.error_handler import MLError

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = _load_config()
ML_CFG     = config["ml"]
LINREG_CFG = config["linreg"]

MIN_SAMPLES  = ML_CFG["min_training_samples"]
HIGH_PROB_THRESHOLD = ML_CFG["high_probability_threshold"]   # 0.70
MODEL_DIR    = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH   = MODEL_DIR / "signal_ranker.pkl"
GAP = ML_CFG["label_forward_periods"]
LINREG_PERIOD = LINREG_CFG["period"]

# The train/test and CV gap must be AT LEAST as large as the biggest
# feature lookback window, or validation rows can share overlapping
# history with training rows (leakage) even though they're on
# "different" dates. LinReg features look back up to LINREG_PERIOD
# (450) candles — larger than the SMC lookback this gap was originally
# sized for (previously hardcoded to 131) and larger than the label's
# forward window (GAP/130). Use the largest of the two to be safe.
SAFETY_GAP = max(LINREG_PERIOD, GAP)

# =============================================================================
# MODEL BUILDER
# =============================================================================

def _build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """
    Build the sklearn Pipeline for the Signal Ranker.

    Slightly deeper than the Volume Classifier (max_depth=5)
    because the Signal Ranker has more features and more complex
    interactions to learn (sentiment × market × volume).

    Args:
        scale_pos_weight: Ratio of negative to positive samples

    Returns:
        sklearn Pipeline
            """
    
    
    model = XGBClassifier(
        n_estimators       = 400,
        max_depth          = 5,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        reg_alpha          = 4.0,
        reg_lambda         = 4.0,
        scale_pos_weight   = scale_pos_weight,
        eval_metric        = "logloss",
        random_state       = 42,
        n_jobs             = -1,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model),
    ])

    return pipeline

# =============================================================================
# TRAINING
# =============================================================================

def train_signal_ranker(
    feature_matrix: pd.DataFrame,
) -> Tuple[Pipeline, dict]:
    """
    Train the Signal Ranker on the labelled feature matrix.

    FLOW:
    1. Validate minimum sample count
    2. Separate features (X) from labels (y)
    3. Drop identifier columns (ticker, date, direction)
       — these are not features, just row identifiers
    4. Compute class imbalance ratio
    5. Build pipeline
    6. 5-fold Walk-Forward cross-validation
    7. Train final model on all data
    8. Compute final metrics
    9. Save model to disk

    Args:
        feature_matrix: Output of build_signal_feature_matrix()
                        Must contain SIGNAL_FEATURE_COLS + 'label'

    Returns:
        Tuple of (trained Pipeline, metrics dict)
    """
    logger.info("=" * 60)
    logger.info("SIGNAL RANKER TRAINING STARTING")
    logger.info("=" * 60)

    # ── Step 1: Validate sample count ────────────────────────────────────────
    if len(feature_matrix) < MIN_SAMPLES:
        raise MLError(
            f"Insufficient training samples: {len(feature_matrix)} "
            f"(need {MIN_SAMPLES})"
        )

    # ── Step 2 & 3: Separate features, labels, AND DATE ─────────────────────
    # Keep date out of X. We only use it to split time
    df = feature_matrix.copy()
    X = df[SIGNAL_FEATURE_COLS].copy()
    y = df["label"].values


    # Use index as time proxy if date column not present
    # Features are already sorted chronologically by build_volume_feature_matrix()
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        # No date column — use row index as time proxy (already sorted chronologically)
        dates = pd.Series(range(len(df)), index=df.index)
        logger.warning(
            "No 'date' column in feature matrix — "
            "using row index as time proxy for walk-forward split"
        )
    # ── Step 4: Class imbalance ratio ─────────────────────────────────────────
    n_negative       = (y == 0).sum()
    n_positive       = (y == 1).sum()
    raw_ratio        = n_negative / n_positive if n_positive > 0 else 1.0
    scale_pos_weight = min(20.0, raw_ratio) # <- CAP IT. Ranking models hate >20

    logger.info(
        f"Class balance | "
        f"Positive: {n_positive} | Negative: {n_negative} | "
        f"Raw ratio: {raw_ratio:.2f} | scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # ── Step 5: Build pipeline ────────────────────────────────────────────────
    pipeline = _build_pipeline(scale_pos_weight) # set as one for now

    # ── Step 6: Cut a FINAL HOLDOUT FIRST. Never touch this during CV ───────
    # Last 30% by time = walk-forward reality check.
    #
    # A buffer of SAFETY_GAP rows is dropped between train and test —
    # previously there was NO gap here at all, so test rows near the
    # boundary could share overlapping LinReg lookback history (up to
    # 450 candles back) with the training rows right before them. That's
    # leakage: the model could see part of what it's later "tested" on.
    split_idx  = int(len(dates) * 0.70)
    test_start = split_idx + SAFETY_GAP

    if test_start >= len(dates):
        logger.warning(
            f"SAFETY_GAP ({SAFETY_GAP}) leaves no room for an OOS test set "
            f"with only {len(dates)} samples — falling back to no gap."
        )
        test_start = split_idx

    train_mask = dates.index < dates.index[split_idx]
    test_mask  = dates.index >= dates.index[test_start]

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(
        f"Walk-Forward Split | Train: {len(X_train)} | "
        f"Gap: {test_start - split_idx} rows | "
        f"OOS Test: {len(X_test)} | Ratio: {raw_ratio:.2f}:1"
    )

    # Diagnostic: log the actual date ranges + label balance of train vs
    # OOS test. CV folds are drawn entirely from the TRAIN window below —
    # they never see the OOS period at all — so if these two windows cover
    # meaningfully different market conditions (volatility, label balance),
    # that's a real, checkable signal of regime shift rather than pure
    # overfitting. Compare this across successive training runs to see if
    # it's a one-off artifact of this particular OOS window or persistent.
    if "date" in df.columns:
        train_dates = dates[train_mask]
        test_dates  = dates[test_mask]
        train_pos_rate = y_train.mean() if len(y_train) > 0 else float("nan")
        test_pos_rate  = y_test.mean()  if len(y_test)  > 0 else float("nan")
        logger.info(
            f"Train window | {train_dates.min()} → {train_dates.max()} | "
            f"Positive rate: {train_pos_rate:.4f}"
        )
        logger.info(
            f"OOS window   | {test_dates.min()} → {test_dates.max()} | "
            f"Positive rate: {test_pos_rate:.4f}"
        )

    # ── Step 7: Cross-validation on TRAIN only ───────────────────────────────
    # SAFETY_GAP = max(LinReg lookback, label forward window) — previously
    # hardcoded to 131 (sized only for SMC's lookback), which was smaller
    # than the 450-candle LinReg lookback and allowed leakage between CV
    # folds. NOTE: CV folds are all drawn from X_train (the oldest 70% of
    # data) — none of them ever see the OOS test window above. A tight CV
    # std here reflects consistency WITHIN the training period only, not
    # agreement with the true held-out future period.
    
    cv = TimeSeriesSplit(n_splits=5, gap=SAFETY_GAP)

    cv_auc_roc  = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
    cv_pr_auc   = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring="average_precision", n_jobs=-1)
    
        # Zero division safe precision scorer
    precision_scorer = make_scorer(
        precision_score,
        zero_division=0    # returns 0 instead of warning when no positives predicted
    )

    cv_precision = cross_val_score(pipeline, X_train, y_train,
                                   cv=cv, scoring=precision_scorer, n_jobs=-1)

    logger.info(
        f"Cross-validation | "
        f"AUC-ROC:   {cv_auc_roc.mean():.4f} ± {cv_auc_roc.std():.4f} | "
        f"PR-AUC:    {cv_pr_auc.mean():.4f} ± {cv_pr_auc.std():.4f} | "
        f"Precision: {cv_precision.mean():.4f} ± {cv_precision.std():.4f}"
    )


    # ── Step 8: Train FINAL model on TRAIN only ─────────────────────────────
    logger.info("Training final model on TRAIN set only...")
    pipeline.fit(X_train, y_train)


    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    # Use 0.5 for now, but for 8.92 imbalance you'll want to tune this later
    y_pred       = (y_pred_proba >= 0.7).astype(int) 

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc_roc   = roc_auc_score(y_test, y_pred_proba)
    pr_auc    = average_precision_score(y_test, y_pred_proba) 

    logger.info(
        f"OOS Test Metrics | Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}"
    )

    metrics = {
        "model_name"     : "signal_ranker",
        "train_date"     : datetime.today().strftime("%Y-%m-%d"),
        "precision"      : round(precision, 4), # <- This is your real one
        "recall"         : round(recall, 4),
        "f1"             : round(f1, 4),
        "auc_roc"        : round(auc_roc, 4),
        "pr_auc"         : round(pr_auc, 4),    # <- Target > 0.28
        "cv_auc_mean"    : round(cv_auc_roc.mean(), 4),
        "cv_auc_std"     : round(cv_auc_roc.std(), 4),
        "cv_pr_mean"     : round(cv_pr_auc.mean(), 4),
        "cv_pr_std"      : round(cv_pr_auc.std(), 4),
        "cv_precision_mean" : round(cv_precision.mean(), 4),
        "cv_precision_std"  : round(cv_precision.std(), 4),
        "n_train"        : len(X_train),
        "n_test"         : len(X_test),
    }

    logger.info(
        f"Final metrics | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"PR-AUC: {pr_auc:.4f} | "  
        f"AUC-ROC: {auc_roc:.4f} | "
        f"CV AUC-ROC: {cv_auc_roc.mean():.4f} ± {cv_auc_roc.std():.4f}"
    )

    # ── Step 9: Save model to disk ────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Signal Ranker saved to {MODEL_PATH}")
    logger.info("=" * 60)
    logger.info("SIGNAL RANKER TRAINING COMPLETE")
    logger.info("=" * 60)

    return pipeline, metrics


# =============================================================================
# INFERENCE
# =============================================================================

def load_signal_ranker() -> Optional[Pipeline]:
    """
    Load the trained Signal Ranker from disk.

    Returns:
        Trained Pipeline or None if model not found
    """
    if not MODEL_PATH.exists():
        logger.warning(
            f"Signal Ranker model not found at {MODEL_PATH}. "
            f"Train the model first."
        )
        return None

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    logger.info(f"Signal Ranker loaded from {MODEL_PATH}")
    return pipeline


def score_candidates(
    candidates_df  : pd.DataFrame,
    prices_df      : pd.DataFrame,
    indicators_df  : pd.DataFrame,
    market_ind_df  : pd.DataFrame,
    vol_scores     : dict,
    signal_date    : str,
    pipeline       : Optional[Pipeline] = None,
) -> pd.DataFrame:
    """
    Score all scanner candidates and rank them by probability.

    FLOW:
    1. Load model if not provided
    2. For each candidate (ticker + direction):
       a. Compute full signal feature vector
       b. Run predict_proba() → success probability
       c. Tag as high/normal probability
    3. Sort by ml_score descending
    4. Assign ml_rank (1 = best)
    5. Return updated candidates DataFrame

    If model is not available yet (first run before training):
    - All candidates get ml_score = 0.5 (neutral)
    - Ranked by SD position instead

    Args:
        candidates_df : Output of run_scanner() — unranked candidates
        prices_df     : Full OHLCV data
        indicators_df : Full indicator results
        sentiment_df  : Sentiment datacv
        market_ind_df : Indicator results for SPY, QQQ, DIA
        vol_scores    : Output of score_volume_signals()
        signal_date   : Today's date YYYY-MM-DD
        pipeline      : Optional pre-loaded model

    Returns:
        candidates_df with ml_score and ml_rank columns filled
    """
    if pipeline is None:
        pipeline = load_signal_ranker()

    if pipeline is None:
        logger.warning(
            "Signal Ranker not available — "
            "using SD position as preliminary ranking"
        )
        # Fallback: rank by how close price is to the mean
        candidates_df = candidates_df.copy()
        candidates_df["ml_score"] = 0.5
        candidates_df = candidates_df.sort_values(
            "sd_position",
            key=lambda x: x.abs(),
            ascending=True
        )
        candidates_df["ml_rank"] = range(1, len(candidates_df) + 1)
        return candidates_df

    results = []

    for _, row in candidates_df.iterrows():
        ticker    = row["ticker"]
        direction = row["direction"]

        # Get volume classifier score for this ticker
        vol_score = vol_scores.get(ticker, 0.5)

        # Get price data for this ticker
        px = prices_df[
            prices_df["ticker"] == ticker
        ].sort_values("date")

        try:
            features = compute_signal_features(
                ticker        = ticker,
                signal_date   = signal_date,
                direction     = direction,
                prices_df     = px,
                indicators_df = indicators_df,
                market_ind_df = market_ind_df,
                vol_clf_score = vol_score,
            )

            if features is None:
                ml_score = 0.5
            else:
                # Build single-row feature DataFrame
                X        = pd.DataFrame([features])[SIGNAL_FEATURE_COLS]
                ml_score = float(pipeline.predict_proba(X)[0][1])

        except Exception as e:
            logger.warning(f"{ticker} | Signal scoring failed: {e}")
            ml_score = 0.5

        result_row = row.to_dict()
        result_row["ml_score"] = round(ml_score, 4)
        results.append(result_row)

    result_df = pd.DataFrame(results)

    # ── Sort by ml_score descending within each direction ─────────────────────
    longs  = result_df[result_df["direction"] == "long"].sort_values(
        "ml_score", ascending=False
    ).reset_index(drop=True)

    shorts = result_df[result_df["direction"] == "short"].sort_values(
        "ml_score", ascending=False
    ).reset_index(drop=True)

    # ── Assign rank within each direction ────────────────────────────────────
    longs["ml_rank"]  = longs.index + 1
    shorts["ml_rank"] = shorts.index + 1

    final = pd.concat([longs, shorts], ignore_index=True)

    # ── Log high probability candidates ──────────────────────────────────────
    high_prob = final[final["ml_score"] >= HIGH_PROB_THRESHOLD]
    logger.info(
        f"Signal Ranker scoring complete | "
        f"Total candidates: {len(final)} | "
        f"High probability (≥{HIGH_PROB_THRESHOLD}): {len(high_prob)}"
    )

    if not high_prob.empty:
        logger.info(
            f"Top candidates:\n"
            f"{high_prob[['ticker','direction','ml_score','ml_rank']].to_string(index=False)}"
        )

    return final
