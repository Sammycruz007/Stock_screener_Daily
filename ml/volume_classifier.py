"""
ml/volume_classifier.py
------------------------
Volume Classifier ML model for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This model learns to distinguish TRUE accumulation/distribution
patterns from FALSE ones using historical volume data.

TRAINING FLOW:
1. Load labelled volume examples from labeller.py
2. Build feature matrix from features.py
3. Handle class imbalance (usually more neutral than clear signals)
4. Train XGBoost classifier with cross-validation
5. Evaluate on holdout set (Precision + AUC-ROC)
6. Save model to disk
7. Write metrics to SQLite

INFERENCE FLOW (daily pipeline):
1. Load saved model from disk
2. For each stock in today's filtered universe:
   - Compute volume features for today
   - Run model.predict_proba()
   - Get probability of true accumulation (class 1)
3. Return scores — fed into Signal Ranker as a feature

MODEL CHOICE — XGBoost:
   XGBoost consistently outperforms on tabular financial data.
   It handles:
   - Mixed feature types (ratios, counts, flags)
   - Missing values natively
   - Non-linear relationships between volume patterns
   - Class imbalance via scale_pos_weight parameter

RETRAINING:
   Model is retrained every MODEL_RETRAIN_DAYS (7 days) by Airflow.
   New labelled data accumulates daily as new scanner results
   are written to SQLite.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.features import (
    VOLUME_FEATURE_COLS,
    compute_volume_features,
)
from utils.logging import get_ml_logger
from utils.error_handler import MLError, handle_critical_error

logger = get_ml_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config  = _load_config()
ML_CFG  = config["ml"]

MIN_SAMPLES    = ML_CFG["min_training_samples"]    # 500
MODEL_DIR      = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH     = MODEL_DIR / "volume_classifier.pkl"


# =============================================================================
# MODEL BUILDER
# =============================================================================

def _build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """
    Build the sklearn Pipeline for the Volume Classifier.

    PIPELINE STEPS:
    1. SimpleImputer  → fills any NaN values with median
       (some volume stats may be NaN for very new stocks)
    2. XGBClassifier  → the actual model

    XGBOOST PARAMETERS:
    - n_estimators    : 300 trees (enough depth without overfitting)
    - max_depth       : 4 (shallow trees = less overfit on financial data)
    - learning_rate   : 0.05 (slow learning = better generalisation)
    - subsample       : 0.8 (row sampling = reduces overfit)
    - colsample_bytree: 0.8 (feature sampling per tree)
    - scale_pos_weight: handles class imbalance
      (if 70% negative, 30% positive → scale = 70/30 = 2.33)

    Args:
        scale_pos_weight: Ratio of negative to positive samples

    Returns:
        sklearn Pipeline ready for fit/predict
    """
    model = XGBClassifier(
        n_estimators       = 300,
        max_depth          = 4,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale_pos_weight,
        use_label_encoder  = False,
        eval_metric        = "logloss",
        random_state       = 42,
        n_jobs             = -1,       # Use all CPU cores
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model),
    ])

    return pipeline


# =============================================================================
# TRAINING
# =============================================================================

def train_volume_classifier(
    feature_matrix: pd.DataFrame,
) -> Tuple[Pipeline, dict]:
    """
    Train the Volume Classifier on the labelled feature matrix.

    FLOW:
    1. Validate minimum sample count
    2. Separate features (X) from labels (y)
    3. Compute class imbalance ratio for scale_pos_weight
    4. Build pipeline
    5. Evaluate with 5-fold stratified cross-validation
       (stratified = preserves class balance in each fold)
    6. Train final model on ALL data
    7. Compute final metrics on full training set
    8. Save model to disk
    9. Return model + metrics dict

    WHY TRAIN FINAL ON ALL DATA:
    Cross-validation gives us reliable metric estimates.
    The final model trains on ALL data to maximise the
    amount of patterns it has learned before deployment.

    Args:
        feature_matrix: Output of build_volume_feature_matrix()
                        Must contain VOLUME_FEATURE_COLS + 'label'

    Returns:
        Tuple of (trained Pipeline, metrics dict)
    """
    logger.info("=" * 60)
    logger.info("VOLUME CLASSIFIER TRAINING STARTING")
    logger.info("=" * 60)

    # ── Step 1: Validate sample count ────────────────────────────────────────
    if len(feature_matrix) < MIN_SAMPLES:
        raise MLError(
            f"Insufficient training samples: {len(feature_matrix)} "
            f"(need {MIN_SAMPLES})"
        )

    # ── Step 2: Separate features and labels ──────────────────────────────────
    X = feature_matrix[VOLUME_FEATURE_COLS].copy()
    y = feature_matrix["label"].values

    logger.info(
        f"Training data | Samples: {len(X)} | "
        f"Features: {len(VOLUME_FEATURE_COLS)} | "
        f"Positive: {y.sum()} | Negative: {(y==0).sum()}"
    )

    # ── Step 3: Compute class imbalance ratio ─────────────────────────────────
    n_negative      = (y == 0).sum()
    n_positive      = (y == 1).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    logger.info(f"Class balance | scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Step 4: Build pipeline ────────────────────────────────────────────────
    pipeline = _build_pipeline(scale_pos_weight)

    # ── Step 5: Cross-validation ──────────────────────────────────────────────
    # 5-fold stratified CV gives reliable performance estimate
    cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_roc  = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    logger.info(
        f"Cross-validation AUC-ROC | "
        f"Mean: {cv_auc_roc.mean():.4f} | "
        f"Std: {cv_auc_roc.std():.4f} | "
        f"Folds: {cv_auc_roc.tolist()}"
    )

    # ── Step 6: Train final model on all data ─────────────────────────────────
    logger.info("Training final model on full dataset...")
    pipeline.fit(X, y)

    # ── Step 7: Compute final metrics ─────────────────────────────────────────
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    precision = precision_score(y, y_pred, zero_division=0)
    auc_roc   = roc_auc_score(y, y_pred_proba)

    metrics = {
        "model_name"     : "volume_classifier",
        "train_date"     : datetime.today().strftime("%Y-%m-%d"),
        "precision_score": round(precision, 4),
        "auc_roc_score"  : round(auc_roc, 4),
        "cv_auc_mean"    : round(cv_auc_roc.mean(), 4),
        "cv_auc_std"     : round(cv_auc_roc.std(), 4),
        "n_samples"      : len(X),
        "n_features"     : len(VOLUME_FEATURE_COLS),
    }

    logger.info(
        f"Final metrics | "
        f"Precision: {precision:.4f} | "
        f"AUC-ROC: {auc_roc:.4f}"
    )

    # ── Step 8: Save model to disk ────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info("=" * 60)
    logger.info("VOLUME CLASSIFIER TRAINING COMPLETE")
    logger.info("=" * 60)

    return pipeline, metrics


# =============================================================================
# INFERENCE
# =============================================================================

def load_volume_classifier() -> Optional[Pipeline]:
    """
    Load the trained Volume Classifier from disk.

    Returns:
        Trained Pipeline or None if model file not found
    """
    if not MODEL_PATH.exists():
        logger.warning(
            f"Volume Classifier model not found at {MODEL_PATH}. "
            f"Train the model first."
        )
        return None

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    logger.info(f"Volume Classifier loaded from {MODEL_PATH}")
    return pipeline


def score_volume_signals(
    tickers_data : dict[str, pd.DataFrame],
    signal_date  : str,
    pipeline     : Optional[Pipeline] = None,
) -> dict[str, float]:
    """
    Score volume patterns for all tickers on a given date.

    FLOW:
    1. Load model if not provided
    2. For each ticker, compute volume features for today
    3. Run predict_proba() to get accumulation probability
    4. Return dict of ticker → score

    This is called during the daily pipeline AFTER the scanner
    identifies candidates. The scores are then passed as a feature
    to the Signal Ranker.

    Args:
        tickers_data: Dict of ticker → OHLCV DataFrame
        signal_date : Today's date YYYY-MM-DD
        pipeline    : Optional pre-loaded model pipeline

    Returns:
        Dict of ticker → probability score (0.0 to 1.0)
        Higher score = stronger accumulation/distribution signal
    """
    if pipeline is None:
        pipeline = load_volume_classifier()

    if pipeline is None:
        logger.warning(
            "Volume Classifier not available — "
            "returning 0.5 (neutral) for all tickers"
        )
        return {ticker: 0.5 for ticker in tickers_data}

    scores = {}

    for ticker, df in tickers_data.items():
        try:
            features = compute_volume_features(df, signal_date)

            if features is None:
                scores[ticker] = 0.5
                continue

            # Build single-row feature DataFrame
            X = pd.DataFrame([features])[VOLUME_FEATURE_COLS]

            # Get probability of class 1 (true accumulation/distribution)
            prob        = pipeline.predict_proba(X)[0][1]
            scores[ticker] = round(float(prob), 4)

        except Exception as e:
            logger.warning(f"{ticker} | Volume scoring failed: {e}")
            scores[ticker] = 0.5

    logger.info(
        f"Volume scoring complete | "
        f"Scored: {len(scores)} tickers | "
        f"Date: {signal_date}"
    )

    return scores