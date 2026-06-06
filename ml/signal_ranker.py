"""
ml/signal_ranker.py
-------------------
Signal Ranker ML model for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This model answers ONE question:
"Given this setup, what is the probability that price reaches
the LinReg mean within 15 days?"

It is a BINARY CLASSIFIER used as a PROBABILITY ESTIMATOR:
- Output is a float between 0 and 1
- Higher score = higher probability of mean reversion
- Used to RANK candidates, not just classify them

WHY THIS IS MORE USEFUL THAN PURE CLASSIFICATION:
   A binary "yes/no" output would give us:
   "AAPL: yes, TSLA: yes, NVDA: yes"
   We can't prioritise from that.

   A probability output gives us:
   "AAPL: 0.81, NVDA: 0.73, TSLA: 0.54"
   We know exactly which to focus on first.

MODEL: XGBoost Classifier with predict_proba()
   Same model family as Volume Classifier for consistency.
   The probability calibration is naturally reasonable for XGBoost
   without additional calibration steps.

KEY DIFFERENCE FROM VOLUME CLASSIFIER:
   The Signal Ranker uses MORE features:
   - All volume features PLUS the Volume Classifier score
   - Sentiment (Put/Call, Short Interest)
   - Market and sector context
   - RSI momentum
   The Volume Classifier feeds INTO the Signal Ranker as a feature.
   This is a two-stage ML pipeline (stacking pattern).

TRAINING SCHEDULE:
   Same as Volume Classifier — every 7 days.
   Trained AFTER Volume Classifier so vol_classifier scores
   are available as features.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import precision_score, roc_auc_score

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

config    = _load_config()
ML_CFG    = config["ml"]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH  = MODEL_DIR / "signal_ranker.pkl"
MIN_SAMPLES = ML_CFG["min_training_samples"]    # 500
HIGH_PROB   = ML_CFG["high_probability_threshold"]  # 0.70


# =============================================================================
# MODEL DEFINITION
# =============================================================================

def _build_model(n_positive: int, n_negative: int) -> XGBClassifier:
    """
    Build Signal Ranker XGBoost model.

    Slightly deeper than Volume Classifier (max_depth=5 vs 4)
    because it has more features and more complex interactions
    to learn (market context, sentiment, momentum all combined).

    Args:
        n_positive: Positive sample count for class weighting
        n_negative: Negative sample count for class weighting

    Returns:
        Configured XGBClassifier
    """
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    return XGBClassifier(
        n_estimators      = 300,        # More trees than vol classifier
        max_depth         = 5,          # Slightly deeper — more features
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,          # Prevent overfitting on small groups
        scale_pos_weight  = scale_pos_weight,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_signal_ranker(
    X : pd.DataFrame,
    y : pd.Series,
) -> Tuple[XGBClassifier, float, float]:
    """
    Train the Signal Ranker on labelled scanner hit data.

    FLOW:
    1. Validate minimum sample count
    2. Compute class balance
    3. Build model
    4. 5-fold stratified cross-validation for honest evaluation
    5. Train final model on all data
    6. Log feature importances (useful for understanding the model)
    7. Save model to disk
    8. Return model + metrics

    Args:
        X: Feature matrix from build_signal_ranker_matrix()
        y: Labels (1 = price reached LinReg in 15 days, 0 = didn't)

    Returns:
        Tuple of (trained model, precision, auc_roc)
    """
    if len(X) < MIN_SAMPLES:
        raise MLError(
            f"Insufficient samples for Signal Ranker: "
            f"{len(X)} < {MIN_SAMPLES} minimum"
        )

    logger.info(
        f"Training Signal Ranker | "
        f"Samples: {len(X)} | "
        f"Features: {X.shape[1]} | "
        f"Positive rate: {y.mean():.2%}"
    )

    n_positive = int(y.sum())
    n_negative = len(y) - n_positive

    model = _build_model(n_positive, n_negative)

    # ── 5-fold stratified cross-validation ───────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y,
        cv      = cv,
        scoring = ["precision", "roc_auc"],
    )

    precision = float(np.mean(cv_results["test_precision"]))
    auc_roc   = float(np.mean(cv_results["test_roc_auc"]))

    logger.info(
        f"Signal Ranker CV results | "
        f"Precision: {precision:.4f} | "
        f"AUC-ROC: {auc_roc:.4f}"
    )

    # ── Train final model on ALL data ─────────────────────────────────────────
    model.fit(X, y)

    # ── Log top feature importances ───────────────────────────────────────────
    # This tells us which features the model finds most predictive
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    logger.info(
        f"Top 5 features:\n"
        f"{importances.head(5).to_string()}"
    )

    # ── Save model ────────────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Signal Ranker saved to {MODEL_PATH}")
    return model, precision, auc_roc


# =============================================================================
# INFERENCE
# =============================================================================

def load_signal_ranker() -> Optional[XGBClassifier]:
    """
    Load trained Signal Ranker from disk.

    Returns:
        Loaded model or None if not yet trained
    """
    if not MODEL_PATH.exists():
        logger.warning(
            "Signal Ranker model not found — "
            "will train on first run with sufficient data"
        )
        return None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("Signal Ranker loaded from disk")
    return model


def predict_signal_ranker(
    model : XGBClassifier,
    X     : pd.DataFrame,
) -> np.ndarray:
    """
    Generate probability scores for scanner candidates.

    Returns probability of price reaching LinReg mean within 15 days.
    This IS the ML Score shown on the Streamlit dashboard.

    Args:
        model: Trained Signal Ranker
        X    : Inference feature matrix

    Returns:
        Array of probabilities (0-1) — one per candidate
    """
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities


def should_retrain(last_train_date: Optional[str]) -> bool:
    """
    Check if Signal Ranker needs retraining.
    Same logic as Volume Classifier.

    Args:
        last_train_date: Date of last training or None

    Returns:
        True if retraining needed
    """
    retrain_days = ML_CFG["model_retrain_days"]

    if last_train_date is None:
        return True

    last_date = datetime.strptime(last_train_date, "%Y-%m-%d")
    days_ago  = (datetime.today() - last_date).days

    needs_retrain = days_ago >= retrain_days
    logger.info(
        f"Signal Ranker: trained {days_ago} days ago | "
        f"Retrain: {needs_retrain}"
    )
    return needs_retrain


# =============================================================================
# RESULTS RANKER
# Takes probability scores and attaches them to the candidates DataFrame
# Re-ranks everything by ML score descending
# =============================================================================

def rank_candidates(
    candidates_df : pd.DataFrame,
    scores        : np.ndarray,
    tickers       : list[str],
) -> pd.DataFrame:
    """
    Attach ML scores to candidates and re-rank by probability.

    FLOW:
    1. Map scores to tickers
    2. Attach ml_score to each candidate row
    3. Sort by ml_score descending within each direction
    4. Re-assign ml_rank (1 = highest probability)
    5. Flag high probability candidates (score >= threshold)

    Args:
        candidates_df: Scanner results DataFrame
        scores       : Probability array from predict_signal_ranker()
        tickers      : List of tickers corresponding to scores

    Returns:
        Updated candidates DataFrame with ml_score and ml_rank
    """
    if len(scores) != len(tickers):
        logger.error(
            f"Score/ticker mismatch: "
            f"{len(scores)} scores vs {len(tickers)} tickers"
        )
        return candidates_df

    # ── Build score lookup ────────────────────────────────────────────────────
    score_map = dict(zip(tickers, scores))

    # ── Attach scores ─────────────────────────────────────────────────────────
    candidates_df = candidates_df.copy()
    candidates_df["ml_score"] = candidates_df["ticker"].map(score_map).fillna(0.0)
    candidates_df["ml_score"] = candidates_df["ml_score"].round(4)

    # ── Flag high probability setups ──────────────────────────────────────────
    candidates_df["high_probability"] = candidates_df["ml_score"] >= HIGH_PROB

    # ── Sort and re-rank within each direction ────────────────────────────────
    ranked_parts = []

    for direction in ["long", "short"]:
        part = candidates_df[candidates_df["direction"] == direction].copy()
        part = part.sort_values("ml_score", ascending=False).reset_index(drop=True)
        part["ml_rank"] = part.index + 1
        ranked_parts.append(part)

    if not ranked_parts:
        return candidates_df

    result = pd.concat(ranked_parts, ignore_index=True)

    high_prob_count = result["high_probability"].sum()
    logger.info(
        f"Candidates ranked | "
        f"Total: {len(result)} | "
        f"High probability (≥{HIGH_PROB}): {high_prob_count}"
    )

    return result


# =============================================================================
# MAIN ENTRY POINT
# Called by Airflow DAG Task 8 (after Volume Classifier)
# =============================================================================

def run_signal_ranker_pipeline(
    X              : pd.DataFrame,
    y              : pd.Series,
    last_train_date: Optional[str],
    candidates_df  : pd.DataFrame,
    inference_X    : pd.DataFrame,
) -> pd.DataFrame:
    """
    Full Signal Ranker pipeline: retrain if needed + rank candidates.

    FLOW:
    1. Retrain if schedule requires it
    2. Load model
    3. Score today's candidates
    4. Rank by score
    5. Return ranked candidates ready for database write

    Args:
        X              : Training features
        y              : Training labels
        last_train_date: Last training date from SQLite
        candidates_df  : Today's scanner candidates
        inference_X    : Feature matrix for today's candidates

    Returns:
        Ranked candidates DataFrame with ml_score and ml_rank filled in
    """
    from data.database import write_model_metrics

    today     = datetime.today().strftime("%Y-%m-%d")
    retrained = False

    # ── Step 1: Retrain if needed ─────────────────────────────────────────────
    if should_retrain(last_train_date) and len(X) >= MIN_SAMPLES:
        try:
            model, precision, auc_roc = train_signal_ranker(X, y)
            write_model_metrics(
                "signal_ranker", today,
                precision, auc_roc, len(X)
            )
            retrained = True
            logger.info(f"Signal Ranker retrained | Precision: {precision:.4f}")
        except MLError as e:
            logger.warning(f"Signal Ranker training failed: {e}")

    # ── Step 2: Load model ────────────────────────────────────────────────────
    model = load_signal_ranker()

    # ── Step 3: Score and rank candidates ────────────────────────────────────
    if model is None or inference_X.empty or candidates_df.empty:
        logger.warning(
            "Signal Ranker: Cannot score — "
            "model not available or no candidates"
        )
        return candidates_df

    # Drop ticker column before inference
    feature_cols = [c for c in inference_X.columns if c != "ticker"]
    tickers      = inference_X["ticker"].tolist()
    scores       = predict_signal_ranker(model, inference_X[feature_cols])

    # ── Step 4: Attach scores and rank ────────────────────────────────────────
    ranked = rank_candidates(candidates_df, scores, tickers)

    logger.info(
        f"Signal Ranker pipeline complete | "
        f"Retrained: {retrained} | "
        f"Candidates ranked: {len(ranked)}"
    )

    return ranked