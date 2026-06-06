"""
ml/volume_classifier.py
-----------------------
Volume Classifier ML model for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This model answers ONE question:
"Does this volume pattern indicate accumulation or distribution?"

It is a BINARY CLASSIFIER:
- Output 1 = Accumulation (price likely to bounce up)
- Output 0 = Distribution (price likely to continue down)

WHY ML OVER RULES:
   Our rules-based volume engine in volume.py uses hardcoded
   thresholds (e.g. volume must be 1.5x average to be a shakeout).
   The ML model LEARNS the optimal thresholds from historical data.
   It may discover that 1.3x average is actually more predictive
   than 1.5x for certain combinations of conditions.
   It also learns INTERACTIONS between conditions that rules can't capture.
   e.g. "Condition 1 + Condition 3 together are much more powerful
        than either alone" — a tree model captures this naturally.

MODEL: XGBoost Classifier
   - Handles tabular data extremely well
   - Fast training and inference
   - Built-in feature importance
   - Robust to class imbalance with scale_pos_weight
   - No need for feature scaling (tree-based)

TRAINING SCHEDULE:
   Retrained every 7 days (configurable in config.yaml)
   Uses all available labelled history — not just recent data
   Model saved to disk after training for reuse

EVALUATION:
   Primary metric: Precision (are the positives it flags correct?)
   Secondary metric: AUC-ROC (overall ranking quality)
   Both written to model_metrics table in SQLite
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
from sklearn.preprocessing import StandardScaler

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

config    = _load_config()
ML_CFG    = config["ml"]
MODEL_DIR = Path("models")   # Local directory for saved models
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODEL_DIR / "volume_classifier.pkl"
MIN_SAMPLES  = ML_CFG["min_training_samples"]    # 500


# =============================================================================
# MODEL DEFINITION
# XGBoost parameters tuned for this specific use case.
# =============================================================================

def _build_model(n_positive: int, n_negative: int) -> XGBClassifier:
    """
    Build and configure XGBoost classifier.

    KEY PARAMETERS:
    - scale_pos_weight: Handles class imbalance.
      Set to n_negative / n_positive so the model
      treats the minority class with more importance.
      e.g. if 70% are failures (0) and 30% success (1):
      scale_pos_weight = 0.7 / 0.3 ≈ 2.33

    - max_depth: 4 keeps trees shallow — reduces overfitting
      on financial data which is inherently noisy.

    - n_estimators: 200 trees — enough for good performance
      without being too slow.

    - learning_rate: 0.05 — small steps, more robust.

    - subsample + colsample_bytree: Randomisation to prevent
      overfitting — each tree sees a random subset of data and features.

    Args:
        n_positive: Number of positive (success) samples
        n_negative: Number of negative (failure) samples

    Returns:
        Configured XGBClassifier instance
    """
    # Compute class weight for imbalance handling
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

    return XGBClassifier(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,          # Use all CPU cores
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_volume_classifier(
    X : pd.DataFrame,
    y : pd.Series,
) -> Tuple[XGBClassifier, float, float]:
    """
    Train the Volume Classifier on labelled volume pattern data.

    FLOW:
    1. Check minimum sample requirement
    2. Compute class balance for scale_pos_weight
    3. Build XGBoost model
    4. Evaluate with 5-fold stratified cross-validation
       (stratified = maintains class ratio in each fold)
    5. Train final model on ALL data
       (cross-val is for evaluation only — final model uses everything)
    6. Save model to disk
    7. Return model + metrics

    WHY CROSS-VALIDATION:
       We evaluate on cross-val folds to get honest performance estimates.
       But we train the final model on ALL data because more data = better model.
       This is the standard production ML pattern.

    Args:
        X: Feature matrix from build_volume_classifier_matrix()
        y: Labels Series (1 = accumulation success, 0 = failure)

    Returns:
        Tuple of (trained model, precision score, auc_roc score)
    """
    if len(X) < MIN_SAMPLES:
        raise MLError(
            f"Insufficient samples for Volume Classifier: "
            f"{len(X)} < {MIN_SAMPLES} minimum"
        )

    logger.info(
        f"Training Volume Classifier | "
        f"Samples: {len(X)} | "
        f"Features: {X.shape[1]} | "
        f"Positive rate: {y.mean():.2%}"
    )

    # ── Class counts for scale_pos_weight ────────────────────────────────────
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive

    # ── Build model ───────────────────────────────────────────────────────────
    model = _build_model(n_positive, n_negative)

    # ── 5-fold stratified cross-validation ───────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = cross_validate(
        model, X, y,
        cv            = cv,
        scoring       = ["precision", "roc_auc"],
        return_train_score = False,
    )

    precision = float(np.mean(cv_results["test_precision"]))
    auc_roc   = float(np.mean(cv_results["test_roc_auc"]))

    logger.info(
        f"Volume Classifier CV results | "
        f"Precision: {precision:.4f} | "
        f"AUC-ROC: {auc_roc:.4f}"
    )

    # ── Train final model on ALL data ─────────────────────────────────────────
    model.fit(X, y)

    # ── Save model to disk ────────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Volume Classifier saved to {MODEL_PATH}")

    return model, precision, auc_roc


# =============================================================================
# INFERENCE
# =============================================================================

def load_volume_classifier() -> Optional[XGBClassifier]:
    """
    Load the trained Volume Classifier from disk.

    Returns:
        Loaded XGBClassifier or None if model file doesn't exist yet
        (first run before any training has occurred)
    """
    if not MODEL_PATH.exists():
        logger.warning(
            "Volume Classifier model not found — "
            "model will be trained on first run with sufficient data"
        )
        return None

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("Volume Classifier loaded from disk")
    return model


def predict_volume_classifier(
    model : XGBClassifier,
    X     : pd.DataFrame,
) -> np.ndarray:
    """
    Generate volume classifier probability scores for inference.

    Returns PROBABILITIES (not binary predictions) so the signal
    ranker can use the volume score as a continuous feature.

    Args:
        model: Trained XGBClassifier
        X    : Feature matrix (same columns as training)

    Returns:
        Array of probabilities between 0 and 1
        (probability of being accumulation = class 1)
    """
    # predict_proba returns [prob_class_0, prob_class_1] per sample
    # We take column 1 = probability of accumulation
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities


# =============================================================================
# SHOULD WE RETRAIN?
# =============================================================================

def should_retrain(last_train_date: Optional[str]) -> bool:
    """
    Check if the model needs retraining based on config schedule.

    LOGIC:
    - No model exists yet → always retrain
    - Model trained within last 7 days → skip
    - Model older than 7 days → retrain

    Args:
        last_train_date: Date string YYYY-MM-DD of last training,
                         or None if never trained

    Returns:
        True if retraining is needed
    """
    retrain_days = ML_CFG["model_retrain_days"]   # 7

    if last_train_date is None:
        return True

    last_date = datetime.strptime(last_train_date, "%Y-%m-%d")
    days_ago  = (datetime.today() - last_date).days

    if days_ago >= retrain_days:
        logger.info(
            f"Volume Classifier: {days_ago} days since last training "
            f"(threshold: {retrain_days}) — retraining"
        )
        return True

    logger.info(
        f"Volume Classifier: trained {days_ago} days ago — "
        f"skipping retraining"
    )
    return False


# =============================================================================
# MAIN ENTRY POINT
# Called by Airflow DAG Task 8
# =============================================================================

def run_volume_classifier_pipeline(
    X             : pd.DataFrame,
    y             : pd.Series,
    last_train_date: Optional[str],
    inference_X   : Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full Volume Classifier pipeline: retrain if needed + inference.

    FLOW:
    1. Check if retraining is needed
    2. If yes → train, save, write metrics to SQLite
    3. Load model (from disk — whether just trained or previously saved)
    4. If inference_X provided → score today's candidates
    5. Return dict with scores and metrics

    Args:
        X              : Training feature matrix
        y              : Training labels
        last_train_date: Date of last model training (from SQLite)
        inference_X    : Today's candidate features for scoring

    Returns:
        Dict with:
        - 'scores'    : array of probabilities for inference_X (or None)
        - 'precision' : most recent precision score
        - 'auc_roc'   : most recent AUC-ROC score
        - 'retrained' : bool — whether model was retrained today
    """
    from data.database import write_model_metrics

    today     = datetime.today().strftime("%Y-%m-%d")
    retrained = False
    precision = None
    auc_roc   = None

    # ── Step 1: Retrain if needed ─────────────────────────────────────────────
    if should_retrain(last_train_date) and len(X) >= MIN_SAMPLES:
        try:
            model, precision, auc_roc = train_volume_classifier(X, y)
            write_model_metrics(
                "volume_classifier", today,
                precision, auc_roc, len(X)
            )
            retrained = True
        except MLError as e:
            logger.warning(f"Volume Classifier training failed: {e}")

    # ── Step 2: Load model ────────────────────────────────────────────────────
    model = load_volume_classifier()

    # ── Step 3: Score today's candidates ─────────────────────────────────────
    scores = None
    if model is not None and inference_X is not None and not inference_X.empty:
        # Drop ticker column before inference — not a model feature
        feature_cols = [c for c in inference_X.columns if c != "ticker"]
        scores       = predict_volume_classifier(model, inference_X[feature_cols])
        logger.info(f"Volume Classifier scored {len(scores)} candidates")

    return {
        "scores"   : scores,
        "precision": precision,
        "auc_roc"  : auc_roc,
        "retrained": retrained,
    }