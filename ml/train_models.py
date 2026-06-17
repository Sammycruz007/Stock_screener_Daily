"""
ml/train_models.py
-------------------
Backfills a rolling indicator history from the 59-day raw_prices data
already fetched, generates labels, builds feature matrices, and trains
both ML models.

WHY THIS IS SEPARATE FROM run_pipeline.py:
indicator_results only stores ONE row per ticker per daily run (today's
candle). The labeller needs MANY historical (ticker, date) indicator rows
to derive labels. This script regenerates that history by re-running the
LinReg/SMC/Volume engines on rolling windows of the price data we already
have — a one-time/periodic batch job, not part of the daily 6AM DAG.

STRIDE:
backfill_stride=4 means we compute indicators every 4th 15m candle
(~hourly) instead of every candle. Cuts compute time ~4x while still
producing thousands of labelled examples per ticker.

Run with:
    python ml/train_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
import pandas as pd
import yaml

from data.database import (
    initialise_database,
    read_filtered_universe,
    read_raw_prices,
    read_sector_metadata,
    write_model_metrics,
)
from engines.linreg import compute_linreg_latest, PERIOD as LINREG_PERIOD
from engines.smc    import compute_smc
from engines.volume import compute_volume_signal
from scanner.screener import (
    _check_sector_health,
    _is_long_candidate,
    _is_short_candidate,
)
from ml.labeller import label_volume_patterns, label_scanner_hits
from ml.features import build_volume_feature_matrix, build_signal_feature_matrix
from ml.volume_classifier import train_volume_classifier, score_volume_signals
from ml.signal_ranker     import train_signal_ranker
from utils.logging import get_ml_logger

logger = get_ml_logger()


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config        = _load_config()
ML_CFG        = config["ml"]
FORWARD       = ML_CFG["label_forward_periods"]
STRIDE        = ML_CFG.get("backfill_stride", 4)
UNIVERSE_CFG  = config["universe"]

# Optional cap for a quick first test run — set to None for full universe
MAX_TICKERS_FOR_TRAINING = None


# =============================================================================
# ROLLING BACKFILL — re-run engines across history for one ticker
# =============================================================================

def backfill_ticker(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run LinReg, SMC, Volume engines on rolling windows of df.

    For each step i (i = LINREG_PERIOD .. len(df)-1, stride STRIDE):
        window = df.iloc[:i+1]   (everything up to and including candle i)
        date   = df.iloc[i]["date"]
        -> compute_linreg_latest, compute_smc (with SD bands), compute_volume_signal

    Returns:
        DataFrame with one row per (ticker, date) — same shape as
        indicator_results, covering the full backfilled history.
    """
    rows = []
    n = len(df)

    if n < LINREG_PERIOD + FORWARD + 1:
        return pd.DataFrame()

    for i in range(LINREG_PERIOD, n, STRIDE):
        window = df.iloc[: i + 1]
        date   = df.iloc[i]["date"]

        lr = compute_linreg_latest(ticker, window, date)
        if lr is None:
            continue

        smc = compute_smc(
            ticker, window, date,
            sd1_lower=lr["sd1_lower"], sd3_lower=lr["sd3_lower"],
            sd1_upper=lr["sd1_upper"], sd3_upper=lr["sd3_upper"],
        )
        if smc is None:
            continue

        vol = compute_volume_signal(ticker, window, date)
        if vol is None:
            continue

        rows.append({
            **lr,
            "smc_structure" : smc["smc_structure"],
            "choch_detected": smc["choch_detected"],
            "has_valid_zone": smc["has_valid_zone"],
            "volume_signal" : vol["volume_signal"],
        })

    return pd.DataFrame(rows)


# =============================================================================
# HISTORICAL SCAN HITS — replay scanner conditions across backfilled history
# =============================================================================

def build_historical_scan_hits(
    indicators_history_df: pd.DataFrame,
    ticker_to_etf: dict,
) -> pd.DataFrame:
    """
    For every (ticker, date) row in the backfilled history, check whether
    it would have qualified as a long or short candidate at that point
    in time. Used as the input to the Signal Ranker labeller.

    Market-level gating (scan_long/scan_short) is intentionally skipped here
    — market context is already captured as features (market_slope_avg,
    market_choch_count). We label every individually-qualifying setup.
    """
    excluded = set(UNIVERSE_CFG["indices"] + UNIVERSE_CFG["sectors"])

    # Precompute sector health per date
    sector_health_by_date = {}
    for date, group in indicators_history_df.groupby("date"):
        sector_health_by_date[date] = _check_sector_health(group)

    hits = []
    stock_rows = indicators_history_df[~indicators_history_df["ticker"].isin(excluded)]

    for _, row in stock_rows.iterrows():
        sector_health = sector_health_by_date.get(row["date"], {})

        if _is_long_candidate(row, sector_health, ticker_to_etf):
            hits.append({"ticker": row["ticker"], "date": row["date"], "direction": "long"})

        if _is_short_candidate(row, sector_health, ticker_to_etf):
            hits.append({"ticker": row["ticker"], "date": row["date"], "direction": "short"})

    logger.info(f"build_historical_scan_hits: {len(hits)} historical setups found")
    return pd.DataFrame(hits)


# =============================================================================
# MAIN
# =============================================================================

def run_training():
    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    initialise_database()
    today = datetime.today().strftime("%Y-%m-%d")

    # ── Step 1: ticker universe (filtered + indices + sectors) ───────────────
    filtered = read_filtered_universe(today)
    if filtered.empty:
        logger.error("No filtered universe found — run run_pipeline.py first")
        return

    tickers = filtered["ticker"].tolist()
    if MAX_TICKERS_FOR_TRAINING:
        tickers = tickers[:MAX_TICKERS_FOR_TRAINING]

    logger.info(f"Backfilling indicators for {len(tickers)} tickers (stride={STRIDE})")

    # ── Step 2: rolling backfill per ticker ───────────────────────────────────
    indicator_history = []
    prices_all = []

    for n, ticker in enumerate(tickers, 1):
        df = read_raw_prices(ticker)
        if df.empty:
            continue

        prices_all.append(df)

        hist = backfill_ticker(ticker, df)
        if not hist.empty:
            hist.insert(0, "ticker", ticker) if "ticker" not in hist.columns else None
            indicator_history.append(hist)

        if n % 100 == 0:
            logger.info(f"Backfill progress: {n}/{len(tickers)} tickers")

    if not indicator_history:
        logger.error("No indicator history produced — aborting")
        return

    indicators_history_df = pd.concat(indicator_history, ignore_index=True)
    prices_all_df          = pd.concat(prices_all, ignore_index=True)

    logger.info(f"Backfill complete | Total rows: {len(indicators_history_df)}")

    # ── Step 3: market indicator history (SPY/QQQ/DIA) for signal features ───
    market_ind_df = indicators_history_df[
        indicators_history_df["ticker"].isin(UNIVERSE_CFG["indices"])
    ]

    # ── Step 4: TRAIN VOLUME CLASSIFIER ───────────────────────────────────────
    vol_labels = label_volume_patterns(prices_all_df, indicators_history_df)

    if vol_labels.empty:
        logger.error("No volume labels generated — aborting")
        return

    vol_matrix = build_volume_feature_matrix(prices_all_df, indicators_history_df, vol_labels)

    vol_pipeline, vol_metrics = train_volume_classifier(vol_matrix)
    write_model_metrics(
        model_name      = vol_metrics["model_name"],
        train_date      = vol_metrics["train_date"],
        precision       = vol_metrics["precision_score"],
        auc_roc         = vol_metrics["auc_roc_score"],
        n_samples       = vol_metrics["n_samples"],
    )

    # ── Step 5: build vol_clf_score lookup for signal features ────────────────
    # Score every (ticker, date) row in the backfilled history
    vol_scores = {}
    for ticker, group in indicators_history_df.groupby("ticker"):
        px = prices_all_df[prices_all_df["ticker"] == ticker].sort_values("date")
        for date in group["date"]:
            tickers_data = {ticker: px[px["date"] <= date]}
            s = score_volume_signals(tickers_data, date, vol_pipeline)
            vol_scores[(ticker, date)] = s.get(ticker, 0.5)

    # ── Step 6: TRAIN SIGNAL RANKER ───────────────────────────────────────────
    ticker_meta   = read_sector_metadata()
    ticker_to_etf = dict(zip(ticker_meta["ticker"], ticker_meta["sector_etf"])) if not ticker_meta.empty else {}

    scan_hits = build_historical_scan_hits(indicators_history_df, ticker_to_etf)

    if scan_hits.empty:
        logger.warning("No historical scan hits found — Signal Ranker not trained")
    else:
        sig_labels = label_scanner_hits(prices_all_df, indicators_history_df, scan_hits)

        if sig_labels.empty:
            logger.warning("No signal labels generated — Signal Ranker not trained")
        else:
            sig_matrix = build_signal_feature_matrix(
                prices_df     = prices_all_df,
                indicators_df = indicators_history_df,
                sentiment_df  = pd.DataFrame(columns=["ticker","date","put_call_ratio","short_interest_pct"]),
                market_ind_df = market_ind_df,
                labels_df     = sig_labels,
                vol_scores    = vol_scores,
            )

            sig_pipeline, sig_metrics = train_signal_ranker(sig_matrix)
            write_model_metrics(
                model_name = sig_metrics["model_name"],
                train_date = sig_metrics["train_date"],
                precision  = sig_metrics["precision_score"],
                auc_roc    = sig_metrics["auc_roc_score"],
                n_samples  = sig_metrics["n_samples"],
            )

    logger.info("=" * 60)
    logger.info("ML TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training()