"""
run_pipeline_cloud.py
----------------------
Cloud version of run_pipeline.py for GitHub Actions — DAILY candles,
8-year full-history universe.

KEY DIFFERENCES from run_pipeline.py:
1. Uses database_cloud.py (PostgreSQL) instead of database.py (SQLite)
2. Raw daily prices are NOT stored in Postgres — they're written to
   Supabase Storage as Parquet (see data/storage_cloud.py), which is
   what makes keeping full 8-year history affordable on the free tier
3. Genuine incremental fetch in cloud mode (data/fetcher.py tracks
   each ticker's last-fetched date via the fetch_tracker table in
   database_cloud.py) — first run does a full 8-year backfill per
   ticker, every run after only fetches new daily candles
4. No TEST_MODE — always runs full universe
5. Reads SUPABASE_DB_URL from environment variable (GitHub Secret)

FLOW:
  Step 1  → Initialise Supabase tables
  Step 2  → Fetch full US stock universe (NASDAQ FTP)
  Step 3  → Fetch daily OHLCV data (full 8yr backfill or incremental
            per ticker), write snapshot to Storage — NEVER pruned,
            since full regime history is the whole point of this project
  Step 4  → Apply Stage 1 filter
  Step 5  → Fetch sector metadata → write to Supabase
  Step 6  → Run LinReg + SMC + Volume engines (in memory)
  Step 7  → Write indicator_results to Supabase
  Step 8  → Run scanner waterfall
  Step 9  → Score with ML models
  Step 10 → Write scan_results to Supabase
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# ── Use cloud database layer ──────────────────────────────────────────────────
# Import cloud DB functions under the same names the pipeline uses
# This way the rest of the code is identical to run_pipeline.py
from data.database_cloud import (
    initialise_database,
    write_indicator_results,
    write_scan_results,
    write_filtered_universe,
    write_sector_metadata,
    read_latest_indicator_results,
    read_sector_metadata,
)
# NOTE: get_last_fetch_date and prune_old_prices were imported here in
# the 15m project's version of this file, but neither is actually
# called anywhere in this file's body (confirmed dead imports) — and
# database_cloud.py no longer exports get_last_fetch_date at all (it
# was replaced by get_last_fetch_dates_bulk/write_last_fetch_dates,
# used internally by fetcher.py's smart_fetch() for real incremental
# fetching). Importing a name that no longer exists would crash this
# script on load, so both are removed here.

from utils.logging import get_logger
from utils.error_handler import handle_critical_error

logger = get_logger("pipeline_runner", "pipeline_runner.log")



def run_full_pipeline():
    today   = datetime.today().strftime("%Y-%m-%d")
    start   = datetime.now()

    logger.info("=" * 70)
    logger.info("EAGLE LOGIC SYSTEM — CLOUD PIPELINE RUN")
    logger.info(f"Date: {today}")
    logger.info("=" * 70)

    # ── STEP 1: Initialise Supabase ───────────────────────────────────────────
    logger.info("\n[STEP 1] Initialising Supabase...")
    try:
        initialise_database()
        logger.info("✅ Supabase initialised")
    except Exception as e:
        handle_critical_error(e, "Step 1: Supabase init", reraise=True)

    # ── STEP 2: Fetch universe ────────────────────────────────────────────────
    logger.info("\n[STEP 2] Fetching full US stock universe...")
    try:
        from data.fetcher import get_full_universe
        tickers = get_full_universe()
        logger.info(f"✅ Universe: {len(tickers)} tickers")
    except Exception as e:
        handle_critical_error(e, "Step 2: Universe fetch", reraise=True)

    # ── STEP 3: Fetch OHLCV data (in memory) + snapshot to Storage ───────────
    logger.info("\n[STEP 3] Fetching daily OHLCV data (full 8yr backfill or incremental)...")
    try:
        from data.fetcher import smart_fetch
        raw_df = smart_fetch(tickers)

        if raw_df is None or raw_df.empty:
            logger.error("No data fetched — aborting")
            return

        logger.info(f"✅ Fetched {len(raw_df)} rows for {raw_df['ticker'].nunique()} tickers")

        # Snapshot today's fetch to Supabase Storage — this is what makes
        # full 8-year history available to train_models.py later, since
        # Postgres never stores raw prices. Non-fatal if it fails —
        # pipeline continues either way.
        #
        # Deliberately NOT calling prune_old_snapshots() here — this
        # project keeps ALL history for full regime coverage, unlike the
        # 15m project's rolling 60-day window. Pruning would delete
        # exactly the data this project exists to preserve.
        from data.storage_cloud import write_daily_snapshot
        write_daily_snapshot(raw_df, today)
    except Exception as e:
        handle_critical_error(e, "Step 3: Data fetch", reraise=True)

    # ── STEP 4: Stage 1 filter ────────────────────────────────────────────────
    logger.info("\n[STEP 4] Applying Stage 1 filter...")
    try:
        from data.fetcher import apply_stage1_filter
        filtered_df = apply_stage1_filter(raw_df, today)

        if filtered_df.empty:
            logger.error("Stage 1: 0 tickers passed — aborting")
            return

        write_filtered_universe(filtered_df, today)
        passed_tickers = filtered_df["ticker"].tolist()
        logger.info(f"✅ Stage 1: {len(passed_tickers)} tickers passed")
    except Exception as e:
        handle_critical_error(e, "Step 4: Stage 1 filter", reraise=True)

    # ── STEP 5: Sector metadata ───────────────────────────────────────────────
    logger.info("\n[STEP 5] Fetching sector metadata...")
    try:
        from data.fetcher import fetch_sector_metadata
        sector_df = fetch_sector_metadata(passed_tickers)
        write_sector_metadata(sector_df, today)
        logger.info(f"✅ Sector metadata: {len(sector_df)} tickers classified")
    except Exception as e:
        handle_critical_error(e, "Step 5: Sector metadata", reraise=True)

    # ── STEP 6: Load price data for engines (from in-memory raw_df) ──────────
    logger.info("\n[STEP 6] Loading price data for indicator engines...")
    try:
        tickers_data = {}
        for ticker in passed_tickers:
            df = raw_df[raw_df["ticker"] == ticker].sort_values("date").copy()
            if not df.empty:
                tickers_data[ticker] = df

        logger.info(f"✅ Loaded data for {len(tickers_data)} tickers")
    except Exception as e:
        handle_critical_error(e, "Step 6: Load prices", reraise=True)

    # ── STEP 7: Run indicator engines ─────────────────────────────────────────
    logger.info("\n[STEP 7] Running indicator engines...")
    try:
        from engines.linreg import run_linreg_engine
        from engines.smc    import run_smc_engine
        from engines.volume import run_volume_engine

        linreg_df = run_linreg_engine(tickers_data, today)
        logger.info(f"✅ LinReg: {len(linreg_df)} tickers computed")

        smc_df = run_smc_engine(tickers_data, today, linreg_df=linreg_df)
        logger.info(f"✅ SMC: {len(smc_df)} tickers computed")

        volume_df = run_volume_engine(tickers_data, today)
        logger.info(f"✅ Volume: {len(volume_df)} tickers computed")

        # Merge engine outputs
        indicator_df = linreg_df.merge(
            smc_df[["ticker", "date", "smc_structure", "has_valid_zone"]],
            on=["ticker", "date"], how="left"
        ).merge(
            volume_df[["ticker", "date", "volume_signal"]],
            on=["ticker", "date"], how="left"
        )

        indicator_df["smc_structure"]  = indicator_df["smc_structure"].fillna("broken")
        indicator_df["has_valid_zone"] = indicator_df["has_valid_zone"].fillna(0)
        indicator_df["volume_signal"]  = indicator_df["volume_signal"].fillna("neutral")

        write_indicator_results(indicator_df)
        logger.info(f"✅ Indicator results written: {len(indicator_df)} rows")
    except Exception as e:
        handle_critical_error(e, "Step 7: Indicator engines", reraise=True)

    # ── STEP 8: Run scanner waterfall ─────────────────────────────────────────
    logger.info("\n[STEP 8] Running scanner waterfall...")
    try:
        from scanner.screener import run_scanner
        candidates_df = run_scanner(indicator_df, today)

        if candidates_df.empty:
            logger.warning("Scanner: No candidates found today")
        else:
            logger.info(
                f"✅ Scanner: {len(candidates_df)} candidates | "
                f"Longs: {len(candidates_df[candidates_df['direction']=='long'])} | "
                f"Shorts: {len(candidates_df[candidates_df['direction']=='short'])}"
            )
    except Exception as e:
        handle_critical_error(e, "Step 8: Scanner", reraise=True)

    # ── STEP 9: ML scoring ────────────────────────────────────────────────────
    logger.info("\n[STEP 9] Running ML scoring...")
    try:
        if not candidates_df.empty:
            from ml.volume_classifier import load_volume_classifier, score_volume_signals
            from ml.signal_ranker     import load_signal_ranker, score_candidates

            vol_pipeline = load_volume_classifier()
            sig_pipeline = load_signal_ranker()

            if vol_pipeline and sig_pipeline:
                # Score volume patterns for candidates only (matches run_pipeline.py)
                candidate_tickers = {
                    t: tickers_data[t]
                    for t in candidates_df["ticker"].tolist()
                    if t in tickers_data
                }
                vol_scores = score_volume_signals(candidate_tickers, today, vol_pipeline)

                market_ind_df = indicator_df[
                    indicator_df["ticker"].isin(["SPY", "QQQ", "DIA"])
                ]

                candidates_df = score_candidates(
                    candidates_df  = candidates_df,
                    prices_df      = raw_df,
                    indicators_df  = indicator_df,
                    market_ind_df  = market_ind_df,
                    vol_scores     = vol_scores,
                    signal_date    = today,
                    pipeline       = sig_pipeline,
                )
                logger.info("✅ ML scoring complete")
            else:
                logger.warning("ML models not found — using SD position ranking")
                candidates_df["ml_score"] = candidates_df["sd_position"].abs()
                candidates_df["ml_rank"]  = candidates_df["ml_score"].rank(
                    ascending=False
                ).astype(int)
    except Exception as e:
        logger.warning(f"ML scoring failed: {e} — using SD position ranking")
        candidates_df["ml_score"] = candidates_df.get("sd_position", 0)
        candidates_df["ml_rank"]  = 1

    # ── STEP 10: Write results to Supabase ────────────────────────────────────
    logger.info("\n[STEP 10] Writing scan results to Supabase...")
    try:
        if not candidates_df.empty:
            rows = write_scan_results(candidates_df, today)
            logger.info(f"✅ Scan results written: {rows} candidates")
        else:
            logger.info("No candidates to write")
    except Exception as e:
        handle_critical_error(e, "Step 10: Write results", reraise=True)

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start).seconds // 60
    logger.info("\n" + "=" * 70)
    logger.info("CLOUD PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed}m")
    logger.info("=" * 70)

    # Print summary to GitHub Actions console
    if not candidates_df.empty:
        print(f"\n📊 EAGLE LOGIC — {today}")
        print(f"Longs:  {len(candidates_df[candidates_df['direction']=='long'])}")
        print(f"Shorts: {len(candidates_df[candidates_df['direction']=='short'])}")
        top = candidates_df.nlargest(5, "ml_score")[["ticker","direction","ml_score"]]
        print("\nTop 5:")
        print(top.to_string(index=False))
    else:
        print(f"\n📊 No candidates found — {today}")


if __name__ == "__main__":
    run_full_pipeline()
