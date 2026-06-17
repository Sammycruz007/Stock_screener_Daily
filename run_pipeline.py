"""
run_pipeline.py
---------------
Manual end-to-end pipeline runner.
Use this to test the full pipeline outside of Airflow.

FLOW:
1.  Initialise database
2.  Fetch full universe from NASDAQ FTP
3.  Smart fetch OHLCV data
4.  Write raw prices to SQLite
5.  Apply Stage 1 filter
6.  Write filtered universe to SQLite
7.  Fetch sector metadata
8.  Write sector metadata to SQLite
9.  Run LinReg engine
10. Run SMC engine
11. Run Volume engine
12. Write indicator results to SQLite
13. Fetch sentiment data
14. Write sentiment to SQLite
15. Run scanner waterfall
16. Score with Volume Classifier (if model exists)
17. Score with Signal Ranker (if model exists)
18. Write scan results to SQLite
19. Print final results summary

Run with:
    python run_pipeline.py
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.logging import get_logger
from utils.error_handler import handle_critical_error

logger = get_logger("pipeline_runner", "pipeline_runner.log")


def run_full_pipeline():
    """
    Run the complete stock scanner pipeline end to end.
    """
    start_time = datetime.now()
    today      = datetime.today().strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info("STOCK SCANNER PIPELINE — FULL RUN")
    logger.info(f"Date: {today}")
    logger.info("=" * 70)

    # =========================================================================
    # STEP 1: Initialise database
    # =========================================================================
    logger.info("\n[STEP 1/12] Initialising database...")
    try:
        from data.database import initialise_database
        initialise_database()
        logger.info("✅ Database initialised")
    except Exception as e:
        handle_critical_error(e, "Step 1: Database init", reraise=True)

    # =========================================================================
    # STEP 2: Fetch full universe
    # =========================================================================
    logger.info("\n[STEP 2/12] Fetching full US stock universe...")
    try:
        from data.fetcher import get_full_universe
        universe = get_full_universe()
        logger.info(f"✅ Universe: {len(universe)} tickers")
    except Exception as e:
        handle_critical_error(e, "Step 2: Universe fetch", reraise=True)

    # =========================================================================
    # STEP 3 & 4: Smart fetch OHLCV + write to database
    # NOTE: Full first-run fetch of 6,500+ tickers will take 20-40 minutes.
    # For testing we limit to a small subset first.
    # Change TEST_MODE = False to run the full universe.
    # =========================================================================
    TEST_MODE    = False
    TEST_TICKERS = [
        # Indices (always needed)
        "SPY", "QQQ", "DIA",
        # Sector ETFs (always needed)
        "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        # Sample stocks — mix of sectors
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",   # Tech / Comm
        "JPM",  "BAC",  "GS",                        # Financials
        "XOM",  "CVX",                               # Energy
        "JNJ",  "UNH",  "LLY",                       # Healthcare
        "AMZN", "TSLA", "HD",                        # Consumer
        "CAT",  "BA",   "HON",                       # Industrials
        "NEE",  "DUK",                               # Utilities
        "PLD",  "AMT",                               # Real Estate
        "LIN",  "SHW",                               # Materials
        "WMT",  "PG",   "KO",                        # Staples
    ]

    if TEST_MODE:
        logger.info(
            f"\n[STEP 3/12] TEST MODE: Fetching {len(TEST_TICKERS)} tickers "
            f"(set TEST_MODE=False for full universe)"
        )
        fetch_tickers = TEST_TICKERS
    else:
        logger.info(f"\n[STEP 3/12] FULL MODE: Fetching {len(universe)} tickers")
        fetch_tickers = universe

    try:
        from data.fetcher import smart_fetch, apply_stage1_filter
        from data.database import write_raw_prices, write_filtered_universe

        logger.info("Fetching OHLCV data...")
        raw_df = smart_fetch(fetch_tickers)

        if raw_df.empty:
            logger.error("No data fetched — aborting")
            return

        logger.info(f"✅ Fetched {len(raw_df)} rows for {raw_df['ticker'].nunique()} tickers")

        logger.info("\n[STEP 4/12] Writing raw prices to database...")
        rows_written = write_raw_prices(raw_df)
        logger.info(f"✅ {rows_written} rows written to raw_prices")

        logger.info("Pruning old price data...")
        from data.database import prune_old_prices
        deleted = prune_old_prices(max_days=60)
        logger.info(f"✅ Pruned {deleted} old rows")

    except Exception as e:
        handle_critical_error(e, "Step 3/4: Data fetch", reraise=True)

    # =========================================================================
    # STEP 5 & 6: Stage 1 filter + write filtered universe
    # =========================================================================
    logger.info("\n[STEP 5/12] Applying Stage 1 filter...")
    try:
        filtered_df = apply_stage1_filter(raw_df, today)
        write_filtered_universe(filtered_df, today)
        logger.info(
            f"✅ Stage 1 filter | "
            f"Input: {raw_df['ticker'].nunique()} | "
            f"Passed: {len(filtered_df)}"
        )
        passed_tickers = filtered_df["ticker"].tolist()
    except Exception as e:
        handle_critical_error(e, "Step 5/6: Stage 1 filter", reraise=True)

    # =========================================================================
    # STEP 6: Fetch and write sector metadata
    # =========================================================================
    logger.info("\n[STEP 6/12] Fetching sector metadata...")
    try:
        from data.fetcher import fetch_sector_metadata
        from data.database import write_sector_metadata

        sector_df = fetch_sector_metadata(passed_tickers)
        write_sector_metadata(sector_df, today)
        logger.info(f"✅ Sector metadata: {len(sector_df)} tickers classified")
    except Exception as e:
        logger.warning(f"Step 6 sector metadata failed: {e} — continuing")

    # =========================================================================
    # STEP 7: Load OHLCV data from database for engines
    # =========================================================================
    logger.info("\n[STEP 7/12] Loading price data for indicator engines...")
    try:
        from data.database import read_raw_prices

        tickers_data = {}
        for ticker in passed_tickers:
            df = read_raw_prices(ticker)
            if not df.empty:
                tickers_data[ticker] = df

        logger.info(f"✅ Loaded data for {len(tickers_data)} tickers")
    except Exception as e:
        handle_critical_error(e, "Step 7: Load prices", reraise=True)

    # =========================================================================
    # STEP 8: Run all three indicator engines
    # =========================================================================
    logger.info("\n[STEP 8/12] Running indicator engines...")
    try:
        from engines.linreg import run_linreg_engine
        from engines.smc    import run_smc_engine
        from engines.volume import run_volume_engine
        from data.database  import write_indicator_results

        # LinReg engine
        logger.info("Running LinReg engine...")
        linreg_df = run_linreg_engine(tickers_data, today)
        logger.info(f"✅ LinReg: {len(linreg_df)} tickers computed")

        # SMC engine — now uses linreg_df for demand/supply zone SD band checks
        logger.info("Running SMC engine...")
        smc_df = run_smc_engine(tickers_data, today, linreg_df=linreg_df)
        logger.info(f"✅ SMC: {len(smc_df)} tickers computed")

        # Volume engine
        logger.info("Running Volume engine...")
        volume_df = run_volume_engine(tickers_data, today)
        logger.info(f"✅ Volume: {len(volume_df)} tickers computed")

        # Merge all engine outputs into one indicator DataFrame
        logger.info("Merging engine outputs...")

        indicator_df = linreg_df.merge(
            smc_df[["ticker", "date", "smc_structure", "choch_detected", "has_valid_zone"]],
            on=["ticker", "date"], how="left"
        ).merge(
            volume_df[["ticker", "date", "volume_signal"]],
            on=["ticker", "date"], how="left"
        )

        # Fill any missing SMC or volume values with safe defaults
        indicator_df["smc_structure"]  = indicator_df["smc_structure"].fillna("broken")
        indicator_df["choch_detected"] = indicator_df["choch_detected"].fillna(1)
        indicator_df["has_valid_zone"] = indicator_df["has_valid_zone"].fillna(0)
        indicator_df["volume_signal"]  = indicator_df["volume_signal"].fillna("neutral")

        # Write merged results to database
        write_indicator_results(indicator_df)
        logger.info(f"✅ Indicator results written: {len(indicator_df)} rows")

    except Exception as e:
        handle_critical_error(e, "Step 8: Indicator engines", reraise=True)

    # =========================================================================
    # STEP 9: Fetch sentiment data
    # =========================================================================
    logger.info("\n[STEP 9/12] Fetching sentiment data...")
    try:
        from sentiment.sentiment import run_sentiment_pipeline
        from data.database       import write_sentiment_data

        sentiment_df = run_sentiment_pipeline(passed_tickers, today)
        write_sentiment_data(sentiment_df)
        logger.info(f"✅ Sentiment data: {len(sentiment_df)} tickers")
    except Exception as e:
        logger.warning(f"Step 9 sentiment failed: {e} — continuing with empty sentiment")
        sentiment_df = pd.DataFrame(columns=[
            "ticker", "date", "put_call_ratio", "short_interest_pct"
        ])

    # =========================================================================
    # STEP 10: Run scanner waterfall
    # =========================================================================
    logger.info("\n[STEP 10/12] Running scanner waterfall...")
    try:
        from scanner.screener import run_scanner

        candidates_df = run_scanner(indicator_df, sentiment_df, today)

        if candidates_df.empty:
            logger.warning(
                "Scanner returned no candidates today. "
                "This is normal — market conditions may not favour any setups."
            )
        else:
            logger.info(
                f"✅ Scanner complete | "
                f"Total candidates: {len(candidates_df)} | "
                f"Longs: {len(candidates_df[candidates_df['direction']=='long'])} | "
                f"Shorts: {len(candidates_df[candidates_df['direction']=='short'])}"
            )
    except Exception as e:
        handle_critical_error(e, "Step 10: Scanner", reraise=True)

    # =========================================================================
    # STEP 11: ML scoring (if models exist)
    # =========================================================================
    logger.info("\n[STEP 11/12] Running ML scoring...")
    try:
        from ml.volume_classifier import score_volume_signals, load_volume_classifier
        from ml.signal_ranker     import score_candidates,    load_signal_ranker

        vol_pipeline = load_volume_classifier()
        sig_pipeline = load_signal_ranker()

        if vol_pipeline is None or sig_pipeline is None:
            logger.warning(
                "ML models not trained yet — "
                "candidates will use SD position as preliminary ranking. "
                "Run model training first to enable ML scoring."
            )
        else:
            if not candidates_df.empty:
                # Score volume patterns for all candidates
                candidate_tickers = {
                    t: tickers_data[t]
                    for t in candidates_df["ticker"].tolist()
                    if t in tickers_data
                }
                vol_scores = score_volume_signals(
                    candidate_tickers, today, vol_pipeline
                )

                # Get market indicator data for context features
                market_tickers = ["SPY", "QQQ", "DIA"]
                market_ind_df  = indicator_df[
                    indicator_df["ticker"].isin(market_tickers)
                ]

                # Score and rank all candidates
                candidates_df = score_candidates(
                    candidates_df  = candidates_df,
                    prices_df      = raw_df,
                    indicators_df  = indicator_df,
                    sentiment_df   = sentiment_df,
                    market_ind_df  = market_ind_df,
                    vol_scores     = vol_scores,
                    signal_date    = today,
                    pipeline       = sig_pipeline,
                )
                logger.info("✅ ML scoring complete")

    except Exception as e:
        logger.warning(f"Step 11 ML scoring failed: {e} — continuing without ML scores")

    # =========================================================================
    # STEP 12: Write final results to database
    # =========================================================================
    logger.info("\n[STEP 12/12] Writing scan results to database...")
    try:
        from data.database import write_scan_results

        if not candidates_df.empty:
            write_scan_results(candidates_df, today)
            logger.info(f"✅ Scan results written: {len(candidates_df)} candidates")
        else:
            logger.info("No candidates to write")
    except Exception as e:
        handle_critical_error(e, "Step 12: Write results", reraise=True)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = (datetime.now() - start_time).seconds
    minutes = elapsed // 60
    seconds = elapsed % 60

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {minutes}m {seconds}s")
    logger.info("=" * 70)

    if not candidates_df.empty:
        import pandas as pd
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)

        print("\n" + "=" * 70)
        print(f"📊 SCAN RESULTS — {today}")
        print("=" * 70)

        longs  = candidates_df[candidates_df["direction"] == "long"]
        shorts = candidates_df[candidates_df["direction"] == "short"]

        if not longs.empty:
            print(f"\n📗 LONG CANDIDATES ({len(longs)})")
            print("-" * 70)
            print(longs[[
                "ml_rank", "ticker", "sector",
                "sd_position", "volume_signal",
                "put_call_ratio", "short_interest_pct", "ml_score"
            ]].to_string(index=False))

        if not shorts.empty:
            print(f"\n📕 SHORT CANDIDATES ({len(shorts)})")
            print("-" * 70)
            print(shorts[[
                "ml_rank", "ticker", "sector",
                "sd_position", "volume_signal",
                "put_call_ratio", "short_interest_pct", "ml_score"
            ]].to_string(index=False))

        print("\n" + "=" * 70)
    else:
        print("\n📊 No candidates found today.")

    return candidates_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    try:
        results = run_full_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)