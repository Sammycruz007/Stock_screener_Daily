"""
engines/relative_strength.py
-----------------------------
Relative Strength (RS) engine — measures how a ticker has performed
relative to a benchmark (SPY) over a lookback window. NOT the same
thing as RSI (Relative Strength Index) — this is a cross-sectional
outperformance/underperformance measure against the broader market.

LOGICAL FLOW:
─────────────
RS is NOT just (ticker_price / SPY_price) — that raw ratio isn't
comparable across tickers, since it's dominated by each side's
absolute price level rather than actual relative performance.

Instead, RS measures the CHANGE in that ratio over the lookback
window — mathematically equivalent to comparing each side's return:

    RS = (ticker_price_today / ticker_price_N_days_ago)
         / (SPY_price_today / SPY_price_N_days_ago)
         - 1

    Positive RS → ticker outperformed SPY over the window
    Negative RS → ticker underperformed SPY over the window
    RS near 0   → ticker moved roughly in line with SPY

This is comparable across every ticker regardless of price level,
which is what makes it usable as an ML feature.

OUTPUT per ticker:
   - relative_strength : outperformance vs SPY over LOOKBACK_PERIOD

WHY THIS MATTERS FOR THE MODEL:
   A stock breaking out of a LinReg band while also strongly
   outperforming the market is a meaningfully different setup than
   one doing the same while lagging the market — this feature lets
   the model learn that distinction rather than treating all setups
   the same regardless of market-relative context.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_relative_strength_logger
from utils.error_handler import graceful, EngineError

logger = get_relative_strength_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config           = _load_config()
RS_CFG           = config["relative_strength"]

BENCHMARK        = RS_CFG["benchmark"]         # "SPY"
LOOKBACK_PERIOD  = RS_CFG["lookback_period"]   # 63 trading days (~1 quarter)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

@graceful(default_return=None, exceptions=(Exception,), log_level="warning")
def compute_relative_strength_latest(
    ticker            : str,
    df                : pd.DataFrame,
    benchmark_df      : pd.DataFrame,
    date              : str,
) -> Optional[dict]:
    """
    Compute this ticker's relative strength vs the benchmark (SPY) as
    of the given date, over LOOKBACK_PERIOD trading days.

    Matches compute_linreg_latest / compute_adx_latest's calling
    convention, with one addition: needs the benchmark's own price
    series passed in alongside the ticker's.

    Args:
        ticker      : Ticker symbol e.g. 'AAPL'
        df          : Full OHLCV DataFrame for this ticker, sorted date
                      ascending, filtered to <= the target date by the caller
        benchmark_df: Full OHLCV DataFrame for the benchmark (SPY),
                      same filtering — sorted date ascending, <= target date
        date        : Signal date string YYYY-MM-DD (for database keying)

    Returns:
        Dict with ticker, date, relative_strength —
        or None if insufficient data on either side
    """
    if len(df) < LOOKBACK_PERIOD + 1:
        logger.warning(
            f"{ticker} | Only {len(df)} rows — need {LOOKBACK_PERIOD + 1}+ "
            f"for {LOOKBACK_PERIOD}-day relative strength"
        )
        return None

    if len(benchmark_df) < LOOKBACK_PERIOD + 1:
        logger.warning(
            f"{BENCHMARK} | Only {len(benchmark_df)} rows — need "
            f"{LOOKBACK_PERIOD + 1}+ for relative strength calc"
        )
        return None

    ticker_now  = float(df["close"].iloc[-1])
    ticker_then = float(df["close"].iloc[-(LOOKBACK_PERIOD + 1)])

    bench_now  = float(benchmark_df["close"].iloc[-1])
    bench_then = float(benchmark_df["close"].iloc[-(LOOKBACK_PERIOD + 1)])

    if ticker_then == 0 or bench_then == 0 or bench_now == 0:
        return None

    ticker_return = ticker_now / ticker_then
    bench_return  = bench_now  / bench_then

    relative_strength = (ticker_return / bench_return) - 1

    result = {
        "ticker"            : ticker,
        "date"              : date,
        "relative_strength" : round(relative_strength, 6),
    }

    logger.debug(
        f"{ticker} | RS vs {BENCHMARK}: {result['relative_strength']:+.4f} "
        f"over {LOOKBACK_PERIOD}d"
    )

    return result
