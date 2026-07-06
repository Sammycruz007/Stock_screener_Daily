"""
scanner/screener.py
-------------------
Top-down waterfall scanner for the Stock Scanner pipeline.

LOGICAL FLOW:
─────────────
This is the brain of the scanner. It takes all the indicator engine
outputs and applies our trading rules in a strict top-down hierarchy.

THE WATERFALL:

LEVEL 1 — MARKET HEALTH CHECK:
   We check SPY, QQQ and DIA simultaneously.
   For LONG bias:
   - All three must have LinReg sloping UP
   - None of the three can have a CHoCH detected
   - If ANY index fails → no long setups surfaced today
   For SHORT bias:
   - All three must have LinReg sloping DOWN
   - None can have a CHoCH detected
   - If ANY index fails → no short setups surfaced today
   MIXED market (some up, some down) → surface both with a warning flag

LEVEL 2 — SECTOR HEALTH CHECK:
   Each stock belongs to a sector (via the SECTOR_MAP).
   The stock's sector ETF must pass the same check as the market:
   - For long: sector LinReg sloping UP + no CHoCH
   - For short: sector LinReg sloping DOWN + no CHoCH
   If the sector fails → that stock is filtered out regardless of
   how good its individual setup looks.

LEVEL 3 — STOCK SETUP CHECK:
   For LONG candidates:
   - LinReg sloping UP
   - No CHoCH on the stock itself
   - Price SD position between -1 and -3 (in the buy zone)
   - Volume signal = 'accumulation'

   For SHORT candidates:
   - LinReg sloping DOWN
   - No CHoCH on the stock itself
   - Price SD position between +1 and +3 (in the sell zone)
   - Volume signal = 'distribution'

FINAL OUTPUT:
   A ranked list of long and short candidates with all their
   indicator values and sentiment data attached.
   ML scoring is applied in Phase 5 — here we just filter and tag.
   Stocks without ML scores yet get ml_score = 0.0 as placeholder.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import yaml

from utils.logging import get_scanner_logger
from utils.error_handler import ScannerError, handle_critical_error

logger = get_scanner_logger()


# =============================================================================
# CONFIG
# =============================================================================

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config      = _load_config()
SCANNER_CFG = config["scanner"]
LINREG_CFG  = config["linreg"]

# Entry zone boundaries
LONG_SD_MIN  = SCANNER_CFG["long_entry_sd_min"]    # -1 (upper boundary)
LONG_SD_MAX  = SCANNER_CFG["long_entry_sd_max"]    # -3 (lower boundary)
SHORT_SD_MIN = SCANNER_CFG["short_entry_sd_min"]   # +1 (lower boundary)
SHORT_SD_MAX = SCANNER_CFG["short_entry_sd_max"]   # +3 (upper boundary)




# =============================================================================
# SECTOR LOOKUP — dynamic from database
# Replaces the old hardcoded SECTOR_MAP.
# Loaded once per scanner run from the ticker_metadata SQLite table.
# =============================================================================

def _load_sector_lookup() -> tuple[dict, dict]:
    """
    Load sector mappings from the ticker_metadata table in SQLite.

    FLOW:
    1. Read ticker_metadata table
    2. Build two lookup dicts:
       - ticker_to_etf  : {"AAPL": "XLK", "JPM": "XLF", ...}
       - ticker_to_name : {"AAPL": "Technology", "JPM": "Financials", ...}
    3. Unclassified tickers have etf = None, name = "Unclassified"

    Returns:
        Tuple of (ticker_to_etf dict, ticker_to_name dict)
    """
    from data.database import read_sector_metadata

    df = read_sector_metadata()

    if df.empty:
        logger.warning(
            "Sector metadata table is empty — "
            "run data pipeline first to populate it"
        )
        return {}, {}

    ticker_to_etf  = {}
    ticker_to_name = {}

    for _, row in df.iterrows():
        ticker = row["ticker"]
        ticker_to_etf[ticker]  = row["sector_etf"]    # May be None
        ticker_to_name[ticker] = row["sector_name"]   # May be 'Unclassified'

    classified   = sum(1 for v in ticker_to_etf.values() if v is not None)
    unclassified = sum(1 for v in ticker_to_etf.values() if v is None)

    logger.info(
        f"Sector lookup loaded | "
        f"Classified: {classified} | "
        f"Unclassified: {unclassified}"
    )

    return ticker_to_etf, ticker_to_name

# =============================================================================
# SECTOR ETF DISPLAY NAMES
# Fixed set of 11 — not the per-stock map removed earlier, just display labels
# for the sector grid on the dashboard.
# =============================================================================

SECTOR_ETF_NAMES = {
    "XLK" : "Technology",
    "XLF" : "Financials",
    "XLE" : "Energy",
    "XLV" : "Healthcare",
    "XLI" : "Industrials",
    "XLY" : "Consumer Discretionary",
    "XLP" : "Consumer Staples",
    "XLU" : "Utilities",
    "XLB" : "Materials",
    "XLRE": "Real Estate",
    "XLC" : "Communication Services",
}


# =============================================================================
# LEVEL 1 — MARKET HEALTH CHECK
# =============================================================================

def _check_market_health(indicator_df: pd.DataFrame) -> dict:
    """
    Check health of the three market indices: SPY, QQQ, DIA.

    LOGIC:
    For each index, we check:
    - linreg_slope_up == 1 → uptrend
    - choch_detected  == 0 → no structure break

    An index is:
    - 'bullish' : slope up AND no CHoCH
    - 'bearish' : slope down AND no CHoCH
    - 'broken'  : CHoCH detected regardless of slope

    Overall market bias:
    - All three bullish → market = 'bullish' → surface long setups
    - All three bearish → market = 'bearish' → surface short setups
    - Mixed            → market = 'mixed'   → surface both with warning

    Args:
        indicator_df: DataFrame from indicator_results table
                      containing all tickers including SPY, QQQ, DIA

    Returns:
        Dict with status per index and overall market bias
    """
    indices = ["SPY", "QQQ", "DIA"]
    result  = {}

    for idx in indices:
        row = indicator_df[indicator_df["ticker"] == idx]

        if row.empty:
            logger.warning(f"Market check: {idx} not found in indicator results")
            result[idx] = "unknown"
            continue

        row = row.iloc[0]

        slope_up = int(row["linreg_slope_up"]) == 1
        #no_choch = int(row["choch_detected"])  == 0

        if slope_up:
            result[idx] = "bullish"
        else:
            result[idx] = "bearish"
        
        logger.info(
            f"Market | {idx}: {result[idx].upper()} | "
            f"Slope up: {slope_up} |"
        )

    # ── Determine overall market bias ─────────────────────────────────────────
    # Exclude "unknown" — missing indices don't override found ones
    known_statuses = [s for s in result.values() if s != "unknown"]

    if not known_statuses:
        market_bias = "mixed"
    elif all(s == "bullish" for s in known_statuses):
        market_bias = "bullish"
    elif all(s == "bearish" for s in known_statuses):
        market_bias = "bearish"
    else:
        market_bias = "mixed"

    result["market_bias"] = market_bias
    logger.info(f"Overall market bias: {market_bias.upper()}")

    return result


# =============================================================================
# LEVEL 2 — SECTOR HEALTH CHECK
# =============================================================================

def _check_sector_health(indicator_df: pd.DataFrame) -> dict:
    """
    Check health of all 11 sector ETFs.

    Same logic as market health but applied to each sector ETF.
    Result is used to:
    1. Power the sector health grid on the dashboard
    2. Filter individual stocks — stock's sector must be healthy

    Args:
        indicator_df: DataFrame from indicator_results table

    Returns:
        Dict mapping sector ETF → status ('bullish', 'bearish', 'broken')
        e.g. {"XLK": "bullish", "XLE": "bearish", "XLF": "broken", ...}
    """
    sectors = config["universe"]["sectors"]
    result  = {}

    for sector in sectors:
        row = indicator_df[indicator_df["ticker"] == sector]

        if row.empty:
            logger.warning(f"Sector check: {sector} not found in indicator results")
            result[sector] = "unknown"
            continue

        row = row.iloc[0]

        slope_up = int(row["linreg_slope_up"]) == 1
        #no_choch = int(row["choch_detected"])  == 0

        if slope_up:
            result[sector] = "bullish"
        else:
            result[sector] = "bearish"
        
        logger.info(
            f"Sector | {sector}: {result[sector].upper()}"
        )
        

    return result


# =============================================================================
# LEVEL 3 — STOCK SETUP CHECK
# =============================================================================

def _is_long_candidate(
    row            : pd.Series,
    sector_health  : dict,
    ticker_to_etf  : dict,          # ← new parameter
) -> bool:
    """
    Check if a stock qualifies as a LONG candidate.
    Now uses dynamic sector lookup from database instead of hardcoded map.
    """
    ticker = row["ticker"]

    # ── Condition 1: Uptrend ──────────────────────────────────────────────────
    if int(row["linreg_slope_up"]) != 1:
        return False

    # ── Condition 2: No CHoCH ─────────────────────────────────────────────────
   # if int(row["choch_detected"]) != 0:
    #    return False

    # ── Condition 3: Price in buy zone (-1 to -3 SD) ─────────────────────────
    sd_pos = float(row["price_sd_position"])
    if not (LONG_SD_MAX <= sd_pos <= LONG_SD_MIN):
        return False

    # ── Condition 4: Accumulation volume ─────────────────────────────────────
    #if row["volume_signal"] != "accumulation":
    #    return False

    # ── Condition 5: Sector health (dynamic lookup) ───────────────────────────
    sector_etf = ticker_to_etf.get(ticker)

    # ── Condition 6: Valid demand zone (BOS + engulfing in SD band) ───────────
    #if int(row.get("has_valid_zone", 0)) != 1:
    #    return False

    if sector_etf is not None:
        # Sector found — apply the health check
        if sector_health.get(sector_etf) != "bullish":
            return False
    else:
        # Sector is Unclassified — stock passes but will be flagged on dashboard
        logger.debug(f"{ticker} | Unclassified sector — passing without sector check")

    return True


def _is_short_candidate(
    row           : pd.Series,
    sector_health : dict,
    ticker_to_etf : dict,           # ← new parameter
) -> bool:
    """
    Check if a stock qualifies as a SHORT candidate.
    Mirror of _is_long_candidate with reversed conditions.
    """
    ticker = row["ticker"]

    if int(row["linreg_slope_up"]) != 0:
        return False

    #if int(row["choch_detected"]) != 0:
    #    return False

    sd_pos = float(row["price_sd_position"])
    if not (SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX):
        return False

    #if row["volume_signal"] != "distribution":
    #    return False

    sector_etf = ticker_to_etf.get(ticker)

    # ── Condition 6: Valid supply zone (BOS + engulfing in SD band) ───────────
    #if int(row.get("has_valid_zone", 0)) != 1:
    #    return False

    if sector_etf is not None:
        if sector_health.get(sector_etf) != "bearish":
            return False
    else:
        logger.debug(f"{ticker} | Unclassified sector — passing without sector check")

    return True


# =============================================================================
# CANDIDATE BUILDER
# Assembles the final candidate rows with all data attached
# =============================================================================

def _build_candidate_row(
    row            : pd.Series,
    direction      : str,
    sector_health  : dict,
    ticker_to_etf  : dict,
    ticker_to_name : dict,
) -> dict:
    """
    Build a complete candidate row.
    Now uses dynamic sector lookup from database.
    Unclassified stocks are flagged with ⚠️ in sector name.
    """
    ticker      = row["ticker"]
    sector_etf  = ticker_to_etf.get(ticker)
    sector_name = ticker_to_name.get(ticker, "Unclassified")

    # Flag unclassified stocks for dashboard warning
    if sector_etf is None:
        sector_name = f"⚠️ {sector_name}"

    

    return {
        "ticker"            : ticker,
        "direction"         : direction,
        "sector"            : sector_name,
        "sd_position"       : float(row["price_sd_position"]),
        "volume_signal"     : row["volume_signal"],
        "has_valid_zone"    : int(row.get("has_valid_zone", 0)),
        "ml_score"          : 0.0,
        "ml_rank"           : 0,
    }

# =============================================================================
# MAIN SCANNER WATERFALL
# =============================================================================

def run_scanner(
    indicator_df : pd.DataFrame,
    date         : str,
) -> pd.DataFrame:
    """
    Run the full top-down scanner waterfall.
    Now loads sector lookup dynamically from database.
    """
    logger.info("=" * 60)
    logger.info("SCANNER WATERFALL STARTING")
    logger.info("=" * 60)

    excluded = set(
        config["universe"]["indices"] +
        config["universe"]["sectors"]
    )

    # ── Load sector lookup from database ─────────────────────────────────────
    ticker_to_etf, ticker_to_name = _load_sector_lookup()

    # ── Level 1: Market health ────────────────────────────────────────────────
    market_health = _check_market_health(indicator_df)
    market_bias   = market_health["market_bias"]

    scan_long  = market_bias in ["bullish", "mixed"]
    scan_short = market_bias in ["bearish", "mixed"]

    logger.info(
        f"Market bias: {market_bias.upper()} | "
        f"Scan longs: {scan_long} | Scan shorts: {scan_short}"
    )

    # ── Level 2: Sector health ────────────────────────────────────────────────
    sector_health = _check_sector_health(indicator_df)

    # ── Level 3: Stock scan ───────────────────────────────────────────────────
    long_candidates  = []
    short_candidates = []

    stock_rows = indicator_df[~indicator_df["ticker"].isin(excluded)]
    logger.info(f"Scanning {len(stock_rows)} stocks...")

    for _, row in stock_rows.iterrows():
        ticker = row["ticker"]

        if scan_long and _is_long_candidate(row, sector_health, ticker_to_etf):
            candidate = _build_candidate_row(
                row, "long", sector_health, ticker_to_etf, ticker_to_name
            )
            long_candidates.append(candidate)

        if scan_short and _is_short_candidate(row, sector_health, ticker_to_etf):
            candidate = _build_candidate_row(
                row, "short", sector_health,
                         ticker_to_etf, ticker_to_name
            )
            short_candidates.append(candidate)

    logger.info(
        f"Scanner complete | "
        f"Long: {len(long_candidates)} | "
        f"Short: {len(short_candidates)}"
    )

    all_candidates = long_candidates + short_candidates

    if not all_candidates:
        logger.warning("Scanner: No candidates found today")
        return pd.DataFrame()

    df = pd.DataFrame(all_candidates)

    longs  = df[df["direction"] == "long"].sort_values("sd_position", ascending=False)
    shorts = df[df["direction"] == "short"].sort_values("sd_position", ascending=True)

    longs  = longs.reset_index(drop=True)
    shorts = shorts.reset_index(drop=True)
    longs["ml_rank"]  = longs.index + 1
    shorts["ml_rank"] = shorts.index + 1

    result = pd.concat([longs, shorts], ignore_index=True)

    logger.info(f"Final candidates | Total: {len(result)}")
    return result


# =============================================================================
# MARKET + SECTOR STATUS EXPORT
# Used by the Streamlit dashboard to populate:
# - Section 1: Market Pulse cards
# - Section 2: Sector health grid
# =============================================================================

def get_market_sector_status(indicator_df: pd.DataFrame) -> dict:
    """
    Export market and sector health for the Streamlit dashboard.

    Called by the dashboard directly — not part of the Airflow pipeline.
    The dashboard reads indicator_results from SQLite and calls this
    to get the color-coded status for each index and sector.

    Args:
        indicator_df: Today's indicator results from SQLite

    Returns:
        Dict with:
        - 'market'  : {SPY: status, QQQ: status, DIA: status, market_bias: bias}
        - 'sectors' : {XLK: status, XLF: status, ..., sector_name: name}
    """
    market_health = _check_market_health(indicator_df)
    sector_health = _check_sector_health(indicator_df)

    # Add human-readable names to sector health
    sector_health_named = {
        etf: {
            "status": status,
            "name"  : SECTOR_ETF_NAMES.get(etf, etf),
        }
        for etf, status in sector_health.items()
    }

    return {
        "market" : market_health,
        "sectors": sector_health_named,
    }