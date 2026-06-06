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
# SECTOR MAP
# Maps each stock's sector ETF so we can check sector health.
# Source: S&P 500 GICS sector classifications.
# Stocks not in this map default to None (sector check skipped).
# =============================================================================

SECTOR_MAP = {
    # Technology
    "XLK": "XLK",
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "AVGO": "XLK", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "QCOM": "XLK", "TXN": "XLK", "INTC": "XLK", "MU": "XLK",
    "AMAT": "XLK", "LRCX": "XLK", "KLAC": "XLK", "SNPS": "XLK",
    "CDNS": "XLK", "MRVL": "XLK", "FTNT": "XLK", "PANW": "XLK",

    # Communication Services
    "XLC": "XLC",
    "META": "XLC", "GOOGL": "XLC", "GOOG": "XLC", "NFLX": "XLC",
    "DIS": "XLC",  "CMCSA": "XLC", "T": "XLC",    "VZ": "XLC",
    "TMUS": "XLC", "CHTR": "XLC",  "EA": "XLC",   "TTWO": "XLC",

    # Consumer Discretionary
    "XLY": "XLY",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY",   "MCD": "XLY",
    "NKE": "XLY",  "SBUX": "XLY", "TJX": "XLY",  "BKNG": "XLY",
    "LOW": "XLY",  "GM": "XLY",   "F": "XLY",    "EBAY": "XLY",

    # Consumer Staples
    "XLP": "XLP",
    "WMT": "XLP",  "PG": "XLP",   "KO": "XLP",   "PEP": "XLP",
    "COST": "XLP", "PM": "XLP",   "MO": "XLP",   "CL": "XLP",
    "MDLZ": "XLP", "GIS": "XLP",  "KHC": "XLP",  "SYY": "XLP",

    # Healthcare
    "XLV": "XLV",
    "LLY": "XLV",  "UNH": "XLV",  "JNJ": "XLV",  "ABBV": "XLV",
    "MRK": "XLV",  "TMO": "XLV",  "ABT": "XLV",  "DHR": "XLV",
    "PFE": "XLV",  "AMGN": "XLV", "ISRG": "XLV", "GILD": "XLV",
    "VRTX": "XLV", "REGN": "XLV", "CVS": "XLV",  "CI": "XLV",

    # Financials
    "XLF": "XLF",
    "BRK-B": "XLF","JPM": "XLF",  "V": "XLF",    "MA": "XLF",
    "BAC": "XLF",  "WFC": "XLF",  "GS": "XLF",   "MS": "XLF",
    "BLK": "XLF",  "SCHW": "XLF", "AXP": "XLF",  "C": "XLF",
    "CB": "XLF",   "PGR": "XLF",  "MMC": "XLF",  "AON": "XLF",

    # Industrials
    "XLI": "XLI",
    "CAT": "XLI",  "RTX": "XLI",  "HON": "XLI",  "UNP": "XLI",
    "GE": "XLI",   "LMT": "XLI",  "BA": "XLI",   "DE": "XLI",
    "MMM": "XLI",  "UPS": "XLI",  "FDX": "XLI",  "ETN": "XLI",
    "EMR": "XLI",  "PH": "XLI",   "ITW": "XLI",  "GD": "XLI",

    # Energy
    "XLE": "XLE",
    "XOM": "XLE",  "CVX": "XLE",  "COP": "XLE",  "EOG": "XLE",
    "SLB": "XLE",  "MPC": "XLE",  "PSX": "XLE",  "VLO": "XLE",
    "PXD": "XLE",  "OXY": "XLE",  "DVN": "XLE",  "HAL": "XLE",

    # Materials
    "XLB": "XLB",
    "LIN": "XLB",  "APD": "XLB",  "SHW": "XLB",  "FCX": "XLB",
    "NEM": "XLB",  "NUE": "XLB",  "CTVA": "XLB", "DOW": "XLB",
    "DD": "XLB",   "PPG": "XLB",  "ALB": "XLB",  "CF": "XLB",

    # Utilities
    "XLU": "XLU",
    "NEE": "XLU",  "DUK": "XLU",  "SO": "XLU",   "D": "XLU",
    "AEP": "XLU",  "EXC": "XLU",  "SRE": "XLU",  "XEL": "XLU",
    "PEG": "XLU",  "WEC": "XLU",  "ES": "XLU",   "ETR": "XLU",

    # Real Estate
    "XLRE": "XLRE",
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE","CCI": "XLRE",
    "SPG": "XLRE", "O": "XLRE",   "WELL": "XLRE","DLR": "XLRE",
    "PSA": "XLRE", "EQR": "XLRE", "AVB": "XLRE", "WY": "XLRE",
}

# Human-readable sector names for dashboard display
SECTOR_NAMES = {
    "XLK" : "Technology",
    "XLC" : "Communication Services",
    "XLY" : "Consumer Discretionary",
    "XLP" : "Consumer Staples",
    "XLV" : "Healthcare",
    "XLF" : "Financials",
    "XLI" : "Industrials",
    "XLE" : "Energy",
    "XLB" : "Materials",
    "XLU" : "Utilities",
    "XLRE": "Real Estate",
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
        no_choch = int(row["choch_detected"])  == 0

        if slope_up and no_choch:
            result[idx] = "bullish"
        elif not slope_up and no_choch:
            result[idx] = "bearish"
        else:
            result[idx] = "broken"   # CHoCH detected

        logger.info(
            f"Market | {idx}: {result[idx].upper()} | "
            f"Slope up: {slope_up} | CHoCH: {not no_choch}"
        )

    # ── Determine overall market bias ─────────────────────────────────────────
    statuses = list(result.values())

    if all(s == "bullish" for s in statuses):
        market_bias = "bullish"
    elif all(s == "bearish" for s in statuses):
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
        no_choch = int(row["choch_detected"])  == 0

        if slope_up and no_choch:
            result[sector] = "bullish"
        elif not slope_up and no_choch:
            result[sector] = "bearish"
        else:
            result[sector] = "broken"

        logger.info(
            f"Sector | {sector} ({SECTOR_NAMES.get(sector, sector)}): "
            f"{result[sector].upper()}"
        )

    return result


# =============================================================================
# LEVEL 3 — STOCK SETUP CHECK
# =============================================================================

def _is_long_candidate(
    row            : pd.Series,
    sector_health  : dict,
) -> bool:
    """
    Check if a stock qualifies as a LONG candidate.

    ALL conditions must be True:
    1. LinReg sloping UP (stock is in an uptrend)
    2. No CHoCH on the stock (uptrend structure intact)
    3. Price SD position between -1 and -3 (in the buy zone)
    4. Volume signal = 'accumulation' (smart money buying)
    5. Stock's sector is 'bullish' (sector confirms the trade)

    Args:
        row           : Single row from indicator_results for this stock
        sector_health : Dict of sector ETF → health status

    Returns:
        True if all conditions pass, False otherwise
    """
    ticker = row["ticker"]

    # ── Condition 1: Uptrend ──────────────────────────────────────────────────
    if int(row["linreg_slope_up"]) != 1:
        return False

    # ── Condition 2: No CHoCH ─────────────────────────────────────────────────
    if int(row["choch_detected"]) != 0:
        return False

    # ── Condition 3: Price in buy zone (-1 to -3 SD) ─────────────────────────
    sd_pos = float(row["price_sd_position"])
    # sd_pos must be negative (below LinReg) and between -1 and -3
    if not (LONG_SD_MAX <= sd_pos <= LONG_SD_MIN):
        return False

    # ── Condition 4: Accumulation volume ─────────────────────────────────────
    if row["volume_signal"] != "accumulation":
        return False

    # ── Condition 5: Sector health ────────────────────────────────────────────
    sector_etf = SECTOR_MAP.get(ticker)
    if sector_etf:
        if sector_health.get(sector_etf) != "bullish":
            return False

    return True


def _is_short_candidate(
    row           : pd.Series,
    sector_health : dict,
) -> bool:
    """
    Check if a stock qualifies as a SHORT candidate.
    Mirror of _is_long_candidate with reversed conditions.

    ALL conditions must be True:
    1. LinReg sloping DOWN
    2. No CHoCH on the stock
    3. Price SD position between +1 and +3 (in the sell zone)
    4. Volume signal = 'distribution'
    5. Stock's sector is 'bearish'

    Args:
        row           : Single row from indicator_results for this stock
        sector_health : Dict of sector ETF → health status

    Returns:
        True if all conditions pass, False otherwise
    """
    ticker = row["ticker"]

    # ── Condition 1: Downtrend ────────────────────────────────────────────────
    if int(row["linreg_slope_up"]) != 0:
        return False

    # ── Condition 2: No CHoCH ─────────────────────────────────────────────────
    if int(row["choch_detected"]) != 0:
        return False

    # ── Condition 3: Price in sell zone (+1 to +3 SD) ────────────────────────
    sd_pos = float(row["price_sd_position"])
    if not (SHORT_SD_MIN <= sd_pos <= SHORT_SD_MAX):
        return False

    # ── Condition 4: Distribution volume ─────────────────────────────────────
    if row["volume_signal"] != "distribution":
        return False

    # ── Condition 5: Sector health ────────────────────────────────────────────
    sector_etf = SECTOR_MAP.get(ticker)
    if sector_etf:
        if sector_health.get(sector_etf) != "bearish":
            return False

    return True


# =============================================================================
# CANDIDATE BUILDER
# Assembles the final candidate rows with all data attached
# =============================================================================

def _build_candidate_row(
    row           : pd.Series,
    direction     : str,
    sector_health : dict,
    sentiment_df  : pd.DataFrame,
) -> dict:
    """
    Build a complete candidate row ready for the scan_results table.

    Joins indicator data with sentiment data for this ticker.
    ML score is set to 0.0 here — Phase 5 will fill it in.

    Args:
        row           : Indicator results row for this ticker
        direction     : 'long' or 'short'
        sector_health : Sector health dict
        sentiment_df  : Sentiment DataFrame for put/call + short interest

    Returns:
        Dict with all candidate data
    """
    ticker     = row["ticker"]
    sector_etf = SECTOR_MAP.get(ticker)
    sector_name = SECTOR_NAMES.get(sector_etf, "Unknown") if sector_etf else "Unknown"

    # ── Attach sentiment data ─────────────────────────────────────────────────
    sentiment_row = sentiment_df[sentiment_df["ticker"] == ticker]

    if not sentiment_row.empty:
        put_call_ratio     = sentiment_row.iloc[0]["put_call_ratio"]
        short_interest_pct = sentiment_row.iloc[0]["short_interest_pct"]
    else:
        put_call_ratio     = None
        short_interest_pct = None

    return {
        "ticker"            : ticker,
        "direction"         : direction,
        "sector"            : sector_name,
        "sd_position"       : float(row["price_sd_position"]),
        "volume_signal"     : row["volume_signal"],
        "put_call_ratio"    : put_call_ratio,
        "short_interest_pct": short_interest_pct,
        "ml_score"          : 0.0,   # Placeholder — filled by ML in Phase 5
        "ml_rank"           : 0,     # Placeholder — filled after ML scoring
    }


# =============================================================================
# MAIN SCANNER WATERFALL
# =============================================================================

def run_scanner(
    indicator_df : pd.DataFrame,
    sentiment_df : pd.DataFrame,
    date         : str,
) -> pd.DataFrame:
    """
    Run the full top-down scanner waterfall.

    FLOW:
    1. Level 1: Check market health (SPY, QQQ, DIA)
    2. Level 2: Check all sector health
    3. Level 3: For each stock in filtered universe:
       - Skip indices and sector ETFs
       - Check long conditions if market is bullish/mixed
       - Check short conditions if market is bearish/mixed
       - Build candidate row if conditions pass
    4. Combine long and short candidates
    5. Sort by SD position proximity to LinReg
       (price closer to -1 SD = closer to mean reversion = higher priority)
    6. Assign preliminary rank
    7. Return combined DataFrame

    Args:
        indicator_df : DataFrame with all indicator results for today
        sentiment_df : DataFrame with sentiment data for today
        date         : Today's date string YYYY-MM-DD

    Returns:
        DataFrame with all long and short candidates
        Sorted by direction then sd_position
    """
    logger.info("=" * 60)
    logger.info("SCANNER WATERFALL STARTING")
    logger.info("=" * 60)

    # Tickers to exclude from stock scanning
    # (indices and sectors are used for filtering only)
    excluded = set(
        config["universe"]["indices"] +
        config["universe"]["sectors"]
    )

    # ── Level 1: Market health ────────────────────────────────────────────────
    market_health = _check_market_health(indicator_df)
    market_bias   = market_health["market_bias"]

    # Determine which directions to scan based on market bias
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

    # Only scan actual stocks — not indices or sector ETFs
    stock_rows = indicator_df[~indicator_df["ticker"].isin(excluded)]

    logger.info(f"Scanning {len(stock_rows)} stocks...")

    for _, row in stock_rows.iterrows():
        ticker = row["ticker"]

        # ── Check long conditions ─────────────────────────────────────────────
        if scan_long and _is_long_candidate(row, sector_health):
            candidate = _build_candidate_row(
                row, "long", sector_health, sentiment_df
            )
            long_candidates.append(candidate)
            logger.debug(f"LONG candidate: {ticker} | SD: {row['price_sd_position']}")

        # ── Check short conditions ────────────────────────────────────────────
        if scan_short and _is_short_candidate(row, sector_health):
            candidate = _build_candidate_row(
                row, "short", sector_health, sentiment_df
            )
            short_candidates.append(candidate)
            logger.debug(f"SHORT candidate: {ticker} | SD: {row['price_sd_position']}")

    logger.info(
        f"Scanner complete | "
        f"Long candidates: {len(long_candidates)} | "
        f"Short candidates: {len(short_candidates)}"
    )

    # ── Combine and sort ──────────────────────────────────────────────────────
    all_candidates = long_candidates + short_candidates

    if not all_candidates:
        logger.warning("Scanner: No candidates found today")
        return pd.DataFrame()

    df = pd.DataFrame(all_candidates)

    # Sort longs by sd_position descending (closest to -1 first = nearest mean)
    # Sort shorts by sd_position ascending (closest to +1 first = nearest mean)
    longs  = df[df["direction"] == "long"].sort_values(
        "sd_position", ascending=False
    )
    shorts = df[df["direction"] == "short"].sort_values(
        "sd_position", ascending=True
    )

    # Assign preliminary rank within each direction
    # ML Phase will re-rank by ml_score
    longs  = longs.reset_index(drop=True)
    shorts = shorts.reset_index(drop=True)
    longs["ml_rank"]  = longs.index + 1
    shorts["ml_rank"] = shorts.index + 1

    result = pd.concat([longs, shorts], ignore_index=True)

    logger.info(
        f"Final candidates | "
        f"Longs: {len(longs)} | "
        f"Shorts: {len(shorts)} | "
        f"Total: {len(result)}"
    )

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
            "name"  : SECTOR_NAMES.get(etf, etf),
        }
        for etf, status in sector_health.items()
    }

    return {
        "market" : market_health,
        "sectors": sector_health_named,
    }