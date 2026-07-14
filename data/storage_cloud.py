"""
data/storage_cloud.py
----------------------
FULL raw daily-price history for cloud training, stored as Parquet
snapshots in Supabase Storage (NOT Postgres — keeps Supabase DB rows
free-tier friendly).

WHY THIS EXISTS:
run_pipeline_cloud.py fetches OHLCV and discards it after computing
indicators — Supabase Postgres was never meant to hold raw_prices (see
database_cloud.py docstring). But train_models.py needs years of raw
daily OHLCV history to backfill indicators and give the model exposure
to multiple market regimes. This module holds that history cheaply.

NOT A ROLLING WINDOW — unlike the 15m project's version of this file,
this one accumulates and keeps ALL history by design. The whole point
of the 8-year daily fetch is full regime coverage in training, so
pruning old snapshots here would actively work against that goal.
read_price_history() defaults to reading everything ever written.

INCREMENTAL FETCH CHANGES THE SNAPSHOT SHAPE:
Day 1's snapshot is huge (full 8-year backfill for every ticker, since
nothing is tracked yet). Every day after is tiny (just each ticker's
one new row, via fetcher.py's genuine cloud incremental fetch). Both
shapes are handled by the same chunked-write / read-everything logic
below — no special-casing needed.

BUCKET LAYOUT (chunked — see note below):
    prices/
        2026-07-14_part000.parquet   ← day 1: huge, many chunks
        2026-07-14_part001.parquet
        ...
        2026-07-15_part000.parquet   ← day 2+: tiny, usually one chunk
        ...

WHY CHUNKED, NOT ONE FILE PER DAY:
Supabase Storage free-tier buckets cap individual file uploads at
50MB. Day 1's full-universe, full-history snapshot is far larger than
that as a single Parquet file (this exact 413 "Payload too large"
error is what forced chunking in the 15m project too). So each
snapshot is split into multiple smaller part files (CHUNK_ROWS rows
each) instead of one large file. Reading transparently reassembles
all parts.

USAGE:
    write_daily_snapshot(raw_df, "2026-07-14")  # called from run_pipeline_cloud.py
    df = read_price_history()                    # called from train_models.py — reads ALL history
    # prune_old_snapshots() exists but should NOT be called in this
    # project's normal flow — see docstring on that function.

REQUIRES:
    No extra package — uses plain HTTP calls to Supabase's Storage
    REST API via `requests` (already a pinned dependency), instead of
    the `supabase` client package. This avoids any ambiguity around
    client-library version support for Supabase's newer sb_secret_...
    API key format.

    Two new GitHub Secrets needed: SUPABASE_URL, SUPABASE_KEY
    (SUPABASE_KEY = your sb_secret_... key — NOT the sb_publishable_...
    key, and NOT the same value as SUPABASE_DB_URL, which is the
    separate Postgres connection string).

DEBUGGING "Invalid API key" ERRORS:
    Because this module uses plain HTTP, you can test the exact same
    credentials directly with curl, outside of the pipeline:

        curl "$SUPABASE_URL/storage/v1/bucket" \\
             -H "apikey: $SUPABASE_KEY" \\
             -H "Authorization: Bearer $SUPABASE_KEY"

    If that curl call also returns "Invalid API key", the problem is
    the secret value itself (wrong key copied, extra whitespace, or
    using the publishable key instead of the secret key) — not this
    code. If curl succeeds, re-check the GitHub secret values for
    typos or trailing whitespace.
"""

import os
import io
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from utils.logging import get_database_logger
from utils.error_handler import DatabaseError

logger = get_database_logger()

BUCKET_NAME = "price-history"
PREFIX      = "prices"

# Rows per chunk file. Supabase free-tier caps individual uploads at
# 50MB. 500k rows of OHLCV data compresses (snappy+parquet) to well
# under that with headroom — adjust down if you still see 413s.
CHUNK_ROWS = 500_000


# =============================================================================
# CONFIG / HEADERS
# =============================================================================

def _get_config():
    """
    Read SUPABASE_URL and SUPABASE_KEY from the environment and build
    the base URL + auth headers used by every Storage REST call.

    Returns:
        Tuple of (base_url, headers)
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise DatabaseError(
            "SUPABASE_URL / SUPABASE_KEY environment variables not set. "
            "Add them to GitHub Secrets (Storage needs the project URL + "
            "the sb_secret_... key, separate from SUPABASE_DB_URL)."
        )

    base_url = url.rstrip("/") + "/storage/v1"
    headers  = {
        "apikey"       : key,
        "Authorization": f"Bearer {key}",
    }
    return base_url, headers


def _parse_date_from_filename(name: str) -> Optional[datetime]:
    """
    Extract the snapshot date from a chunk filename, e.g.
    "2026-07-12_part003.parquet" -> datetime(2026, 7, 12)
    """
    if not name.endswith(".parquet"):
        return None

    stem = name.replace(".parquet", "")
    date_str = stem.split("_part")[0] if "_part" in stem else stem

    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


# =============================================================================
# WRITE — called daily from run_pipeline_cloud.py after OHLCV fetch
# =============================================================================

def write_daily_snapshot(df: pd.DataFrame, date: str) -> bool:
    """
    Write today's raw OHLCV DataFrame to Supabase Storage as chunked
    Parquet files (each under the 50MB per-file Storage limit).

    Args:
        df  : Raw OHLCV DataFrame (ticker, date, open, high, low, close, volume)
        date: Today's date string YYYY-MM-DD — used in the filenames

    Returns:
        True if ALL chunks were written successfully, False otherwise
        (non-fatal — pipeline should continue even if this fails)
    """
    if df.empty:
        logger.warning("write_daily_snapshot: empty DataFrame, skipping")
        return False

    try:
        base_url, headers = _get_config()

        n_chunks = max(1, (len(df) + CHUNK_ROWS - 1) // CHUNK_ROWS)
        upload_headers = {
            **headers,
            "Content-Type": "application/octet-stream",
            "x-upsert"    : "true",
        }

        total_bytes    = 0
        chunks_written = 0

        for i in range(n_chunks):
            chunk = df.iloc[i * CHUNK_ROWS : (i + 1) * CHUNK_ROWS]
            if chunk.empty:
                continue

            buffer = io.BytesIO()
            chunk.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
            buffer.seek(0)
            raw_bytes = buffer.read()

            path = f"{PREFIX}/{date}_part{i:03d}.parquet"

            resp = requests.post(
                f"{base_url}/object/{BUCKET_NAME}/{path}",
                headers=upload_headers,
                data=raw_bytes,
                timeout=60,
            )

            if resp.status_code not in (200, 201):
                logger.warning(
                    f"write_daily_snapshot: chunk {i} failed | "
                    f"HTTP {resp.status_code} — {resp.text[:300]}"
                )
                continue

            total_bytes    += len(raw_bytes)
            chunks_written  += 1

        if chunks_written == 0:
            logger.warning("write_daily_snapshot: no chunks written successfully")
            return False

        size_mb = total_bytes / (1024 * 1024)
        logger.info(
            f"write_daily_snapshot: {len(df)} rows written across "
            f"{chunks_written}/{n_chunks} chunks for {date} ({size_mb:.1f} MB total)"
        )
        return chunks_written == n_chunks

    except Exception as e:
        logger.warning(f"write_daily_snapshot failed: {e} — continuing without it")
        return False


# =============================================================================
# READ — called from train_models.py to rebuild history for backfilling
# =============================================================================

def _list_snapshot_files(base_url: str, headers: dict) -> list[dict]:
    """
    List all files under the prices/ prefix in the bucket via the
    Storage REST API's list endpoint.
    """
    resp = requests.post(
        f"{base_url}/object/list/{BUCKET_NAME}",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "prefix": PREFIX,
            "limit": 1000,
            "sortBy": {"column": "name", "order": "asc"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def read_price_history(days: Optional[int] = None) -> pd.DataFrame:
    """
    Download and concatenate ALL accumulated Parquet snapshot chunks by
    default — unlike the 15m project, this one is NOT a rolling window.

    WHY NO DEFAULT WINDOW: the incremental fetcher means day 1's snapshot
    is huge (full 8-year backfill for every ticker, since nothing is
    tracked yet) and every day after is tiny (just each ticker's new
    row). A rolling "last N days" read would silently lose almost the
    entire dataset once day 1 aged out of the window. Reading everything
    and relying on drop_duplicates(subset=["ticker","date"]) to merge
    correctly is the simplest fix at this data scale (~200MB total).

    Args:
        days: Optional — if given, restricts to snapshots from the last
              N days only (rolling window). Leave as None (default) to
              read the full accumulated history, which is what training
              needs for full regime coverage.

    Returns:
        Combined DataFrame across all available snapshot chunks.
        Empty DataFrame if none found or on failure.
    """
    try:
        base_url, headers = _get_config()

        files = _list_snapshot_files(base_url, headers)
        if not files:
            logger.warning("read_price_history: no snapshots found in bucket")
            return pd.DataFrame()

        cutoff = (datetime.today() - timedelta(days=days)) if days is not None else None

        frames      = []
        dates_seen  = set()

        for f in files:
            name = f["name"]  # e.g. "2026-07-12_part003.parquet"
            file_date = _parse_date_from_filename(name)
            if file_date is None:
                continue
            if cutoff is not None and file_date < cutoff:
                continue

            dl_resp = requests.get(
                f"{base_url}/object/{BUCKET_NAME}/{PREFIX}/{name}",
                headers=headers,
                timeout=60,
            )
            dl_resp.raise_for_status()

            df = pd.read_parquet(io.BytesIO(dl_resp.content), engine="pyarrow")
            frames.append(df)
            dates_seen.add(file_date.strftime("%Y-%m-%d"))

        if not frames:
            window_desc = f"within last {days} days" if days is not None else "at all"
            logger.warning(f"read_price_history: no snapshots found {window_desc}")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"])

        logger.info(
            f"read_price_history: {len(combined)} rows loaded from "
            f"{len(frames)} chunks across {len(dates_seen)} snapshot dates | "
            f"Tickers: {combined['ticker'].nunique()}"
        )
        return combined

    except Exception as e:
        logger.error(f"read_price_history failed: {e}")
        return pd.DataFrame()


def read_raw_prices_cloud(ticker: str, days: Optional[int] = None) -> pd.DataFrame:
    """
    Drop-in replacement for database.py's read_raw_prices(ticker),
    but backed by Parquet snapshots instead of SQLite.

    NOTE: Less efficient than a real per-ticker query — this reads
    the full rolling window then filters. For a one-off call this is
    fine; train_models.py should prefer read_price_history() once and
    filter in memory across the whole ticker loop instead of calling
    this per ticker.

    Args:
        ticker: Ticker symbol
        days  : Rolling window size

    Returns:
        DataFrame for this ticker only, sorted by date
    """
    history = read_price_history(days=days)
    if history.empty:
        return pd.DataFrame()

    df = history[history["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    return df


# =============================================================================
# PRUNE — called daily from run_pipeline_cloud.py to keep the bucket small
# =============================================================================

def prune_old_snapshots(max_days: int = 60) -> int:
    """
    Delete Parquet snapshot chunks older than max_days from Supabase Storage.

    ⚠️ DO NOT call this in this project's normal daily pipeline flow.
    Unlike the 15m project (which only needed a rolling 60-day window
    and pruned aggressively), this project's entire purpose is
    preserving FULL 8-year history for regime coverage in training.
    Calling this would delete exactly the data you're trying to keep.

    Kept here only in case retention policy changes deliberately in
    the future (e.g. capping to a fixed N-year rolling window once
    the model is mature) — not wired into run_pipeline_cloud.py.

    Args:
        max_days: Keep snapshots within this many days, delete the rest

    Returns:
        Number of chunk files deleted
    """
    try:
        base_url, headers = _get_config()

        files = _list_snapshot_files(base_url, headers)
        if not files:
            return 0

        cutoff    = datetime.today() - timedelta(days=max_days)
        to_delete = []

        for f in files:
            name = f["name"]
            file_date = _parse_date_from_filename(name)
            if file_date is None:
                continue

            if file_date < cutoff:
                to_delete.append(f"{PREFIX}/{name}")

        if not to_delete:
            logger.info("prune_old_snapshots: nothing to prune")
            return 0

        resp = requests.delete(
            f"{base_url}/object/{BUCKET_NAME}",
            headers={**headers, "Content-Type": "application/json"},
            json={"prefixes": to_delete},
            timeout=30,
        )
        resp.raise_for_status()

        logger.info(f"prune_old_snapshots: deleted {len(to_delete)} old snapshot chunks")
        return len(to_delete)

    except Exception as e:
        logger.warning(f"prune_old_snapshots failed: {e} — continuing")
        return 0
