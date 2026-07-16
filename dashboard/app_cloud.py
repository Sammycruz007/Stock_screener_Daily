"""
dashboard/app_cloud.py
-----------------------
Streamlit Cloud version of dashboard/app.py.

DIFFERENCES from app.py:
- Reads from Supabase (PostgreSQL) instead of SQLite
- Gets DB credentials from st.secrets (Streamlit Cloud secrets)
- Sets SUPABASE_DB_URL env var so database_cloud.py can connect
- Entry point for Streamlit Cloud deployment

DEPLOYMENT:
- Streamlit Cloud looks for a file called streamlit_app.py at root
- We create streamlit_app.py that simply imports this file
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# =============================================================================
# PAGE CONFIG — must be the FIRST Streamlit command in the script, before
# any other st.* call (including st.error/st.stop below). Calling it later
# throws StreamlitAPIException and masks the real secrets error underneath.
# =============================================================================

SYSTEM_NAME = "Eagle Logic System"
EAGLE_ICON  = "🦅"

st.set_page_config(
    page_title = SYSTEM_NAME,
    page_icon  = EAGLE_ICON,
    layout     = "wide",
)

# ── Set Supabase credentials from Streamlit secrets ───────────────────────────
# Streamlit Cloud reads from .streamlit/secrets.toml
# GitHub Actions reads from environment variable directly
if "SUPABASE_DB_URL" in st.secrets:
    os.environ["SUPABASE_DB_URL"] = st.secrets["SUPABASE_DB_URL"]
elif "SUPABASE_DB_URL" not in os.environ:
    st.error(
        "SUPABASE_DB_URL not found in secrets or environment. "
        "Add it to Streamlit Cloud secrets or .streamlit/secrets.toml"
    )
    st.stop()

# ── Import cloud DB functions ─────────────────────────────────────────────────
from data.database_cloud import (
    initialise_database,
    read_latest_indicator_results,
    read_latest_scan_results,
    read_latest_model_metrics,
)
from scanner.screener  import get_market_sector_status
# NOTE: compute_linreg_series was imported here but never used anywhere in
# this file — removed. An unused import of a function that doesn't exist
# (or was renamed) would crash the whole app on load with an ImportError
# before anything renders.

# ── Initialise DB tables (safe — IF NOT EXISTS) ───────────────────────────────
initialise_database()


# =============================================================================
# HEADER
# =============================================================================

header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    st.markdown(
        f"<div style='font-size:64px;line-height:1;'>{EAGLE_ICON}</div>",
        unsafe_allow_html=True,
    )
with header_col2:
    st.markdown(
        f"<h1 style='margin-bottom:0;'>{SYSTEM_NAME}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#9ca3af;margin-top:0;'>"
        "LinReg Channel + Smart Money Concepts Scanner — Daily"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Load indicator data ───────────────────────────────────────────────────────
indicator_df = read_latest_indicator_results()

if indicator_df.empty:
    st.warning("No scan data yet. Pipeline has not run or Supabase is empty.")
    st.stop()

latest_date = indicator_df["date"].max()
st.caption(
    f"Last scan: {latest_date} at 06:00 WAT — "
    f"prices may have moved since scan time"
)

status  = get_market_sector_status(indicator_df)
market  = status["market"]
sectors = status["sectors"]

BADGE      = {"bullish": "🟢", "bearish": "🔴", "broken": "🟡", "unknown": "⚪"}
TILE_COLOR = {
    "bullish": "#dcfce7",
    "bearish": "#fee2e2",
    "broken" : "#fef9c3",
    "unknown": "#f3f4f6",
}


# =============================================================================
# SECTION 1 — MARKET PULSE
# =============================================================================

st.header("Market Pulse")

market_cols = st.columns(3)
for i, idx in enumerate(["SPY", "QQQ", "DIA"]):
    with market_cols[i]:
        st_status = market.get(idx, "unknown")
        st.subheader(f"{idx}  {BADGE.get(st_status,'⚪')} {st_status.upper()}")

        idx_row = indicator_df[indicator_df["ticker"] == idx]
        if not idx_row.empty:
            row = idx_row.iloc[0]
            slope    = float(row.get("linreg_slope", 0))
            sd_pos   = float(row.get("price_sd_position", 0))
            st.caption(
                f"Slope: {slope:+.6f} | "
                f"SD Position: {sd_pos:+.2f}"
            )

        # Chart placeholder — in cloud we don't have raw_prices
        # Show SD band position as a simple gauge instead
        if not idx_row.empty:
            sd_pos = float(idx_row.iloc[0].get("price_sd_position", 0))
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = sd_pos,
                title = {"text": "SD Position", "font": {"color": "#e5e7eb"}},
                gauge = {
                    "axis"   : {"range": [-3.5, 3.5], "tickcolor": "#e5e7eb"},
                    "bar"    : {"color": "#16a34a" if sd_pos < 0 else "#dc2626"},
                    "bgcolor": "#1c2128",
                    "steps"  : [
                        {"range": [-3.5, -1], "color": "#14532d"},
                        {"range": [-1, 1],    "color": "#1c2128"},
                        {"range": [1, 3.5],   "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line" : {"color": "white", "width": 2},
                        "value": sd_pos,
                    },
                },
                number = {"font": {"color": "#e5e7eb"}},
            ))
            fig.update_layout(
                height       = 220,
                margin       = dict(l=20, r=20, t=40, b=20),
                paper_bgcolor= "#1c2128",
                font         = {"color": "#e5e7eb"},
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SECTION 2 — SECTOR HEALTH
# =============================================================================

st.header("Sector Health")

sector_cols = st.columns(4)
for i, (etf, info) in enumerate(sectors.items()):
    with sector_cols[i % 4]:
        s     = info["status"]
        name  = info["name"]
        color = TILE_COLOR.get(s, "#f3f4f6")
        st.markdown(
            f"<div style='background-color:{color};color:#111827;padding:10px;"
            f"border-radius:6px;margin-bottom:8px;font-weight:500;'>"
            f"<b>{etf}</b> — {name}<br>"
            f"{BADGE.get(s,'⚪')} {s.upper()}</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# SECTION 3 — SCANNER RESULTS
# =============================================================================

st.header("Scanner Results")

tab_long, tab_short = st.tabs(["📗 Long Candidates", "📕 Short Candidates"])

cols_to_show = [
    "ml_rank", "ticker", "sector", "sd_position",
    "volume_signal", "has_valid_zone", "ml_score",
]

with tab_long:
    longs = read_latest_scan_results(direction="long")
    if longs.empty:
        st.info("No long candidates today.")
    else:
        # Style ml_score column
        display_cols = [c for c in cols_to_show if c in longs.columns]
        st.dataframe(
            longs[display_cols].style.background_gradient(
                subset=["ml_score"], cmap="Greens"
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_short:
    shorts = read_latest_scan_results(direction="short")
    if shorts.empty:
        st.info("No short candidates today.")
    else:
        display_cols = [c for c in cols_to_show if c in shorts.columns]
        st.dataframe(
            shorts[display_cols].style.background_gradient(
                subset=["ml_score"], cmap="Reds"
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# SECTION 4 — MODEL HEALTH
# =============================================================================

st.header("Model Health")

metrics = read_latest_model_metrics()

if metrics.empty:
    st.info("ML models not trained yet.")
else:
    mcols = st.columns(2)
    for i, row in metrics.iterrows():
        with mcols[i % 2]:
            st.subheader(row["model_name"])
            st.metric("Precision", f"{row['precision_score']:.2%}")
            st.metric("AUC-ROC",   f"{row['auc_roc_score']:.3f}")

            if "recall_score" in row and pd.notna(row.get("recall_score")):
                st.metric("Recall", f"{row['recall_score']:.2%}")
            if "pr_auc_score" in row and pd.notna(row.get("pr_auc_score")):
                st.metric("PR-AUC", f"{row['pr_auc_score']:.3f}")

            st.caption(
                f"Trained: {row['train_date']} | "
                f"Samples: {row['n_samples']}"
            )
