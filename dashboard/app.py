"""
dashboard/app.py
-----------------
Streamlit dashboard for the Stock Scanner.
Read-only — displays results computed by the pipeline.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import plotly.graph_objects as go

from data.database import (
    initialise_database,
    read_raw_prices,
    read_latest_indicator_results,
    read_latest_scan_results,
    read_latest_model_metrics,
)
from scanner.screener import get_market_sector_status
from engines.linreg import compute_linreg_series
from pathlib import Path


LOGO_PATH  = Path(__file__).resolve().parent / "assets" / "logo.png"
SYSTEM_NAME = "Eagle Logic System"

st.set_page_config(
    page_title=SYSTEM_NAME,
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
)

initialise_database()

header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
with header_col2:
    st.markdown(f"<h1 style='margin-bottom:0;'>{SYSTEM_NAME}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9ca3af;margin-top:0;'>LinReg Channel + Smart Money Concepts Scanner — 15m</p>",
        unsafe_allow_html=True,
    )

indicator_df = read_latest_indicator_results()

if indicator_df.empty:
    st.warning("No indicator data yet. Run the pipeline first.")
    st.stop()

latest_date = indicator_df["date"].max()
st.caption(f"Last scan: {latest_date}")

status  = get_market_sector_status(indicator_df)
market  = status["market"]
sectors = status["sectors"]

BADGE = {"bullish": "🟢", "bearish": "🔴", "broken": "🟡", "unknown": "⚪"}
TILE_COLOR = {"bullish": "#dcfce7", "bearish": "#fee2e2", "broken": "#fef9c3", "unknown": "#f3f4f6"}


# ============================================================
# SECTION 1 — MARKET PULSE
# ============================================================
st.header("Market Pulse")

cols = st.columns(3)
for i, idx in enumerate(["SPY", "QQQ", "DIA"]):
    with cols[i]:
        st_status = market.get(idx, "unknown")
        st.subheader(f"{idx}  {BADGE.get(st_status,'⚪')} {st_status.upper()}")

        df = read_raw_prices(idx, days=200)
        if len(df) >= 50:
            closes = df["close"].values
            lr = compute_linreg_series(closes)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df["close"], mode="lines", name="Price",
                                      line=dict(color="#1f2937", width=1)))
            fig.add_trace(go.Scatter(y=lr["linreg"], mode="lines", name="LinReg",
                                      line=dict(color="#b91c1c", width=2)))
            fig.add_trace(go.Scatter(y=lr["sd1_upper"], mode="lines", name="+1SD",
                                      line=dict(color="#0d9488", dash="dash", width=1)))
            fig.add_trace(go.Scatter(y=lr["sd1_lower"], mode="lines", name="-1SD",
                                      line=dict(color="#0d9488", dash="dash", width=1)))
            fig.add_trace(go.Scatter(y=lr["sd2_upper"], mode="lines", name="+2SD",
                                      line=dict(color="#0d9488", width=1)))
            fig.add_trace(go.Scatter(y=lr["sd2_lower"], mode="lines", name="-2SD",
                                      line=dict(color="#0d9488", width=1)))
            fig.add_trace(go.Scatter(y=lr["sd3_upper"], mode="lines", name="+3SD",
                                      line=dict(color="#0d9488", dash="dot", width=1)))
            fig.add_trace(go.Scatter(y=lr["sd3_lower"], mode="lines", name="-3SD",
                                      line=dict(color="#0d9488", dash="dot", width=1)))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for chart")


# ============================================================
# SECTION 2 — SECTOR HEALTH
# ============================================================
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


# ============================================================
# SECTION 3 — SCANNER RESULTS
# ============================================================
st.header("Scanner Results")

tab_long, tab_short = st.tabs(["📗 Long Candidates", "📕 Short Candidates"])

cols_to_show = [
    "ml_rank", "ticker", "sector", "sd_position", "volume_signal",
    "put_call_ratio", "short_interest_pct", "ml_score",
]

with tab_long:
    longs = read_latest_scan_results(direction="long")
    if longs.empty:
        st.info("No long candidates today.")
    else:
        st.dataframe(longs[cols_to_show], use_container_width=True, hide_index=True)

with tab_short:
    shorts = read_latest_scan_results(direction="short")
    if shorts.empty:
        st.info("No short candidates today.")
    else:
        st.dataframe(shorts[cols_to_show], use_container_width=True, hide_index=True)


# ============================================================
# SECTION 4 — MODEL HEALTH
# ============================================================
st.header("Model Health")

metrics = read_latest_model_metrics()

if metrics.empty:
    st.info("ML models not trained yet — candidates ranked by SD position.")
else:
    mcols = st.columns(2)
    for i, row in metrics.iterrows():
        with mcols[i % 2]:
            st.subheader(row["model_name"])
            st.metric("Precision", f"{row['precision_score']:.2%}")
            st.metric("AUC-ROC",   f"{row['auc_roc_score']:.3f}")

            # Recall and PR-AUC — show if available
            if "recall_score" in row and pd.notna(row["recall_score"]):
                st.metric("Recall", f"{row['recall_score']:.2%}")
            if "pr_auc_score" in row and pd.notna(row["pr_auc_score"]):
                st.metric("PR-AUC", f"{row['pr_auc_score']:.3f}")

            st.caption(f"Trained: {row['train_date']} | Samples: {row['n_samples']}")