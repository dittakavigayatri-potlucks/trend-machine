"""
app.py — Trend Machine: Stock Intelligence Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from analysis.factors import compute_factors
from analysis.ml_model import run_ml_model
from analysis.signals import compute_signals
from analysis.stats import compute_stats

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trend Machine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}

.stApp { background-color: #0a0e1a; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}
.metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 26px;
    font-weight: 600;
    color: #f1f5f9;
    font-family: 'Space Mono', monospace;
}
.metric-delta {
    font-size: 12px;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}
.positive { color: #10b981; }
.negative { color: #ef4444; }
.neutral  { color: #94a3b8; }

.signal-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}
.badge-long  { background: #052e16; color: #34d399; border: 1px solid #10b981; }
.badge-short { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
.badge-flat  { background: #0f172a; color: #94a3b8; border: 1px solid #334155; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}

.stTextInput > div > div > input {
    background-color: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 18px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

.stSelectbox > div > div {
    background-color: #111827 !important;
    border-color: #1e293b !important;
}

div[data-testid="stSidebar"] {
    background-color: #060912 !important;
    border-right: 1px solid #1e293b;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.insight-box {
    background: #0f1729;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.6;
    color: #cbd5e1;
}

.ticker-display {
    font-family: 'Space Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.company-name {
    font-size: 15px;
    color: #64748b;
    margin-top: -4px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 TREND MACHINE")
    st.markdown("<div style='font-size:11px;color:#475569;letter-spacing:2px;margin-bottom:24px'>TSMOM SIGNAL EXPLORER</div>", unsafe_allow_html=True)

    ticker_input = st.text_input("TICKER", value="GEV", placeholder="e.g. GEV, AAPL, SPY").upper().strip()

    period = st.selectbox("LOOKBACK PERIOD", ["1y", "2y", "3y", "5y"], index=2)

    st.markdown("---")
    st.markdown("<div class='section-header'>TSMOM PARAMETERS</div>", unsafe_allow_html=True)
    momentum_window = st.slider("Momentum Window (days)", 60, 504, 252)
    ema_fast = st.slider("EMA Fast (days)", 5, 50, 12)
    ema_slow = st.slider("EMA Slow (days)", 20, 200, 50)
    vol_window = st.slider("Vol Window (EWMA span)", 20, 120, 60)
    vol_target = st.slider("Vol Target (%)", 10, 60, 40) / 100

    st.markdown("---")
    run_btn = st.button("▶  RUN ANALYSIS")

    st.markdown("<div style='margin-top:32px;font-size:10px;color:#334155;text-align:center'>Based on Moskowitz, Ooi & Pedersen (2012)</div>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='ticker-display'></div>", unsafe_allow_html=True)

if not ticker_input:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

# Fetch data
@st.cache_data(ttl=300)
def load_data(ticker, period):
    tk = yf.Ticker(ticker)
    df = tk.history(period=period)
    info = {}
    try:
        info = tk.info
    except:
        pass
    return df, info

with st.spinner(f"Fetching {ticker_input}..."):
    try:
        df_raw, info = load_data(ticker_input, period)
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker_input}**: {e}")
        st.stop()

if df_raw.empty or len(df_raw) < 60:
    st.error(f"Not enough data for **{ticker_input}**. Try a different ticker or longer period.")
    st.stop()

df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)

# ── Header ────────────────────────────────────────────────────────────────────
company_name = info.get("longName", ticker_input)
sector       = info.get("sector", "—")
industry     = info.get("industry", "—")

st.markdown(f"<div class='ticker-display'>{ticker_input}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='company-name'>{company_name} &nbsp;·&nbsp; {sector} &nbsp;·&nbsp; {industry}</div>", unsafe_allow_html=True)

# ── Compute everything ────────────────────────────────────────────────────────
params = dict(
    momentum_window=momentum_window,
    ema_fast=ema_fast,
    ema_slow=ema_slow,
    vol_window=vol_window,
    vol_target=vol_target,
)

df       = compute_signals(df_raw.copy(), params)
stats    = compute_stats(df)
factors  = compute_factors(df, params)
ml_out   = run_ml_model(df, params)

# ── Top KPI row ───────────────────────────────────────────────────────────────
signal_val  = df["tsmom_signal"].iloc[-1]
signal_html = (
    "<span class='signal-badge badge-long'>▲ LONG</span>"  if signal_val > 0 else
    "<span class='signal-badge badge-short'>▼ SHORT</span>" if signal_val < 0 else
    "<span class='signal-badge badge-flat'>— FLAT</span>"
)

ret_1d  = df["Close"].pct_change().iloc[-1]
ret_1m  = df["Close"].pct_change(21).iloc[-1]
ret_ytd = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1)

def fmt_pct(v):
    sign = "+" if v >= 0 else ""
    cls  = "positive" if v >= 0 else "negative"
    return f"<span class='{cls}'>{sign}{v*100:.2f}%</span>"

col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    ("TSMOM SIGNAL",    signal_html,                                    ""),
    ("LAST CLOSE",      f"${df['Close'].iloc[-1]:,.2f}",               fmt_pct(ret_1d) + " today"),
    ("1-MONTH RETURN",  fmt_pct(ret_1m),                               "vs. prior month"),
    ("SHARPE (PERIOD)", f"<span class='neutral'>{stats['sharpe']:.2f}</span>", "annualized"),
    ("ML EDGE SCORE",   f"<span class='{'positive' if ml_out['score']>0.5 else 'negative'}'>{ml_out['score']:.2f}</span>", "0–1 scale"),
]
for col, (label, val, delta) in zip([col1,col2,col3,col4,col5], cards):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-delta neutral'>{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Price + EMAs chart ────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>PRICE & TREND SIGNALS</div>", unsafe_allow_html=True)

fig_price = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.25, 0.20],
    vertical_spacing=0.03,
    subplot_titles=("Price + EMAs + MACD Signal", "Volume", "TSMOM Position Weight"),
)

# Candlestick
fig_price.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="OHLC",
    increasing_line_color="#10b981", decreasing_line_color="#ef4444",
    increasing_fillcolor="#052e16",  decreasing_fillcolor="#450a0a",
), row=1, col=1)

# EMAs
for col_name, color, width, dash in [
    (f"ema_{ema_fast}", "#60a5fa", 1.5, "solid"),
    (f"ema_{ema_slow}", "#f59e0b", 1.5, "solid"),
    ("ema_200",         "#a78bfa", 1,   "dash"),
]:
    if col_name in df.columns:
        fig_price.add_trace(go.Scatter(
            x=df.index, y=df[col_name],
            name=col_name.upper().replace("_", " "),
            line=dict(color=color, width=width, dash=dash),
        ), row=1, col=1)

# Bollinger bands
if "bb_upper" in df.columns:
    fig_price.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
        line=dict(color="#334155", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig_price.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
        line=dict(color="#334155", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(51,65,85,0.07)", showlegend=False), row=1, col=1)

# Volume
colors_vol = ["#052e16" if c >= o else "#450a0a" for c, o in zip(df["Close"], df["Open"])]
fig_price.add_trace(go.Bar(
    x=df.index, y=df["Volume"], name="Volume",
    marker_color=colors_vol, opacity=0.7,
), row=2, col=1)

# TSMOM weight
if "position_weight" in df.columns:
    fig_price.add_trace(go.Scatter(
        x=df.index, y=df["position_weight"],
        name="Position Weight",
        line=dict(color="#8b5cf6", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(139,92,246,0.1)",
    ), row=3, col=1)
    fig_price.add_hline(y=0, line_color="#334155", line_width=1, row=3, col=1)

fig_price.update_layout(
    height=580,
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0a0e1a",
    font=dict(family="DM Sans", color="#94a3b8", size=11),
    legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)", font_size=10),
    xaxis_rangeslider_visible=False,
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)
for i in range(1, 4):
    fig_price.update_xaxes(gridcolor="#0f1729", row=i, col=1)
    fig_price.update_yaxes(gridcolor="#0f1729", row=i, col=1)

st.plotly_chart(fig_price, use_container_width=True)

# ── Factor analysis ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>FACTOR & MOMENTUM DECOMPOSITION</div>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    # Rolling Sharpe
    fig_rs = go.Figure()
    if "rolling_sharpe" in df.columns:
        fig_rs.add_trace(go.Scatter(
            x=df.index, y=df["rolling_sharpe"],
            name="Rolling Sharpe (63d)",
            line=dict(color="#3b82f6", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
        ))
        fig_rs.add_hline(y=0, line_color="#334155", line_width=1)
        fig_rs.add_hline(y=1, line_color="#10b981", line_width=0.8, line_dash="dash")
        fig_rs.add_hline(y=-1, line_color="#ef4444", line_width=0.8, line_dash="dash")
    fig_rs.update_layout(
        title=dict(text="Rolling 63-Day Sharpe", font_size=12, font_color="#94a3b8"),
        height=280, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8"), showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(gridcolor="#0f1729"), yaxis=dict(gridcolor="#0f1729"),
    )
    st.plotly_chart(fig_rs, use_container_width=True)

with col_b:
    # Realized vol
    fig_vol = go.Figure()
    if "realized_vol" in df.columns:
        fig_vol.add_trace(go.Scatter(
            x=df.index, y=df["realized_vol"] * 100,
            name="Realized Vol (ann.)",
            line=dict(color="#f59e0b", width=1.5),
        ))
    if "ewma_vol" in df.columns:
        fig_vol.add_trace(go.Scatter(
            x=df.index, y=df["ewma_vol"] * 100,
            name="EWMA Vol (ex-ante)",
            line=dict(color="#8b5cf6", width=1.5, dash="dash"),
        ))
    fig_vol.update_layout(
        title=dict(text="Realized vs Ex-Ante Volatility (%)", font_size=12, font_color="#94a3b8"),
        height=280, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font_size=10),
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(gridcolor="#0f1729"), yaxis=dict(gridcolor="#0f1729"),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ── Momentum & RSI ────────────────────────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    fig_mom = go.Figure()
    if "momentum_12m" in df.columns:
        mom = df["momentum_12m"] * 100
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in mom]
        fig_mom.add_trace(go.Bar(x=df.index, y=mom, name="12M Momentum", marker_color=colors, opacity=0.8))
        fig_mom.add_hline(y=0, line_color="#334155", line_width=1)
    fig_mom.update_layout(
        title=dict(text="12-Month Trailing Momentum (%)", font_size=12, font_color="#94a3b8"),
        height=260, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8"), showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(gridcolor="#0f1729"), yaxis=dict(gridcolor="#0f1729"),
    )
    st.plotly_chart(fig_mom, use_container_width=True)

with col_d:
    fig_rsi = go.Figure()
    if "rsi" in df.columns:
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI (14)",
            line=dict(color="#60a5fa", width=1.5),
        ))
        fig_rsi.add_hline(y=70, line_color="#ef4444", line_dash="dash", line_width=0.8)
        fig_rsi.add_hline(y=30, line_color="#10b981", line_dash="dash", line_width=0.8)
        fig_rsi.add_hline(y=50, line_color="#334155", line_width=0.8)
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.05)", line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.05)", line_width=0)
    fig_rsi.update_layout(
        title=dict(text="RSI (14)", font_size=12, font_color="#94a3b8"),
        height=260, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8"), showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(gridcolor="#0f1729"),
        yaxis=dict(gridcolor="#0f1729", range=[0, 100]),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ── Return distribution ───────────────────────────────────────────────────────
st.markdown("<div class='section-header'>RETURN DISTRIBUTION & DRAWDOWN</div>", unsafe_allow_html=True)

col_e, col_f = st.columns(2)

with col_e:
    rets = df["Close"].pct_change().dropna() * 100
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=rets, nbinsx=80, name="Daily Returns",
        marker_color="#3b82f6", opacity=0.7,
    ))
    fig_dist.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig_dist.add_vline(x=rets.mean(), line_color="#f59e0b", line_dash="dash", line_width=1.5)
    fig_dist.update_layout(
        title=dict(text="Daily Return Distribution (%)", font_size=12, font_color="#94a3b8"),
        height=260, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8"), showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(gridcolor="#0f1729"), yaxis=dict(gridcolor="#0f1729"),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_f:
    roll_max  = df["Close"].cummax()
    drawdown  = (df["Close"] - roll_max) / roll_max * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df.index, y=drawdown,
        name="Drawdown",
        line=dict(color="#ef4444", width=1),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.1)",
    ))
    fig_dd.update_layout(
        title=dict(text="Underwater Drawdown (%)", font_size=12, font_color="#94a3b8"),
        height=260, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(color="#94a3b8"), showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
        xaxis=dict(gridcolor="#0f1729"), yaxis=dict(gridcolor="#0f1729"),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ── ML Model output ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>ML FACTOR MODEL (RANDOM FOREST)</div>", unsafe_allow_html=True)

col_g, col_h = st.columns([1.2, 1])

with col_g:
    if ml_out.get("feature_importance") is not None:
        fi = ml_out["feature_importance"].sort_values(ascending=True).tail(12)
        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index,
            orientation="h",
            marker=dict(
                color=fi.values,
                colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#60a5fa"]],
            ),
        ))
        fig_fi.update_layout(
            title=dict(text="Feature Importance (Top 12)", font_size=12, font_color="#94a3b8"),
            height=340, paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
            font=dict(color="#94a3b8"), showlegend=False,
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis=dict(gridcolor="#0f1729"), yaxis=dict(gridcolor="#0f1729"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

with col_h:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    ml_metrics = [
        ("DIRECTION ACCURACY",  f"{ml_out.get('accuracy', 0)*100:.1f}%"),
        ("PRECISION (UP)",      f"{ml_out.get('precision', 0)*100:.1f}%"),
        ("ML SIGNAL",           "UP ▲" if ml_out.get("prediction") == 1 else "DOWN ▼"),
        ("CONFIDENCE SCORE",    f"{ml_out.get('score', 0):.3f}"),
        ("FEATURES USED",       str(ml_out.get("n_features", 0))),
        ("TRAIN WINDOW",        f"{ml_out.get('train_size', 0)} days"),
    ]
    for label, val in ml_metrics:
        color = "#10b981" if "UP" in val else ("#ef4444" if "DOWN" in val else "#94a3b8")
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             padding:10px 14px;margin:4px 0;background:#0f1729;border-radius:6px;
             border-left:2px solid #1e293b'>
            <span style='font-size:11px;letter-spacing:1px;color:#475569;font-family:Space Mono'>{label}</span>
            <span style='font-family:Space Mono;font-size:13px;font-weight:600;color:{color}'>{val}</span>
        </div>""", unsafe_allow_html=True)

# ── Key stats table ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>FULL STATISTICS</div>", unsafe_allow_html=True)

stat_cols = st.columns(4)
stat_items = [
    ("Annualized Return",  f"{stats.get('ann_return',0)*100:.2f}%"),
    ("Annualized Vol",     f"{stats.get('ann_vol',0)*100:.2f}%"),
    ("Sharpe Ratio",       f"{stats.get('sharpe',0):.3f}"),
    ("Sortino Ratio",      f"{stats.get('sortino',0):.3f}"),
    ("Max Drawdown",       f"{stats.get('max_dd',0)*100:.2f}%"),
    ("Calmar Ratio",       f"{stats.get('calmar',0):.3f}"),
    ("Win Rate",           f"{stats.get('win_rate',0)*100:.1f}%"),
    ("Skewness",           f"{stats.get('skew',0):.3f}"),
    ("Kurtosis",           f"{stats.get('kurt',0):.3f}"),
    ("VaR 95% (1d)",       f"{stats.get('var_95',0)*100:.2f}%"),
    ("CVaR 95% (1d)",      f"{stats.get('cvar_95',0)*100:.2f}%"),
    ("Avg Daily Volume",   f"{stats.get('avg_vol',0):,.0f}"),
]
for i, (label, val) in enumerate(stat_items):
    with stat_cols[i % 4]:
        st.markdown(f"""
        <div style='padding:12px 14px;margin:3px 0;background:#0d1526;border-radius:6px;
             border:1px solid #1e293b'>
            <div style='font-size:10px;letter-spacing:1.5px;color:#475569;
                 font-family:Space Mono;margin-bottom:4px'>{label.upper()}</div>
            <div style='font-family:Space Mono;font-size:15px;font-weight:600;color:#e2e8f0'>{val}</div>
        </div>""", unsafe_allow_html=True)

# ── AI Insights ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>STRATEGY INSIGHTS</div>", unsafe_allow_html=True)

insights = []

mom_12m = df["momentum_12m"].iloc[-1] if "momentum_12m" in df.columns else 0
if mom_12m > 0.10:
    insights.append(f"📈 <b>Strong momentum signal:</b> {ticker_input} has a 12-month trailing return of {mom_12m*100:.1f}%, well above the long threshold. TSMOM strategy is currently <b>LONG</b>.")
elif mom_12m < -0.10:
    insights.append(f"📉 <b>Negative momentum:</b> {ticker_input} shows a 12-month return of {mom_12m*100:.1f}%. TSMOM strategy is signaling <b>SHORT / AVOID</b>.")
else:
    insights.append(f"↔️ <b>Weak momentum regime:</b> 12-month return is {mom_12m*100:.1f}%. TSMOM position weight is near zero — consider waiting for a clearer trend.")

if stats.get("sharpe", 0) > 1.0:
    insights.append(f"⚡ <b>Strong risk-adjusted performance:</b> Sharpe of {stats['sharpe']:.2f} over the selected period, above the 1.0 threshold used in the tearsheet framework.")
elif stats.get("sharpe", 0) < 0:
    insights.append(f"⚠️ <b>Negative Sharpe:</b> Returns have not compensated for risk over this period. The TSMOM strategy would reduce or short this position.")

rsi_val = df["rsi"].iloc[-1] if "rsi" in df.columns else 50
if rsi_val > 70:
    insights.append(f"🔴 <b>Overbought conditions:</b> RSI at {rsi_val:.1f} — above the 70 threshold. Short-term mean-reversion risk elevated despite positive momentum.")
elif rsi_val < 30:
    insights.append(f"🟢 <b>Oversold conditions:</b> RSI at {rsi_val:.1f} — below 30. While momentum may still be negative, oversold readings can precede reversals.")

ewma_vol = df["ewma_vol"].iloc[-1] if "ewma_vol" in df.columns else None
if ewma_vol:
    insights.append(f"📊 <b>Ex-ante volatility:</b> Current EWMA vol is {ewma_vol*100:.1f}% annualized. At the {int(vol_target*100)}% portfolio vol target, the position scalar is <b>{(vol_target/ewma_vol):.2f}x</b>.")

if ml_out.get("score", 0.5) > 0.65:
    insights.append(f"🤖 <b>ML model agrees:</b> Random Forest returns an edge score of {ml_out['score']:.2f} — the model predicts an UP move in the next period with {ml_out.get('accuracy',0)*100:.1f}% historical directional accuracy.")
elif ml_out.get("score", 0.5) < 0.40:
    insights.append(f"🤖 <b>ML model is bearish:</b> Edge score of {ml_out['score']:.2f} — the model leans toward a DOWN move. This diverges from any positive price momentum and warrants caution.")

max_dd = stats.get("max_dd", 0)
if abs(max_dd) > 0.30:
    insights.append(f"🚨 <b>Significant drawdown history:</b> Max drawdown of {max_dd*100:.1f}% in this period. The Calmar ratio of {stats.get('calmar',0):.2f} measures whether returns justify this tail risk.")

for ins in insights:
    st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:48px;padding-top:16px;border-top:1px solid #1e293b;font-size:11px;color:#334155;text-align:center;font-family:Space Mono'>TREND MACHINE · Dittakavi Gayatri · Columbia QMSS · Based on Moskowitz, Ooi & Pedersen (2012)</div>", unsafe_allow_html=True)
