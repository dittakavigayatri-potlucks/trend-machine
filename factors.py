"""
analysis/factors.py
Computes cross-sectional and time-series factor exposures.
"""

import numpy as np
import pandas as pd


def compute_factors(df: pd.DataFrame, params: dict) -> dict:
    """
    Returns a dict of factor metrics for the sidebar summary.
    """
    close = df["Close"]
    rets  = close.pct_change().dropna()

    factors = {}

    # 1-month, 3-month, 12-month momentum
    for window, label in [(21, "mom_1m"), (63, "mom_3m"), (252, "mom_12m")]:
        if len(close) > window:
            factors[label] = float(close.pct_change(window).iloc[-1])

    # Trend strength: ratio of EMA fast to EMA slow
    ema_fast_col = f"ema_{params['ema_fast']}"
    ema_slow_col = f"ema_{params['ema_slow']}"
    if ema_fast_col in df.columns and ema_slow_col in df.columns:
        fast = df[ema_fast_col].iloc[-1]
        slow = df[ema_slow_col].iloc[-1]
        factors["ema_ratio"] = fast / slow if slow != 0 else 1.0
        factors["trend_up"]  = fast > slow

    # MACD crossover
    if "macd" in df.columns and "macd_signal" in df.columns:
        factors["macd_bullish"] = bool(df["macd"].iloc[-1] > df["macd_signal"].iloc[-1])

    # Volume trend (20-day vs 60-day avg vol)
    if "Volume" in df.columns:
        v20 = df["Volume"].rolling(20).mean().iloc[-1]
        v60 = df["Volume"].rolling(60).mean().iloc[-1]
        factors["vol_trend"] = float(v20 / v60) if v60 != 0 else 1.0

    # Autocorrelation of returns (1-day lag) — momentum persistence
    factors["autocorr_1d"] = float(rets.autocorr(lag=1)) if len(rets) > 10 else 0.0
    factors["autocorr_5d"] = float(rets.autocorr(lag=5)) if len(rets) > 10 else 0.0

    return factors
