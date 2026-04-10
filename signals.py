"""
analysis/signals.py
Computes all technical signals used in the TSMOM dashboard.
"""

import numpy as np
import pandas as pd


def compute_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds all indicator columns to df in-place and returns it.
    """
    close = df["Close"]
    ema_fast = params["ema_fast"]
    ema_slow = params["ema_slow"]
    vol_window = params["vol_window"]
    vol_target = params["vol_target"]
    momentum_window = params["momentum_window"]

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df[f"ema_{ema_fast}"] = close.ewm(span=ema_fast, adjust=False).mean()
    df[f"ema_{ema_slow}"] = close.ewm(span=ema_slow, adjust=False).mean()
    df["ema_200"]          = close.ewm(span=200, adjust=False).mean()

    # ── Bollinger Bands (20-day, 2σ) ─────────────────────────────────────────
    bb_window = 20
    bb_mean   = close.rolling(bb_window).mean()
    bb_std    = close.rolling(bb_window).std()
    df["bb_upper"] = bb_mean + 2 * bb_std
    df["bb_lower"] = bb_mean - 2 * bb_std
    df["bb_mid"]   = bb_mean

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    df["rsi"] = _rsi(close, 14)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Daily returns ─────────────────────────────────────────────────────────
    df["ret"] = close.pct_change()

    # ── Ex-ante vol (EWMA) ───────────────────────────────────────────────────
    df["ewma_vol"]     = df["ret"].ewm(span=vol_window).std() * np.sqrt(252)
    df["realized_vol"] = df["ret"].rolling(63).std() * np.sqrt(252)

    # ── TSMOM signal ─────────────────────────────────────────────────────────
    # Moskowitz et al.: sign of trailing 12-month return (skip most recent day)
    trailing_ret = close.pct_change(momentum_window).shift(1)
    df["momentum_12m"]  = trailing_ret
    df["tsmom_signal"]  = np.sign(trailing_ret)

    # ── Vol-scaled position weight ────────────────────────────────────────────
    ewma_vol_safe       = df["ewma_vol"].replace(0, np.nan).fillna(method="ffill")
    raw_weight          = df["tsmom_signal"] * (vol_target / ewma_vol_safe)
    df["position_weight"] = raw_weight.clip(-5, 5)   # cap individual leverage at 5×

    # ── Rolling Sharpe (63-day) ───────────────────────────────────────────────
    roll_mean = df["ret"].rolling(63).mean() * 252
    roll_std  = df["ret"].rolling(63).std()  * np.sqrt(252)
    df["rolling_sharpe"] = (roll_mean / roll_std.replace(0, np.nan)).fillna(0)

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
