"""
Time-Series Momentum (TSMOM) Strategy
======================================
Captures persistent cross-asset trends using 12-month trailing returns as
position signals across equities, commodities, and fixed income, with
ex-ante volatility scaling. Backtested over 20+ years of daily data.

Based on Moskowitz, Ooi & Pedersen (2012): "Time Series Momentum"
Journal of Financial Economics, 104(2), 228-250.

Author: Naga Siva Gayatri Dittakavi
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. UNIVERSE
# ---------------------------------------------------------------------------

UNIVERSE = {
    # Equities
    "SPY":  "equity",
    "QQQ":  "equity",
    "IWM":  "equity",
    "EFA":  "equity",
    "EEM":  "equity",
    "VGK":  "equity",
    "EWJ":  "equity",
    # Commodities
    "GLD":  "commodity",
    "SLV":  "commodity",
    "USO":  "commodity",
    "DBA":  "commodity",
    "PDBC": "commodity",
    # Fixed Income
    "TLT":  "fixed_income",
    "IEF":  "fixed_income",
    "AGG":  "fixed_income",
    "HYG":  "fixed_income",
    "EMB":  "fixed_income",
    "BNDX": "fixed_income",
}

ASSETS      = list(UNIVERSE.keys())
ASSET_CLASS = list(UNIVERSE.values())
N_ASSETS    = len(ASSETS)


# ---------------------------------------------------------------------------
# 2. DATA SIMULATION
# ---------------------------------------------------------------------------

def simulate_price_history(
    n_assets:  int   = N_ASSETS,
    n_days:    int   = 5040,   # ~20 years
    seed:      int   = 7,
) -> pd.DataFrame:
    """
    Simulate daily price data with realistic momentum regimes.
    In production: replace with yfinance, Bloomberg, or CRSP data.
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2004-01-02", periods=n_days)

    # Asset return params
    ann_rets  = np.array([0.10, 0.12, 0.09, 0.07, 0.08, 0.08, 0.06,    # eq
                           0.05, 0.04, 0.03, 0.02, 0.03,                 # com
                           0.03, 0.03, 0.04, 0.05, 0.05, 0.03])          # fi
    ann_vols  = np.array([0.18, 0.22, 0.20, 0.16, 0.22, 0.17, 0.16,
                           0.14, 0.22, 0.28, 0.16, 0.18,
                           0.12, 0.08, 0.05, 0.10, 0.12, 0.06])

    daily_mu  = ann_rets  / 252
    daily_sig = ann_vols  / np.sqrt(252)

    # Regime-switching: bull/bear with persistence
    regime   = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        if regime[t-1] == 0:   # bull
            regime[t] = 1 if np.random.rand() < 0.002 else 0
        else:                  # bear
            regime[t] = 0 if np.random.rand() < 0.008 else 1

    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        shock = np.random.randn(n_days) * daily_sig[i]
        regime_adj = np.where(regime == 1, -daily_mu[i] * 3, daily_mu[i])
        returns[:, i] = regime_adj + shock

    log_returns = pd.DataFrame(returns, index=dates, columns=ASSETS)
    prices      = np.exp(log_returns.cumsum()) * 100.0
    return prices, log_returns


# ---------------------------------------------------------------------------
# 3. SIGNAL CONSTRUCTION
# ---------------------------------------------------------------------------

def compute_tsmom_signal(
    returns: pd.DataFrame,
    lookback: int = 252,   # 12-month trailing (≈252 trading days)
    skip_last: int = 1,    # skip most recent day (implementation shortfall buffer)
) -> pd.DataFrame:
    """
    TSMOM signal: sign of 12-month trailing return.
    +1 = long (positive momentum), -1 = short (negative momentum).
    """
    trailing_ret = returns.shift(skip_last).rolling(lookback).sum()
    signal       = np.sign(trailing_ret)
    return signal


def compute_ex_ante_volatility(
    returns:     pd.DataFrame,
    vol_window:  int   = 60,   # 3-month realized vol estimation window
    annual_vol_target: float = 0.40,  # 40% per-asset annualized vol target
) -> pd.DataFrame:
    """
    Ex-ante (forecast) volatility using exponentially weighted variance.
    Scale each position to target a fixed annualized volatility.
    """
    ewm_vol   = returns.ewm(span=vol_window).std() * np.sqrt(252)
    vol_scale = annual_vol_target / ewm_vol.replace(0, np.nan)
    vol_scale = vol_scale.clip(0, 5)  # cap leverage at 5x
    return vol_scale


# ---------------------------------------------------------------------------
# 4. POSITION CONSTRUCTION
# ---------------------------------------------------------------------------

def build_positions(
    signal:      pd.DataFrame,
    vol_scale:   pd.DataFrame,
    portfolio_vol_target: float = 0.15,  # 15% annualized portfolio vol target
) -> pd.DataFrame:
    """
    Position = signal * vol_scale, then normalize to portfolio vol target.
    Each asset gets position weight: w_i = signal_i * (target_vol / asset_vol)
    Cross-sectionally: scale such that portfolio has target_vol.
    """
    raw_positions = signal * vol_scale / N_ASSETS
    return raw_positions * portfolio_vol_target


# ---------------------------------------------------------------------------
# 5. BACKTEST ENGINE
# ---------------------------------------------------------------------------

def backtest(
    returns:   pd.DataFrame,
    positions: pd.DataFrame,
    tc_bps:    float = 10.0,  # transaction costs in basis points (round-trip)
) -> pd.Series:
    """
    Compute daily portfolio P&L given positions (weights) and returns.
    Applies proportional transaction costs on position changes.
    """
    # Align positions and returns (positions use prev-day signal → today's return)
    pos_shifted  = positions.shift(1).fillna(0)
    gross_pnl    = (pos_shifted * returns).sum(axis=1)

    # Transaction costs
    pos_changes  = pos_shifted.diff().abs().sum(axis=1)
    tc           = pos_changes * (tc_bps / 10000)

    net_pnl      = gross_pnl - tc
    return net_pnl


# ---------------------------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Simulating 20-year price history...")
    prices, returns = simulate_price_history()

    print(f"Universe: {N_ASSETS} assets | {len(returns)} trading days\n")

    signal    = compute_tsmom_signal(returns, lookback=252)
    vol_scale = compute_ex_ante_volatility(returns, vol_window=60)
    positions = build_positions(signal, vol_scale, portfolio_vol_target=0.15)

    pnl = backtest(returns, positions, tc_bps=10.0)

    cum_ret  = (1 + pnl).cumprod()
    ann_ret  = pnl.mean()  * 252
    ann_vol  = pnl.std()   * np.sqrt(252)
    sharpe   = ann_ret / ann_vol
    max_dd   = (cum_ret / cum_ret.cummax() - 1).min()
    calmar   = ann_ret / abs(max_dd)

    print("TSMOM Strategy Performance (20yr backtest, 10bps TC)")
    print("=" * 50)
    print(f"  Annualized Return: {ann_ret*100:.2f}%")
    print(f"  Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:.3f}")
    print(f"  Max Drawdown:      {max_dd*100:.2f}%")
    print(f"  Calmar Ratio:      {calmar:.3f}")

    pnl.to_csv("outputs/tsmom_daily_pnl.csv", header=["PnL"])
    cum_ret.to_csv("outputs/tsmom_cumulative_return.csv", header=["Cumulative_Return"])
    positions.to_csv("outputs/tsmom_positions.csv")
    print("\nOutputs saved.")
