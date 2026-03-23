"""
tearsheet.py
============
Generates a full performance tearsheet for the TSMOM strategy:
  - Sharpe, Calmar, Sortino, Max Drawdown
  - Rolling attribution (per asset class)
  - Regime-conditional return profiles (bull/bear)
  - Key risk catalysts analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from strategy.tsmom import (
    simulate_price_history, compute_tsmom_signal,
    compute_ex_ante_volatility, build_positions, backtest,
    UNIVERSE, ASSET_CLASS, ASSETS, N_ASSETS
)


# ---------------------------------------------------------------------------
# 1. PERFORMANCE METRICS
# ---------------------------------------------------------------------------

def compute_metrics(pnl: pd.Series, rf_daily: float = 0.0) -> dict:
    excess   = pnl - rf_daily
    cum_ret  = (1 + pnl).cumprod()
    ann_ret  = pnl.mean()  * 252
    ann_vol  = pnl.std()   * np.sqrt(252)
    sharpe   = (excess.mean() * 252) / (excess.std() * np.sqrt(252))

    downside = pnl[pnl < 0].std() * np.sqrt(252)
    sortino  = (ann_ret - rf_daily * 252) / downside if downside > 0 else np.nan

    drawdowns = cum_ret / cum_ret.cummax() - 1
    max_dd    = drawdowns.min()
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    # Win rate
    win_rate  = (pnl > 0).mean()

    # Skewness, kurtosis
    skew = pnl.skew()
    kurt = pnl.kurt()

    return {
        "Ann_Return_%":  round(ann_ret  * 100, 2),
        "Ann_Vol_%":     round(ann_vol  * 100, 2),
        "Sharpe":        round(sharpe, 3),
        "Sortino":       round(sortino, 3),
        "Max_DD_%":      round(max_dd  * 100, 2),
        "Calmar":        round(calmar, 3),
        "Win_Rate_%":    round(win_rate * 100, 1),
        "Skewness":      round(skew, 3),
        "Kurtosis":      round(kurt, 3),
    }


# ---------------------------------------------------------------------------
# 2. ROLLING ATTRIBUTION
# ---------------------------------------------------------------------------

def rolling_attribution(
    returns:   pd.DataFrame,
    positions: pd.DataFrame,
    window:    int = 252,
) -> pd.DataFrame:
    """Rolling 1-year Sharpe by asset class."""
    gross_pnl = (positions.shift(1).fillna(0) * returns)

    class_pnl = {}
    classes   = list(set(ASSET_CLASS))
    for cls in classes:
        assets_in_class = [a for a, c in UNIVERSE.items() if c == cls]
        class_pnl[cls]  = gross_pnl[assets_in_class].sum(axis=1)

    df_class = pd.DataFrame(class_pnl)

    # Rolling Sharpe
    roll_sharpe = df_class.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    )
    return roll_sharpe


# ---------------------------------------------------------------------------
# 3. REGIME CONDITIONAL ANALYSIS
# ---------------------------------------------------------------------------

def regime_conditional_returns(pnl: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Split performance into bull/bear regimes using 200-day SPY MA.
    """
    # Use first equity (SPY proxy) for regime flag
    spy_ret   = returns.iloc[:, 0]
    spy_cum   = (1 + spy_ret).cumprod()
    spy_ma200 = spy_cum.rolling(200).mean()
    bull_flag = spy_cum > spy_ma200

    pnl_aligned = pnl.reindex(bull_flag.index).dropna()
    bull_mask   = bull_flag.reindex(pnl_aligned.index)

    results = {}
    for label, mask in [("Bull", bull_mask), ("Bear", ~bull_mask)]:
        sub = pnl_aligned[mask]
        if len(sub) == 0:
            continue
        ann_r = sub.mean() * 252
        ann_v = sub.std()  * np.sqrt(252)
        results[label] = {
            "Ann_Return_%": round(ann_r * 100, 2),
            "Ann_Vol_%":    round(ann_v * 100, 2),
            "Sharpe":       round(ann_r / ann_v, 3) if ann_v > 0 else np.nan,
            "N_Days":       len(sub),
        }
    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# 4. TEARSHEET PLOT
# ---------------------------------------------------------------------------

def plot_tearsheet(
    pnl:          pd.Series,
    roll_sharpe:  pd.DataFrame,
    save_path:    str = "outputs/tsmom_tearsheet.png",
):
    cum_ret   = (1 + pnl).cumprod()
    drawdowns = cum_ret / cum_ret.cummax() - 1
    roll_ret  = pnl.rolling(252).mean() * 252 * 100

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Cumulative return
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum_ret.index, cum_ret.values, color="#1f77b4", lw=1.5, label="TSMOM")
    ax1.axhline(1, color="gray", linestyle="--", lw=0.8)
    ax1.set_title("TSMOM Cumulative Return (20yr backtest)", fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(); ax1.grid(alpha=0.2)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(drawdowns.index, drawdowns.values * 100, 0,
                     color="#d62728", alpha=0.6, label="Drawdown")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("%"); ax2.grid(alpha=0.2)

    # 3. Rolling 1yr return
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(roll_ret.index, roll_ret.values, color="#2ca02c", lw=1.2)
    ax3.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax3.set_title("Rolling 1yr Annualized Return (%)")
    ax3.set_ylabel("%"); ax3.grid(alpha=0.2)

    # 4. Rolling Sharpe by asset class
    ax4 = fig.add_subplot(gs[2, :])
    colors = {"equity": "#1f77b4", "commodity": "#ff7f0e", "fixed_income": "#2ca02c"}
    for cls in roll_sharpe.columns:
        ax4.plot(roll_sharpe.index, roll_sharpe[cls],
                 label=cls.replace("_", " ").title(),
                 color=colors.get(cls, "gray"), lw=1.2, alpha=0.85)
    ax4.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax4.set_title("Rolling 1yr Sharpe by Asset Class")
    ax4.set_ylabel("Sharpe Ratio")
    ax4.legend(loc="upper left"); ax4.grid(alpha=0.2)

    plt.suptitle("TSMOM Strategy — Full Tearsheet", fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Tearsheet saved to {save_path}")


# ---------------------------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running TSMOM tearsheet generation...\n")

    prices, returns = simulate_price_history()
    signal          = compute_tsmom_signal(returns, lookback=252)
    vol_scale       = compute_ex_ante_volatility(returns, vol_window=60)
    positions       = build_positions(signal, vol_scale)
    pnl             = backtest(returns, positions, tc_bps=10.0)

    # Full-period metrics
    metrics = compute_metrics(pnl)
    print("Full-Period Performance:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    # Rolling attribution
    roll_sharpe = rolling_attribution(returns, positions)

    # Regime analysis
    regime_df = regime_conditional_returns(pnl, returns)
    print("\nRegime-Conditional Performance:")
    print(regime_df.to_string())

    # Save outputs
    pd.DataFrame([metrics]).T.rename(columns={0: "Value"}).to_csv(
        "outputs/tearsheet_metrics.csv")
    regime_df.to_csv("outputs/regime_conditional.csv")

    # Plot
    plot_tearsheet(pnl, roll_sharpe)
    print("\nAll tearsheet outputs saved to outputs/")
