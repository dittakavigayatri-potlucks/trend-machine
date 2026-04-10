"""
analysis/stats.py
Full performance statistics matching the tearsheet in the TSMOM strategy.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_stats(df: pd.DataFrame) -> dict:
    """
    Returns a dict of annualized performance metrics.
    """
    rets = df["Close"].pct_change().dropna()

    if len(rets) < 10:
        return {}

    ann_return = rets.mean() * 252
    ann_vol    = rets.std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol != 0 else 0

    # Sortino (downside dev)
    neg_rets   = rets[rets < 0]
    down_dev   = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 1 else 1e-9
    sortino    = ann_return / down_dev

    # Max drawdown
    cum        = (1 + rets).cumprod()
    roll_max   = cum.cummax()
    dd_series  = (cum - roll_max) / roll_max
    max_dd     = dd_series.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (rets > 0).mean()

    # Higher moments
    skew = float(scipy_stats.skew(rets.dropna()))
    kurt = float(scipy_stats.kurtosis(rets.dropna()))

    # VaR & CVaR (historical, 95%)
    var_95  = float(np.percentile(rets, 5))
    cvar_95 = float(rets[rets <= var_95].mean())

    avg_vol = float(df["Volume"].mean()) if "Volume" in df.columns else 0

    return dict(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_dd=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        skew=skew,
        kurt=kurt,
        var_95=var_95,
        cvar_95=cvar_95,
        avg_vol=avg_vol,
    )
