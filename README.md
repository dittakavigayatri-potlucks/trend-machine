# Time-Series Momentum (TSMOM) Strategy

Captures persistent cross-asset trends by engineering a TSMOM strategy using 12-month trailing returns as position signals across equities, commodities, and fixed income with ex-ante volatility scaling. Backtested over 20+ years of daily data with a full tearsheet covering Sharpe, Calmar, max drawdown, rolling attribution by asset class, regime-conditional return profiles, and key risk catalysts.

Based on Moskowitz, Ooi & Pedersen (2012): *Time Series Momentum*, Journal of Financial Economics.

## Structure

```
tsmom_strategy/
├── strategy/
│   └── tsmom.py          # Signal construction, vol scaling, backtest engine
├── tearsheet/
│   └── tearsheet.py      # Full performance tearsheet generation
├── outputs/              # CSVs and plots 
├── data/                 # Historical price data 
└── requirements.txt
```

## Methodology

### Signal Construction
- **Lookback**: 12-month trailing return (252 trading days), skipping the most recent day
- **Signal**: `sign(R_{t-252,t-1})` → +1 (long) or -1 (short)
- **Universe**: 18 liquid ETFs spanning equities, commodities, and fixed income

### Volatility Scaling (Ex-Ante)
- Estimate per-asset realized vol using exponentially weighted moving average (60-day span)
- Scale each position to a 40% annualized vol target
- Cap individual leverage at 5x

### Position Construction
```
w_i = signal_i × (target_vol / σ̂_i) / N
```
Portfolio normalized to 15% annualized volatility target.

### Backtest Engine
- Positions entered at next-day open (shift by 1)
- Proportional transaction costs: 10 bps round-trip
- Gross and net P&L computed daily

## Asset Universe

| Asset Class | Tickers |
|---|---|
| Equity | SPY, QQQ, IWM, EFA, EEM, VGK, EWJ |
| Commodity | GLD, SLV, USO, DBA, PDBC |
| Fixed Income | TLT, IEF, AGG, HYG, EMB, BNDX |

## Usage

```bash
# Run strategy and generate core output
python strategy/tsmom.py

# Generate full tearsheet with plots
python tearsheet/tearsheet.py
```

## Key Outputs

| File | Contents |
|---|---|
| `tsmom_daily_pnl.csv` | Daily net P&L series |
| `tsmom_cumulative_return.csv` | Cumulative growth of $1 |
| `tsmom_positions.csv` | Daily position weights by asset |
| `tearsheet_metrics.csv` | Sharpe, Sortino, Calmar, Max DD, Win Rate |
| `regime_conditional.csv` | Bull/Bear split performance |
| `tsmom_tearsheet.png` | Full visual tearsheet |

## Performance Summary (Simulated)

| Metric | Value |
|---|---|
| Annualized Return | ~9–11% |
| Annualized Vol | ~15% |
| Sharpe Ratio | ~0.6–0.8 |
| Max Drawdown | ~-20% to -30% |
| Transaction Costs | 10 bps round-trip |

Actual values depend on random seed and regime simulation. In production, replace simulated data with live historical prices.

## Notes

Simulated data uses a regime-switching model (bull/bear with empirically calibrated transition probabilities) to produce realistic momentum environments. In production, replace `simulate_price_history()` with yfinance, Bloomberg, or CRSP daily price series. Ledoit-Wolf covariance shrinkage is recommended for live covariance estimation.

## Requirements

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scipy>=1.10
```
