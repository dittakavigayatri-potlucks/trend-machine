# Trend Machine — Stock Intelligence Dashboard

> Interactive Streamlit frontend for the TSMOM strategy. Enter any ticker to get a full momentum signal breakdown, ML factor model output, and tearsheet-grade statistics.

---

## Dashboard (New)

### What it does

| Feature | Details |
|---|---|
| **TSMOM Signal** | Long / Short / Flat based on 12-month trailing return (Moskowitz et al. 2012) |
| **EMA Overlay** | Configurable fast/slow EMAs + EMA-200 on candlestick chart |
| **Bollinger Bands** | 20-day, 2σ bands |
| **RSI** | 14-period RSI with overbought/oversold zones |
| **MACD** | 12/26/9 MACD histogram |
| **Ex-Ante Vol** | EWMA volatility + vol-scaled position weight |
| **ML Model** | Random Forest on 14 technical features, time-series CV, feature importance chart |
| **Full Tearsheet** | Sharpe, Sortino, Calmar, Max DD, VaR 95%, CVaR 95%, Win Rate, Skew, Kurtosis |
| **Strategy Insights** | Auto-generated written interpretation of all signals |

### Setup

```bash
# Install dashboard dependencies (separate from core strategy)
pip install -r requirements_dashboard.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Usage

1. Enter any ticker in the sidebar (e.g. `GEV`, `AAPL`, `SPY`, `TLT`)
2. Select a lookback period (1y–5y)
3. Adjust TSMOM parameters (momentum window, EMA spans, vol target)
4. Click **▶ RUN ANALYSIS**

### File structure (dashboard)

```
trend-machine/
├── app.py                      # Streamlit entry point
├── requirements_dashboard.txt  # Dashboard-specific dependencies
└── analysis/
    ├── __init__.py
    ├── signals.py              # EMA, RSI, MACD, Bollinger, TSMOM signal, position weight
    ├── stats.py                # Sharpe, Sortino, Calmar, VaR, CVaR, Skew, Kurtosis
    ├── factors.py              # Momentum factor decomposition
    └── ml_model.py             # Random Forest direction predictor
```

---

*Core TSMOM strategy files (tsmom.py, tearsheet.py) remain unchanged.*
