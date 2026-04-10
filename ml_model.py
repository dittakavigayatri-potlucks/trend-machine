"""
analysis/ml_model.py
Random Forest classifier to predict next-day directional move.
Features: technical indicators + momentum factors.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
import warnings
warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "ret",
    "ret_5d",
    "ret_21d",
    "momentum_12m",
    "ewma_vol",
    "realized_vol",
    "rolling_sharpe",
    "rsi",
    "macd",
    "macd_hist",
    "bb_position",   # where price sits within bollinger band
    "ema_ratio",
    "vol_ratio",
    "ret_skew_21d",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    rets  = df["ret"] if "ret" in df.columns else close.pct_change()

    feat["ret"]        = rets
    feat["ret_5d"]     = close.pct_change(5)
    feat["ret_21d"]    = close.pct_change(21)

    # Copy pre-computed columns if available
    for col in ["momentum_12m", "ewma_vol", "realized_vol", "rolling_sharpe",
                "rsi", "macd", "macd_hist"]:
        if col in df.columns:
            feat[col] = df[col]

    # Bollinger position: 0 = at lower band, 1 = at upper band
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        band_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        feat["bb_position"] = (close - df["bb_lower"]) / band_width

    # EMA ratio
    ema_keys = [c for c in df.columns if c.startswith("ema_") and c != "ema_200"]
    if len(ema_keys) >= 2:
        ema_keys_sorted = sorted(ema_keys, key=lambda x: int(x.split("_")[1]))
        fast = df[ema_keys_sorted[0]]
        slow = df[ema_keys_sorted[-1]]
        feat["ema_ratio"] = (fast / slow.replace(0, np.nan)) - 1

    # Volume ratio (20d vs 60d)
    if "Volume" in df.columns:
        v20 = df["Volume"].rolling(20).mean()
        v60 = df["Volume"].rolling(60).mean()
        feat["vol_ratio"] = (v20 / v60.replace(0, np.nan)) - 1

    # Return skewness over 21 days
    feat["ret_skew_21d"] = rets.rolling(21).skew()

    # Lag features (t-1, t-2)
    for col in ["ret", "rsi", "macd_hist"]:
        if col in feat.columns:
            feat[f"{col}_lag1"] = feat[col].shift(1)
            feat[f"{col}_lag2"] = feat[col].shift(2)

    return feat


def run_ml_model(df: pd.DataFrame, params: dict) -> dict:
    """
    Trains a Random Forest on historical features and returns:
    - prediction: 1 (up) or 0 (down) for next period
    - score: probability of up move
    - accuracy: out-of-sample directional accuracy
    - feature_importance: pd.Series
    """
    feat_df = build_features(df)

    # Target: 1 if next day's return > 0
    target = (df["Close"].pct_change().shift(-1) > 0).astype(int)

    # Align and drop NaN
    combined = pd.concat([feat_df, target.rename("target")], axis=1).dropna()
    if len(combined) < 100:
        return _empty_result()

    available_cols = [c for c in combined.columns if c != "target"]
    X = combined[available_cols]
    y = combined["target"]

    # Time-series cross-val (last fold = most recent)
    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores, prec_scores = [], []

    scaler = StandardScaler()

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5,
            min_samples_leaf=10, random_state=42, n_jobs=-1
        )
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        acc_scores.append(accuracy_score(y_te, preds))
        prec_scores.append(precision_score(y_te, preds, zero_division=0))

    # Final model on all data
    X_all = scaler.fit_transform(X)
    clf_final = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    clf_final.fit(X_all, y)

    # Predict on last available row
    last_feat = feat_df.dropna().iloc[[-1]]
    last_feat_s = scaler.transform(last_feat[available_cols])
    prediction = int(clf_final.predict(last_feat_s)[0])
    proba      = clf_final.predict_proba(last_feat_s)[0]
    score      = float(proba[1])  # prob of up

    # Feature importance
    fi = pd.Series(clf_final.feature_importances_, index=available_cols)

    return dict(
        prediction=prediction,
        score=score,
        accuracy=float(np.mean(acc_scores)),
        precision=float(np.mean(prec_scores)),
        feature_importance=fi,
        n_features=len(available_cols),
        train_size=len(combined),
    )


def _empty_result():
    return dict(
        prediction=0, score=0.5, accuracy=0.5,
        precision=0.5, feature_importance=None,
        n_features=0, train_size=0,
    )
