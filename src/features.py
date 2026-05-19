from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CEASEFIRE_START, FEATURE_COLUMNS, TRAIN_END, WAR_START


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]

    df["return_1d"] = close.pct_change()
    df["return_3d"] = close.pct_change(3)
    df["return_5d"] = close.pct_change(5)

    for window in [5, 10, 20]:
        ma = close.rolling(window).mean()
        df[f"ma_ratio_{window}"] = close / ma - 1
        df[f"volatility_{window}"] = df["return_1d"].rolling(window).std()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["rsi_14"] = _rsi(close)

    if "Volume" in df.columns and df["Volume"].fillna(0).abs().sum() > 0:
        df["volume_change"] = df["Volume"].replace(0, np.nan).pct_change().fillna(0)
    else:
        df["volume_change"] = 0.0

    df["prev_direction"] = (df["return_1d"] > 0).astype(int)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["period"] = df.index.to_series().apply(assign_period)

    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["target"])


def assign_period(date: pd.Timestamp) -> str:
    date = pd.Timestamp(date)
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    if date < pd.Timestamp(WAR_START):
        return "pre_war_eval"
    if date < pd.Timestamp(CEASEFIRE_START):
        return "war"
    return "post_ceasefire"


def build_feature_table(downloaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for ticker, frame in downloaded.items():
        enriched = add_features(frame)
        enriched["Ticker"] = ticker
        frames.append(enriched.reset_index())
    return pd.concat(frames, ignore_index=True)
