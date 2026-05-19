from __future__ import annotations

import pandas as pd
import yfinance as yf

from .config import ASSETS, DATA_RAW, END_DATE, START_DATE


def _normalize_download(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.rename(columns=str.title)
    keep = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in frame.columns]
    frame = frame[keep].copy()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "Date"
    return frame.dropna(subset=["Close"])


def download_asset(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    frame = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No data returned for {ticker}.")
    frame = _normalize_download(frame)
    frame["Ticker"] = ticker
    frame["Asset"] = ASSETS.get(ticker, ticker)
    return frame


def download_all_assets() -> dict[str, pd.DataFrame]:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, pd.DataFrame] = {}
    for ticker in ASSETS:
        frame = download_asset(ticker)
        downloaded[ticker] = frame
        safe_ticker = ticker.replace("^", "").replace("=", "-")
        frame.to_csv(DATA_RAW / f"{safe_ticker}.csv")
    return downloaded
