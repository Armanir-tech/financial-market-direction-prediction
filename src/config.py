from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = REPORTS / "results"

START_DATE = "2024-01-01"
END_DATE = "2026-05-19"

TRAIN_END = "2025-12-31"
PRE_WAR_EVAL_START = "2026-01-01"
WAR_START = "2026-02-28"
CEASEFIRE_START = "2026-04-08"

ASSETS = {
    "CL=F": "Crude Oil Futures",
    "GC=F": "Gold Futures",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^VIX": "CBOE Volatility Index",
    "BTC-USD": "Bitcoin",
    "EURUSD=X": "EUR/USD",
}

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "ma_ratio_5",
    "ma_ratio_10",
    "ma_ratio_20",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "volume_change",
    "prev_direction",
]


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str | None


EVALUATION_PERIODS = [
    Period("pre_war_eval", PRE_WAR_EVAL_START, "2026-02-27"),
    Period("war", WAR_START, "2026-04-07"),
    Period("post_ceasefire", CEASEFIRE_START, None),
]
