from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .config import CEASEFIRE_START, FIGURES, REPORTS, RESULTS, WAR_START


PERIOD_LABELS = {
    "pre_war_eval": "Pre-war evaluation",
    "war": "War period",
    "post_ceasefire": "Post-ceasefire",
}


def save_metric_plots(metrics: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        return

    for metric in ["accuracy", "f1", "roc_auc"]:
        plot_data = metrics.pivot_table(index=["Asset", "Model"], columns="Period", values=metric, aggfunc="mean")
        plot_data = plot_data.rename(columns=PERIOD_LABELS)
        height = max(5, len(plot_data) * 0.32)
        plt.figure(figsize=(9, height))
        sns.heatmap(plot_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.35, vmax=0.70, linewidths=0.4)
        plt.title(f"{metric.replace('_', ' ').title()} by Asset, Model, and Period")
        plt.tight_layout()
        plt.savefig(FIGURES / f"{metric}_heatmap.png", dpi=180)
        plt.close()


def save_price_plots(feature_table: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ticker, asset_df in feature_table.groupby("Ticker", sort=False):
        asset_df = asset_df.sort_values("Date")
        asset_name = asset_df["Asset"].iloc[0]
        plt.figure(figsize=(10, 4))
        plt.plot(asset_df["Date"], asset_df["Close"], linewidth=1.4)
        plt.axvline(pd.Timestamp(WAR_START), color="#b91c1c", linestyle="--", linewidth=1, label="War start")
        plt.axvline(pd.Timestamp(CEASEFIRE_START), color="#047857", linestyle="--", linewidth=1, label="Ceasefire")
        plt.title(f"{asset_name} close price")
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.legend()
        plt.tight_layout()
        safe_ticker = ticker.replace("^", "").replace("=", "-")
        plt.savefig(FIGURES / f"{safe_ticker}_price.png", dpi=180)
        plt.close()


def write_final_report(metrics: pd.DataFrame, feature_table: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if metrics.empty:
        (REPORTS / "final_report.md").write_text("# Final Report\n\nNo model results were generated.\n", encoding="utf-8")
        return

    best = metrics.sort_values(["Period", "f1", "accuracy"], ascending=[True, False, False]).groupby("Period").head(3)
    period_counts = feature_table.groupby(["Ticker", "period"]).size().unstack(fill_value=0)
    summary = (
        metrics.groupby(["Model", "Period"])[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        .mean()
        .round(3)
        .reset_index()
    )

    text = f"""# Financial Market Direction Prediction During the US-Iran Conflict

## Project Aim

This project predicts whether selected financial assets will close higher on the next trading day. It compares k-NN, Random Forest, and LSTM models across three event-based periods around the 2026 US-Iran conflict:

- Pre-war evaluation: 2026-01-01 to 2026-02-27
- War period: 2026-02-28 to 2026-04-07
- Post-ceasefire: 2026-04-08 to 2026-05-19

Models are trained only on earlier historical observations ending on 2025-12-31, then evaluated on each event period. This avoids random splitting and keeps the time-series order realistic.

## Data

Daily market data is collected from Yahoo Finance with `yfinance`. The analyzed assets are crude oil, gold, S&P 500, Nasdaq Composite, VIX, Bitcoin, and EUR/USD.

Event-date context is based on public reporting and briefings, including Al Jazeera's conflict timeline, Axios' ceasefire report, and the UK House of Commons Library briefing.

## Method

The target variable is binary: `1` if tomorrow's closing price is higher than today's closing price, otherwise `0`. Features include short-term returns, moving-average ratios, rolling volatility, RSI, MACD, volume change, and previous direction.

## Average Model Performance

{summary.to_markdown(index=False)}

## Best Results by Period

{best[["Period", "Asset", "Model", "N", "accuracy", "f1", "roc_auc"]].round(3).to_markdown(index=False)}

## Observation Counts

{period_counts.to_markdown()}

## Interpretation

The project should be interpreted as a direction-prediction experiment under geopolitical stress, not as a trading recommendation. The war and post-ceasefire windows are intentionally short because the project uses up-to-date data. This makes the analysis current, but also means that post-ceasefire results should be read as early market reaction rather than a long-term conclusion.

If accuracy or F1 decreases during the war period, the likely explanation is that geopolitical shocks increase volatility and weaken patterns learned from calmer historical data. If gold, oil, or VIX behave differently from equity indices, that is economically meaningful because these assets are directly connected to safe-haven demand, energy-supply risk, and market fear. A strong Random Forest result would suggest that engineered technical indicators capture nonlinear short-term effects better than distance-based k-NN. If LSTM does not dominate, the likely reason is limited training data and the noisy nature of daily financial direction labels.

## Limitation

Financial direction prediction is inherently difficult because daily prices contain substantial noise. The short event windows are useful for a current geopolitical case study, but they limit statistical certainty. A future extension could update the dataset after more post-ceasefire observations become available.

## Event Context Sources

- Al Jazeera: https://www.aljazeera.com/news/2026/2/28/us-israel-bomb-iran-a-timeline-of-talks-and-threats-leading-up-to-attacks
- Axios: https://www.axios.com/2026/04/07/iran-2-week-ceasfire-trump-pakistan
- UK House of Commons Library: https://commonslibrary.parliament.uk/research-briefings/cbp-10637/
"""
    (REPORTS / "final_report.md").write_text(text, encoding="utf-8")
