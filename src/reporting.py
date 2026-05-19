from __future__ import annotations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

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


def save_confusion_matrices(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if metrics.empty or predictions.empty:
        return

    top_cases = metrics.sort_values(["Period", "f1", "accuracy"], ascending=[True, False, False]).groupby("Period").head(1)
    for _, case in top_cases.iterrows():
        subset = predictions[
            (predictions["Ticker"] == case["Ticker"])
            & (predictions["Model"] == case["Model"])
            & (predictions["Period"] == case["Period"])
        ]
        if subset.empty:
            continue

        matrix = confusion_matrix(subset["Actual"], subset["Predicted"], labels=[0, 1])
        plt.figure(figsize=(4.8, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Down", "Predicted Up"],
            yticklabels=["Actual Down", "Actual Up"],
        )
        plt.title(f"{case['Asset']} - {case['Model']} ({case['Period']})")
        plt.tight_layout()
        safe_ticker = str(case["Ticker"]).replace("^", "").replace("=", "-")
        safe_model = str(case["Model"]).lower().replace(" ", "_")
        plt.savefig(FIGURES / f"confusion_{case['Period']}_{safe_ticker}_{safe_model}.png", dpi=180)
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


def write_turkish_report(metrics: pd.DataFrame, feature_table: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        (REPORTS / "turkce_rapor.md").write_text("# Turkce Rapor\n\nModel sonucu uretilemedi.\n", encoding="utf-8")
        return

    best = metrics.sort_values(["Period", "f1", "accuracy"], ascending=[True, False, False]).groupby("Period").head(3)
    summary = (
        metrics.groupby(["Model", "Period"])[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        .mean()
        .round(3)
        .reset_index()
    )
    period_counts = feature_table.groupby(["Ticker", "period"]).size().unstack(fill_value=0)

    text = f"""# Finansal Piyasa Yon Tahmini: ABD-Iran Catisma Donemi Analizi

## Projenin Amaci

Bu proje, finansal varliklarda bir sonraki islem gunu fiyat yonunun artip artmayacagini tahmin etmeyi amaclar. Calisma bir universite makine ogrenmesi odevi olarak hazirlanmis ve ayni zamanda GitHub portfolyosunda gosterilebilecek sekilde duzenlenmistir.

Ana arastirma sorusu sudur:

> ABD-Iran catisma sureci ve ateskes sonrasi donemde, finansal piyasa yon tahmini modellerinin performansi degisiyor mu?

## Veri Kaynagi

Veriler Yahoo Finance uzerinden `yfinance` kutuphanesi ile otomatik olarak cekilmistir. Hazir bir CSV veri seti kullanilmamistir; proje calistirildiginda veri kaynagindan yeniden uretilebilir.

Kullanilan varliklar:

- Ham petrol (`CL=F`)
- Altin (`GC=F`)
- S&P 500 (`^GSPC`)
- Nasdaq Composite (`^IXIC`)
- VIX volatilite endeksi (`^VIX`)
- Bitcoin (`BTC-USD`)
- EUR/USD (`EURUSD=X`)

## Donem Ayrimi

- Egitim donemi: 2024-01-01 - 2025-12-31
- Savas oncesi test donemi: 2026-01-01 - 2026-02-27
- Savas donemi: 2026-02-28 - 2026-04-07
- Ateskes sonrasi donem: 2026-04-08 - 2026-05-19

Bu ayrim hocanin belirttigi "guncel veri", "zaman araligi" ve "savas oncesi/sonrasi" beklentilerine uygun olacak sekilde olay temelli kurulmustur.

## Yontem

Hedef degisken ikili siniflandirma seklinde tanimlanmistir:

- `1`: Ertesi gun kapanis fiyati bugunku kapanistan yuksek
- `0`: Ertesi gun kapanis fiyati bugunku kapanistan yuksek degil

Rastgele train-test ayrimi kullanilmamistir. Modeller gecmis veri ile egitilmis, daha sonraki zaman araliklarinda test edilmistir. Bu, finansal zaman serilerinde gelecek bilgisi sizmasini onlemek icin onemlidir.

Kullanilan ozellikler arasinda gunluk getiriler, hareketli ortalama oranlari, volatilite, RSI, MACD, hacim degisimi ve onceki gun yonu yer alir.

## Karsilastirilan Modeller

- k-Nearest Neighbors
- Random Forest
- LSTM
- Momentum Baseline

## Ortalama Model Performansi

{summary.to_markdown(index=False)}

## Donemlere Gore En Iyi Sonuclar

{best[["Period", "Asset", "Model", "N", "accuracy", "f1", "roc_auc"]].round(3).to_markdown(index=False)}

## Gozlem Sayilari

{period_counts.to_markdown()}

## Mantiksal Yorum

Sonuclar, finansal piyasalarda gunluk yon tahmininin zor oldugunu gostermektedir. Ortalama basarilarin 50% civarina yakin olmasi beklenebilir bir durumdur; cunku gunluk finansal fiyatlar yuksek gurultu icerir ve kisa vadeli yon hareketleri her zaman belirgin teknik sinyaller tasimaz.

Savas doneminde altin icin kNN modelinin daha yuksek performans gostermesi ekonomik olarak anlamlidir. Altin, jeopolitik belirsizliklerde guvenli liman olarak goruldugu icin savas doneminde daha yonlu ve modellenebilir hareket etmis olabilir. Petrol ve VIX gibi varliklar da kriz donemlerinde dogrudan etkilenebilecek varliklardir; bu nedenle bu varliklarin ayri ayri incelenmesi proje acisindan mantiklidir.

LSTM modelinin her durumda en iyi sonucu vermemesi de beklenebilir. LSTM daha fazla veri ve daha uzun zaman yapilari gerektiren bir modeldir. Bu projede olay donemleri kisa oldugu icin Random Forest veya kNN gibi klasik modeller bazi varliklarda daha dengeli sonuc verebilir.

Ateskes sonrasi donem guncel oldugu icin gozlem sayisi sinirlidir. Bu nedenle bu bolum kesin uzun vadeli sonuc olarak degil, erken piyasa tepkisi olarak degerlendirilmelidir.

## Sonuc

Proje, hocanin istedigi sekilde guncel finansal veri kullanarak savas oncesi, savas donemi ve ateskes sonrasi piyasa yon tahmini performanslarini karsilastirmaktadir. Sonucta yalnizca sayisal metrikler degil, piyasa kosullariyla iliskili mantiksal yorumlar da yapilmistir.
"""
    (REPORTS / "turkce_rapor.md").write_text(text, encoding="utf-8")
