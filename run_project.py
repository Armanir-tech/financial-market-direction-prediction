from __future__ import annotations

import pandas as pd

from src.config import DATA_PROCESSED, RESULTS
from src.data import download_all_assets
from src.features import build_feature_table
from src.models import evaluate_classical_models, evaluate_lstm
from src.reporting import (
    save_confusion_matrices,
    save_metric_plots,
    save_price_plots,
    write_final_report,
    write_turkish_report,
)


def main() -> None:
    downloaded = download_all_assets()
    feature_table = build_feature_table(downloaded)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(DATA_PROCESSED / "features.csv", index=False)

    classical_metrics, classical_predictions = evaluate_classical_models(feature_table)
    lstm_metrics, lstm_predictions = evaluate_lstm(feature_table)
    metrics = pd.concat([classical_metrics, lstm_metrics], ignore_index=True)
    predictions = pd.concat([classical_predictions, lstm_predictions], ignore_index=True)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    predictions.to_csv(RESULTS / "predictions.csv", index=False)

    save_price_plots(feature_table)
    save_metric_plots(metrics)
    save_confusion_matrices(metrics, predictions)
    write_final_report(metrics, feature_table)
    write_turkish_report(metrics, feature_table)

    print("Project run completed.")
    print(f"Rows in feature table: {len(feature_table)}")
    print(f"Metric rows: {len(metrics)}")


if __name__ == "__main__":
    main()
