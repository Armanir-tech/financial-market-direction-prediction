from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import EVALUATION_PERIODS, FEATURE_COLUMNS


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def _sklearn_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=8,
                        max_depth=6,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_classical_models(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_rows = []
    for ticker, asset_df in feature_table.groupby("Ticker", sort=False):
        asset_name = asset_df["Asset"].iloc[0]
        train = asset_df[asset_df["period"] == "train"].sort_values("Date")
        if len(train) < 120:
            continue

        x_train = train[FEATURE_COLUMNS]
        y_train = train["target"].astype(int)

        models = _sklearn_models()
        for model_name, model in models.items():
            model.fit(x_train, y_train)

            for period in EVALUATION_PERIODS:
                test = asset_df[asset_df["period"] == period.name].sort_values("Date")
                if len(test) < 5:
                    continue
                x_test = test[FEATURE_COLUMNS]
                y_test = test["target"].astype(int).to_numpy()
                y_pred = model.predict(x_test)
                y_score = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
                metrics = _classification_metrics(y_test, y_pred, y_score)
                rows.append({"Ticker": ticker, "Asset": asset_name, "Model": model_name, "Period": period.name, "N": len(test), **metrics})
                prediction_rows.extend(
                    _prediction_rows(ticker, asset_name, model_name, period.name, test["Date"], y_test, y_pred, y_score)
                )

        for period in EVALUATION_PERIODS:
            test = asset_df[asset_df["period"] == period.name].sort_values("Date")
            if len(test) < 5:
                continue
            y_true = test["target"].astype(int).to_numpy()
            y_pred = test["prev_direction"].astype(int).to_numpy()
            metrics = _classification_metrics(y_true, y_pred)
            rows.append({"Ticker": ticker, "Asset": asset_name, "Model": "Momentum Baseline", "Period": period.name, "N": len(test), **metrics})
            prediction_rows.extend(
                _prediction_rows(ticker, asset_name, "Momentum Baseline", period.name, test["Date"], y_true, y_pred, None)
            )

    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def evaluate_lstm(
    feature_table: pd.DataFrame, sequence_length: int = 20, epochs: int = 20, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception as exc:
        warnings.warn(f"TensorFlow is unavailable; LSTM evaluation skipped: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    tf.keras.utils.set_random_seed(random_state)
    rows = []
    prediction_rows = []

    for ticker, asset_df in feature_table.groupby("Ticker", sort=False):
        asset_df = asset_df.sort_values("Date").reset_index(drop=True)
        asset_name = asset_df["Asset"].iloc[0]
        train = asset_df[asset_df["period"] == "train"]
        if len(train) < sequence_length + 120:
            continue

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(train[FEATURE_COLUMNS])
        all_scaled = scaler.transform(asset_df[FEATURE_COLUMNS])

        train_indices = train.index.to_numpy()
        x_train, y_train = _make_sequences(all_scaled, asset_df["target"].to_numpy(), train_indices, sequence_length)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(sequence_length, len(FEATURE_COLUMNS))),
                keras.layers.LSTM(32, dropout=0.15, recurrent_dropout=0.0),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy")
        model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        for period in EVALUATION_PERIODS:
            indices = asset_df.index[asset_df["period"] == period.name].to_numpy()
            indices = indices[indices >= sequence_length]
            if len(indices) < 5:
                continue
            x_test, y_test = _make_sequences(all_scaled, asset_df["target"].to_numpy(), indices, sequence_length)
            y_score = model.predict(x_test, verbose=0).ravel()
            y_pred = (y_score >= 0.5).astype(int)
            metrics = _classification_metrics(y_test, y_pred, y_score)
            rows.append({"Ticker": ticker, "Asset": asset_name, "Model": "LSTM", "Period": period.name, "N": len(y_test), **metrics})
            dates = asset_df.loc[indices[: len(y_test)], "Date"]
            prediction_rows.extend(_prediction_rows(ticker, asset_name, "LSTM", period.name, dates, y_test, y_pred, y_score))

    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def _make_sequences(features: np.ndarray, target: np.ndarray, end_indices: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for end_idx in end_indices:
        start_idx = end_idx - sequence_length + 1
        if start_idx < 0:
            continue
        x_values.append(features[start_idx : end_idx + 1])
        y_values.append(target[end_idx])
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.int32)


def _prediction_rows(
    ticker: str,
    asset_name: str,
    model_name: str,
    period: str,
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> list[dict[str, object]]:
    scores = y_score if y_score is not None else np.full(len(y_true), np.nan)
    return [
        {
            "Date": pd.Timestamp(date).date().isoformat(),
            "Ticker": ticker,
            "Asset": asset_name,
            "Model": model_name,
            "Period": period,
            "Actual": int(actual),
            "Predicted": int(predicted),
            "Score": float(score) if not np.isnan(score) else np.nan,
        }
        for date, actual, predicted, score in zip(dates, y_true, y_pred, scores)
    ]
