# Project Walkthrough

This walkthrough explains the project flow in a simple order. It is useful for reviewing the assignment, presenting it to the instructor, or explaining the repository in a portfolio context.

## 1. Research Question

The project asks whether machine learning models can predict next-day financial market direction during different geopolitical periods around the US-Iran conflict.

The event windows are:

- Training period: 2024-01-01 to 2025-12-31
- Pre-war evaluation: 2026-01-01 to 2026-02-27
- War period: 2026-02-28 to 2026-04-07
- Post-ceasefire: 2026-04-08 to 2026-05-19

## 2. Data Collection

Data is collected automatically from Yahoo Finance with `yfinance`. The project does not depend on a manually downloaded CSV file.

Main file:

```text
src/data.py
```

The selected assets are crude oil, gold, S&P 500, Nasdaq, VIX, Bitcoin, and EUR/USD.

## 3. Feature Engineering

The raw OHLCV data is transformed into model features.

Main file:

```text
src/features.py
```

Features include:

- Daily returns
- Moving-average ratios
- Rolling volatility
- RSI
- MACD
- Volume change
- Previous-day direction

The target variable is:

```text
1 = tomorrow's close is higher than today's close
0 = tomorrow's close is lower or equal
```

## 4. Model Training

Main file:

```text
src/models.py
```

The compared models are:

- k-NN
- Random Forest
- LSTM
- Momentum baseline

The training process is time-aware. The models are trained only on earlier data and evaluated on later event periods.

## 5. Evaluation

Main outputs:

```text
reports/results/metrics.csv
reports/results/predictions.csv
reports/figures/
```

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrices

## 6. Interpretation

The results should not be interpreted as a trading strategy. The important academic point is that market direction prediction is difficult, especially during short and volatile geopolitical periods.

Gold performing well during the war period is economically meaningful because gold is often considered a safe-haven asset. If LSTM does not consistently outperform classical models, this can be explained by limited training data and noisy daily direction labels.

## 7. How to Re-run

```powershell
python run_project.py
```

This command downloads data, creates features, trains models, evaluates results, and regenerates reports and figures.
