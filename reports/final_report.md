# Financial Market Direction Prediction During the US-Iran Conflict

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

| Model             | Period         |   accuracy |   precision |   recall |    f1 |   roc_auc |
|:------------------|:---------------|-----------:|------------:|---------:|------:|----------:|
| LSTM              | post_ceasefire |      0.577 |       0.546 |    0.398 | 0.453 |     0.59  |
| LSTM              | pre_war_eval   |      0.456 |       0.346 |    0.37  | 0.317 |     0.558 |
| LSTM              | war            |      0.526 |       0.514 |    0.476 | 0.419 |     0.565 |
| Momentum Baseline | post_ceasefire |      0.504 |       0.491 |    0.51  | 0.5   |   nan     |
| Momentum Baseline | pre_war_eval   |      0.492 |       0.517 |    0.507 | 0.512 |   nan     |
| Momentum Baseline | war            |      0.499 |       0.479 |    0.487 | 0.483 |   nan     |
| Random Forest     | post_ceasefire |      0.484 |       0.533 |    0.439 | 0.437 |     0.542 |
| Random Forest     | pre_war_eval   |      0.479 |       0.632 |    0.436 | 0.417 |     0.496 |
| Random Forest     | war            |      0.468 |       0.337 |    0.464 | 0.38  |     0.537 |
| kNN               | post_ceasefire |      0.568 |       0.567 |    0.577 | 0.558 |     0.579 |
| kNN               | pre_war_eval   |      0.448 |       0.524 |    0.445 | 0.423 |     0.45  |
| kNN               | war            |      0.482 |       0.377 |    0.535 | 0.428 |     0.55  |

## Best Results by Period

| Period         | Asset             | Model             |   N |   accuracy |    f1 |   roc_auc |
|:---------------|:------------------|:------------------|----:|-----------:|------:|----------:|
| post_ceasefire | S&P 500           | kNN               |  29 |      0.586 | 0.667 |     0.505 |
| post_ceasefire | Nasdaq Composite  | Momentum Baseline |  29 |      0.552 | 0.667 |   nan     |
| post_ceasefire | Nasdaq Composite  | kNN               |  29 |      0.586 | 0.647 |     0.584 |
| pre_war_eval   | Nasdaq Composite  | kNN               |  39 |      0.59  | 0.704 |     0.511 |
| pre_war_eval   | S&P 500           | LSTM              |  39 |      0.564 | 0.702 |     0.389 |
| pre_war_eval   | S&P 500           | Random Forest     |  39 |      0.564 | 0.679 |     0.505 |
| war            | Gold Futures      | kNN               |  26 |      0.731 | 0.774 |     0.732 |
| war            | Crude Oil Futures | Momentum Baseline |  26 |      0.5   | 0.649 |   nan     |
| war            | S&P 500           | LSTM              |  26 |      0.462 | 0.632 |     0.548 |

## Observation Counts

| Ticker   |   post_ceasefire |   pre_war_eval |   train |   war |
|:---------|-----------------:|---------------:|--------:|------:|
| BTC-USD  |               41 |             58 |     711 |    39 |
| CL=F     |               29 |             39 |     484 |    26 |
| EURUSD=X |               29 |             41 |     499 |    27 |
| GC=F     |               29 |             39 |     484 |    26 |
| ^GSPC    |               29 |             39 |     482 |    26 |
| ^IXIC    |               29 |             39 |     482 |    26 |
| ^VIX     |               29 |             39 |     482 |    26 |

## Interpretation

The project should be interpreted as a direction-prediction experiment under geopolitical stress, not as a trading recommendation. The war and post-ceasefire windows are intentionally short because the project uses up-to-date data. This makes the analysis current, but also means that post-ceasefire results should be read as early market reaction rather than a long-term conclusion.

If accuracy or F1 decreases during the war period, the likely explanation is that geopolitical shocks increase volatility and weaken patterns learned from calmer historical data. If gold, oil, or VIX behave differently from equity indices, that is economically meaningful because these assets are directly connected to safe-haven demand, energy-supply risk, and market fear. A strong Random Forest result would suggest that engineered technical indicators capture nonlinear short-term effects better than distance-based k-NN. If LSTM does not dominate, the likely reason is limited training data and the noisy nature of daily financial direction labels.

## Limitation

Financial direction prediction is inherently difficult because daily prices contain substantial noise. The short event windows are useful for a current geopolitical case study, but they limit statistical certainty. A future extension could update the dataset after more post-ceasefire observations become available.

## Event Context Sources

- Al Jazeera: https://www.aljazeera.com/news/2026/2/28/us-israel-bomb-iran-a-timeline-of-talks-and-threats-leading-up-to-attacks
- Axios: https://www.axios.com/2026/04/07/iran-2-week-ceasfire-trump-pakistan
- UK House of Commons Library: https://commonslibrary.parliament.uk/research-briefings/cbp-10637/
