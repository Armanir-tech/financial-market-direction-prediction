# Proje Ozeti

## Baslik

Finansal Piyasa Yon Tahmini: ABD-Iran Catisma Donemi Uzerinden k-NN, Random Forest ve LSTM Karsilastirmasi

## Proje Amaci

Bu proje, finansal varliklarin bir sonraki islem gununde yukselip yukselmeyecegini tahmin etmeyi amaclayan bir siniflandirma calismasidir. Calisma, guncel finansal veri kullanarak savas oncesi, savas donemi ve ateskes sonrasi piyasa kosullarinda model performanslarini karsilastirir.

## Veri Kaynagi

Veriler Yahoo Finance uzerinden `yfinance` Python kutuphanesiyle otomatik olarak cekilmistir. Hazir ve statik bir veri dosyasi kullanilmamistir.

## Kullanilan Varliklar

- `CL=F`: Ham petrol
- `GC=F`: Altin
- `^GSPC`: S&P 500
- `^IXIC`: Nasdaq Composite
- `^VIX`: VIX volatilite endeksi
- `BTC-USD`: Bitcoin
- `EURUSD=X`: Euro/Dolar

## Donemler

- Egitim donemi: 2024-01-01 - 2025-12-31
- Savas oncesi test donemi: 2026-01-01 - 2026-02-27
- Savas donemi: 2026-02-28 - 2026-04-07
- Ateskes sonrasi donem: 2026-04-08 - 2026-05-19

## Karsilastirilan Modeller

- k-Nearest Neighbors
- Random Forest
- LSTM
- Momentum Baseline

## Degerlendirme Metrikleri

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Kisa Sonuc

Sonuclar, finansal piyasalarda gunluk yon tahmininin zor oldugunu gostermektedir. Ortalama model performanslari 50% civarina yakin seyretmistir. Bu durum, gunluk finansal fiyat hareketlerinin yuksek gurultu icermesi nedeniyle beklenen bir durumdur.

Savas doneminde altin icin kNN modelinin daha yuksek performans gostermesi dikkat cekicidir. Bu bulgu, altinin belirsizlik donemlerinde guvenli liman olarak daha yonlu davranabilmesiyle iliskilendirilebilir.

Ateskes sonrasi donem kisa oldugu icin bu bolum erken piyasa tepkisi olarak yorumlanmalidir.

## GitHub Linki

https://github.com/Armanir-tech/financial-market-direction-prediction
