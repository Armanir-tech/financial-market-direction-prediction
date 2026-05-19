# Sunum Taslagi

## 1. Proje Konusu

Finansal piyasalarda bir sonraki gun fiyat yonunu tahmin etmek icin k-NN, Random Forest ve LSTM modelleri karsilastirildi.

## 2. Arastirma Sorusu

ABD-Iran catisma sureci etrafinda piyasa yon tahmini modellerinin performansi degisiyor mu?

## 3. Veri Kaynagi

Veriler Yahoo Finance uzerinden `yfinance` kutuphanesiyle otomatik cekildi.

Kullanilan varliklar:

- Petrol
- Altin
- S&P 500
- Nasdaq
- VIX
- Bitcoin
- EUR/USD

## 4. Donemler

- Egitim: 2024-01-01 - 2025-12-31
- Savas oncesi test: 2026-01-01 - 2026-02-27
- Savas donemi: 2026-02-28 - 2026-04-07
- Ateskes sonrasi: 2026-04-08 - 2026-05-19

## 5. Yontem

Hedef degisken:

- `1`: Ertesi gun fiyat artar
- `0`: Ertesi gun fiyat artmaz

Rastgele bolme yerine zaman sirasi korundu. Modeller gecmis veriyle egitildi, daha sonraki donemlerde test edildi.

## 6. Modeller

- k-NN
- Random Forest
- LSTM
- Momentum Baseline

## 7. Bulgular

Genel basarilar 50% civarina yakindir. Bu durum gunluk finansal yon tahmininin zor ve gurultulu bir problem oldugunu gosterir.

En dikkat cekici bulgu: savas doneminde altin icin kNN modeli daha yuksek performans verdi.

## 8. Mantiksal Yorum

Altin savas ve belirsizlik donemlerinde guvenli liman olarak goruldugu icin daha yonlu hareket etmis olabilir. LSTM her durumda daha iyi cikmadi; bunun nedeni veri miktarinin sinirli olmasi ve gunluk fiyat yonunun gurultulu olmasidir.

## 9. Sinirliliklar

Savas ve ateskes sonrasi donemler kisa oldugu icin sonuclar uzun vadeli kesin kararlar olarak yorumlanmamalidir. Bu calisma daha cok guncel piyasa tepkisini analiz eder.

## 10. Sonuc

Proje guncel veri, olay temelli zaman araligi, model karsilastirmasi ve mantiksal yorum icerir. Bu nedenle hem universite odevi hem de GitHub portfolyo projesi olarak kullanilabilir.
