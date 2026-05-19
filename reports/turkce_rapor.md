# Finansal Piyasa Yon Tahmini: ABD-Iran Catisma Donemi Analizi

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

## Donemlere Gore En Iyi Sonuclar

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

## Gozlem Sayilari

| Ticker   |   post_ceasefire |   pre_war_eval |   train |   war |
|:---------|-----------------:|---------------:|--------:|------:|
| BTC-USD  |               41 |             58 |     711 |    39 |
| CL=F     |               29 |             39 |     484 |    26 |
| EURUSD=X |               29 |             41 |     499 |    27 |
| GC=F     |               29 |             39 |     484 |    26 |
| ^GSPC    |               29 |             39 |     482 |    26 |
| ^IXIC    |               29 |             39 |     482 |    26 |
| ^VIX     |               29 |             39 |     482 |    26 |

## Mantiksal Yorum

Sonuclar, finansal piyasalarda gunluk yon tahmininin zor oldugunu gostermektedir. Ortalama basarilarin 50% civarina yakin olmasi beklenebilir bir durumdur; cunku gunluk finansal fiyatlar yuksek gurultu icerir ve kisa vadeli yon hareketleri her zaman belirgin teknik sinyaller tasimaz.

Savas doneminde altin icin kNN modelinin daha yuksek performans gostermesi ekonomik olarak anlamlidir. Altin, jeopolitik belirsizliklerde guvenli liman olarak goruldugu icin savas doneminde daha yonlu ve modellenebilir hareket etmis olabilir. Petrol ve VIX gibi varliklar da kriz donemlerinde dogrudan etkilenebilecek varliklardir; bu nedenle bu varliklarin ayri ayri incelenmesi proje acisindan mantiklidir.

LSTM modelinin her durumda en iyi sonucu vermemesi de beklenebilir. LSTM daha fazla veri ve daha uzun zaman yapilari gerektiren bir modeldir. Bu projede olay donemleri kisa oldugu icin Random Forest veya kNN gibi klasik modeller bazi varliklarda daha dengeli sonuc verebilir.

Ateskes sonrasi donem guncel oldugu icin gozlem sayisi sinirlidir. Bu nedenle bu bolum kesin uzun vadeli sonuc olarak degil, erken piyasa tepkisi olarak degerlendirilmelidir.

## Sonuc

Proje, hocanin istedigi sekilde guncel finansal veri kullanarak savas oncesi, savas donemi ve ateskes sonrasi piyasa yon tahmini performanslarini karsilastirmaktadir. Sonucta yalnizca sayisal metrikler degil, piyasa kosullariyla iliskili mantiksal yorumlar da yapilmistir.
