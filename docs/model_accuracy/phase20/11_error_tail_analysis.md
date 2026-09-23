# Phase 20 — Error Tail Analysis

## Tail Percentiles

| Model | p90 AE | p95 AE | Max AE | N > 10% |
|---|---|---|---|---|
| M0 | 7.5021 | 10.6249 | 91.0735 | 271 |
| A5a | 7.173 | 10.1295 | 86.7574 | 250 |


## Top 15 Worst Predictions

| Target Year | Country | Model | Actual | Predicted | Abs Error |
|---|---|---|---|---|---|
| 2012 | LBY | M0 | 86.8 | -4.2 | 91.1 |
| 2012 | LBY | A5a | 86.8 | 0.1 | 86.8 |
| 2023 | MAC | A5a | 75.3 | 1.7 | 73.7 |
| 2023 | MAC | M0 | 75.3 | 1.8 | 73.5 |
| 2017 | TCA | A5a | 74.6 | 3.6 | 71.0 |
| 2017 | TCA | M0 | 74.6 | 4.3 | 70.3 |
| 2022 | GUY | M0 | 63.3 | -0.4 | 63.7 |
| 2022 | GUY | A5a | 63.3 | 2.9 | 60.4 |
| 2011 | LBY | M0 | -50.3 | 8.7 | 59.1 |
| 2004 | IRQ | M0 | 53.4 | -4.4 | 57.8 |
| 2020 | MAC | A5a | -54.4 | 3.2 | 57.6 |
| 2011 | LBY | A5a | -50.3 | 5.7 | 56.1 |
| 2020 | MAC | M0 | -54.4 | 1.3 | 55.7 |
| 2012 | SSD | M0 | -46.1 | 8.7 | 54.8 |
| 2004 | IRQ | A5a | 53.4 | -0.6 | 54.0 |


> Outliers are NOT removed. They represent real forecasting failures.
> A model that improves mean RMSE by concentrating large errors in a few
> observations is not considered robust.
