# Phase 19 — Shock Period Analysis

## COVID-19 Shock: Target Year 2020

All models are expected to struggle with the 2020 shock, as COVID-19 was
unprecedented and no pre-2020 feature can capture it.

| candidate | period | rmse | mae | n |
| --- | --- | --- | --- | --- |
| Phase 11 (M0) | Shock (2020) | 11.6646 | 9.0854 | 209 |
| Phase 11 (M0) | Non-Shock | 5.4568 | 3.1069 | 4528 |
| Ridge (A1) | Shock (2020) | 11.9440 | 9.1024 | 209 |
| Ridge (A1) | Non-Shock | 5.4387 | 3.0216 | 4528 |
| Elastic Net (A2) | Shock (2020) | 12.1104 | 9.2733 | 209 |
| Elastic Net (A2) | Non-Shock | 5.4309 | 3.0700 | 4528 |
| Huber (A3) | Shock (2020) | 12.1057 | 9.2558 | 209 |
| Huber (A3) | Non-Shock | 5.3928 | 2.9856 | 4528 |
| Random Forest (A4) | Shock (2020) | 11.9736 | 9.1675 | 209 |
| Random Forest (A4) | Non-Shock | 5.3392 | 2.9778 | 4528 |
| Ensemble Equal (A5a) | Shock (2020) | 11.8385 | 9.1046 | 209 |
| Ensemble Equal (A5a) | Non-Shock | 5.2815 | 2.9141 | 4528 |
| Ensemble Val-Weighted (A5b) | Shock (2020) | 11.6646 | 9.0854 | 209 |
| Ensemble Val-Weighted (A5b) | Non-Shock | 5.4568 | 3.1069 | 4528 |
| Residual Model (A6) | Shock (2020) | 11.6224 | 9.0159 | 209 |
| Residual Model (A6) | Non-Shock | 5.5976 | 3.1780 | 4528 |


> Note: 2020 performance is informative but not a primary selection criterion,
> since no model could plausibly predict a global pandemic from Y-1 features.
