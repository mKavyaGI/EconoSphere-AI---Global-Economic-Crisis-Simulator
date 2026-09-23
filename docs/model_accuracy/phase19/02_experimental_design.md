# Phase 19 — Experimental Design

## Dataset
- Source: `data/processed/master_panel_t1_missingness.csv` (read-only)
- World Bank aggregate rows excluded.
- Target: `gdp_growth_next_year`
- Features: 31 locked Phase 11 features

## T+1 Design
```
Feature year = Y  →  Target year = Y+1
Train: year < Y   →  Eval: year == Y (with valid target labels)
```
This is enforced in code with explicit assertions.

## Valid Origins (Dynamically Determined)
| Feature Year | Target Year | N Train | N Eval |
|---|---|---|---|
| 2001 | 2002 | 200 | 200 |
| 2002 | 2003 | 400 | 204 |
| 2003 | 2004 | 604 | 204 |
| 2004 | 2005 | 808 | 204 |
| 2005 | 2006 | 1012 | 204 |
| 2006 | 2007 | 1216 | 205 |
| 2007 | 2008 | 1421 | 205 |
| 2008 | 2009 | 1626 | 208 |
| 2009 | 2010 | 1834 | 209 |
| 2010 | 2011 | 2043 | 209 |
| 2011 | 2012 | 2252 | 208 |
| 2012 | 2013 | 2460 | 208 |
| 2013 | 2014 | 2668 | 209 |
| 2014 | 2015 | 2877 | 210 |
| 2015 | 2016 | 3087 | 209 |
| 2016 | 2017 | 3296 | 209 |
| 2017 | 2018 | 3505 | 210 |
| 2018 | 2019 | 3715 | 209 |
| 2019 | 2020 | 3924 | 209 |
| 2020 | 2021 | 4133 | 209 |
| 2021 | 2022 | 4342 | 208 |
| 2022 | 2023 | 4550 | 203 |
| 2023 | 2024 | 4753 | 199 |
| 2024 | 2025 | 4952 | 185 |

## Skipped Origins
| Feature Year | Target Year | Reason |
|---|---|---|
| 2000 | 2001 | insufficient training rows (0 < 50) |
| 2025 | 2026 | no evaluation rows with valid targets |

## Candidates
| ID | Name | Architecture |
|---|---|---|
| M0 | Phase 11 (Control) | HistGradientBoosting (Phase 11 exact config) |
| A1 | Ridge | StandardScaler + Ridge (alpha from {0.1,1,10,100}) |
| A2 | Elastic Net | StandardScaler + ElasticNet (alpha × l1_ratio grid) |
| A3 | Huber | StandardScaler + HuberRegressor (epsilon from {1.35,1.5,2.0}) |
| A4 | Random Forest | SimpleImputer + RandomForest (300 trees, max_depth=8) |
| A5a | Ensemble Equal | Average of M0, A1, A4 predictions |
| A5b | Ensemble Val-Weighted | Val-optimised weights (M0, A1, A4) |
| A6 | Residual Model | M0 + Ridge(OOF residuals) |

## Hyperparameter Selection Rule
All hyperparameter selection uses the last 20% of training years as validation.
Test/forecast-origin rows are never inspected during selection.

## Ensemble Weight Rule (A5b)
Weights minimise val-set RMSE using scipy SLSQP (non-negative, sum=1).
Test-set predictions are not used for weight selection.

## Residual Model Rule (A6)
OOF M0 predictions generated via 5-fold chronological KFold inside training window.
Residual = actual_target - OOF_m0_pred.
Ridge residual model trained on those residuals.
Test/eval residuals are never used.
