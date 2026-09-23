# Phase 19 — Architecture Comparison

## Master Result Table
| Model | Mean RMSE | Median RMSE | Mean MAE | Origin Wins | Phase 11 Wins | ΔRMSE |
|---|---|---|---|---|---|---|
| Phase 11 (M0) | 5.4883 | 4.6924 | 3.3604 | 0 | 0 | +0.0000 |
| Ridge (A1) | 5.4709 | 4.5687 | 3.2790 | 11 | 12 | -0.0174 |
| Elastic Net (A2) | 5.4980 | 4.5211 | 3.3333 | 10 | 13 | +0.0097 |
| Huber (A3) | 5.4388 | 4.6796 | 3.2507 | 11 | 12 | -0.0494 |
| Random Forest (A4) | 5.3878 | 4.8107 | 3.2398 | 14 | 9 | -0.1005 |
| Ensemble Equal (A5a) | 5.3175 | 4.6462 | 3.1757 | 18 | 5 | -0.1708 |
| Ensemble Val-Weighted (A5b) | 5.4883 | 4.6924 | 3.3604 | 0 | 0 | +0.0000 |
| Residual Model (A6) | 5.6133 | 4.6974 | 3.4250 | 5 | 18 | +0.1250 |

## Interpretation Guide
- **ΔRMSE < 0**: Candidate beats Phase 11 (lower RMSE is better)
- **ΔRMSE > 0**: Phase 11 beats candidate
- **Origin Wins**: Number of chronological windows where candidate RMSE < Phase 11 RMSE

> A robust improvement requires winning across multiple origins, not just one.
