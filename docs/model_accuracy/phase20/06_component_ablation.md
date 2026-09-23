# Phase 20 — Component Ablation

## Purpose
Does A5a genuinely benefit from architectural diversity?
Does the three-model ensemble outperform any two-model subset?

## Results

| Subset | Mean RMSE | Mean MAE | ΔRMSE vs M0 | Origin Wins |
|---|---|---|---|---|
| M0              | 5.4883 | 3.3604 | +0.0000 | 0/23 |
| Ridge           | 5.4709 | 3.279 | -0.0174 | 11/23 |
| RF              | 5.3878 | 3.2398 | -0.1005 | 14/23 |
| M0+Ridge        | 5.3311 | 3.1899 | -0.1572 | 16/23 |
| M0+RF           | 5.3715 | 3.2402 | -0.1168 | 17/23 |
| Ridge+RF        | 5.3501 | 3.1917 | -0.1382 | 15/23 |
| M0+Ridge+RF     | 5.3175 | 3.1757 | -0.1708 | 18/23 |


## Interpretation
- M0+Ridge+RF should be the best or near-best combination.
- If a two-model subset matches the three-model ensemble, diversity is limited.
- Error cancellation is verified separately in the error diversity report.

> The ablation tests the Phase 19 claim that architectural diversity
> causes partial error cancellation.
