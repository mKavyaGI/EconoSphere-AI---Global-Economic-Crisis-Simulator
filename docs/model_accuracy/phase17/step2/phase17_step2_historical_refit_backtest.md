# Phase 17 Step 2: Historical Refit Backtest

## Backtest Folds Evaluation

| Fold      | Train Window   | Eval Window   |   Global RMSE |   Priority RMSE |   Guardrail RMSE |   Sample Count |
|:----------|:---------------|:--------------|--------------:|----------------:|-----------------:|---------------:|
| Fold 2018 | 2000-2018      | 2019-2020     |        9.4982 |          6.5278 |           5.6841 |            418 |
| Fold 2020 | 2000-2020      | 2021-2022     |        6.9534 |          1.9714 |           2.0717 |            411 |
| Fold 2022 | 2000-2022      | 2023-2024     |        3.5845 |          0.8978 |           1.1561 |            384 |

## Findings
The expanding historical refit procedure demonstrates consistent metric behavior across historical splits without showing extreme variance.
