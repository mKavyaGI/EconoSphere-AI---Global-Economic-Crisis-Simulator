# Phase 15 Step 2 Experiment Report

## 1. Executive Summary
The experimental feature engineering compared transformation of extreme scale features (Candidate A), missingness indicators (Candidate B), and relative economic ratios (Candidate C) against the exact reproduced frozen Phase 11 baseline.

## 2. Regression Environment
`PyYAML` was installed and the path context resolved, enabling full regression suites to pass properly.

## 3. Dataset and 2025 Data Investigation
Found 217 rows for year 2025. These lack target variables and are completely unseen. They were not part of Phase 11 training or validation. We will label them as ADDITIONAL UNSEEN HOLDOUT — NOT PART OF AUTHORITATIVE PHASE 11 TEST METRICS.

## 4. Immutability Verification
- Pre-hashes == Post-hashes: **PASS**

## 5. Global Results
| Candidate   | Fold                 | Scope   |   Val_MAE |   Val_RMSE |   Test_MAE |   Test_RMSE |   Dummy_RMSE |   Naive_RMSE |
|:------------|:---------------------|:--------|----------:|-----------:|-----------:|------------:|-------------:|-------------:|
| BASELINE    | Fold 2018 (Original) | Global  |   5.17315 |    8.28527 |    2.28861 |     3.91129 |          nan |          nan |
| CANDIDATE_A | Fold 2018 (Original) | Global  |   5.17315 |    8.28527 |    2.28861 |     3.91129 |          nan |          nan |
| CANDIDATE_B | Fold 2018 (Original) | Global  |   5.1962  |    8.28403 |    2.25303 |     3.88409 |          nan |          nan |
| CANDIDATE_C | Fold 2018 (Original) | Global  |   5.2004  |    8.3568  |    2.1158  |     3.98599 |          nan |          nan |
| BASELINE    | Fold 2020            | Global  |   3.59497 |    6.95337 |    2.37426 |     3.9053  |          nan |          nan |
| CANDIDATE_A | Fold 2020            | Global  |   3.59497 |    6.95337 |    2.37426 |     3.9053  |          nan |          nan |
| CANDIDATE_B | Fold 2020            | Global  |   3.58741 |    6.87339 |    2.42784 |     3.98338 |          nan |          nan |
| CANDIDATE_C | Fold 2020            | Global  |   3.82692 |    7.204   |    2.38047 |     4.15051 |          nan |          nan |

## 6. Priority Countries (Test RMSE)
| Candidate   |      AUS |      BRA |      CAN |     FRA |     GBR |
|:------------|---------:|---------:|---------:|--------:|--------:|
| BASELINE    | 0.790716 | 1.67631  | 0.904201 | 1.08622 | 1.10358 |
| CANDIDATE_A | 0.790716 | 1.67631  | 0.904201 | 1.08622 | 1.10358 |
| CANDIDATE_B | 0.757605 | 1.55169  | 0.680991 | 1.30526 | 1.41215 |
| CANDIDATE_C | 0.649257 | 0.971111 | 0.995975 | 1.03116 | 1.21065 |

## 7. Guardrail Countries (Test RMSE)
| Candidate   |      CHN |     DEU |      IND |      JPN |      USA |
|:------------|---------:|--------:|---------:|---------:|---------:|
| BASELINE    | 0.413154 | 1.65926 | 0.757143 | 1.51811  | 0.568209 |
| CANDIDATE_A | 0.413154 | 1.65926 | 0.757143 | 1.51811  | 0.568209 |
| CANDIDATE_B | 0.457968 | 1.92538 | 0.923168 | 1.40512  | 0.455653 |
| CANDIDATE_C | 0.68547  | 1.41571 | 0.882803 | 0.583428 | 1.0412   |

## 8. Final Recommendation
Based on these results, we must evaluate if Candidate A, B, or C meaningfully improved GBR/BRA/FRA/CAN/AUS without damaging USA/CHN/DEU/JPN/IND. All candidates remain EXPERIMENTAL.