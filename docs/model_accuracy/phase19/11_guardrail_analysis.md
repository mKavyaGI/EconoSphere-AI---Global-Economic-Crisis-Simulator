# Phase 19 — Guardrail Country Analysis

Guardrail countries: USA, CHN, DEU, JPN, IND

A candidate that improves overall RMSE but degrades guardrail countries is NOT robust.

| country | candidate | rmse | mae | delta_rmse | n |
| --- | --- | --- | --- | --- | --- |
| USA | Ridge (A1) | 2.6258 | 2.4059 | 0.1367 | 23 |
| USA | Elastic Net (A2) | 1.9978 | 1.5796 | -0.4913 | 23 |
| USA | Huber (A3) | 2.2620 | 1.9804 | -0.2270 | 23 |
| USA | Random Forest (A4) | 2.1062 | 1.5456 | -0.3828 | 23 |
| USA | Ensemble Equal (A5a) | 2.2307 | 1.8281 | -0.2584 | 23 |
| USA | Ensemble Val-Weighted (A5b) | 2.4891 | 1.9045 | 0.0000 | 23 |
| USA | Residual Model (A6) | 2.4172 | 1.9012 | -0.0719 | 23 |
| USA | Phase 11 (M0) | 2.4891 | 1.9045 | 0.0000 | 23 |
| CHN | Ridge (A1) | 2.6418 | 2.0688 | 0.1943 | 23 |
| CHN | Elastic Net (A2) | 3.0025 | 2.5389 | 0.5550 | 23 |
| CHN | Huber (A3) | 2.1026 | 1.5792 | -0.3449 | 23 |
| CHN | Random Forest (A4) | 2.2616 | 1.6517 | -0.1859 | 23 |
| CHN | Ensemble Equal (A5a) | 2.1910 | 1.5834 | -0.2565 | 23 |
| CHN | Ensemble Val-Weighted (A5b) | 2.4475 | 1.7787 | -0.0000 | 23 |
| CHN | Residual Model (A6) | 1.9947 | 1.5700 | -0.4528 | 23 |
| CHN | Phase 11 (M0) | 2.4475 | 1.7787 | 0.0000 | 23 |
| DEU | Ridge (A1) | 2.5953 | 1.8864 | -0.2265 | 23 |
| DEU | Elastic Net (A2) | 2.9097 | 2.2430 | 0.0879 | 23 |
| DEU | Huber (A3) | 2.5799 | 1.8872 | -0.2419 | 23 |
| DEU | Random Forest (A4) | 2.6174 | 1.8258 | -0.2044 | 23 |
| DEU | Ensemble Equal (A5a) | 2.6383 | 1.9022 | -0.1835 | 23 |
| DEU | Ensemble Val-Weighted (A5b) | 2.8218 | 2.0341 | 0.0000 | 23 |
| DEU | Residual Model (A6) | 2.9064 | 2.1162 | 0.0846 | 23 |
| DEU | Phase 11 (M0) | 2.8218 | 2.0341 | 0.0000 | 23 |
| JPN | Ridge (A1) | 2.6231 | 1.8285 | -0.3389 | 23 |
| JPN | Elastic Net (A2) | 2.8561 | 2.1220 | -0.1059 | 23 |
| JPN | Huber (A3) | 2.6599 | 1.8820 | -0.3021 | 23 |
| JPN | Random Forest (A4) | 2.9102 | 2.0838 | -0.0518 | 23 |
| JPN | Ensemble Equal (A5a) | 2.7347 | 1.8685 | -0.2274 | 23 |
| JPN | Ensemble Val-Weighted (A5b) | 2.9620 | 1.9661 | 0.0000 | 23 |
| JPN | Residual Model (A6) | 2.8203 | 1.9844 | -0.1418 | 23 |
| JPN | Phase 11 (M0) | 2.9620 | 1.9661 | 0.0000 | 23 |
| IND | Ridge (A1) | 3.8361 | 2.9044 | 0.3271 | 23 |
| IND | Elastic Net (A2) | 3.8815 | 3.0312 | 0.3726 | 23 |
| IND | Huber (A3) | 3.5818 | 2.6213 | 0.0728 | 23 |
| IND | Random Forest (A4) | 3.6984 | 2.7479 | 0.1894 | 23 |
| IND | Ensemble Equal (A5a) | 3.6238 | 2.6412 | 0.1149 | 23 |
| IND | Ensemble Val-Weighted (A5b) | 3.5089 | 2.4213 | 0.0000 | 23 |
| IND | Residual Model (A6) | 3.6463 | 2.3207 | 0.1374 | 23 |
| IND | Phase 11 (M0) | 3.5089 | 2.4213 | 0.0000 | 23 |

