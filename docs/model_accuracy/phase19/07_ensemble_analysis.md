# Phase 19 — Ensemble Analysis

## A5a: Equal-Weight Ensemble
Weights: M0=1/3, Ridge=1/3, RandomForest=1/3

## A5b: Validation-Selected Weights
Weights determined by SLSQP optimisation on validation-split RMSE.
Test/forecast-origin targets never influence weight selection (L7 satisfied).

## Per-Origin Results
| origin | model | rmse | delta | w_M0 | w_Ridge | w_RF |
| --- | --- | --- | --- | --- | --- | --- |
| 2002 | A5a_Ensemble | 5.6512 | -0.3165 | 1/3 | 1/3 | 1/3 |
| 2002 | A5b_Ensemble | 5.9677 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2003 | A5a_Ensemble | 6.1151 | -0.1394 | 1/3 | 1/3 | 1/3 |
| 2003 | A5b_Ensemble | 6.2545 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2004 | A5a_Ensemble | 3.7787 | -0.4035 | 1/3 | 1/3 | 1/3 |
| 2004 | A5b_Ensemble | 4.1822 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2005 | A5a_Ensemble | 4.0735 | -0.1585 | 1/3 | 1/3 | 1/3 |
| 2005 | A5b_Ensemble | 4.2321 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2006 | A5a_Ensemble | 3.6852 | -0.1779 | 1/3 | 1/3 | 1/3 |
| 2006 | A5b_Ensemble | 3.8632 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2007 | A5a_Ensemble | 4.2570 | -0.3745 | 1/3 | 1/3 | 1/3 |
| 2007 | A5b_Ensemble | 4.6315 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2008 | A5a_Ensemble | 7.6463 | -0.1407 | 1/3 | 1/3 | 1/3 |
| 2008 | A5b_Ensemble | 7.7870 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2009 | A5a_Ensemble | 3.9606 | -0.3730 | 1/3 | 1/3 | 1/3 |
| 2009 | A5b_Ensemble | 4.3337 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2010 | A5a_Ensemble | 5.4507 | -0.4078 | 1/3 | 1/3 | 1/3 |
| 2010 | A5b_Ensemble | 5.8585 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2011 | A5a_Ensemble | 8.1390 | -0.4484 | 1/3 | 1/3 | 1/3 |
| 2011 | A5b_Ensemble | 8.5873 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2012 | A5a_Ensemble | 4.8194 | -0.3008 | 1/3 | 1/3 | 1/3 |
| 2012 | A5b_Ensemble | 5.1203 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2013 | A5a_Ensemble | 3.9241 | -0.0095 | 1/3 | 1/3 | 1/3 |
| 2013 | A5b_Ensemble | 3.9335 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2014 | A5a_Ensemble | 4.6462 | -0.0463 | 1/3 | 1/3 | 1/3 |
| 2014 | A5b_Ensemble | 4.6924 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2015 | A5a_Ensemble | 3.8564 | -0.0728 | 1/3 | 1/3 | 1/3 |
| 2015 | A5b_Ensemble | 3.9293 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2016 | A5a_Ensemble | 6.3074 | 0.1066 | 1/3 | 1/3 | 1/3 |
| 2016 | A5b_Ensemble | 6.2008 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2017 | A5a_Ensemble | 3.0772 | -0.0220 | 1/3 | 1/3 | 1/3 |
| 2017 | A5b_Ensemble | 3.0992 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2018 | A5a_Ensemble | 3.5967 | 0.1022 | 1/3 | 1/3 | 1/3 |
| 2018 | A5b_Ensemble | 3.4945 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2019 | A5a_Ensemble | 11.8385 | 0.1739 | 1/3 | 1/3 | 1/3 |
| 2019 | A5b_Ensemble | 11.6646 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2020 | A5a_Ensemble | 7.3880 | -0.4192 | 1/3 | 1/3 | 1/3 |
| 2020 | A5b_Ensemble | 7.8073 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2021 | A5a_Ensemble | 7.2933 | 0.1404 | 1/3 | 1/3 | 1/3 |
| 2021 | A5b_Ensemble | 7.1529 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2022 | A5a_Ensemble | 6.5970 | 0.0943 | 1/3 | 1/3 | 1/3 |
| 2022 | A5b_Ensemble | 6.5027 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2023 | A5a_Ensemble | 3.7884 | -0.1477 | 1/3 | 1/3 | 1/3 |
| 2023 | A5b_Ensemble | 3.9361 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2024 | A5a_Ensemble | 2.4120 | -0.5875 | 1/3 | 1/3 | 1/3 |
| 2024 | A5b_Ensemble | 2.9995 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |


## Governance
Ensemble weights for A5b were determined **exclusively** on validation data
(last 20% of training years). This satisfies L7 of the leakage audit.
