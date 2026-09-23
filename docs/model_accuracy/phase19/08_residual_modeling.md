# Phase 19 — Residual Modeling (A6)

## Method
1. OOF Phase 11 predictions generated via 5-fold chronological KFold inside training window.
2. OOF residuals = actual_target − OOF_M0_prediction.
3. Ridge residual model trained on (X_train, OOF_residuals).
4. Final prediction = M0_prediction(eval) + Residual_prediction(eval).

## Leakage Audit
- OOF predictions generated WITHIN training window only (L8 satisfied).
- Eval/test residuals never used to train the residual model (L9 satisfied).
- Residual model hyperparameter (alpha) selected on validation split (L6 satisfied).

## Hypothesis
Phase 11 may systematically under- or over-predict in certain regimes.
A residual model can correct these patterns if they are learnable from historical data.

## Per-Origin Results
| origin | target | rmse | delta | oof_valid | oof_nan |
| --- | --- | --- | --- | --- | --- |
| 2002.0000 | 2003.0000 | 6.0135 | 0.0458 | 400.0000 | 0.0000 |
| 2003.0000 | 2004.0000 | 6.4930 | 0.2385 | 604.0000 | 0.0000 |
| 2004.0000 | 2005.0000 | 4.6572 | 0.4750 | 808.0000 | 0.0000 |
| 2005.0000 | 2006.0000 | 4.2443 | 0.0123 | 1012.0000 | 0.0000 |
| 2006.0000 | 2007.0000 | 3.9283 | 0.0651 | 1216.0000 | 0.0000 |
| 2007.0000 | 2008.0000 | 4.3721 | -0.2594 | 1421.0000 | 0.0000 |
| 2008.0000 | 2009.0000 | 8.1077 | 0.3207 | 1626.0000 | 0.0000 |
| 2009.0000 | 2010.0000 | 4.2442 | -0.0895 | 1834.0000 | 0.0000 |
| 2010.0000 | 2011.0000 | 5.8841 | 0.0256 | 2043.0000 | 0.0000 |
| 2011.0000 | 2012.0000 | 8.5661 | -0.0213 | 2252.0000 | 0.0000 |
| 2012.0000 | 2013.0000 | 5.2318 | 0.1116 | 2460.0000 | 0.0000 |
| 2013.0000 | 2014.0000 | 4.2240 | 0.2904 | 2668.0000 | 0.0000 |
| 2014.0000 | 2015.0000 | 4.6974 | 0.0049 | 2877.0000 | 0.0000 |
| 2015.0000 | 2016.0000 | 3.9647 | 0.0355 | 3087.0000 | 0.0000 |
| 2016.0000 | 2017.0000 | 6.2994 | 0.0987 | 3296.0000 | 0.0000 |
| 2017.0000 | 2018.0000 | 3.2481 | 0.1489 | 3505.0000 | 0.0000 |
| 2018.0000 | 2019.0000 | 3.6349 | 0.1403 | 3715.0000 | 0.0000 |
| 2019.0000 | 2020.0000 | 11.6224 | -0.0422 | 3924.0000 | 0.0000 |
| 2020.0000 | 2021.0000 | 8.2927 | 0.4855 | 4133.0000 | 0.0000 |
| 2021.0000 | 2022.0000 | 7.5282 | 0.3753 | 4342.0000 | 0.0000 |
| 2022.0000 | 2023.0000 | 6.6446 | 0.1419 | 4550.0000 | 0.0000 |
| 2023.0000 | 2024.0000 | 3.8879 | -0.0482 | 4753.0000 | 0.0000 |
| 2024.0000 | 2025.0000 | 3.3188 | 0.3192 | 4952.0000 | 0.0000 |

